#!/usr/bin/env python
# coding: utf-8

"""
Two-split, two-stage SVGP calibration script.

硬约束 / 对齐点：
  - 1D toy f0(x) = sin(2πx) + 0.5 cos(3x)，无噪声 latent f。
  - 训练区间 [train_low, train_high]，测试区间 [test_low, test_high]。
  - 对称线性归一化：x_std = (x_raw - mid) / half，把 train 区间映射到 [-1, 1]。
  - Z: 训练区间上等距 M=50 个点，经同一 transform；learn_inducing_locations=False。
  - Stage-1 (alpha=1): 训练 kernel + variational mean (+likelihood)，带 early stopping。
  - Stage-2 (给定 alpha): 固定 Stage-1 的 mean/kernel/likelihood，只训 variational covariance，
    ELBO 的 beta = 1/alpha。
  - calibration: two-split + bootstrap：
      Split-1: Stage-1 (alpha=1) -> Stage-2(beta=1/alpha) 得到 pseudo-truth mu_truth(x*)
      Split-2: Stage-1 anchor (alpha=1)，每次 bootstrap 在此基础上做 Stage-2(beta=1/alpha)
               区间中心是 Split-2 的 posterior mean，
               现在同时比较：
                 (1) mu_truth(x*) 的 coverage（原有）
                 (2) 真实 f0(x*) 的 coverage（新增）

  - 诊断新增：
      * baseline_true_cov_mean / baseline_true_cov_min:
          Split-1 两阶段模型在 x* 上，对真值 f0(x*) 的 95% CI 覆盖率
      * mean_bias_split1 / max_bias_split1:
          |mu_truth - f0(x*)| 的平均/最大
      * mean_std_split1:
          Split-1 模型在 x* 的平均 std
      * mean_std_bootstrap:
          bootstrap CI 的平均 std （= mean_bandwidth / 1.96）
      * ratio_std_boot_over_split1:
          mean_std_bootstrap / mean_std_split1
"""

import math, time, os, argparse
from contextlib import ExitStack
from dataclasses import dataclass

import torch, gpytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# -------- helpers --------
def now(): return time.strftime("%H:%M:%S")
def log(msg): print(f"[{now()}] {msg}", flush=True)
def has_setting(name: str) -> bool: return hasattr(gpytorch.settings, name)

def ciq_context():
    """Optional CIQ settings（默认 use_ciq=False，不会启用）"""
    stack = ExitStack()
    for name, val in [
        ("num_contour_quadrature", 15),
        ("ciq_samples", 16),
        ("ciq_max_iter", 30),
        ("ciq_forward_precision", 0.01),
        ("eval_cg_tolerance", 1e-4),
        ("max_root_decomposition_size", 64),
    ]:
        if has_setting(name):
            stack.enter_context(getattr(gpytorch.settings, name)(val))
    stack.enter_context(gpytorch.settings.fast_pred_var(True))
    return stack

def clone_state_dict(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().clone() if torch.is_tensor(v) else v
    return out

# -------- toy truth (latent) --------
def f0(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)

def make_stratified_points_1d(n, low, high, generator, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + torch.rand(n, generator=generator, device=device, dtype=dtype)) / n
    x = low + (high - low) * u
    perm = torch.randperm(n, device=device)
    return x[perm].unsqueeze(-1)

# -------- config --------
@dataclass
class CFG:
    # data
    n_train: int = 500
    n_test: int = 50
    train_low: float = -2.5
    train_high: float = 2.5
    test_low: float = -2.5
    test_high: float = 2.5

    # model/train
    use_fp64: bool = True
    use_ciq: bool = False
    inducing_points: int = 50
    stage1_epochs: int = 5000
    stage2_epochs: int = 1500
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2
    seed: int = 13

    # early stopping (Stage-1 only)
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

# -------- model --------
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_std: torch.Tensor):
        M = inducing_points_std.size(0)
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(M)
        q_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points_std, q_dist, learn_inducing_locations=False
        )
        super().__init__(q_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x_std):
        mean_x = self.mean_module(x_std)
        covar_x = self.covar_module(x_std)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ---- two-stage param helpers ----
def freeze_all_params(model, likelihood):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in likelihood.parameters():
        p.requires_grad_(False)

def pick_covariance_params_svgp(model):
    """只选 variational covariance 相关参数，不动 mean/kernel/likelihood。"""
    trainables = []
    vs = getattr(model, "variational_strategy", None)
    vdist_mod = None
    if vs is not None:
        vdist_mod = getattr(vs, "_variational_distribution", None)
        if vdist_mod is None:
            cand = getattr(vs, "variational_distribution", None)
            if isinstance(cand, torch.nn.Module):
                vdist_mod = cand
    if isinstance(vdist_mod, torch.nn.Module):
        for n, p in vdist_mod.named_parameters(recurse=True):
            low = n.lower()
            if "mean" in low:
                p.requires_grad_(False)
            elif any(k in low for k in ["chol", "std", "var", "covar", "factor"]):
                p.requires_grad_(True); trainables.append(p)
            else:
                p.requires_grad_(False)
    if not trainables:
        for n, p in model.named_parameters():
            low = n.lower()
            if (
                "variational_strategy" in low
                and any(
                    k in low
                    for k in [
                        "chol_variational_covar",
                        "raw_covar",
                        "chol_factor",
                        "covar_factor",
                        "variational_stddev",
                        "raw_var",
                        "qchol",
                        "q_scale_tril",
                    ]
                )
                and "mean" not in low
            ):
                p.requires_grad_(True); trainables.append(p)
    return trainables

# ---- Stage-1 ----
def stage1_train(
    model,
    likelihood,
    x_train_std,
    y_train,
    lr,
    epochs,
    jitter,
    batch_size,
    early_stop_patience,
    early_stop_min_delta,
):
    """
    Stage-1: alpha=1，训练 kernel + variational mean (+likelihood)，带 early stopping。
    """
    N = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=1.0)

    likelihood.noise_covar.raw_noise.requires_grad_(False)
    opt = torch.optim.Adam(
        [
            {"params": model.variational_parameters()},
            {"params": model.hyperparameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    loader = DataLoader(
        TensorDataset(x_train_std, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()
    best_loss, best_epoch = float("inf"), 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())
    patience = 0

    with gpytorch.settings.cholesky_jitter(jitter):
        for ep in range(1, epochs + 1):
            total, n_batches = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = -mll(out, yb)
                if loss.numel() > 1:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += float(loss.item())
                n_batches += 1
            avg_loss = total / max(1, n_batches)
            if best_loss - avg_loss > early_stop_min_delta:
                best_loss, best_epoch = avg_loss, ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_epoch, best_loss

# ---- Stage-2 (covariance only) ----
def stage2_train_cov_only(
    model,
    likelihood,
    x_train_std,
    y_train,
    beta,
    lr,
    epochs,
    jitter,
    batch_size,
):
    """
    Stage-2: 固定 Stage-1 的 mean/kernel/likelihood，只训练 variational covariance。
    ELBO 的 beta = 1/alpha。
    """
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params = pick_covariance_params_svgp(model)
    assert cov_params, "No covariance params found to train in Stage-2."
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=beta)
    opt = torch.optim.Adam(cov_params, lr=lr)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()
    best_loss, best_epoch = float("inf"), 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())

    with gpytorch.settings.cholesky_jitter(jitter):
        for ep in range(1, epochs + 1):
            total, n_batches = 0.0, 0
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = -mll(out, yb)
                if loss.numel() > 1:
                    loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cov_params, 5.0)
                opt.step()
                total += float(loss.item())
                n_batches += 1
            avg_loss = total / max(1, n_batches)
            if avg_loss < best_loss:
                best_loss, best_epoch = avg_loss, ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_epoch, best_loss

# ---- calibration (single alpha, two-split, B bootstraps) ----
def calibrate_alpha_once(
    alpha=0.5,
    B=200,
    seed=2025,
    save_curve_csv=True,
    save_summary_csv=True,
    verbose=True,
):
    """
    完整 two-split, two-stage calibration（单个 alpha）：

    Split-1:
      Stage-1(alpha=1)  -> 训练 mean + kernel
      Stage-2(beta=1/alpha, cov-only) -> 得到 pseudo-truth mu_truth(x*)

      诊断：对真值 f0(x*) 构造 CI，看 baseline_true_cov_mean (~ baseline coverage)

    Split-2:
      Stage-1(alpha=1)  -> anchor 模型
      对 b=1..B:
        - 在 Split-2 上 bootstrap
        - 从 anchor state 加载，做 Stage-2(beta=1/alpha, cov-only)
        - 用 95% CI 覆盖：
             (1) mu_truth(x*)         -> mean_coverage, min_coverage
             (2) 真实 y_test_true(x*) -> mean_coverage_true, min_coverage_true
    """
    beta = 1.0 / alpha
    cfg = CFG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)

    # global seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # === 1) global data + symmetric norm + Z/X* ===
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    x_train_raw = make_stratified_points_1d(
        n=cfg.n_train,
        low=cfg.train_low,
        high=cfg.train_high,
        generator=gen,
        device=device,
        dtype=torch.get_default_dtype(),
    )
    x_test_raw = torch.linspace(
        cfg.test_low,
        cfg.test_high,
        cfg.n_test,
        device=device,
        dtype=torch.get_default_dtype(),
    ).unsqueeze(1)
    # 真正的 noiseless latent truth
    y_test_true = f0(x_test_raw).squeeze(-1)

    mid = torch.tensor(
        [(cfg.train_low + cfg.train_high) / 2.0],
        device=device,
        dtype=torch.get_default_dtype(),
    )
    half = torch.tensor(
        [(cfg.train_high - cfg.train_low) / 2.0],
        device=device,
        dtype=torch.get_default_dtype(),
    ).clamp_min(1e-12)

    def to_std(x_raw): return (x_raw - mid) / half

    x_train_std_full = to_std(x_train_raw)
    x_test_std = to_std(x_test_raw)

    Z_raw = torch.linspace(
        cfg.train_low,
        cfg.train_high,
        cfg.inducing_points,
        device=device,
        dtype=torch.get_default_dtype(),
    ).unsqueeze(1)
    Z_std = to_std(Z_raw)

    y_train_full = f0(x_train_raw).squeeze(-1)

    # === 2) equal split: 250 / 250 ===
    perm = torch.randperm(cfg.n_train, device=device)
    idx1 = perm[: cfg.n_train // 2]
    idx2 = perm[cfg.n_train // 2 : cfg.n_train]

    x1, y1 = x_train_std_full[idx1], y_train_full[idx1]
    x2, y2 = x_train_std_full[idx2], y_train_full[idx2]

    # === 3) Split-1: full two-stage => mu_truth ===
    model_s1 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s1 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s1.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s1.covar_module.outputscale.fill_(1.0)
        lik_s1.noise = torch.as_tensor(
            3e-3, dtype=torch.get_default_dtype(), device=device
        )
    lik_s1.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _be1_s1, _bl1_s1 = stage1_train(
            model_s1,
            lik_s1,
            x_train_std=x1,
            y_train=y1,
            lr=cfg.lr_adam_stage1,
            epochs=cfg.stage1_epochs,
            jitter=cfg.jitter,
            batch_size=min(cfg.batch_size, x1.size(0)),
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _be1_s2, _bl1_s2 = stage2_train_cov_only(
            model_s1,
            lik_s1,
            x_train_std=x1,
            y_train=y1,
            beta=beta,
            lr=cfg.lr_adam_stage2,
            epochs=cfg.stage2_epochs,
            jitter=cfg.jitter,
            batch_size=min(cfg.batch_size, x1.size(0)),
        )

    # --- Split-1 两阶段模型在 x* 上的 posterior，用于：
    #     (1) pseudo-truth mu_truth
    #     (2) 对真值 f0(x*) 的 baseline coverage 诊断
    with torch.no_grad():
        pf1 = model_s1(x_test_std)
        mu_truth = pf1.mean.detach()
        std_truth = pf1.variance.sqrt().clamp_min(1e-12)

        # baseline coverage: 对真实 f0(x*) 的 95% CI
        lower1, upper1 = mu_truth - 1.96 * std_truth, mu_truth + 1.96 * std_truth
        cover_true_split1 = ((y_test_true >= lower1) & (y_test_true <= upper1)).to(torch.float64)
        baseline_true_cov_mean = float(cover_true_split1.mean().item())
        baseline_true_cov_min  = float(cover_true_split1.min().item())

        bias_split1 = (mu_truth - y_test_true).abs()
        mean_bias_split1 = float(bias_split1.mean().item())
        max_bias_split1  = float(bias_split1.max().item())
        mean_std_split1  = float(std_truth.mean().item())

    if verbose:
        log(
            f"[Split-1 α={alpha:.3f}] baseline_true_cov_mean={baseline_true_cov_mean:.4f}, "
            f"mean_std_split1={mean_std_split1:.4f}, mean_bias={mean_bias_split1:.4e}"
        )

    # === 4) Split-2: Stage-1 anchor ===
    model_anchor = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_anchor = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_anchor.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_anchor.covar_module.outputscale.fill_(1.0)
        lik_anchor.noise = torch.as_tensor(
            3e-3, dtype=torch.get_default_dtype(), device=device
        )
    lik_anchor.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _be2_s1, _bl2_s1 = stage1_train(
            model_anchor,
            lik_anchor,
            x_train_std=x2,
            y_train=y2,
            lr=cfg.lr_adam_stage1,
            epochs=cfg.stage1_epochs,
            jitter=cfg.jitter,
            batch_size=min(cfg.batch_size, x2.size(0)),
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
        )

    s2_model_sd = clone_state_dict(model_anchor.state_dict())
    s2_lik_sd = clone_state_dict(lik_anchor.state_dict())

    # === 5) Bootstrap on Split-2 with Stage-2 (cov-only) ===
    J = x_test_std.size(0)
    indicators_pseudo = np.zeros((B, J), dtype=np.float64)  # vs mu_truth
    indicators_true  = np.zeros((B, J), dtype=np.float64)   # vs f0(x)
    avg_bandwidth = np.zeros(B, dtype=np.float64)

    idx_all = torch.arange(x2.size(0), device=device)
    for b in range(B):
        if verbose and (b % 20 == 0 or b == B - 1):
            log(f"[alpha={alpha:.3f}] bootstrap {b+1}/{B}")

        boot_idx = idx_all[
            torch.randint(
                low=0,
                high=x2.size(0),
                size=(x2.size(0),),
                device=device,
            )
        ]
        xb, yb = x2[boot_idx], y2[boot_idx]

        model_b = SVGPModel(Z_std.clone().contiguous()).to(device=device)
        lik_b = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
        model_b.load_state_dict(s2_model_sd)
        lik_b.load_state_dict(s2_lik_sd)

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _be_s2, _bl_s2 = stage2_train_cov_only(
                model_b,
                lik_b,
                x_train_std=xb,
                y_train=yb,
                beta=beta,
                lr=cfg.lr_adam_stage2,
                epochs=cfg.stage2_epochs,
                jitter=cfg.jitter,
                batch_size=min(cfg.batch_size, xb.size(0)),
            )

        with torch.no_grad():
            pf = model_b(x_test_std)
            mu_b = pf.mean
            std_b = pf.variance.sqrt().clamp_min(1e-12)
            lower, upper = mu_b - 1.96 * std_b, mu_b + 1.96 * std_b

            cover_pseudo = ((mu_truth    >= lower) & (mu_truth    <= upper)).to(torch.float64)
            cover_true   = ((y_test_true >= lower) & (y_test_true <= upper)).to(torch.float64)

            indicators_pseudo[b, :] = cover_pseudo.cpu().numpy()
            indicators_true[b, :]   = cover_true.cpu().numpy()
            avg_bandwidth[b] = float((1.96 * std_b).mean().item())

    # === 6) aggregate ===
    C_point_pseudo = indicators_pseudo.mean(axis=0)
    C_point_true   = indicators_true.mean(axis=0)

    C_mean_pseudo = float(C_point_pseudo.mean())
    C_min_pseudo  = float(C_point_pseudo.min())

    C_mean_true = float(C_point_true.mean())
    C_min_true  = float(C_point_true.min())

    bw_mean = float(avg_bandwidth.mean())
    mean_std_bootstrap = bw_mean / 1.96
    ratio_std_boot_over_split1 = (
        mean_std_bootstrap / mean_std_split1 if mean_std_split1 > 0 else float("nan")
    )

    xs = x_test_raw.squeeze(-1).cpu().numpy()
    tag = f"calib_alpha{alpha:.3f}_B{B}"

    if save_curve_csv:
        pd.DataFrame({"x": xs, "coverage_pseudo": C_point_pseudo}).to_csv(
            f"{tag}_coverage_curve_pseudo.csv", index=False
        )
        pd.DataFrame({"x": xs, "coverage_true": C_point_true}).to_csv(
            f"{tag}_coverage_curve_true.csv", index=False
        )

    if save_summary_csv:
        pd.DataFrame(
            [
                {
                    "alpha": alpha,
                    "B": B,
                    "mean_coverage": C_mean_pseudo,      # vs mu_truth（保持原字段名）
                    "min_coverage":  C_min_pseudo,
                    "mean_coverage_true": C_mean_true,  # 新增：vs f0(x)
                    "min_coverage_true":  C_min_true,
                    "baseline_true_cov_mean": baseline_true_cov_mean,
                    "baseline_true_cov_min":  baseline_true_cov_min,
                    "mean_std_split1": mean_std_split1,
                    "mean_bias_split1": mean_bias_split1,
                    "max_bias_split1":  max_bias_split1,
                    "mean_std_bootstrap": mean_std_bootstrap,
                    "ratio_std_boot_over_split1": ratio_std_boot_over_split1,
                    "mean_bandwidth": bw_mean,
                }
            ]
        ).to_csv(f"{tag}_summary.csv", index=False)

    if verbose:
        log(
            f"[RESULT {tag}] "
            f"mean_cov_pseudo={C_mean_pseudo:.4f}, "
            f"mean_cov_true={C_mean_true:.4f}, "
            f"baseline_true_cov={baseline_true_cov_mean:.4f}, "
            f"mean_std_split1={mean_std_split1:.4f}, "
            f"mean_std_boot={mean_std_bootstrap:.4f}, "
            f"ratio_std_boot/split1={ratio_std_boot_over_split1:.3f}, "
            f"mean_bw={bw_mean:.4f}"
        )

    return {
        "alpha": alpha,
        "B": B,
        "x": xs,
        "coverage_curve": C_point_pseudo,      # 保留原字段（vs mu_truth）
        "coverage_curve_true": C_point_true,   # 新增（vs f0）
        "mean_coverage": C_mean_pseudo,
        "min_coverage": C_min_pseudo,
        "mean_coverage_true": C_mean_true,
        "min_coverage_true": C_min_true,
        "baseline_true_cov_mean": baseline_true_cov_mean,
        "baseline_true_cov_min":  baseline_true_cov_min,
        "mean_std_split1": mean_std_split1,
        "mean_bias_split1": mean_bias_split1,
        "max_bias_split1":  max_bias_split1,
        "mean_std_bootstrap": mean_std_bootstrap,
        "ratio_std_boot_over_split1": ratio_std_boot_over_split1,
        "mean_bandwidth": bw_mean,
    }

# === driver for many replications of one alpha ===
def run_replications_for_one_alpha(
    alpha: float,
    B: int = 200,
    R: int = 100,
    base_seed: int = 2025,
    out_dir: str = "C:/users/yangy/calibration",  # 默认保存到你指定的目录
    tag: str = "calib",
    verbose: bool = True,
    save_curves: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    all_curves = []

    for r in range(R):
        seed_r = base_seed + r
        if verbose:
            log(f"[RUN] alpha={alpha:.3f}, rep={r+1}/{R}, seed={seed_r}")

        res = calibrate_alpha_once(
            alpha=alpha,
            B=B,
            seed=seed_r,
            save_curve_csv=False,
            save_summary_csv=False,
            verbose=verbose,
        )

        row = {
            "alpha": alpha,
            "B": B,
            "rep": r,
            "seed": seed_r,
            # pseudo truth (mu_truth)
            "mean_coverage": res["mean_coverage"],
            "min_coverage": res["min_coverage"],
            # true f0
            "mean_coverage_true": res["mean_coverage_true"],
            "min_coverage_true": res["min_coverage_true"],
            # baseline (Split-1，两阶段，对真值)
            "baseline_true_cov_mean": res["baseline_true_cov_mean"],
            "baseline_true_cov_min":  res["baseline_true_cov_min"],
            # bias / std 诊断
            "mean_std_split1": res["mean_std_split1"],
            "mean_bias_split1": res["mean_bias_split1"],
            "max_bias_split1":  res["max_bias_split1"],
            "mean_std_bootstrap": res["mean_std_bootstrap"],
            "ratio_std_boot_over_split1": res["ratio_std_boot_over_split1"],
            # overall band
            "mean_bandwidth": res["mean_bandwidth"],
        }
        rows.append(row)

        if save_curves:
            all_curves.append(res["coverage_curve"])

    df = pd.DataFrame(rows)
    summary_path = os.path.join(
        out_dir, f"{tag}_alpha{alpha:.3f}_R{R}_summary_reps.csv"
    )
    df.to_csv(summary_path, index=False)
    log(f"[SAVE] summaries -> {summary_path}")

    agg = {
        "alpha": alpha,
        "B": B,
        "R": R,
        "mean_coverage_mean": float(df["mean_coverage"].mean()),
        "mean_coverage_std":  float(df["mean_coverage"].std(ddof=1)),
        "mean_coverage_true_mean": float(df["mean_coverage_true"].mean()),
        "mean_coverage_true_std":  float(df["mean_coverage_true"].std(ddof=1)),
        "baseline_true_cov_mean_mean": float(df["baseline_true_cov_mean"].mean()),
        "baseline_true_cov_mean_std":  float(df["baseline_true_cov_mean"].std(ddof=1)),
        "min_coverage_mean": float(df["min_coverage"].mean()),
        "min_coverage_true_mean": float(df["min_coverage_true"].mean()),
        "min_coverage_min": float(df["min_coverage"].min()),
        "mean_bandwidth_mean": float(df["mean_bandwidth"].mean()),
        # std / bias 对比
        "mean_std_split1_mean": float(df["mean_std_split1"].mean()),
        "mean_std_bootstrap_mean": float(df["mean_std_bootstrap"].mean()),
        "ratio_std_boot_over_split1_mean": float(df["ratio_std_boot_over_split1"].mean()),
        "mean_bias_split1_mean": float(df["mean_bias_split1"].mean()),
        "max_bias_split1_mean":  float(df["max_bias_split1"].mean()),
    }
    agg_df = pd.DataFrame([agg])
    agg_path = os.path.join(
        out_dir, f"{tag}_alpha{alpha:.3f}_R{R}_summary_over_reps.csv"
    )
    agg_df.to_csv(agg_path, index=False)
    log(f"[SAVE] per-alpha summary (over reps) -> {agg_path}")

    if save_curves and len(all_curves) > 0:
        curves_arr = np.stack(all_curves, axis=0)
        curves_path = os.path.join(
            out_dir, f"{tag}_alpha{alpha:.3f}_R{R}_coverage_curves.npy"
        )
        np.save(curves_path, curves_arr)
        log(f"[SAVE] coverage curves -> {curves_path}")

    return df

# === sweep over an alpha grid ===
def run_alpha_grid(
    B: int = 200,
    R: int = 100,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    alpha_num: int = 100,
    base_seed: int = 2025,
    out_dir: str = "C:/users/yangy/calibration",
    tag: str = "calib_grid",
    verbose: bool = True,
    save_curves: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    alpha_grid = np.linspace(alpha_min, alpha_max, alpha_num, endpoint=True)
    big_rows = []

    for a in alpha_grid:
        df_alpha = run_replications_for_one_alpha(
            alpha=float(a),
            B=B,
            R=R,
            base_seed=base_seed,
            out_dir=out_dir,
            tag=tag,
            verbose=verbose,
            save_curves=save_curves,
        )
        big_rows.append(df_alpha)

    df_all = pd.concat(big_rows, ignore_index=True)
    grid_path = os.path.join(out_dir, f"{tag}_alpha_grid_summary_all.csv")
    df_all.to_csv(grid_path, index=False)
    log(f"[SAVE] full alpha-grid summary -> {grid_path}")

    return df_all

# === argument parser & main ===
def parse_args():
    p = argparse.ArgumentParser(
        description="Two-split, two-stage SVGP calibration (B bootstraps, R replications)."
    )
    p.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "single_alpha"],
        help="grid: sweep many alphas; single_alpha: only one alpha with R replications.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha used in single_alpha mode.",
    )
    p.add_argument(
        "--B",
        type=int,
        default=200,
        help="Number of bootstraps per replication.",
    )
    p.add_argument(
        "--R",
        type=int,
        default=100,
        help="Number of replications for each alpha.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=2025,
        help="Base random seed; actual seed = base_seed + rep.",
    )
    p.add_argument(
        "--alpha-min",
        type=float,
        default=0.05,
        help="Minimum alpha in grid mode.",
    )
    p.add_argument(
        "--alpha-max",
        type=float,
        default=0.95,
        help="Maximum alpha in grid mode.",
    )
    p.add_argument(
        "--alpha-num",
        type=int,
        default=100,
        help="Number of alpha points in grid mode.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="C:/users/yangy/calibration",  # 本地 Windows 默认；NOTS 上用参数覆盖
        help="Directory to save csv/npy outputs.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="calib",
        help="Tag prefix for output file names.",
    )
    p.add_argument(
        "--no-curves",
        action="store_true",
        help="If set, do NOT save coverage curves npy (only summaries).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="If set, reduce logging.",
    )
    return p.parse_args()

def main():
    args = parse_args()
    verbose = not args.quiet
    save_curves = not args.no_curves

    if args.mode == "single_alpha":
        log(
            f"[MAIN] mode=single_alpha, alpha={args.alpha:.3f}, "
            f"B={args.B}, R={args.R}"
        )
        run_replications_for_one_alpha(
            alpha=args.alpha,
            B=args.B,
            R=args.R,
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            tag=args.tag,
            verbose=verbose,
            save_curves=save_curves,
        )
    else:
        log(
            f"[MAIN] mode=grid, B={args.B}, R={args.R}, "
            f"alpha_min={args.alpha_min}, alpha_max={args.alpha_max}, "
            f"alpha_num={args.alpha_num}"
        )
        run_alpha_grid(
            B=args.B,
            R=args.R,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_num=args.alpha_num,
            base_seed=args.base_seed,
            out_dir=args.out_dir,
            tag=args.tag,
            verbose=verbose,
            save_curves=save_curves,
        )

if __name__ == "__main__":
    main()
