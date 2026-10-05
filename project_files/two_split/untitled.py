#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-split + bootstrap + alpha-grid + R-replication SVGP calibration.

Outputs:
- indicators_array_R{R}_B{B}.npy  # shape (A, R, B, J) of 0/1 inclusion indicators
- coverage_array_R{R}_B{B}.npy    # shape (A, R, J) = mean over B
- calib_two_split_coverage_R{R}_B{B}.png
- pointwise_coverage_all_alphas_R{R}_B{B}.csv
"""

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
import argparse

import torch
import gpytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")  # for cluster (no display)
import matplotlib.pyplot as plt


# ---------- logging / utils ----------

def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def has_setting(name: str) -> bool:
    return hasattr(gpytorch.settings, name)


def ciq_context():
    """Optional CIQ settings. If cfg.use_ciq=False we just use ExitStack()."""
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


# ---------- data-generating process ----------

def f0(x: torch.Tensor) -> torch.Tensor:
    """True latent function (noiseless)."""
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)


def make_stratified_points_1d(n, low, high, generator, device, dtype):
    """1D stratified (LHS-like) points on [low, high]."""
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + torch.rand(n, generator=generator, device=device, dtype=dtype)) / n
    x = low + (high - low) * u
    perm = torch.randperm(n, generator=generator, device=device)
    return x[perm].unsqueeze(-1)


# ---------- config ----------

@dataclass
class CFG:
    # data
    n_train: int = 500
    n_test: int = 50
    train_low: float = -2.5
    train_high: float = 2.5
    test_low: float = -2.5
    test_high: float = 2.5

    # model / training
    use_fp64: bool = True
    use_ciq: bool = False
    inducing_points: int = 50
    stage1_epochs: int = 2000   # 可以在 NOTS 上再调大
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2
    seed: int = 13

    # early stopping for Stage-1
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    # experiment (defaults; 可以被命令行覆盖)
    alphas: tuple = (1.0, 0.1)
    replications: int = 3
    B_bootstrap: int = 20
    ci_level: float = 0.95

    # logging / plots
    log_epoch: bool = False
    log_every: int = 10
    show_plots: bool = False
    save_plots: bool = True


# ---------- SVGP model + training helpers ----------

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


def freeze_all_params(model, likelihood):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in likelihood.parameters():
        p.requires_grad_(False)


def pick_covariance_params_svgp(model):
    """Select variational covariance parameters only (for Stage-2)."""
    trainables, names = [], []
    vdist_mod = None
    vs = getattr(model, "variational_strategy", None)
    if vs is not None:
        if hasattr(vs, "_variational_distribution"):
            vdist_mod = getattr(vs, "_variational_distribution")
        else:
            cand = getattr(vs, "variational_distribution", None)
            if isinstance(cand, torch.nn.Module):
                vdist_mod = cand
    if isinstance(vdist_mod, torch.nn.Module):
        for n, p in vdist_mod.named_parameters(recurse=True):
            low = n.lower()
            if "mean" in low:
                p.requires_grad_(False)
            elif any(k in low for k in ["chol", "std", "var", "covar", "factor"]):
                p.requires_grad_(True)
                trainables.append(p)
                names.append(f"vdist.{n}")
            else:
                p.requires_grad_(False)
    if not trainables:
        allow = (
            "chol_variational_covar", "variational_distribution.chol_covar",
            "variational_distribution._cholesky_factor", "_variational_distribution.chol_covar",
            "raw_covar", "chol_factor", "covar_factor", "variational_stddev", "raw_var",
            "qchol", "q_scale_tril"
        )
        for n, p in model.named_parameters():
            low = n.lower()
            if "variational_strategy" in low and any(k in low for k in allow) and "mean" not in low:
                p.requires_grad_(True)
                trainables.append(p)
                names.append(n)
    return trainables, names


def stage1_train(model, likelihood, x_train_std, y_train, lr, epochs, jitter,
                 batch_size, early_stop_patience, early_stop_min_delta,
                 log_epoch=False, log_every=10):
    """Stage-1: train kernel + variational mean (alpha=1)."""
    N = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=N, beta=1.0
    )

    likelihood.noise_covar.raw_noise.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": model.variational_parameters()},
        {"params": model.hyperparameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()
    neg_elbo_hist = []
    best_loss = float("inf")
    best_epoch = 0
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
            neg_elbo_hist.append(avg_loss)

            if best_loss - avg_loss > early_stop_min_delta:
                best_loss = avg_loss
                best_epoch = ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())
                patience = 0
            else:
                patience += 1

            if log_epoch and (ep % log_every == 0 or ep == 1 or ep == epochs):
                log(
                    f"[S1] epoch {ep:04d}/{epochs} | "
                    f"-ELBO={avg_loss:.6f} (best {best_loss:.6f}@{best_epoch})"
                )

            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return neg_elbo_hist, best_epoch, best_loss


def stage2_train_cov_only(model, likelihood, x_train_std, y_train, beta, lr,
                          epochs, jitter, batch_size):
    """Stage-2: fix everything, train only variational covariance (beta = 1/alpha)."""
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params, cov_names = pick_covariance_params_svgp(model)
    assert cov_params, "No covariance params found to train in Stage-2."

    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=N, beta=beta
    )
    opt = torch.optim.Adam(cov_params, lr=lr)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()
    neg_elbo_hist = []
    best_loss = float("inf")
    best_epoch = 0
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
            neg_elbo_hist.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return neg_elbo_hist, best_epoch, best_loss, cov_names


# ---------- main calibration routine ----------

def run_two_split_bootstrap_calibration(cfg: CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    log(f"Device: {device} | dtype: {torch.get_default_dtype()}")
    if torch.cuda.is_available():
        log(f"CUDA mem (start): {gpu_mem_mb():.0f} MB")

    log(f"Config: R={cfg.replications}, B={cfg.B_bootstrap}, "
    f"A={len(cfg.alphas)} alphas (min={min(cfg.alphas):.3g}, max={max(cfg.alphas):.3g})")

    # ---- global train/test ----
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

    y_train_full = f0(x_train_raw).squeeze(-1)
    y_test_true = f0(x_test_raw).squeeze(-1)

    # symmetric linear normalization
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

    def to_std(x_raw):
        return (x_raw - mid) / half

    x_train_std_full = to_std(x_train_raw)
    x_test_std = to_std(x_test_raw)

    # fixed inducing points Z
    Z_raw = torch.linspace(
        cfg.train_low,
        cfg.train_high,
        cfg.inducing_points,
        device=device,
        dtype=torch.get_default_dtype(),
    ).unsqueeze(1)
    Z_std = to_std(Z_raw)

    left_std = float(to_std(torch.tensor([[cfg.train_low]], device=device)))
    right_std = float(to_std(torch.tensor([[cfg.train_high]], device=device)))
    log(
        f"Symmetric norm: raw [{cfg.train_low:.2f},{cfg.train_high:.2f}] "
        f"-> std [{left_std:.2f},{right_std:.2f}] (≈ -1,+1)"
    )
    log(f"Z_std head/tail: {float(Z_std[0]):.3f}, {float(Z_std[-1]):.3f}")

    # ---- storage for inclusion indicators: (A, R, B, J) ----
    alphas = list(cfg.alphas)
    A, R, J = len(alphas), cfg.replications, cfg.n_test
    B = cfg.B_bootstrap
    indicators = np.zeros((A, R, B, J), dtype=np.int8)

    z_ci = 1.96  # for 95% CI

    for r in range(R):
        log(f"\n===== Replication {r + 1:03d}/{R} =====")
        n_train = cfg.n_train
        assert n_train % 2 == 0, "n_train must be even for equal split."

        perm = torch.randperm(n_train, device=device)
        half_n = n_train // 2
        split1_idx = perm[:half_n]
        split2_idx = perm[half_n:]

        x_s1_std = x_train_std_full[split1_idx]
        y_s1 = y_train_full[split1_idx]
        x_s2_std = x_train_std_full[split2_idx]
        y_s2 = y_train_full[split2_idx]

        # ---- Split 1: Stage-1 (alpha=1) to produce pseudo-truth ----
        model_s1 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
        lik_s1 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
        with torch.no_grad():
            model_s1.covar_module.base_kernel.lengthscale.fill_(1.0)
            model_s1.covar_module.outputscale.fill_(1.0)
            lik_s1.noise = torch.as_tensor(
                3e-3,
                dtype=torch.get_default_dtype(),
                device=device,
            )
        lik_s1.noise_covar.raw_noise.requires_grad_(False)

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, best_epoch_s1, best_loss_s1 = stage1_train(
                model_s1,
                lik_s1,
                x_s1_std,
                y_s1,
                lr=cfg.lr_adam_stage1,
                epochs=cfg.stage1_epochs,
                jitter=cfg.jitter,
                batch_size=cfg.batch_size,
                early_stop_patience=cfg.early_stop_patience,
                early_stop_min_delta=cfg.early_stop_min_delta,
                log_epoch=cfg.log_epoch,
                log_every=cfg.log_every,
            )

        with torch.no_grad():
            pred_truth = model_s1(x_test_std)
            mu_truth = pred_truth.mean.detach()  # [J]
        log(
            f"[rep {r + 1:03d}] Split1 Stage-1 done | "
            f"best -ELBO={best_loss_s1:.6f}@{best_epoch_s1}"
        )

        # ---- Split 2: Stage-1 base model (alpha=1) for bootstrap ----
        model_s2 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
        lik_s2 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
        with torch.no_grad():
            model_s2.covar_module.base_kernel.lengthscale.fill_(1.0)
            model_s2.covar_module.outputscale.fill_(1.0)
            lik_s2.noise = torch.as_tensor(
                3e-3,
                dtype=torch.get_default_dtype(),
                device=device,
            )
        lik_s2.noise_covar.raw_noise.requires_grad_(False)

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, best_epoch_s2_base, best_loss_s2_base = stage1_train(
                model_s2,
                lik_s2,
                x_s2_std,
                y_s2,
                lr=cfg.lr_adam_stage1,
                epochs=cfg.stage1_epochs,
                jitter=cfg.jitter,
                batch_size=cfg.batch_size,
                early_stop_patience=cfg.early_stop_patience,
                early_stop_min_delta=cfg.early_stop_min_delta,
                log_epoch=cfg.log_epoch,
                log_every=cfg.log_every,
            )
        base_model_s2_sd = clone_state_dict(model_s2.state_dict())
        base_lik_s2_sd = clone_state_dict(lik_s2.state_dict())
        log(
            f"[rep {r + 1:03d}] Split2 Stage-1 base done | "
            f"best -ELBO={best_loss_s2_base:.6f}@{best_epoch_s2_base}"
        )

        # ---- for each alpha, run bootstrap on Split2 ----
        n_s2 = x_s2_std.size(0)
        for a_idx, alpha in enumerate(alphas):
            beta = 1.0 / alpha
            log(f"[rep {r + 1:03d}]   Alpha {alpha:.3f}: bootstrap B={B}")

            for b in range(B):
                # bootstrap indices from Split2
                boot_idx = torch.randint(
                    low=0,
                    high=n_s2,
                    size=(n_s2,),
                    device=device,
                )
                x_boot_std = x_s2_std[boot_idx]
                y_boot = y_s2[boot_idx]

                # reset model to Stage-1 base
                model_s2.load_state_dict(base_model_s2_sd)
                lik_s2.load_state_dict(base_lik_s2_sd)
                model_s2.eval()
                lik_s2.eval()

                # Stage-2 (covariance only) on bootstrap data
                with (ciq_context() if cfg.use_ciq else ExitStack()):
                    _, _, _, _ = stage2_train_cov_only(
                        model_s2,
                        lik_s2,
                        x_boot_std,
                        y_boot,
                        beta=beta,
                        lr=cfg.lr_adam_stage2,
                        epochs=cfg.stage2_epochs,
                        jitter=cfg.jitter,
                        batch_size=cfg.batch_size,
                    )

                # predict at test grid, form CI, store inclusion indicator
                with torch.no_grad():
                    pred_b = model_s2(x_test_std)
                    mu_b = pred_b.mean
                    std_b = pred_b.variance.sqrt().clamp_min(1e-12)
                    lower = mu_b - z_ci * std_b
                    upper = mu_b + z_ci * std_b
                    inc = ((mu_truth >= lower) & (mu_truth <= upper)).to(torch.int8)

                indicators[a_idx, r, b, :] = inc.cpu().numpy()

                if (b + 1) % max(1, (B // 5)) == 0:
                    log(f"        [alpha={alpha:.3f}] bootstrap {b + 1}/{B}")

    # ===== Post-processing: compute coverage curves & save CSV =====
    x_axis = x_test_raw.squeeze(-1).cpu().numpy()  # [J]

    # mean over B (bootstrap) -> (A, R, J)
    coverage = indicators.mean(axis=2)
    # mean over R (replications) -> (A, J)
    coverage_mean_R = coverage.mean(axis=1)
    overall_cov = coverage_mean_R.mean(axis=1)  # [A]

    # coverage curves plot
    plt.figure(figsize=(7.8, 4.4))
    for a_idx, alpha in enumerate(alphas):
        plt.plot(
            x_axis,
            coverage_mean_R[a_idx],
            label=f"α={alpha:.3f} (mean={overall_cov[a_idx]:.3f})",
        )
    plt.hlines(
        cfg.ci_level,
        x_axis.min(),
        x_axis.max(),
        linestyles="--",
        label=f"Nominal {cfg.ci_level:.2f}",
    )
    plt.ylim(0.0, 1.02)
    plt.title(f"Two-split bootstrap calibration | R={R}, B={B}")
    plt.xlabel("x")
    plt.ylabel("Pointwise coverage (mean over reps)")
    plt.legend()
    plt.tight_layout()
    if cfg.save_plots:
        plt.savefig(f"calib_two_split_coverage_R{R}_B{B}.png", dpi=150)
    plt.close()

    # save arrays
    np.save(f"indicators_array_R{R}_B{B}.npy", indicators)
    np.save(f"coverage_array_R{R}_B{B}.npy", coverage)
    log("Saved indicators and coverage arrays.")

    # save CSV with all alphas as columns
    data_all = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        col_name = f"alpha_{alpha:.3f}"
        data_all[col_name] = coverage_mean_R[a_idx]
    df_all = pd.DataFrame(data_all)
    df_all.to_csv(f"pointwise_coverage_all_alphas_R{R}_B{B}.csv", index=False)
    log("Saved pointwise coverage table for all alphas.")

    # small preview in stdout
    log("\nHead of pointwise coverage table:")
    log(df_all.head().to_string(index=False))

    return indicators, coverage, df_all


# ---------- argument parsing & main ----------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-split SVGP bootstrap calibration"
    )
    parser.add_argument(
        "--R", "--replications", dest="replications",
        type=int, default=None,
        help="Number of replications"
    )
    parser.add_argument(
        "--B", "--bootstrap", dest="B_bootstrap",
        type=int, default=None,
        help="Number of bootstrap samples per alpha"
    )
    parser.add_argument(
        "--alphas", type=str, default=None,
        help="Comma-separated list of alphas, e.g. '1.0,0.8,0.5'. "
             "If not provided, use an equally spaced grid."
    )
    parser.add_argument(
        "--alpha-min", type=float, default=None,
        help="Minimum alpha for equally spaced grid (default 0.01)."
    )
    parser.add_argument(
        "--alpha-max", type=float, default=None,
        help="Maximum alpha for equally spaced grid (default 1.0)."
    )
    parser.add_argument(
        "--alpha-num", type=int, default=None,
        help="Number of alphas in equally spaced grid (default 100)."
    )
    return parser.parse_args()



def main():
    cfg = CFG()
    args = parse_args()

    # override from command line if provided
    if args.replications is not None:
        cfg.replications = args.replications
    if args.B_bootstrap is not None:
        cfg.B_bootstrap = args.B_bootstrap

    if args.alphas is not None:
        # 模式 1：显式给 alpha 列表
        alpha_list = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
        cfg.alphas = tuple(alpha_list)
    else:
        # 模式 2：自动生成等距 alpha-grid
        alpha_min = 0.01 if args.alpha_min is None else args.alpha_min
        alpha_max = 1.0  if args.alpha_max is None else args.alpha_max
        alpha_num = 100  if args.alpha_num is None else args.alpha_num
        # 避免 alpha <= 0
        if alpha_min <= 0.0:
            alpha_min = 0.01
        grid = np.linspace(alpha_min, alpha_max, alpha_num).tolist()
        cfg.alphas = tuple(grid)

    run_two_split_bootstrap_calibration(cfg)


if __name__ == "__main__":
    main()
