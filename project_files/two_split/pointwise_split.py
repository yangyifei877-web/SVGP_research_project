#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-replication two-split + bootstrap + alpha-grid SVGP calibration.

For a given rep_id, this script:
  - generates train/test data (using seed + rep_id),
  - does a 2-way split of the training set,
  - trains Stage-1 on split1 to obtain pseudo-truth at test points,
  - trains Stage-1 base on split2,
  - for each alpha in a grid, performs B bootstraps on split2:
        Stage-2 (covariance only) + CI + inclusion indicator vs pseudo-truth,
  - averages over bootstrap samples to get pointwise coverage for this rep,
  - writes a CSV with columns: x, alpha_..., alpha_..., ... to --out-dir.
"""

import os
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
    """Optional CIQ settings (not really used if cfg.use_ciq=False)."""
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
    stage1_epochs: int = 2000
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2
    seed: int = 13           # base seed; actual seed = seed + rep_id

    # early stopping for Stage-1
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    # experiment
    ci_level: float = 0.95

    # logging / plots
    log_epoch: bool = False
    log_every: int = 10
    save_plots: bool = False   # 单 rep 不必画图，减少 IO


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


# ---------- single-rep main routine ----------

def run_single_rep(cfg: CFG,
                   rep_id: int,
                   B_bootstrap: int,
                   alphas: np.ndarray,
                   out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)

    # seed per replication
    seed = cfg.seed + rep_id
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log(f"=== Single replication rep_id={rep_id} | seed={seed} ===")
    log(f"Device: {device} | dtype: {torch.get_default_dtype()}")
    if torch.cuda.is_available():
        log(f"CUDA mem (start): {gpu_mem_mb():.0f} MB")

    # ---- train/test ----
    gen = torch.Generator(device=device).manual_seed(seed)
    x_train_raw = make_stratified_points_1d(
        n=cfg.n_train, low=cfg.train_low, high=cfg.train_high,
        generator=gen, device=device, dtype=torch.get_default_dtype()
    )
    x_test_raw = torch.linspace(
        cfg.test_low, cfg.test_high, cfg.n_test,
        device=device, dtype=torch.get_default_dtype()
    ).unsqueeze(1)

    y_train_full = f0(x_train_raw).squeeze(-1)
    y_test_true = f0(x_test_raw).squeeze(-1)

    # symmetric normalization
    mid = torch.tensor(
        [(cfg.train_low + cfg.train_high) / 2.0],
        device=device, dtype=torch.get_default_dtype()
    )
    half = torch.tensor(
        [(cfg.train_high - cfg.train_low) / 2.0],
        device=device, dtype=torch.get_default_dtype()
    ).clamp_min(1e-12)

    def to_std(x_raw):
        return (x_raw - mid) / half

    x_train_std_full = to_std(x_train_raw)
    x_test_std = to_std(x_test_raw)

    # fixed inducing points
    Z_raw = torch.linspace(
        cfg.train_low, cfg.train_high, cfg.inducing_points,
        device=device, dtype=torch.get_default_dtype()
    ).unsqueeze(1)
    Z_std = to_std(Z_raw)

    left_std = float(to_std(torch.tensor([[cfg.train_low]], device=device)))
    right_std = float(to_std(torch.tensor([[cfg.train_high]], device=device)))
    log(
        f"Symmetric norm: raw [{cfg.train_low:.2f},{cfg.train_high:.2f}] "
        f"-> std [{left_std:.2f},{right_std:.2f}] (≈ -1,+1)"
    )

    # ---- two-way split ----
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

    # ---- Split1 Stage-1 -> pseudo-truth ----
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
        _, best_epoch_s1, best_loss_s1 = stage1_train(
            model_s1, lik_s1, x_s1_std, y_s1,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every
        )

    with torch.no_grad():
        pred_truth = model_s1(x_test_std)
        mu_truth = pred_truth.mean.detach()  # [J]

    log(f"[rep {rep_id:03d}] Split1 Stage-1 done | "
        f"best -ELBO={best_loss_s1:.6f}@{best_epoch_s1}")

    # ---- Split2 Stage-1 base for bootstrap ----
    model_s2 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s2 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s2.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s2.covar_module.outputscale.fill_(1.0)
        lik_s2.noise = torch.as_tensor(
            3e-3, dtype=torch.get_default_dtype(), device=device
        )
    lik_s2.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_s2_base, best_loss_s2_base = stage1_train(
            model_s2, lik_s2, x_s2_std, y_s2,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every
        )

    base_model_s2_sd = clone_state_dict(model_s2.state_dict())
    base_lik_s2_sd = clone_state_dict(lik_s2.state_dict())

    log(f"[rep {rep_id:03d}] Split2 Stage-1 base done | "
        f"best -ELBO={best_loss_s2_base:.6f}@{best_epoch_s2_base}")

    # ---- bootstrap over alphas ----
    A = len(alphas)
    B = B_bootstrap
    J = cfg.n_test
    indicators = np.zeros((A, B, J), dtype=np.int8)

    z_ci = 1.96  # for 95% CI
    n_s2 = x_s2_std.size(0)

    for a_idx, alpha in enumerate(alphas):
        beta = 1.0 / alpha
        log(f"[rep {rep_id:03d}]   Alpha {alpha:.4f}: bootstrap B={B}")

        for b in range(B):
            # bootstrap indices from split2
            boot_idx = torch.randint(
                low=0, high=n_s2, size=(n_s2,), device=device
            )
            x_boot_std = x_s2_std[boot_idx]
            y_boot = y_s2[boot_idx]

            # reset to Stage-1 base
            model_s2.load_state_dict(base_model_s2_sd)
            lik_s2.load_state_dict(base_lik_s2_sd)
            model_s2.eval()
            lik_s2.eval()

            # Stage-2 on bootstrap data
            with (ciq_context() if cfg.use_ciq else ExitStack()):
                _, _, _, _ = stage2_train_cov_only(
                    model_s2, lik_s2, x_boot_std, y_boot,
                    beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                    jitter=cfg.jitter, batch_size=cfg.batch_size
                )

            # prediction + CI
            with torch.no_grad():
                pred_b = model_s2(x_test_std)
                mu_b = pred_b.mean
                std_b = pred_b.variance.sqrt().clamp_min(1e-12)
                lower = mu_b - z_ci * std_b
                upper = mu_b + z_ci * std_b
                inc = ((mu_truth >= lower) & (mu_truth <= upper)).to(torch.int8)

            indicators[a_idx, b, :] = inc.cpu().numpy()

            if (b + 1) % max(1, (B // 5)) == 0:
                log(f"        [alpha={alpha:.4f}] bootstrap {b + 1}/{B}")

    # ---- aggregate over bootstrap & save ----
    x_axis = x_test_raw.squeeze(-1).cpu().numpy()  # [J]
    coverage_B = indicators.mean(axis=1)           # [A, J]

    # save npy
    np.save(
        os.path.join(out_dir,
                     f"indicators_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"),
        indicators
    )
    np.save(
        os.path.join(out_dir,
                     f"coverage_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"),
        coverage_B
    )

    # save CSV table: x + alpha columns
    data = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        col_name = f"alpha_{alpha:.4f}"
        data[col_name] = coverage_B[a_idx]
    df = pd.DataFrame(data)
    csv_path = os.path.join(
        out_dir,
        f"pointwise_coverage_rep{rep_id:03d}_A{A}_B{B}.csv"
    )
    df.to_csv(csv_path, index=False)
    log(f"[rep {rep_id:03d}] Saved CSV to {csv_path}")
    log(df.head().to_string(index=False))


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-rep two-split SVGP bootstrap calibration"
    )
    p.add_argument("--rep-id", type=int, required=True,
                   help="Replication id (e.g., 0..99 in a Slurm array)")
    p.add_argument("--B", "--bootstrap", dest="B_bootstrap",
                   type=int, default=500,
                   help="Number of bootstrap samples per alpha")
    p.add_argument("--alphas", type=str, default=None,
                   help="Comma-separated alpha list, e.g. '1.0,0.8,0.5'. "
                        "If not set, use equally-spaced grid.")
    p.add_argument("--alpha-min", type=float, default=0.01,
                   help="Minimum alpha for grid (default 0.01)")
    p.add_argument("--alpha-max", type=float, default=1.0,
                   help="Maximum alpha for grid (default 1.0)")
    p.add_argument("--alpha-num", type=int, default=100,
                   help="Number of alphas in grid (default 100)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write npy/csv outputs")
    return p.parse_args()


def main():
    cfg = CFG()
    args = parse_args()

    # build alpha-grid
    if args.alphas is not None:
        alpha_list = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
        alphas = np.array(alpha_list, dtype=float)
    else:
        alpha_min = max(args.alpha_min, 1e-4)
        alpha_max = args.alpha_max
        alpha_num = args.alpha_num
        alphas = np.linspace(alpha_min, alpha_max, alpha_num, dtype=float)

    run_single_rep(
        cfg=cfg,
        rep_id=args.rep_id,
        B_bootstrap=args.B_bootstrap,
        alphas=alphas,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
