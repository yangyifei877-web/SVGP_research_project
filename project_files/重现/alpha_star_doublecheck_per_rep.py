#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-rep double-check for data-dependent alpha*(r, x_j) using two-stage SVGP.

For a given rep_id, this script:
  - reads that rep's alpha-dictionary CSV produced by two-split calibration,
  - for each test point x_j, picks alpha*_r(x_j) according to:
        * sort alphas in descending order;
        * if any coverage(alpha) >= target (0.95), pick the LARGEST alpha
          with coverage >= target (minimal tempering);
        * otherwise pick the alpha with maximal coverage (least under-cover),
  - reconstructs the SAME full training dataset D^(r) as in calibration
    using the same seed + LHS construction,
  - runs Stage-1 (alpha=1) on full data,
  - for each unique alpha in {alpha*_r(x_j)}, runs Stage-2 (covariance only)
    with beta=1/alpha on full data, and computes predictive intervals
    at the subset of test points whose alpha* equals that alpha,
  - records a 0/1 coverage indicator for each test point vs true f0(x),
  - saves per-rep outputs in --out-dir.
"""

import os
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
import argparse
import glob

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
    """Optional CIQ settings (not used if use_ciq=False)."""
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

    # model / training (match calibration code)
    use_fp64: bool = True
    use_ciq: bool = False
    inducing_points: int = 50
    stage1_epochs: int = 2000
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2
    seed: int = 13           # base seed; per rep: seed + rep_id

    # early stopping for Stage-1
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    # experiment
    ci_level: float = 0.95

    # logging
    log_epoch: bool = False
    log_every: int = 10


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


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Per-rep double-check for data-dependent alpha*(r,x)"
    )
    p.add_argument(
        "--rep-id",
        type=int,
        required=True,
        help="Replication id (0-based, e.g. 0..99)."
    )
    p.add_argument(
        "--calib-dir",
        type=str,
        required=True,
        help="Root directory of calibration results "
             "(e.g. /scratch/yy101/pointwise_two_split_R/results)"
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save per-rep double-check outputs."
    )
    return p.parse_args()


# ---------- main per-rep routine ----------

def main():
    args = parse_args()
    cfg = CFG()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)

    os.makedirs(args.out_dir, exist_ok=True)

    rep_id = args.rep_id
    seed_r = cfg.seed + rep_id

    log(f"=== Double-check rep_id={rep_id} | seed={seed_r} ===")
    log(f"Device: {device} | dtype: {torch.get_default_dtype()}")
    if torch.cuda.is_available():
        log(f"CUDA mem (start): {gpu_mem_mb():.0f} MB")

    # ---- load this rep's alpha-dictionary CSV ----
    rep_dir = os.path.join(args.calib_dir, f"rep_{rep_id}")
    pattern = os.path.join(rep_dir, "pointwise_coverage_rep*_A*_B*.csv")
    matches = glob.glob(pattern)
    assert len(matches) == 1, f"Expected 1 CSV in {rep_dir}, found {len(matches)}"
    csv_path = matches[0]
    log(f"Reading calibration dictionary from: {csv_path}")

    df = pd.read_csv(csv_path)

    # first column is x; all other columns alpha_xxxx
    assert df.columns[0].startswith("x"), "First column is expected to be 'x'"
    x_axis = df.iloc[:, 0].to_numpy()    # [J]
    J = x_axis.shape[0]
    assert J == cfg.n_test, f"n_test mismatch: cfg={cfg.n_test}, csv={J}"

    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    assert len(alpha_cols) > 0, "No alpha_* columns found in CSV."

    # parse alphas from column names
    alpha_values = np.array(
        [float(c.split("_")[1]) for c in alpha_cols],
        dtype=float
    )  # [A]
    A = alpha_values.shape[0]
    log(f"Found A={A} alphas: min={alpha_values.min():.4f}, max={alpha_values.max():.4f}")

    # coverage[j, a] from CSV
    coverage_mat = df[alpha_cols].to_numpy()    # shape [J, A]

    # ---- compute alpha_star_r(x_j) for this rep ----
    target = cfg.ci_level   # e.g. 0.95
    alpha_star = np.zeros(J, dtype=float)
    cov_at_alpha_star = np.zeros(J, dtype=float)

    for j in range(J):
        cov_row = coverage_mat[j, :]          # [A]

        # sort alphas descending
        order_desc = np.argsort(alpha_values)[::-1]
        alphas_desc = alpha_values[order_desc]
        cov_desc = cov_row[order_desc]

        # indices where coverage >= target
        idx_ge = np.where(cov_desc >= target)[0]

        if idx_ge.size > 0:
            # pick largest alpha with cov >= target (first in descending order)
            chosen_desc_idx = idx_ge[0]
        else:
            # all < target: pick alpha with maximal coverage
            chosen_desc_idx = int(np.argmax(cov_desc))

        best_idx = int(order_desc[chosen_desc_idx])
        alpha_star[j] = alpha_values[best_idx]
        cov_at_alpha_star[j] = cov_row[best_idx]

    log("Example alpha_star for first 5 test points:")
    for j in range(min(5, J)):
        log(f"  j={j:02d}, x={x_axis[j]: .3f}, alpha*={alpha_star[j]:.4f}, "
            f"cov@alpha*={cov_at_alpha_star[j]:.3f}")

    # group test indices by unique alpha_star (for efficient Stage-2)
    unique_alphas = np.unique(alpha_star)
    log(f"Unique alpha_star values in this rep: {unique_alphas}")

    alpha_to_idxs = {a: np.where(alpha_star == a)[0] for a in unique_alphas}

    # ---- global test grid & normalization (must match calibration) ----
    x_test_raw = torch.tensor(x_axis, device=device,
                              dtype=torch.get_default_dtype()).unsqueeze(1)
    y_test_true = f0(x_test_raw).squeeze(-1)

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

    x_test_std = to_std(x_test_raw)

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

    # ---- reconstruct full training dataset D^(r) (must match calibration) ----
    torch.manual_seed(seed_r)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_r)
    gen = torch.Generator(device=device).manual_seed(seed_r)

    x_train_raw = make_stratified_points_1d(
        n=cfg.n_train, low=cfg.train_low, high=cfg.train_high,
        generator=gen, device=device, dtype=torch.get_default_dtype()
    )
    y_train = f0(x_train_raw).squeeze(-1)
    x_train_std = to_std(x_train_raw)

    log(f"Training set for rep {rep_id} reconstructed: n_train={x_train_raw.size(0)}")

    # ---- build SVGP model & Stage-1 on full data ----
    model = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.fill_(1.0)
        model.covar_module.outputscale.fill_(1.0)
        likelihood.noise = torch.as_tensor(
            3e-3, dtype=torch.get_default_dtype(), device=device
        )
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_s1, best_loss_s1 = stage1_train(
            model, likelihood, x_train_std, y_train,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every,
        )

    with torch.no_grad():
        pred_f = model(x_test_std)
        mu_s1 = pred_f.mean
        mse_s1 = torch.mean((mu_s1 - y_test_true) ** 2).item()

    log(f"[rep {rep_id:03d}] Stage-1 (α=1.000) | best -ELBO={best_loss_s1:.6f} "
        f"@epoch {best_epoch_s1} | MSE={mse_s1:.6f}")

    s1_model_sd = clone_state_dict(model.state_dict())
    s1_lik_sd = clone_state_dict(likelihood.state_dict())

    # z-quantile for CI
    norm = torch.distributions.Normal(
        torch.tensor(0.0, device=device, dtype=torch.get_default_dtype()),
        torch.tensor(1.0, device=device, dtype=torch.get_default_dtype())
    )
    z_ci = float(norm.icdf(torch.tensor([(1.0 + cfg.ci_level) / 2.0],
                                       device=device))[0].item())
    log(f"Using z_ci={z_ci:.4f} for ci_level={cfg.ci_level:.3f}")

    # ---- Stage-2 per unique alpha_star, record indicators ----
    indicators = np.zeros(J, dtype=np.float64)

    for alpha in unique_alphas:
        idxs_np = alpha_to_idxs[alpha]             # numpy indices for this alpha
        idxs_torch = torch.from_numpy(idxs_np).to(
            device=y_test_true.device, dtype=torch.long
        )
        beta = float(1.0 / alpha)

        model.load_state_dict(s1_model_sd)
        likelihood.load_state_dict(s1_lik_sd)
        model.eval()
        likelihood.eval()

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, best_epoch_s2, best_loss_s2, _ = stage2_train_cov_only(
                model, likelihood, x_train_std, y_train,
                beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                jitter=cfg.jitter, batch_size=cfg.batch_size,
            )

        with torch.no_grad():
            pred_f = model(x_test_std)
            mu = pred_f.mean
            var = pred_f.variance
            std = var.sqrt().clamp_min(1e-12)

            lower = mu[idxs_torch] - z_ci * std[idxs_torch]
            upper = mu[idxs_torch] + z_ci * std[idxs_torch]
            y_sel = y_test_true[idxs_torch]

            cover = ((y_sel >= lower) & (y_sel <= upper)).to(torch.float64)
            indicators[idxs_np] = cover.cpu().numpy()

            mse = torch.mean((mu - y_test_true) ** 2).item()

        log(f"[rep {rep_id:03d}] α={float(alpha):.4f} (Stage-2) | "
            f"best -ELBO={best_loss_s2:.6f} @epoch {best_epoch_s2} | MSE={mse:.6f} | "
            f"num_points={len(idxs_np)}")

    # ---- save per-rep results ----
    out_prefix = os.path.join(args.out_dir, f"rep_{rep_id:03d}")

    df_out = pd.DataFrame({
        "x": x_axis,
        "alpha_star": alpha_star,
        "cov_indicator": indicators,
        "cov_at_alpha_star_calib": cov_at_alpha_star,
    })
    csv_out = out_prefix + "_doublecheck.csv"
    df_out.to_csv(csv_out, index=False)
    log(f"Saved per-rep CSV to {csv_out}")

    np.save(out_prefix + "_alpha_star.npy", alpha_star)
    np.save(out_prefix + "_indicators.npy", indicators)
    log(f"Saved alpha_star and indicators .npy for rep {rep_id}")

    log("First few rows of double-check result:")
    log(df_out.head().to_string(index=False))


if __name__ == "__main__":
    main()
