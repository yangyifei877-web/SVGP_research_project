#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
alpha_grid.py

Aligned two-split + bootstrap + alpha-grid SVGP calibration (single replication),
with:
  - fixed global x_train pool (seed)
  - per-rep permutation/split + bootstrap (seed+rep_id)
  - x minmax -> [0,1] then [-1,1]
  - y z-score on full train pool; de-standardize for CI in RAW y
  - two-stage training:
        Stage-1: beta=1 (train variational params + hyperparams)
        Stage-2: cov-only, beta=1/alpha
  - ALIGNED w.r.t. Liu TVB core: pseudo-truth depends on alpha and uses same alpha
        split1 pseudo-truth(alpha) vs split2 bootstrap CI(alpha)

NEW in this version:
  - alpha grid default: 30 log-spaced values from 1.0 down to 0.001
  - save diagnostic plots per rep:
        (i) f0(x) vs split1 pseudo-truth means for selected alphas
        (ii) bias curves: (mu_truth_alpha - f0)

Outputs:
  - indicators_repXXX_AA_BB_JJ.npy     [A,B,J]
  - coverage_repXXX_AA_BB_JJ.npy       [A,J]
  - pointwise_coverage_repXXX_AA_BB.csv
  - pseudo_truth_repXXX_AA_JJ.npy      [A,J] (RAW)
  - bias_truth_repXXX_AA_JJ.npy        [A,J] (RAW)  # mu_truth - f0
  - diag_truth_vs_f0_repXXX.png
  - diag_bias_repXXX.png
  - meta_repXXX.json
"""

import os
import math
import time
import json
from contextlib import ExitStack
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd

import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- logging ----------------
def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


# ---------------- helpers ----------------
def has_setting(name: str) -> bool:
    return hasattr(gpytorch.settings, name)


def ciq_context(use_ciq: bool):
    """Optional CIQ settings (only used if use_ciq=True)."""
    if not use_ciq:
        return ExitStack()
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


def z_from_ci_level(ci_level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Two-sided normal quantile z = Phi^{-1}((1+ci_level)/2)."""
    ci = float(ci_level)
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci_level must be in (0,1), got {ci_level}")
    p = 0.5 * (1.0 + ci)
    t = torch.tensor(2.0 * p - 1.0, device=device, dtype=dtype)
    return math.sqrt(2.0) * torch.erfinv(t)


# ---------------- true latent f (noiseless) ----------------
def f0(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)


# ---------------- 1D stratified points (LHS-like) ----------------
def make_stratified_points_1d(n, low, high, generator, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + torch.rand(n, generator=generator, device=device, dtype=dtype)) / n
    x = low + (high - low) * u
    perm = torch.randperm(n, generator=generator, device=device)
    return x[perm].unsqueeze(-1)


# ---------------- config ----------------
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
    stage1_epochs: int = 2000
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2

    # global seed for FIXED x_train pool
    seed: int = 13  # pool seed; per-rep split seed = seed + rep_id

    # latent-f nugget (RAW y units); converted to standardized units internally
    noise_raw: float = 3e-3

    # early stopping for Stage-1
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    # experiment
    ci_level: float = 0.95

    # logging
    log_epoch: bool = False
    log_every: int = 10


# ---------------- model ----------------
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_std: torch.Tensor):
        M = inducing_points_std.size(0)
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(M)
        q_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points_std, q_dist, learn_inducing_locations=False
        )
        super().__init__(q_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x_std):
        mean_x = self.mean_module(x_std)
        covar_x = self.covar_module(x_std)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------------- two-stage param helpers ----------------
def freeze_all_params(model, likelihood):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in likelihood.parameters():
        p.requires_grad_(False)


def pick_covariance_params_svgp(model):
    """Select ONLY variational covariance parameters (exclude any '*mean*')."""
    trainables = []
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

    return trainables


# ---------------- Stage-1 ----------------
def stage1_train(model, likelihood, x_train_std, y_train_std, lr, epochs, jitter, batch_size,
                 early_stop_patience, early_stop_min_delta, log_epoch=False, log_every=10, use_ciq=False):
    N = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=1.0)

    likelihood.noise_covar.raw_noise.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": model.variational_parameters()},
        {"params": model.hyperparameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    loader = DataLoader(TensorDataset(x_train_std, y_train_std),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    likelihood.train()
    best_loss, best_epoch = float("inf"), 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())
    patience = 0

    with ciq_context(use_ciq):
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

                if log_epoch and (ep % log_every == 0 or ep == 1 or ep == epochs):
                    log(f"[S1] epoch {ep:04d}/{epochs} | -ELBO={avg_loss:.6f} (best {best_loss:.6f}@{best_epoch})")

                if patience >= early_stop_patience:
                    break

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_epoch, best_loss


# ---------------- Stage-2 ----------------
def stage2_train_cov_only(model, likelihood, x_train_std, y_train_std, beta, lr, epochs, jitter, batch_size, use_ciq=False):
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params = pick_covariance_params_svgp(model)
    if not cov_params:
        raise RuntimeError("No covariance params found to train in Stage-2.")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=beta)
    opt = torch.optim.Adam(cov_params, lr=lr)

    loader = DataLoader(TensorDataset(x_train_std, y_train_std),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    likelihood.train()
    best_loss = float("inf")
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())

    with ciq_context(use_ciq):
        with gpytorch.settings.cholesky_jitter(jitter):
            for _ in range(1, epochs + 1):
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
                    best_loss = avg_loss
                    best_model_sd = clone_state_dict(model.state_dict())
                    best_lik_sd = clone_state_dict(likelihood.state_dict())

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_loss


# ---------------- alpha grid ----------------
def make_alpha_grid(A: int, alpha_max: float = 1.0, alpha_min: float = 1e-3) -> np.ndarray:
    if A < 2:
        raise ValueError("A must be >= 2")
    if not (0.0 < alpha_min < alpha_max <= 1.0):
        raise ValueError("Need 0 < alpha_min < alpha_max <= 1")
    grid = np.logspace(np.log10(alpha_max), np.log10(alpha_min), A)
    # ensure exact endpoints
    grid[0] = alpha_max
    grid[-1] = alpha_min
    return grid.astype(float)


def nearest_alphas(grid: np.ndarray, targets=(1.0, 0.7, 0.5, 0.3, 0.1)):
    out = []
    for t in targets:
        idx = int(np.argmin(np.abs(grid - t)))
        out.append(float(grid[idx]))
    # unique in order
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


# ---------------- main routine ----------------
def run_single_rep(cfg: CFG, rep_id: int, B_bootstrap: int, alphas: np.ndarray, out_dir: str, save_plots: bool):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)
    dtype = torch.get_default_dtype()

    # Per-rep seed ONLY affects split permutation + bootstrap resampling
    split_seed = int(cfg.seed + rep_id)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    log(f"=== rep_id={rep_id} | split_seed={split_seed} ===")
    log(f"Device: {device} | dtype: {dtype}")
    if torch.cuda.is_available():
        log(f"CUDA mem (start): {gpu_mem_mb():.0f} MB")

    # ---- FIXED global train pool (RAW): draw once using cfg.seed (pool seed) ----
    pool_gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    x_train_raw = make_stratified_points_1d(
        n=cfg.n_train,
        low=cfg.train_low,
        high=cfg.train_high,
        generator=pool_gen,
        device=device,
        dtype=dtype,
    )

    x_test_raw = torch.linspace(
        cfg.test_low, cfg.test_high, cfg.n_test, device=device, dtype=dtype
    ).unsqueeze(1)

    # noiseless latent targets (RAW)
    y_train_raw = f0(x_train_raw).squeeze(-1)
    y_test_true_raw = f0(x_test_raw).squeeze(-1)

    # ---- A1 INPUT NORMALIZATION ----
    train_low = torch.as_tensor(cfg.train_low, device=device, dtype=dtype)
    train_high = torch.as_tensor(cfg.train_high, device=device, dtype=dtype)
    span = (train_high - train_low).clamp_min(1e-12)

    def x_to_std(x_raw):
        x01 = (x_raw - train_low) / span
        return 2.0 * x01 - 1.0

    x_train_std_full = x_to_std(x_train_raw)
    x_test_std = x_to_std(x_test_raw)

    Z_raw = torch.linspace(cfg.train_low, cfg.train_high, cfg.inducing_points, device=device, dtype=dtype).unsqueeze(1)
    Z_std = x_to_std(Z_raw)

    # ---- Y STANDARDIZATION on FULL TRAIN POOL ----
    y_mean = y_train_raw.mean()
    y_scale = y_train_raw.std(unbiased=False).clamp_min(1e-12)
    y_train_std_full = (y_train_raw - y_mean) / y_scale

    noise_std = torch.as_tensor(cfg.noise_raw, device=device, dtype=dtype) / (y_scale ** 2)

    # ---- Two-way split (equal) ----
    n_train = cfg.n_train
    if n_train % 2 != 0:
        raise ValueError("n_train must be even for equal split.")
    perm = torch.randperm(n_train, device=device)
    half_n = n_train // 2
    s1_idx = perm[:half_n]
    s2_idx = perm[half_n:]

    x_s1_std = x_train_std_full[s1_idx]
    y_s1_std = y_train_std_full[s1_idx]
    x_s2_std = x_train_std_full[s2_idx]
    y_s2_std = y_train_std_full[s2_idx]

    # ---- CI z ----
    z_ci = float(z_from_ci_level(cfg.ci_level, device=device, dtype=dtype).item())

    # =============================================================================
    # Split1 Stage-1 base (beta=1)
    # =============================================================================
    model_s1_base = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s1_base = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s1_base.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s1_base.covar_module.outputscale.fill_(1.0)
        lik_s1_base.noise = noise_std
    lik_s1_base.noise_covar.raw_noise.requires_grad_(False)

    with ciq_context(cfg.use_ciq):
        best_epoch_s1, best_loss_s1 = stage1_train(
            model_s1_base, lik_s1_base, x_s1_std, y_s1_std,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every,
            use_ciq=cfg.use_ciq,
        )

    s1_base_model_sd = clone_state_dict(model_s1_base.state_dict())
    s1_base_lik_sd = clone_state_dict(lik_s1_base.state_dict())

    log(f"[rep {rep_id:03d}] Split1 Stage-1 base done | best -ELBO={best_loss_s1:.6f}@{best_epoch_s1}")

    # =============================================================================
    # Split2 Stage-1 base (beta=1)
    # =============================================================================
    model_s2_base = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s2_base = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s2_base.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s2_base.covar_module.outputscale.fill_(1.0)
        lik_s2_base.noise = noise_std
    lik_s2_base.noise_covar.raw_noise.requires_grad_(False)

    with ciq_context(cfg.use_ciq):
        best_epoch_s2, best_loss_s2 = stage1_train(
            model_s2_base, lik_s2_base, x_s2_std, y_s2_std,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every,
            use_ciq=cfg.use_ciq,
        )

    s2_base_model_sd = clone_state_dict(model_s2_base.state_dict())
    s2_base_lik_sd = clone_state_dict(lik_s2_base.state_dict())

    log(f"[rep {rep_id:03d}] Split2 Stage-1 base done | best -ELBO={best_loss_s2:.6f}@{best_epoch_s2}")

    # =============================================================================
    # Alpha loop: pseudo-truth(alpha) from split1 + bootstrap CI(alpha) from split2
    # =============================================================================
    A = len(alphas)
    B = int(B_bootstrap)
    J = int(cfg.n_test)

    indicators = np.zeros((A, B, J), dtype=np.int8)
    pseudo_truth_raw_all = np.zeros((A, J), dtype=np.float64)

    # truth f0 in RAW
    f0_raw = y_test_true_raw.detach().cpu().numpy()
    bias_raw_all = np.zeros((A, J), dtype=np.float64)

    n_s2 = x_s2_std.size(0)

    # Reuse modules to avoid realloc
    model_s1 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s1 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model_s2 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s2 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)

    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        beta = 1.0 / alpha

        log(f"[rep {rep_id:03d}] Alpha {alpha:.6f}: beta={beta:.6f} | CI={cfg.ci_level:.3f} (z={z_ci:.4f}) | bootstrap B={B}")

        # ---- Split1 pseudo-truth(alpha): reset to split1 base then Stage-2 cov-only with beta ----
        model_s1.load_state_dict(s1_base_model_sd)
        lik_s1.load_state_dict(s1_base_lik_sd)
        model_s1.eval(); lik_s1.eval()

        with ciq_context(cfg.use_ciq):
            _ = stage2_train_cov_only(
                model_s1, lik_s1, x_s1_std, y_s1_std,
                beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                jitter=cfg.jitter, batch_size=cfg.batch_size,
                use_ciq=cfg.use_ciq,
            )

        with torch.no_grad():
            pred_truth_std = model_s1(x_test_std)
            mu_truth_std = pred_truth_std.mean
            mu_truth_raw = mu_truth_std * y_scale + y_mean

        mu_truth_raw_np = mu_truth_raw.detach().cpu().numpy()
        pseudo_truth_raw_all[a_idx, :] = mu_truth_raw_np
        bias_raw_all[a_idx, :] = mu_truth_raw_np - f0_raw

        # ---- Split2 bootstrap: reset to split2 base, Stage-2 cov-only on bootstrap sample ----
        for b in range(B):
            boot_idx = torch.randint(low=0, high=n_s2, size=(n_s2,), device=device)
            x_boot_std = x_s2_std[boot_idx]
            y_boot_std = y_s2_std[boot_idx]

            model_s2.load_state_dict(s2_base_model_sd)
            lik_s2.load_state_dict(s2_base_lik_sd)
            model_s2.eval(); lik_s2.eval()

            with ciq_context(cfg.use_ciq):
                _ = stage2_train_cov_only(
                    model_s2, lik_s2, x_boot_std, y_boot_std,
                    beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                    jitter=cfg.jitter, batch_size=cfg.batch_size,
                    use_ciq=cfg.use_ciq,
                )

            with torch.no_grad():
                pred_b_std = model_s2(x_test_std)
                mu_b_std = pred_b_std.mean
                var_b_std = pred_b_std.variance.clamp_min(1e-24)

                mu_b_raw = mu_b_std * y_scale + y_mean
                std_b_raw = (var_b_std * (y_scale ** 2)).sqrt().clamp_min(1e-12)

                lower = mu_b_raw - z_ci * std_b_raw
                upper = mu_b_raw + z_ci * std_b_raw

                inc = ((mu_truth_raw >= lower) & (mu_truth_raw <= upper)).to(torch.int8)

            indicators[a_idx, b, :] = inc.cpu().numpy()

            if (b + 1) % max(1, (B // 5)) == 0:
                log(f"        [alpha={alpha:.6f}] bootstrap {b + 1}/{B}")

    # ---- aggregate & save ----
    x_axis = x_test_raw.squeeze(-1).detach().cpu().numpy()      # [J] RAW x grid
    coverage_B = indicators.mean(axis=1)                        # [A,J]

    np.save(os.path.join(out_dir, f"indicators_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"), indicators)
    np.save(os.path.join(out_dir, f"coverage_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"), coverage_B)
    np.save(os.path.join(out_dir, f"pseudo_truth_rep{rep_id:03d}_A{A}_J{J}.npy"), pseudo_truth_raw_all)
    np.save(os.path.join(out_dir, f"bias_truth_rep{rep_id:03d}_A{A}_J{J}.npy"), bias_raw_all)

    data = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        data[f"alpha_{float(alpha):.6f}"] = coverage_B[a_idx]
    df = pd.DataFrame(data)
    csv_path = os.path.join(out_dir, f"pointwise_coverage_rep{rep_id:03d}_A{A}_B{B}.csv")
    df.to_csv(csv_path, index=False)

    # ---- diagnostics plots (mean vs f0, bias) ----
    if save_plots:
        sel = nearest_alphas(alphas, targets=(1.0, 0.7, 0.5, 0.3, 0.1))
        # indices for selected
        idxs = [int(np.argmin(np.abs(alphas - a))) for a in sel]

        # Plot 1: f0 vs pseudo-truth means
        plt.figure(figsize=(10, 4))
        plt.plot(x_axis, f0_raw, label="f0 (truth)", linewidth=2.5)
        for a, idx in zip(sel, idxs):
            plt.plot(x_axis, pseudo_truth_raw_all[idx], label=f"mu_split1(alpha≈{a:.4f})", linewidth=1.5)
        plt.title(f"rep {rep_id:03d} | split1 pseudo-truth mean vs f0 (selected alphas)")
        plt.xlabel("x")
        plt.ylabel("y (raw)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"diag_truth_vs_f0_rep{rep_id:03d}.png")
        plt.savefig(p1, dpi=160)
        plt.close()

        # Plot 2: bias curves
        plt.figure(figsize=(10, 4))
        plt.axhline(0.0, linestyle="--", linewidth=1.0)
        for a, idx in zip(sel, idxs):
            plt.plot(x_axis, bias_raw_all[idx], label=f"bias(alpha≈{a:.4f})", linewidth=1.5)
        plt.title(f"rep {rep_id:03d} | bias = mu_split1(alpha) - f0 (selected alphas)")
        plt.xlabel("x")
        plt.ylabel("bias (raw)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"diag_bias_rep{rep_id:03d}.png")
        plt.savefig(p2, dpi=160)
        plt.close()

    meta = {
        "rep_id": rep_id,
        "split_seed": int(cfg.seed + rep_id),
        "pool_seed": int(cfg.seed),
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "B": B,
        "A": A,
        "ci_level": cfg.ci_level,
        "z_ci": z_ci,
        "alphas": [float(a) for a in alphas],
        "alpha_grid": "logspace(alpha_max=1.0, alpha_min=1e-3, A=30) by default",
        "alignment_update": "pseudo-truth depends on alpha; split1 also uses Stage-2 beta=1/alpha (cov-only) before defining pseudo-truth",
        "two_stage": "Stage-1 beta=1 on split1/split2; Stage-2 cov-only beta=1/alpha on split1 (truth) and split2 bootstraps",
        "normalization": "x: minmax->[0,1] then [-1,1]; y: zscore on full train pool; de-standardize for CI",
        "bootstrap_object": "nonparametric bootstrap of split2 indices (x,y pairs)",
        "Z_fixed": True,
        "M": cfg.inducing_points,
        "use_ciq": bool(cfg.use_ciq),
        "dtype": str(dtype),
        "device": str(device),
        "saved_plots": bool(save_plots),
    }
    meta_path = os.path.join(out_dir, f"meta_rep{rep_id:03d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"[rep {rep_id:03d}] Saved CSV to {csv_path}")
    if save_plots:
        log(f"[rep {rep_id:03d}] Saved plots diag_truth_vs_f0_rep{rep_id:03d}.png and diag_bias_rep{rep_id:03d}.png")


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Aligned two-split SVGP bootstrap calibration with alpha grid + bias diagnostics")
    p.add_argument("--rep-id", type=int, required=True, help="Replication id (e.g., SLURM_ARRAY_TASK_ID)")
    p.add_argument("--B", dest="B_bootstrap", type=int, default=500, help="Bootstrap samples per alpha")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory")

    # alpha grid
    p.add_argument("--A", type=int, default=30, help="Number of alphas (default 30)")
    p.add_argument("--alpha-min", type=float, default=1e-3, help="Min alpha for log grid (default 1e-3)")
    p.add_argument("--alpha-max", type=float, default=1.0, help="Max alpha for log grid (default 1.0)")

    # allow explicit alpha list override (comma-separated)
    p.add_argument("--alphas", type=str, default=None,
                   help="Optional comma-separated alpha list. If set, overrides --A/--alpha-min/--alpha-max.")

    # plots
    p.add_argument("--no-plots", action="store_true", help="Disable saving diagnostic plots")

    return p.parse_args()


def main():
    cfg = CFG()
    args = parse_args()

    if args.alphas is not None:
        alpha_list = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
        alphas = np.array(alpha_list, dtype=float)
    else:
        alphas = make_alpha_grid(A=int(args.A), alpha_max=float(args.alpha_max), alpha_min=float(args.alpha_min))

    run_single_rep(
        cfg=cfg,
        rep_id=int(args.rep_id),
        B_bootstrap=int(args.B_bootstrap),
        alphas=alphas,
        out_dir=str(args.out_dir),
        save_plots=(not args.no_plots),
    )


if __name__ == "__main__":
    main()