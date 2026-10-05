#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
svgp_twosplit_bootstrap_alpha_grid_minmax_yzscore_aligned_tvB.py

Single-replication two-split + bootstrap + alpha-grid SVGP calibration
(ready for Slurm array jobs).

ALIGNMENT UPDATE (to match paper's core logic items 1) and 2)):
  1) Reference ("pseudo-truth") MUST depend on alpha:
       In paper: h(theta_hat_{omega}^{X1}) changes with omega.
       Here: for each alpha, we build a split1 model with that same alpha
       (implemented via Stage-2 covariance-only with beta=1/alpha) and define
       pseudo-truth as its predictive mean on test grid.
  2) Both sides use the SAME alpha:
       For each alpha:
         - split1 model -> pseudo-truth(mean) depends on alpha
         - split2 bootstrap models -> CIs depend on alpha
       Coverage compares pseudo-truth(alpha) vs CI(alpha).

KEPT FROM YOUR ORIGINAL CODE (unchanged design choices):
  - Two-stage training structure:
        Stage-1: train kernel + variational mean (beta=1)
        Stage-2: freeze all, train ONLY variational covariance (beta=1/alpha)
  - Fixed global x_train pool (draw once with cfg.seed; NOT rep-dependent)
  - Per-rep randomness ONLY for permute/split + bootstrap resampling (seed=cfg.seed+rep_id)
  - A1 input normalization (mentor-style): raw -> [0,1] -> [-1,1] shared by train/test/Z
  - y_train Z-score on FULL train pool; de-standardize predictions/CI back to RAW y
  - bootstrap object: resample split2 indices (nonparametric bootstrap of (x,y) pairs)
  - "alpha dictionary" output structure:
        indicators[A,B,J], coverage[A,J], pointwise_coverage CSV

Outputs (per rep):
  - indicators_repXXX_AA_BB_JJ.npy     [A,B,J]  inclusion indicators (per alpha, bootstrap, test point)
  - coverage_repXXX_AA_BB_JJ.npy       [A,J]    bootstrap-mean inclusion (pointwise coverage per alpha)
  - pointwise_coverage_repXXX_AA_BB.csv x + alpha columns
  - meta_repXXX.json                  run metadata
  - (optional) pseudo_truth_repXXX_AA_JJ.npy   [A,J] pseudo-truth means in RAW y space

-------------------------------------------------------------------------------
SBATCH template:

#!/bin/bash
#SBATCH -J svgp_calib
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --array=0-99
#SBATCH -o /scratch/$USER/svgp_calib/logs/%x_%A_%a.out
#SBATCH -e /scratch/$USER/svgp_calib/logs/%x_%A_%a.err

module purge
module load anaconda3
source activate svgp311

OUTDIR=/scratch/$USER/svgp_calib/results
mkdir -p ${OUTDIR}/logs

python svgp_twosplit_bootstrap_alpha_grid_minmax_yzscore_aligned_tvB.py \
  --rep-id ${SLURM_ARRAY_TASK_ID} \
  --B 500 \
  --out-dir ${OUTDIR}

-------------------------------------------------------------------------------
"""

import os
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
import argparse
import json

import torch
import gpytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


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


def ciq_context():
    """Optional CIQ settings (only used if cfg.use_ciq=True)."""
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
    alphas: tuple = (1.0, 0.8, 0.6, 0.5, 0.3, 0.1, 0.001)

    # logging
    log_epoch: bool = False
    log_every: int = 10

    # save pseudo-truth array
    save_pseudo_truth: bool = True


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


# ---------------- Stage-1 ----------------
def stage1_train(model, likelihood, x_train_std, y_train_std, lr, epochs, jitter, batch_size,
                 early_stop_patience, early_stop_min_delta, log_epoch=False, log_every=10):
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
    neg_elbo_hist, best_loss, best_epoch = [], float("inf"), 0
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
    return neg_elbo_hist, best_epoch, best_loss


# ---------------- Stage-2 (cov only) ----------------
def stage2_train_cov_only(model, likelihood, x_train_std, y_train_std, beta, lr, epochs, jitter, batch_size):
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params, cov_names = pick_covariance_params_svgp(model)
    if not cov_params:
        raise RuntimeError("No covariance params found to train in Stage-2.")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=beta)
    opt = torch.optim.Adam(cov_params, lr=lr)

    loader = DataLoader(TensorDataset(x_train_std, y_train_std),
                        batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    likelihood.train()
    neg_elbo_hist, best_loss, best_epoch = [], float("inf"), 0
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
                best_loss, best_epoch = avg_loss, ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return neg_elbo_hist, best_epoch, best_loss, cov_names


# ---------------- main routine ----------------
def run_single_rep(cfg: CFG, rep_id: int, B_bootstrap: int, alphas: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)
    dtype = torch.get_default_dtype()

    # Per-rep seed ONLY affects split permutation + bootstrap resampling
    split_seed = int(cfg.seed + rep_id)
    torch.manual_seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    log(f"=== Single replication rep_id={rep_id} | split_seed={split_seed} ===")
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

    left_std = float(x_to_std(torch.tensor([[cfg.train_low]], device=device, dtype=dtype)))
    right_std = float(x_to_std(torch.tensor([[cfg.train_high]], device=device, dtype=dtype)))
    log(f"MinMax norm: raw [{cfg.train_low:.2f},{cfg.train_high:.2f}] -> std [{left_std:.2f},{right_std:.2f}] (≈ -1,+1)")

    # ---- Y STANDARDIZATION (Z-score) on FULL TRAIN POOL (fixed across reps) ----
    y_mean = y_train_raw.mean()
    y_scale = y_train_raw.std(unbiased=False).clamp_min(1e-12)
    y_train_std_full = (y_train_raw - y_mean) / y_scale

    # Convert nugget/noise to standardized y units (variance scales by y_scale^2)
    noise_std = torch.as_tensor(cfg.noise_raw, device=device, dtype=dtype) / (y_scale ** 2)

    # ---- Two-way split: same pool, per-rep random permutation -> different split ----
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

    # ---- CI quantile from ci_level ----
    z_ci = float(z_from_ci_level(cfg.ci_level, device=device, dtype=dtype).item())

    # =============================================================================
    # Split1 Stage-1 base (alpha=1) checkpoint: will be reused for ALL alphas
    # =============================================================================
    model_s1_base = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s1_base = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s1_base.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s1_base.covar_module.outputscale.fill_(1.0)
        lik_s1_base.noise = noise_std
    lik_s1_base.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_s1, best_loss_s1 = stage1_train(
            model_s1_base, lik_s1_base, x_s1_std, y_s1_std,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every,
        )

    s1_base_model_sd = clone_state_dict(model_s1_base.state_dict())
    s1_base_lik_sd = clone_state_dict(lik_s1_base.state_dict())

    log(f"[rep {rep_id:03d}] Split1 Stage-1 base done | best -ELBO={best_loss_s1:.6f}@{best_epoch_s1}")

    # =============================================================================
    # Split2 Stage-1 base (alpha=1) checkpoint: reused for ALL bootstrap runs
    # =============================================================================
    model_s2_base = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s2_base = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_s2_base.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_s2_base.covar_module.outputscale.fill_(1.0)
        lik_s2_base.noise = noise_std
    lik_s2_base.noise_covar.raw_noise.requires_grad_(False)

    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_s2_base, best_loss_s2_base = stage1_train(
            model_s2_base, lik_s2_base, x_s2_std, y_s2_std,
            lr=cfg.lr_adam_stage1, epochs=cfg.stage1_epochs, jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch, log_every=cfg.log_every,
        )

    s2_base_model_sd = clone_state_dict(model_s2_base.state_dict())
    s2_base_lik_sd = clone_state_dict(lik_s2_base.state_dict())

    log(f"[rep {rep_id:03d}] Split2 Stage-1 base done | best -ELBO={best_loss_s2_base:.6f}@{best_epoch_s2_base}")

    # =============================================================================
    # Alpha loop: for EACH alpha, compute pseudo-truth(alpha) from split1
    #           then do bootstrap on split2 and compute inclusion vs that pseudo-truth(alpha).
    # =============================================================================
    A = len(alphas)
    B = int(B_bootstrap)
    J = int(cfg.n_test)
    indicators = np.zeros((A, B, J), dtype=np.int8)

    pseudo_truth_raw_all = np.zeros((A, J), dtype=np.float64)

    n_s2 = x_s2_std.size(0)

    # Reuse these modules to avoid re-alloc each time (we still reset state_dicts)
    model_s1 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s1 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model_s2 = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_s2 = gpytorch.likelihoods.GaussianLikelihood().to(device=device)

    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        beta = 1.0 / alpha

        log(f"[rep {rep_id:03d}] Alpha {alpha:.4f}: beta={beta:.4f} | CI={cfg.ci_level:.3f} (z={z_ci:.4f}) | bootstrap B={B}")

        # ------------------------------
        # Split1: build pseudo-truth(alpha)
        #   reset to split1 Stage-1 base, then Stage-2 cov-only with beta=1/alpha on split1 data
        # ------------------------------
        model_s1.load_state_dict(s1_base_model_sd)
        lik_s1.load_state_dict(s1_base_lik_sd)
        model_s1.eval()
        lik_s1.eval()

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _h1, _be1, _bl1, _names1 = stage2_train_cov_only(
                model_s1, lik_s1, x_s1_std, y_s1_std,
                beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                jitter=cfg.jitter, batch_size=cfg.batch_size,
            )

        with torch.no_grad():
            pred_truth_std = model_s1(x_test_std)
            mu_truth_std = pred_truth_std.mean
            mu_truth_raw = mu_truth_std * y_scale + y_mean  # pseudo-truth(alpha) in RAW y space

        pseudo_truth_raw_all[a_idx, :] = mu_truth_raw.detach().cpu().numpy()

        # ------------------------------
        # Split2 bootstrap: for each b
        #   reset to split2 Stage-1 base, then Stage-2 cov-only on bootstrap sample
        # ------------------------------
        for b in range(B):
            boot_idx = torch.randint(low=0, high=n_s2, size=(n_s2,), device=device)
            x_boot_std = x_s2_std[boot_idx]
            y_boot_std = y_s2_std[boot_idx]

            model_s2.load_state_dict(s2_base_model_sd)
            lik_s2.load_state_dict(s2_base_lik_sd)
            model_s2.eval()
            lik_s2.eval()

            with (ciq_context() if cfg.use_ciq else ExitStack()):
                _h2, _be2, _bl2, _names2 = stage2_train_cov_only(
                    model_s2, lik_s2, x_boot_std, y_boot_std,
                    beta=beta, lr=cfg.lr_adam_stage2, epochs=cfg.stage2_epochs,
                    jitter=cfg.jitter, batch_size=cfg.batch_size,
                )

            with torch.no_grad():
                pred_b_std = model_s2(x_test_std)
                mu_b_std = pred_b_std.mean
                var_b_std = pred_b_std.variance.clamp_min(1e-24)

                mu_b_raw = mu_b_std * y_scale + y_mean
                std_b_raw = (var_b_std * (y_scale ** 2)).sqrt().clamp_min(1e-12)

                lower = mu_b_raw - z_ci * std_b_raw
                upper = mu_b_raw + z_ci * std_b_raw

                # Inclusion: pseudo-truth(alpha) vs CI(alpha)
                inc = ((mu_truth_raw >= lower) & (mu_truth_raw <= upper)).to(torch.int8)

            indicators[a_idx, b, :] = inc.cpu().numpy()

            if (b + 1) % max(1, (B // 5)) == 0:
                log(f"        [alpha={alpha:.4f}] bootstrap {b + 1}/{B}")

    # ---- aggregate & save ----
    x_axis = x_test_raw.squeeze(-1).detach().cpu().numpy()      # [J] RAW x grid
    coverage_B = indicators.mean(axis=1)                        # [A,J]

    # save npy
    np.save(os.path.join(out_dir, f"indicators_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"), indicators)
    np.save(os.path.join(out_dir, f"coverage_rep{rep_id:03d}_A{A}_B{B}_J{J}.npy"), coverage_B)

    if cfg.save_pseudo_truth:
        np.save(os.path.join(out_dir, f"pseudo_truth_rep{rep_id:03d}_A{A}_J{J}.npy"), pseudo_truth_raw_all)

    # save CSV
    data = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        data[f"alpha_{float(alpha):.4f}"] = coverage_B[a_idx]
    df = pd.DataFrame(data)
    csv_path = os.path.join(out_dir, f"pointwise_coverage_rep{rep_id:03d}_A{A}_B{B}.csv")
    df.to_csv(csv_path, index=False)

    log(f"[rep {rep_id:03d}] Saved CSV to {csv_path}")
    log(df.head().to_string(index=False))

    # meta
    meta = {
        "rep_id": rep_id,
        "split_seed": split_seed,
        "pool_seed": int(cfg.seed),
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "B": B,
        "A": A,
        "ci_level": cfg.ci_level,
        "z_ci": z_ci,
        "alphas": [float(a) for a in alphas],
        "alignment_update": "pseudo-truth depends on alpha; split1 also uses Stage-2 beta=1/alpha (cov-only) before defining pseudo-truth",
        "two_stage": "Stage-1 beta=1 on split1 and split2; Stage-2 cov-only beta=1/alpha on split1 (truth) and split2 bootstraps",
        "normalization": "x: minmax->[0,1] then [-1,1]; y: zscore on full train pool; de-standardize for CI",
        "bootstrap_object": "nonparametric bootstrap of split2 indices (x,y pairs)",
        "Z_fixed": True,
        "M": cfg.inducing_points,
        "use_ciq": bool(cfg.use_ciq),
        "dtype": str(dtype),
        "device": str(device),
    }
    meta_path = os.path.join(out_dir, f"meta_rep{rep_id:03d}.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        log(f"[rep {rep_id:03d}] Saved meta to {meta_path}")
    except Exception as e:
        log(f"[rep {rep_id:03d}] WARNING: failed to write meta json: {e}")


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-rep two-split SVGP bootstrap calibration (aligned: truth depends on alpha)")
    p.add_argument("--rep-id", type=int, required=True,
                   help="Replication id (e.g., SLURM_ARRAY_TASK_ID in 0..99)")
    p.add_argument("--B", "--bootstrap", dest="B_bootstrap", type=int, default=500,
                   help="Number of bootstrap samples per alpha (default 500)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Directory to write outputs (recommend /scratch/$USER/...)")
    p.add_argument("--alphas", type=str, default=None,
                   help="Optional comma-separated alpha list, e.g. '1.0,0.8,0.5'. If not set, uses CFG.alphas.")
    p.add_argument("--save-pseudo-truth", action="store_true",
                   help="If set, saves pseudo_truth_repXXX_AA_JJ.npy (default: on in CFG; this flag forces on).")
    p.add_argument("--no-save-pseudo-truth", action="store_true",
                   help="If set, disables saving pseudo-truth array.")
    return p.parse_args()


def main():
    cfg = CFG()
    args = parse_args()

    if args.alphas is not None:
        alpha_list = [float(a) for a in args.alphas.split(",") if a.strip() != ""]
        alphas = np.array(alpha_list, dtype=float)
    else:
        alphas = np.array(list(cfg.alphas), dtype=float)

    if args.save_pseudo_truth:
        cfg.save_pseudo_truth = True
    if args.no_save_pseudo_truth:
        cfg.save_pseudo_truth = False

    run_single_rep(
        cfg=cfg,
        rep_id=int(args.rep_id),
        B_bootstrap=int(args.B_bootstrap),
        alphas=alphas,
        out_dir=str(args.out_dir),
    )


if __name__ == "__main__":
    main()