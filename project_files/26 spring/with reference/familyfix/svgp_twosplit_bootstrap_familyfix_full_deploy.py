#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
svgp_twosplit_bootstrap_familyfix_full_deploy.py

Purpose
-------
Single-replication, two-split, bootstrap, alpha-grid SVGP calibration script
with the key family-identity fix:

    every bootstrap resample on split-2 gets its OWN Stage-1 fit,
    and Stage-2 is then trained on that SAME bootstrap resample.

So the calibration interval generator and the deployment interval generator are
now the same family:

    CI_alpha(D; x) = IntervalGenerator(D, alpha; x)

where IntervalGenerator always means:
    1) Stage-1 on D with beta = 1
    2) Stage-2 (cov-only) on the SAME D with beta = 1 / alpha
    3) latent-f prediction interval on x-grid

This file does four things for one replication:
  1) Split D into D1, D2
  2) Build split-1 pseudo-truth curves for each alpha
  3) Bootstrap D2 and estimate pointwise coverage for each alpha
  4) Select pointwise alpha*(x), then FULL-DEPLOY on full data and
     save the final deployed latent-f intervals + simulation double-check hits

Default alpha grid
------------------
The user asked for 10 equally spaced alpha values on the 0-1 interval.
Since alpha=0 is invalid here, this script uses:

    np.linspace(0.1, 1.0, 10)

which is the natural practical interpretation.

Outputs per replication
-----------------------
Saved under out_dir (recommended scratch/results directory):

  indicators_repXXX_AA_BB_JJ.npy        [A,B,J]
  coverage_repXXX_AA_BB_JJ.npy          [A,J]
  pseudo_truth_repXXX_AA_JJ.npy         [A,J]
  alpha_star_repXXX_JJ.npy              [J]
  alpha_star_curve_repXXX.csv           x, alpha_star
  deploy_full_by_alpha_repXXX_AA_JJ.npz mu/std/lower/upper for every alpha on full data
  deploy_final_repXXX_JJ.csv            x, true_f, alpha_star, mu, std, lower, upper, hit
  meta_repXXX.json

SBATCH template (NOTS)
----------------------
Use a separate .sbatch file like this:

#!/bin/bash
#SBATCH -J svgp_ffix
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --array=0-49
#SBATCH -o /scratch/$USER/15_points_familyfix_strict/logs/%x_%A_%a.out
#SBATCH -e /scratch/$USER/15_points_familyfix_strict/logs/%x_%A_%a.err

module purge
module load anaconda3
source activate svgp311

SCRATCH_BASE=/scratch/$USER/15_points_familyfix_strict
mkdir -p ${SCRATCH_BASE}/results
mkdir -p ${SCRATCH_BASE}/logs

python svgp_twosplit_bootstrap_familyfix_full_deploy.py \
  --rep-id ${SLURM_ARRAY_TASK_ID} \
  --B 100 \
  --out-dir ${SCRATCH_BASE}/results
"""

import argparse
import json
import math
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gpytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def has_setting(name: str) -> bool:
    return hasattr(gpytorch.settings, name)


def ciq_context():
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


def clone_state_dict(sd: Dict[str, torch.Tensor], to_cpu: bool = False) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            vv = v.detach().clone()
            if to_cpu:
                vv = vv.cpu()
            out[k] = vv
        else:
            out[k] = v
    return out


def z_from_ci_level(ci_level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0,1), got {ci_level}")
    p = 0.5 * (1.0 + ci_level)
    t = torch.tensor(2.0 * p - 1.0, device=device, dtype=dtype)
    return math.sqrt(2.0) * torch.erfinv(t)


# -----------------------------------------------------------------------------
# true latent function (simulation ground truth)
# -----------------------------------------------------------------------------
def f0(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)


# -----------------------------------------------------------------------------
# 1D stratified points
# -----------------------------------------------------------------------------
def make_stratified_points_1d(
    n: int,
    low: float,
    high: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + torch.rand(n, generator=generator, device=device, dtype=dtype)) / n
    x = low + (high - low) * u
    perm = torch.randperm(n, generator=generator, device=device)
    return x[perm].unsqueeze(-1)


# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------
@dataclass
class CFG:
    # data
    n_train: int = 500
    n_test: int = 50
    train_low: float = -2.5
    train_high: float = 2.5
    test_low: float = -2.5
    test_high: float = 2.5

    # model / train
    use_fp64: bool = True
    use_ciq: bool = False
    inducing_points: int = 50
    batch_size: int = 500
    jitter: float = 1e-2

    # stage-1 / stage-2 training
    stage1_epochs: int = 1200
    stage2_epochs: int = 600
    lr_stage1: float = 2e-2
    lr_stage2: float = 2e-2
    early_stop_patience: int = 80
    early_stop_min_delta: float = 1e-4

    # randomness
    seed: int = 13

    # latent-f nugget in raw y variance units
    noise_raw: float = 3e-3

    # calibration / deployment
    ci_level: float = 0.95
    alphas: Tuple[float, ...] = tuple(np.linspace(0.1, 1.0, 10).tolist())
    selection_rule: str = "largest_meeting_target"  # or 'closest'

    # saving / logging
    save_pseudo_truth: bool = True
    save_full_alpha_bank: bool = True
    log_epoch: bool = False
    log_every: int = 20


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points_std: torch.Tensor):
        m = inducing_points_std.size(0)
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(m)
        q_strat = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points_std,
            q_dist,
            learn_inducing_locations=False,
        )
        super().__init__(q_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x_std: torch.Tensor):
        mean_x = self.mean_module(x_std)
        covar_x = self.covar_module(x_std)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# -----------------------------------------------------------------------------
# parameter utilities
# -----------------------------------------------------------------------------
def freeze_all_params(model: torch.nn.Module, likelihood: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in likelihood.parameters():
        p.requires_grad_(False)


def pick_covariance_params_svgp(model: SVGPModel) -> Tuple[List[torch.nn.Parameter], List[str]]:
    trainables: List[torch.nn.Parameter] = []
    names: List[str] = []

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
            "chol_variational_covar",
            "variational_distribution.chol_covar",
            "variational_distribution._cholesky_factor",
            "_variational_distribution.chol_covar",
            "raw_covar",
            "chol_factor",
            "covar_factor",
            "variational_stddev",
            "raw_var",
            "qchol",
            "q_scale_tril",
        )
        for n, p in model.named_parameters():
            low = n.lower()
            if "variational_strategy" in low and any(k in low for k in allow) and "mean" not in low:
                p.requires_grad_(True)
                trainables.append(p)
                names.append(n)

    return trainables, names


# -----------------------------------------------------------------------------
# training
# -----------------------------------------------------------------------------
def stage1_train(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_train_std: torch.Tensor,
    y_train_std: torch.Tensor,
    lr: float,
    epochs: int,
    jitter: float,
    batch_size: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    log_epoch: bool,
    log_every: int,
) -> Tuple[List[float], int, float]:
    n = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n, beta=1.0)

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
        TensorDataset(x_train_std, y_train_std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()

    neg_elbo_hist: List[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())
    patience = 0

    with gpytorch.settings.cholesky_jitter(jitter):
        for ep in range(1, epochs + 1):
            total = 0.0
            n_batches = 0
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

            if log_epoch and (ep == 1 or ep % log_every == 0 or ep == epochs):
                log(f"[S1] epoch {ep:04d}/{epochs} | -ELBO={avg_loss:.6f} | best={best_loss:.6f}@{best_epoch}")

            if patience >= early_stop_patience:
                break

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return neg_elbo_hist, best_epoch, best_loss


def stage2_train_cov_only(
    model: SVGPModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    x_train_std: torch.Tensor,
    y_train_std: torch.Tensor,
    beta: float,
    lr: float,
    epochs: int,
    jitter: float,
    batch_size: int,
) -> Tuple[List[float], int, float, List[str]]:
    n = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params, cov_names = pick_covariance_params_svgp(model)
    if not cov_params:
        raise RuntimeError("No covariance params found for Stage-2 training.")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n, beta=beta)
    opt = torch.optim.Adam(cov_params, lr=lr)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train_std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()

    neg_elbo_hist: List[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())

    with gpytorch.settings.cholesky_jitter(jitter):
        for ep in range(1, epochs + 1):
            total = 0.0
            n_batches = 0
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


# -----------------------------------------------------------------------------
# model / prediction helpers
# -----------------------------------------------------------------------------
def build_model_and_likelihood(
    z_std: torch.Tensor,
    noise_std: torch.Tensor,
    device: torch.device,
) -> Tuple[SVGPModel, gpytorch.likelihoods.GaussianLikelihood]:
    model = SVGPModel(z_std.clone().contiguous()).to(device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model.covar_module.base_kernel.lengthscale.fill_(1.0)
        model.covar_module.outputscale.fill_(1.0)
        likelihood.noise = noise_std
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    model.eval()
    likelihood.eval()
    return model, likelihood


@torch.no_grad()
def predict_latent_raw(
    model: SVGPModel,
    x_test_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_scale: torch.Tensor,
    z_ci_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_std = model(x_test_std)
    mu_std = pred_std.mean
    var_std = pred_std.variance.clamp_min(1e-24)

    mu_raw = mu_std * y_scale + y_mean
    std_raw = (var_std * (y_scale ** 2)).sqrt().clamp_min(1e-12)

    lower = mu_raw - z_ci_value * std_raw
    upper = mu_raw + z_ci_value * std_raw
    return mu_raw, std_raw, lower, upper


# -----------------------------------------------------------------------------
# alpha selection
# -----------------------------------------------------------------------------
def choose_alpha_star(
    alphas: np.ndarray,
    coverage_alpha_j: np.ndarray,
    target: float,
    rule: str,
) -> float:
    if rule == "closest":
        idx = int(np.argmin(np.abs(coverage_alpha_j - target)))
        return float(alphas[idx])

    if rule == "largest_meeting_target":
        ok = np.where(coverage_alpha_j >= target)[0]
        if len(ok) > 0:
            return float(alphas[ok[-1]])
        idx = int(np.argmin(np.abs(coverage_alpha_j - target)))
        return float(alphas[idx])

    raise ValueError(f"Unknown selection rule: {rule}")


# -----------------------------------------------------------------------------
# main routine
# -----------------------------------------------------------------------------
def run_single_rep(cfg: CFG, rep_id: int, b_bootstrap: int, alphas: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)
    dtype = torch.get_default_dtype()

    split_seed = int(cfg.seed + rep_id)
    torch.manual_seed(split_seed)
    np.random.seed(split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(split_seed)

    log(f"=== rep_id={rep_id} | split_seed={split_seed} ===")
    log(f"Device={device} | dtype={dtype}")
    if torch.cuda.is_available():
        log(f"CUDA mem start={gpu_mem_mb():.0f} MB")

    # -----------------------------------------------------------------
    # fixed global data pool
    # -----------------------------------------------------------------
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
        cfg.test_low,
        cfg.test_high,
        cfg.n_test,
        device=device,
        dtype=dtype,
    ).unsqueeze(1)
    y_train_raw = f0(x_train_raw).squeeze(-1)
    y_test_true = f0(x_test_raw).squeeze(-1)

    # -----------------------------------------------------------------
    # shared x normalization
    # -----------------------------------------------------------------
    train_low = torch.as_tensor(cfg.train_low, device=device, dtype=dtype)
    train_high = torch.as_tensor(cfg.train_high, device=device, dtype=dtype)
    span = (train_high - train_low).clamp_min(1e-12)

    def x_to_std(x_raw: torch.Tensor) -> torch.Tensor:
        x01 = (x_raw - train_low) / span
        return 2.0 * x01 - 1.0

    x_train_std_full = x_to_std(x_train_raw)
    x_test_std = x_to_std(x_test_raw)
    z_raw = torch.linspace(cfg.train_low, cfg.train_high, cfg.inducing_points, device=device, dtype=dtype).unsqueeze(1)
    z_std = x_to_std(z_raw)

    # -----------------------------------------------------------------
    # shared y standardization
    # -----------------------------------------------------------------
    y_mean = y_train_raw.mean()
    y_scale = y_train_raw.std(unbiased=False).clamp_min(1e-12)
    y_train_std_full = (y_train_raw - y_mean) / y_scale
    noise_std = torch.as_tensor(cfg.noise_raw, device=device, dtype=dtype) / (y_scale ** 2)

    z_ci = float(z_from_ci_level(cfg.ci_level, device=device, dtype=dtype).item())
    target_cov = float(cfg.ci_level)

    # -----------------------------------------------------------------
    # split
    # -----------------------------------------------------------------
    if cfg.n_train % 2 != 0:
        raise ValueError("n_train must be even for 50/50 split.")

    perm = torch.randperm(cfg.n_train, device=device)
    half_n = cfg.n_train // 2
    s1_idx = perm[:half_n]
    s2_idx = perm[half_n:]

    x_s1_std = x_train_std_full[s1_idx]
    y_s1_std = y_train_std_full[s1_idx]
    x_s2_std = x_train_std_full[s2_idx]
    y_s2_std = y_train_std_full[s2_idx]
    n_s2 = x_s2_std.size(0)

    a = len(alphas)
    b = int(b_bootstrap)
    j = int(cfg.n_test)

    indicators = np.zeros((a, b, j), dtype=np.int8)
    coverage = np.zeros((a, j), dtype=np.float64)
    pseudo_truth_raw_all = np.zeros((a, j), dtype=np.float64)

    # -----------------------------------------------------------------
    # split-1 pseudo-truth bank: stage1 base once, stage2 per alpha
    # -----------------------------------------------------------------
    log("Building split-1 Stage-1 base...")
    model_s1_base, lik_s1_base = build_model_and_likelihood(z_std, noise_std, device)
    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_s1, best_loss_s1 = stage1_train(
            model_s1_base,
            lik_s1_base,
            x_s1_std,
            y_s1_std,
            lr=cfg.lr_stage1,
            epochs=cfg.stage1_epochs,
            jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch,
            log_every=cfg.log_every,
        )
    s1_base_model_sd = clone_state_dict(model_s1_base.state_dict(), to_cpu=True)
    s1_base_lik_sd = clone_state_dict(lik_s1_base.state_dict(), to_cpu=True)
    log(f"Split-1 Stage-1 done | best -ELBO={best_loss_s1:.6f}@{best_epoch_s1}")

    model_tmp_s1, lik_tmp_s1 = build_model_and_likelihood(z_std, noise_std, device)
    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        beta = 1.0 / alpha
        model_tmp_s1.load_state_dict(s1_base_model_sd)
        lik_tmp_s1.load_state_dict(s1_base_lik_sd)
        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, _, _, _ = stage2_train_cov_only(
                model_tmp_s1,
                lik_tmp_s1,
                x_s1_std,
                y_s1_std,
                beta=beta,
                lr=cfg.lr_stage2,
                epochs=cfg.stage2_epochs,
                jitter=cfg.jitter,
                batch_size=cfg.batch_size,
            )
        mu_truth_raw, _, _, _ = predict_latent_raw(model_tmp_s1, x_test_std, y_mean, y_scale, z_ci)
        pseudo_truth_raw_all[a_idx, :] = mu_truth_raw.detach().cpu().numpy()
        log(f"Pseudo-truth built for alpha={alpha:.4f}")

    # -----------------------------------------------------------------
    # bootstrap resamples from split-2
    # IMPORTANT FIX: each bootstrap sample gets its own Stage-1 base
    # -----------------------------------------------------------------
    log("Preparing split-2 bootstrap resamples...")
    boot_x_list: List[torch.Tensor] = []
    boot_y_list: List[torch.Tensor] = []
    for b_idx in range(b):
        boot_idx = torch.randint(low=0, high=n_s2, size=(n_s2,), device=device)
        boot_x_list.append(x_s2_std[boot_idx])
        boot_y_list.append(y_s2_std[boot_idx])

    log("Caching bootstrap-specific Stage-1 bases (family-identity fix)...")
    boot_stage1_model_sds: List[Dict[str, torch.Tensor]] = []
    boot_stage1_lik_sds: List[Dict[str, torch.Tensor]] = []

    model_boot_base, lik_boot_base = build_model_and_likelihood(z_std, noise_std, device)
    for b_idx in range(b):
        xb = boot_x_list[b_idx]
        yb = boot_y_list[b_idx]

        model_boot_base, lik_boot_base = build_model_and_likelihood(z_std, noise_std, device)
        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, best_ep, best_loss = stage1_train(
                model_boot_base,
                lik_boot_base,
                xb,
                yb,
                lr=cfg.lr_stage1,
                epochs=cfg.stage1_epochs,
                jitter=cfg.jitter,
                batch_size=cfg.batch_size,
                early_stop_patience=cfg.early_stop_patience,
                early_stop_min_delta=cfg.early_stop_min_delta,
                log_epoch=False,
                log_every=cfg.log_every,
            )
        boot_stage1_model_sds.append(clone_state_dict(model_boot_base.state_dict(), to_cpu=True))
        boot_stage1_lik_sds.append(clone_state_dict(lik_boot_base.state_dict(), to_cpu=True))

        if (b_idx + 1) % max(1, b // 5) == 0 or b_idx == 0 or (b_idx + 1) == b:
            log(f"Bootstrap Stage-1 cache {b_idx + 1}/{b} | best -ELBO={best_loss:.6f}@{best_ep}")

    # -----------------------------------------------------------------
    # alpha loop on cached bootstrap-specific Stage-1 bases
    # -----------------------------------------------------------------
    model_boot_stage2, lik_boot_stage2 = build_model_and_likelihood(z_std, noise_std, device)

    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        beta = 1.0 / alpha
        mu_truth_raw = torch.from_numpy(pseudo_truth_raw_all[a_idx]).to(device=device, dtype=dtype)
        log(f"Coverage loop alpha={alpha:.4f} | beta={beta:.4f}")

        for b_idx in range(b):
            model_boot_stage2.load_state_dict(boot_stage1_model_sds[b_idx])
            lik_boot_stage2.load_state_dict(boot_stage1_lik_sds[b_idx])
            xb = boot_x_list[b_idx]
            yb = boot_y_list[b_idx]

            with (ciq_context() if cfg.use_ciq else ExitStack()):
                _, _, _, _ = stage2_train_cov_only(
                    model_boot_stage2,
                    lik_boot_stage2,
                    xb,
                    yb,
                    beta=beta,
                    lr=cfg.lr_stage2,
                    epochs=cfg.stage2_epochs,
                    jitter=cfg.jitter,
                    batch_size=cfg.batch_size,
                )

            mu_b_raw, std_b_raw, lower_b, upper_b = predict_latent_raw(
                model_boot_stage2,
                x_test_std,
                y_mean,
                y_scale,
                z_ci,
            )
            inc = ((mu_truth_raw >= lower_b) & (mu_truth_raw <= upper_b)).to(torch.int8)
            indicators[a_idx, b_idx, :] = inc.detach().cpu().numpy()

            if (b_idx + 1) % max(1, b // 5) == 0 or (b_idx + 1) == b:
                log(f"  alpha={alpha:.4f} | bootstrap {b_idx + 1}/{b}")

        coverage[a_idx] = indicators[a_idx].mean(axis=0)

    # -----------------------------------------------------------------
    # choose alpha*(x)
    # -----------------------------------------------------------------
    alpha_star = np.zeros(j, dtype=np.float64)
    for j_idx in range(j):
        alpha_star[j_idx] = choose_alpha_star(
            alphas=alphas,
            coverage_alpha_j=coverage[:, j_idx],
            target=target_cov,
            rule=cfg.selection_rule,
        )
    log(f"alpha*(x) selection done | unique={sorted(set(alpha_star.tolist()))}")

    # -----------------------------------------------------------------
    # FULL-DEPLOY: build full-data CI bank for every alpha, then recombine by alpha*(x)
    # -----------------------------------------------------------------
    log("Building full-data Stage-1 base...")
    model_full_base, lik_full_base = build_model_and_likelihood(z_std, noise_std, device)
    with (ciq_context() if cfg.use_ciq else ExitStack()):
        _, best_epoch_full, best_loss_full = stage1_train(
            model_full_base,
            lik_full_base,
            x_train_std_full,
            y_train_std_full,
            lr=cfg.lr_stage1,
            epochs=cfg.stage1_epochs,
            jitter=cfg.jitter,
            batch_size=cfg.batch_size,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            log_epoch=cfg.log_epoch,
            log_every=cfg.log_every,
        )
    full_base_model_sd = clone_state_dict(model_full_base.state_dict(), to_cpu=True)
    full_base_lik_sd = clone_state_dict(lik_full_base.state_dict(), to_cpu=True)
    log(f"Full-data Stage-1 done | best -ELBO={best_loss_full:.6f}@{best_epoch_full}")

    model_full_alpha, lik_full_alpha = build_model_and_likelihood(z_std, noise_std, device)
    mu_full_bank = np.zeros((a, j), dtype=np.float64)
    std_full_bank = np.zeros((a, j), dtype=np.float64)
    lower_full_bank = np.zeros((a, j), dtype=np.float64)
    upper_full_bank = np.zeros((a, j), dtype=np.float64)

    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        beta = 1.0 / alpha
        model_full_alpha.load_state_dict(full_base_model_sd)
        lik_full_alpha.load_state_dict(full_base_lik_sd)

        with (ciq_context() if cfg.use_ciq else ExitStack()):
            _, _, _, _ = stage2_train_cov_only(
                model_full_alpha,
                lik_full_alpha,
                x_train_std_full,
                y_train_std_full,
                beta=beta,
                lr=cfg.lr_stage2,
                epochs=cfg.stage2_epochs,
                jitter=cfg.jitter,
                batch_size=cfg.batch_size,
            )

        mu_raw, std_raw, lower, upper = predict_latent_raw(
            model_full_alpha,
            x_test_std,
            y_mean,
            y_scale,
            z_ci,
        )
        mu_full_bank[a_idx] = mu_raw.detach().cpu().numpy()
        std_full_bank[a_idx] = std_raw.detach().cpu().numpy()
        lower_full_bank[a_idx] = lower.detach().cpu().numpy()
        upper_full_bank[a_idx] = upper.detach().cpu().numpy()
        log(f"Full-deploy alpha bank built for alpha={alpha:.4f}")

    alpha_to_idx = {float(alpha): idx for idx, alpha in enumerate(alphas.tolist())}
    deploy_mu = np.zeros(j, dtype=np.float64)
    deploy_std = np.zeros(j, dtype=np.float64)
    deploy_lower = np.zeros(j, dtype=np.float64)
    deploy_upper = np.zeros(j, dtype=np.float64)

    for j_idx in range(j):
        idx = alpha_to_idx[float(alpha_star[j_idx])]
        deploy_mu[j_idx] = mu_full_bank[idx, j_idx]
        deploy_std[j_idx] = std_full_bank[idx, j_idx]
        deploy_lower[j_idx] = lower_full_bank[idx, j_idx]
        deploy_upper[j_idx] = upper_full_bank[idx, j_idx]

    true_f = y_test_true.detach().cpu().numpy()
    deploy_hit = ((true_f >= deploy_lower) & (true_f <= deploy_upper)).astype(np.int8)

    # -----------------------------------------------------------------
    # save outputs
    # -----------------------------------------------------------------
    x_axis = x_test_raw.squeeze(-1).detach().cpu().numpy()

    np.save(os.path.join(out_dir, f"indicators_rep{rep_id:03d}_A{a}_B{b}_J{j}.npy"), indicators)
    np.save(os.path.join(out_dir, f"coverage_rep{rep_id:03d}_A{a}_B{b}_J{j}.npy"), coverage)
    if cfg.save_pseudo_truth:
        np.save(os.path.join(out_dir, f"pseudo_truth_rep{rep_id:03d}_A{a}_J{j}.npy"), pseudo_truth_raw_all)
    np.save(os.path.join(out_dir, f"alpha_star_rep{rep_id:03d}_J{j}.npy"), alpha_star)

    alpha_star_df = pd.DataFrame({"x": x_axis, "alpha_star": alpha_star})
    alpha_star_df.to_csv(os.path.join(out_dir, f"alpha_star_curve_rep{rep_id:03d}.csv"), index=False)

    pointwise_cov_df = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        pointwise_cov_df[f"alpha_{float(alpha):.4f}"] = coverage[a_idx]
    pd.DataFrame(pointwise_cov_df).to_csv(
        os.path.join(out_dir, f"pointwise_coverage_rep{rep_id:03d}_A{a}_B{b}.csv"),
        index=False,
    )

    deploy_df = pd.DataFrame(
        {
            "x": x_axis,
            "true_f": true_f,
            "alpha_star": alpha_star,
            "mu": deploy_mu,
            "std": deploy_std,
            "lower": deploy_lower,
            "upper": deploy_upper,
            "hit": deploy_hit,
        }
    )
    deploy_csv = os.path.join(out_dir, f"deploy_final_rep{rep_id:03d}_J{j}.csv")
    deploy_df.to_csv(deploy_csv, index=False)

    if cfg.save_full_alpha_bank:
        np.savez_compressed(
            os.path.join(out_dir, f"deploy_full_by_alpha_rep{rep_id:03d}_A{a}_J{j}.npz"),
            x=x_axis,
            alphas=alphas,
            mu=mu_full_bank,
            std=std_full_bank,
            lower=lower_full_bank,
            upper=upper_full_bank,
        )

    meta = {
        "rep_id": rep_id,
        "pool_seed": int(cfg.seed),
        "split_seed": int(split_seed),
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "B": b,
        "A": a,
        "alphas": [float(v) for v in alphas.tolist()],
        "alpha_grid_note": "10 equally spaced practical points on (0,1], implemented as np.linspace(0.1,1.0,10)",
        "ci_level": float(cfg.ci_level),
        "z_ci": float(z_ci),
        "selection_rule": cfg.selection_rule,
        "family_identity_fix": "each split2 bootstrap resample has its own Stage-1 fit, then Stage-2 on the same resample",
        "deployment": "full-data Stage-1 base once + full-data Stage-2 per alpha; recombine pointwise by alpha_star(x)",
        "normalization": "x raw->[-1,1] via minmax; y standardized on full train pool; latent-f intervals de-standardized to raw y",
        "device": str(device),
        "dtype": str(dtype),
        "use_ciq": bool(cfg.use_ciq),
        "avg_final_hit": float(deploy_hit.mean()),
        "unique_alpha_star": sorted(set(float(v) for v in alpha_star.tolist())),
    }

    with open(os.path.join(out_dir, f"meta_rep{rep_id:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"Saved deploy CSV to {deploy_csv}")
    log(f"Final single-rep deploy hit mean = {deploy_hit.mean():.4f}")
    log("Done.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-split bootstrap SVGP calibration with family-identity fix and full deploy"
    )
    parser.add_argument("--rep-id", type=int, required=True, help="Replication id, e.g. SLURM_ARRAY_TASK_ID")
    parser.add_argument("--B", dest="b_bootstrap", type=int, default=100, help="Bootstrap count per replication")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: /scratch/$USER/15_points_familyfix_strict/results",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Optional comma-separated alphas. Default uses np.linspace(0.1,1.0,10).",
    )
    parser.add_argument(
        "--selection-rule",
        type=str,
        default=None,
        choices=["largest_meeting_target", "closest"],
        help="Alpha*(x) selection rule.",
    )
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    cfg = CFG()
    args = parse_args()

    if args.alphas is not None:
        alpha_list = [float(x) for x in args.alphas.split(",") if x.strip() != ""]
        alphas = np.array(alpha_list, dtype=float)
    else:
        alphas = np.array(list(cfg.alphas), dtype=float)

    if np.any(alphas <= 0.0):
        raise ValueError("All alpha values must be > 0. alpha=0 is not allowed in this code.")

    if args.selection_rule is not None:
        cfg.selection_rule = args.selection_rule
    if args.stage1_epochs is not None:
        cfg.stage1_epochs = int(args.stage1_epochs)
    if args.stage2_epochs is not None:
        cfg.stage2_epochs = int(args.stage2_epochs)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.log_epoch:
        cfg.log_epoch = True

    if args.out_dir is None:
        user = os.environ.get("USER", "yy101")
        out_dir = f"/scratch/{user}/15_points_familyfix_strict/results"
    else:
        out_dir = args.out_dir

    run_single_rep(
        cfg=cfg,
        rep_id=int(args.rep_id),
        b_bootstrap=int(args.b_bootstrap),
        alphas=alphas,
        out_dir=str(out_dir),
    )


if __name__ == "__main__":
    main()
