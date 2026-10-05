#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Double-check experiment with pointwise alpha*(x) for SVGP two-stage calibration.

Config/normalization/optimization are aligned with the two-split bootstrap
calibration code, EXCEPT:
  - no two-way split (use all n_train points),
  - no bootstrap (direct frequentist replications vs true f0),
  - coverage is computed vs true f0(x) at test grid.

We still:
  - fix Z (M=50, symmetric normalization on [train_low, train_high]),
  - Stage-1: alpha=1 (ELBO, early stopping),
  - Stage-2: covariance-only training with beta=1/alpha.

We use an alpha_star[x_j] vector (length 50). For each job (specified by
--alpha-idx), we pick ONE unique alpha value and only update the coverage
for test points whose alpha_star == that alpha.
"""

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
import argparse
import os

import torch
import gpytorch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


# ---------------- logging ----------------
def now():
    return time.strftime("%H:%M:%S")


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def gpu_mem_mb():
    return (torch.cuda.memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0


# ---------------- helpers ----------------
def has_setting(name: str) -> bool:
    return hasattr(gpytorch.settings, name)


def ciq_context():
    """Optional CIQ settings (kept for compatibility, default use_ciq=False)."""
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


# true latent f (noiseless)
def f0(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)


# 1D stratified (LHS-like) points
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

    # model/train (match calibration code)
    use_fp64: bool = True
    use_ciq: bool = False
    inducing_points: int = 50
    stage1_epochs: int = 2000
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2
    seed: int = 13  # base seed; per rep: seed + r

    # early stopping for Stage-1
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    # experiment
    replications: int = 100  # R

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


# ---------------- Stage-1 (copied from calibration code) ----------------
def stage1_train(model, likelihood, x_train_std, y_train, lr, epochs, jitter,
                 batch_size, early_stop_patience, early_stop_min_delta,
                 log_epoch=False, log_every=10):
    """Stage-1: train kernel + variational mean (alpha=1)."""
    N = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=1.0)

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


# ---------------- Stage-2 (covariance only, copied from calibration) ----------------
def stage2_train_cov_only(model, likelihood, x_train_std, y_train, beta, lr,
                          epochs, jitter, batch_size):
    """Stage-2: fix everything, train only variational covariance (beta = 1/alpha)."""
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params, cov_names = pick_covariance_params_svgp(model)
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


# ---------------- alpha* (from calibrated CSV, length must be 50) ----------------
alpha_star_np = np.array([
    0.1500,  # 1
    1.0000,  # 2
    0.4300,  # 3
    0.2600,  # 4
    0.7000,  # 5
    1.0000,  # 6
    0.4100,  # 7
    0.5400,  # 8
    1.0000,  # 9
    1.0000,  # 10
    0.4200,  # 11
    0.6700,  # 12
    1.0000,  # 13
    1.0000,  # 14
    0.7000,  # 15
    0.9700,  # 16
    1.0000,  # 17
    0.8700,  # 18
    0.9900,  # 19
    1.0000,  # 20
    1.0000,  # 21
    1.0000,  # 22
    1.0000,  # 23
    1.0000,  # 24
    1.0000,  # 25
    1.0000,  # 26
    1.0000,  # 27
    1.0000,  # 28
    1.0000,  # 29
    1.0000,  # 30
    1.0000,  # 31
    1.0000,  # 32
    1.0000,  # 33
    1.0000,  # 34
    1.0000,  # 35
    1.0000,  # 36
    1.0000,  # 37
    0.9600,  # 38
    1.0000,  # 39
    1.0000,  # 40
    1.0000,  # 41
    1.0000,  # 42
    1.0000,  # 43
    1.0000,  # 44
    0.7600,  # 45
    0.5100,  # 46
    1.0000,  # 47
    1.0000,  # 48
    1.0000,  # 49
    0.3400,  # 50
], dtype=np.float64)


# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--alpha-idx",
        type=int,
        default=-1,
        help=(
            "Index into sorted unique alphas in alpha_star_np (0..K-1). "
            "If -1, run ALL unique alphas in this job (mainly for local debug)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CFG()
    assert cfg.n_test == len(alpha_star_np), "n_test must equal length of alpha_star_np"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)

    # ---- global test grid & normalization params (match calibration) ----
    x_test_raw = torch.linspace(cfg.test_low, cfg.test_high, cfg.n_test,
                                device=device, dtype=torch.get_default_dtype()).unsqueeze(1)
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

    Z_raw = torch.linspace(cfg.train_low, cfg.train_high, cfg.inducing_points,
                           device=device, dtype=torch.get_default_dtype()).unsqueeze(1)
    Z_std = to_std(Z_raw)

    left_std = float(to_std(torch.tensor([[cfg.train_low]], device=device)))
    right_std = float(to_std(torch.tensor([[cfg.train_high]], device=device)))
    log(
        f"Symmetric norm: raw [{cfg.train_low:.2f},{cfg.train_high:.2f}] "
        f"-> std [{left_std:.2f},{right_std:.2f}] (≈ -1,+1)"
    )
    log(f"Z_std head/tail: {float(Z_std[0]):.3f}, {float(Z_std[-1]):.3f}")

    # ---- unique alphas & subset for this job ----
    unique_alphas_all = np.unique(alpha_star_np)
    K = len(unique_alphas_all)
    if args.alpha_idx >= 0:
        assert 0 <= args.alpha_idx < K, f"alpha-idx must be in [0, {K-1}]"
        unique_alphas = np.array([unique_alphas_all[args.alpha_idx]])
        log(f"Running ONLY alpha_idx={args.alpha_idx}, alpha={unique_alphas[0]:.4f}")
    else:
        unique_alphas = unique_alphas_all
        log(f"Running ALL unique alphas: {unique_alphas}")

    idx_by_alpha = {a: np.where(alpha_star_np == a)[0] for a in unique_alphas}
    log("alpha_star counts in THIS job: " +
        ", ".join([f"{a:.3f}: {len(idx_by_alpha[a])}" for a in unique_alphas]))

    # ---- storage (coverage) ----
    R, J = cfg.replications, cfg.n_test
    indicators = np.full((R, J), np.nan, dtype=np.float64)

    # ---- main replication loop (align seeds & data logic with calibration) ----
    for r in range(R):
        seed_r = cfg.seed + r
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

        log(f"\n========== Replication {r+1}/{R} | seed={seed_r} ==========")

        # build model/likelihood fresh with fixed Z_std
        model = SVGPModel(Z_std.clone().contiguous()).to(device=device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
        with torch.no_grad():
            model.covar_module.base_kernel.lengthscale.fill_(1.0)
            model.covar_module.outputscale.fill_(1.0)
            likelihood.noise = torch.as_tensor(3e-3, dtype=torch.get_default_dtype(), device=device)
        likelihood.noise_covar.raw_noise.requires_grad_(False)

        # Stage-1 (alpha=1)
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
            mu_pred = pred_f.mean
            mse_s1 = torch.mean((mu_pred - y_test_true) ** 2).item()

        log(f"[rep {r+1:03d}/{R}] Stage-1 (α=1.000) | best -ELBO={best_loss_s1:.6f} "
            f"@epoch {best_epoch_s1} | MSE={mse_s1:.6f}")

        # checkpoint after Stage-1
        s1_model_sd = clone_state_dict(model.state_dict())
        s1_lik_sd = clone_state_dict(likelihood.state_dict())

        # Stage-2 for the selected alpha values in this job
        for alpha in unique_alphas:
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

                idxs_np = idx_by_alpha[alpha]
                idxs_torch = torch.from_numpy(idxs_np).to(device=mu.device, dtype=torch.long)

                lower = mu[idxs_torch] - 1.96 * std[idxs_torch]
                upper = mu[idxs_torch] + 1.96 * std[idxs_torch]
                y_sel = y_test_true[idxs_torch]

                cover = ((y_sel >= lower) & (y_sel <= upper)).to(torch.float64)
                indicators[r, idxs_np] = cover.cpu().numpy()

                mse = torch.mean((mu - y_test_true) ** 2).item()

            log(f"[rep {r+1:03d}/{R}] α={float(alpha):.3f} (Stage-2) | "
                f"best -ELBO={best_loss_s2:.6f} @epoch {best_epoch_s2} | MSE={mse:.6f}")

    # ---- aggregate coverage under alpha*(x) ----
    x_axis = x_test_raw.squeeze(-1).cpu().numpy()
    coverage_star = np.nanmean(indicators, axis=0)

    # choose suffix for filenames
    if args.alpha_idx >= 0:
        suffix = f"_alphaidx{args.alpha_idx}"
    else:
        suffix = "_allalphas"

    df = pd.DataFrame({
        "x": x_axis,
        "alpha_star": alpha_star_np,
        "coverage_alpha_star": coverage_star,
    })
    out_csv = f"alpha_star_pointwise_coverage{suffix}.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved {out_csv} in {os.getcwd()}")

    out_npy = f"indicators_alpha_star{suffix}.npy"
    np.save(out_npy, indicators)
    log(f"Saved {out_npy} with shape {indicators.shape}")

    log("\n=== Pointwise coverage summary (first few points) ===")
    for j in range(min(10, len(x_axis))):
        cov_val = coverage_star[j]
        log(f"x={x_axis[j]: .3f} | alpha*={alpha_star_np[j]:.3f} | cov={cov_val:.3f}")


if __name__ == "__main__":
    main()
