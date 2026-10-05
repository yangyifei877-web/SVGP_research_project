#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calib_fullci_pointwise.py

Scheme A (align-to-TVb idea): calibrate / evaluate the SAME CI mechanism you will use finally.

For each replication rep_id:
  - Build fixed global training pool X (seed=cfg.seed), latent noiseless y=f0(x).
  - Apply x normalization: minmax->[0,1] then [-1,1]; Z fixed M inducing points.
  - Apply y z-score on full pool; convert noise_raw to standardized units; CI reported in RAW y.
  - Train Stage-1 base on FULL DATA (beta=1).
  - For each alpha in a 15-point uniform grid in (0,1]:
        * Reset to Stage-1 base
        * Stage-2 covariance-only training on FULL DATA with beta=1/alpha
        * Predict pointwise CI on test grid
        * Compute hit indicator: 1{ f0(x*) in CI_alpha(x*) }
  - Save hits[A,J] and a CSV.

This removes the mismatch: "bootstrap-retrained CI" (calibration) vs "full-data CI" (final evaluation).
You can still keep your pointwise alpha* selection in downstream aggregation.

Outputs per rep:
  - hits_repXXX_A15_JJ.npy
  - pointwise_hits_repXXX_A15.csv
  - meta_repXXX.json
  - diag_pointwise_hits_repXXX.png   (optional quick-look)

CPU-ready; also works on GPU if available.
"""

import os
import math
import time
import json
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd

import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def clone_state_dict(sd):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().clone() if torch.is_tensor(v) else v
    return out


def z_from_ci_level(ci_level: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    p = 0.5 * (1.0 + float(ci_level))
    t = torch.tensor(2.0 * p - 1.0, device=device, dtype=dtype)
    return math.sqrt(2.0) * torch.erfinv(t)


def f0(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * math.pi * x) + 0.5 * torch.cos(3 * x)


def make_stratified_points_1d(n, low, high, generator, device, dtype):
    i = torch.arange(n, device=device, dtype=dtype)
    u = (i + torch.rand(n, generator=generator, device=device, dtype=dtype)) / n
    x = low + (high - low) * u
    perm = torch.randperm(n, generator=generator, device=device)
    return x[perm].unsqueeze(-1)


@dataclass
class CFG:
    n_train: int = 500
    n_test: int = 50
    train_low: float = -2.5
    train_high: float = 2.5
    test_low: float = -2.5
    test_high: float = 2.5

    use_fp64: bool = True
    inducing_points: int = 50
    stage1_epochs: int = 2000
    stage2_epochs: int = 800
    batch_size: int = 500
    lr_adam_stage1: float = 2e-2
    lr_adam_stage2: float = 2e-2
    jitter: float = 1e-2

    seed: int = 13
    noise_raw: float = 3e-3

    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4

    ci_level: float = 0.95
    save_plot: bool = True


def alpha_grid_uniform_15(alpha_min: float = 0.0, alpha_max: float = 1.0) -> np.ndarray:
    if alpha_max <= 0.0:
        raise ValueError("alpha_max must be > 0")
    if alpha_min <= 0.0:
        return np.linspace(1.0 / 15.0, 1.0, 15, dtype=float)
    if not (0.0 < alpha_min <= alpha_max <= 1.0):
        raise ValueError("Need 0 < alpha_min <= alpha_max <= 1")
    return np.linspace(alpha_min, alpha_max, 15, dtype=float)


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


def freeze_all_params(model, likelihood):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in likelihood.parameters():
        p.requires_grad_(False)


def pick_covariance_params_svgp(model):
    trainables = []
    vs = getattr(model, "variational_strategy", None)
    vdist_mod = None
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
        allow = ("chol", "covar", "factor", "raw_var", "q_scale_tril", "scale_tril")
        for n, p in model.named_parameters():
            low = n.lower()
            if "variational_strategy" in low and any(k in low for k in allow) and "mean" not in low:
                p.requires_grad_(True)
                trainables.append(p)

    return trainables


def stage1_train(model, likelihood, x_train_std, y_train_std, cfg: CFG):
    N = x_train_std.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=1.0)

    likelihood.noise_covar.raw_noise.requires_grad_(False)
    opt = torch.optim.Adam([
        {"params": model.variational_parameters()},
        {"params": model.hyperparameters()},
        {"params": likelihood.parameters()},
    ], lr=cfg.lr_adam_stage1)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train_std),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()

    best_loss, best_epoch = float("inf"), 0
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())
    patience = 0

    with gpytorch.settings.cholesky_jitter(cfg.jitter):
        for ep in range(1, cfg.stage1_epochs + 1):
            total, nb = 0.0, 0
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
                nb += 1

            avg = total / max(1, nb)

            if best_loss - avg > cfg.early_stop_min_delta:
                best_loss, best_epoch = avg, ep
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())
                patience = 0
            else:
                patience += 1

            if patience >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_epoch, best_loss


def stage2_cov_only(model, likelihood, x_train_std, y_train_std, beta: float, cfg: CFG):
    N = x_train_std.size(0)
    freeze_all_params(model, likelihood)
    cov_params = pick_covariance_params_svgp(model)
    if not cov_params:
        raise RuntimeError("No covariance params found for Stage-2 (cov-only).")

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N, beta=beta)
    opt = torch.optim.Adam(cov_params, lr=cfg.lr_adam_stage2)

    loader = DataLoader(
        TensorDataset(x_train_std, y_train_std),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model.train()
    likelihood.train()

    best_loss = float("inf")
    best_model_sd = clone_state_dict(model.state_dict())
    best_lik_sd = clone_state_dict(likelihood.state_dict())

    with gpytorch.settings.cholesky_jitter(cfg.jitter):
        for _ in range(1, cfg.stage2_epochs + 1):
            total, nb = 0.0, 0
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
                nb += 1

            avg = total / max(1, nb)
            if avg < best_loss:
                best_loss = avg
                best_model_sd = clone_state_dict(model.state_dict())
                best_lik_sd = clone_state_dict(likelihood.state_dict())

    model.load_state_dict(best_model_sd)
    likelihood.load_state_dict(best_lik_sd)
    model.eval()
    likelihood.eval()
    return best_loss


def run_single_rep(rep_id: int, out_dir: str, alpha_min: float, alpha_max: float, no_plot: bool):
    cfg = CFG()
    cfg.save_plot = (not no_plot)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64 if cfg.use_fp64 else torch.float32)
    dtype = torch.get_default_dtype()

    seed_rep = int(cfg.seed + rep_id)
    torch.manual_seed(seed_rep)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_rep)

    log(f"=== rep_id={rep_id:03d} | seed_rep={seed_rep} | device={device} | dtype={dtype} ===")

    pool_gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    x_train_raw = make_stratified_points_1d(cfg.n_train, cfg.train_low, cfg.train_high, pool_gen, device, dtype)
    y_train_raw = f0(x_train_raw).squeeze(-1)

    x_test_raw = torch.linspace(cfg.test_low, cfg.test_high, cfg.n_test, device=device, dtype=dtype).unsqueeze(1)
    y_test_true_raw = f0(x_test_raw).squeeze(-1)

    train_low = torch.as_tensor(cfg.train_low, device=device, dtype=dtype)
    train_high = torch.as_tensor(cfg.train_high, device=device, dtype=dtype)
    span = (train_high - train_low).clamp_min(1e-12)

    def x_to_std(x_raw):
        x01 = (x_raw - train_low) / span
        return 2.0 * x01 - 1.0

    x_train_std = x_to_std(x_train_raw)
    x_test_std = x_to_std(x_test_raw)

    Z_raw = torch.linspace(cfg.train_low, cfg.train_high, cfg.inducing_points, device=device, dtype=dtype).unsqueeze(1)
    Z_std = x_to_std(Z_raw)

    y_mean = y_train_raw.mean()
    y_scale = y_train_raw.std(unbiased=False).clamp_min(1e-12)
    y_train_std = (y_train_raw - y_mean) / y_scale

    noise_std = torch.as_tensor(cfg.noise_raw, device=device, dtype=dtype) / (y_scale ** 2)

    alphas = alpha_grid_uniform_15(alpha_min=alpha_min, alpha_max=alpha_max)
    A = len(alphas)
    J = cfg.n_test
    z_ci = float(z_from_ci_level(cfg.ci_level, device=device, dtype=dtype).item())

    model_base = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik_base = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    with torch.no_grad():
        model_base.covar_module.base_kernel.lengthscale.fill_(1.0)
        model_base.covar_module.outputscale.fill_(1.0)
        lik_base.noise = noise_std
    lik_base.noise_covar.raw_noise.requires_grad_(False)

    be1, bl1 = stage1_train(model_base, lik_base, x_train_std, y_train_std, cfg)
    base_m = clone_state_dict(model_base.state_dict())
    base_l = clone_state_dict(lik_base.state_dict())
    log(f"[rep {rep_id:03d}] Stage-1 full-data done | best -ELBO={bl1:.6f}@{be1}")

    hits = np.zeros((A, J), dtype=np.int8)

    model = SVGPModel(Z_std.clone().contiguous()).to(device=device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device=device)

    for a_idx, alpha in enumerate(alphas):
        alpha = float(alpha)
        beta = 1.0 / alpha

        model.load_state_dict(base_m)
        lik.load_state_dict(base_l)
        model.eval()
        lik.eval()

        _ = stage2_cov_only(model, lik, x_train_std, y_train_std, beta=beta, cfg=cfg)

        with torch.no_grad():
            pred = model(x_test_std)
            mu_std = pred.mean
            var_std = pred.variance.clamp_min(1e-24)

            mu_raw = mu_std * y_scale + y_mean
            std_raw = (var_std * (y_scale ** 2)).sqrt().clamp_min(1e-12)

            lo = mu_raw - z_ci * std_raw
            hi = mu_raw + z_ci * std_raw

            inc = ((y_test_true_raw >= lo) & (y_test_true_raw <= hi)).to(torch.int8)

        hits[a_idx, :] = inc.detach().cpu().numpy()
        log(f"[rep {rep_id:03d}] alpha={alpha:.4f} done")

    npy_path = os.path.join(out_dir, f"hits_rep{rep_id:03d}_A{A}_J{J}.npy")
    np.save(npy_path, hits)

    x_axis = x_test_raw.squeeze(-1).detach().cpu().numpy()
    data = {"x": x_axis}
    for a_idx, alpha in enumerate(alphas):
        data[f"alpha_{float(alpha):.6f}"] = hits[a_idx, :]
    df = pd.DataFrame(data)
    csv_path = os.path.join(out_dir, f"pointwise_hits_rep{rep_id:03d}_A{A}.csv")
    df.to_csv(csv_path, index=False)

    if cfg.save_plot:
        plt.figure(figsize=(10, 4))
        for a_idx, alpha in enumerate(alphas):
            plt.plot(x_axis, hits[a_idx, :], linewidth=1.0)
        plt.axhline(cfg.ci_level, linestyle="--", linewidth=1.0)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("x")
        plt.ylabel("hit (vs f0)")
        plt.title(f"rep {rep_id:03d} | pointwise hits (A=15, full-data CI mechanism)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"diag_pointwise_hits_rep{rep_id:03d}.png"), dpi=160)
        plt.close()

    meta = {
        "rep_id": rep_id,
        "seed_pool": int(cfg.seed),
        "seed_rep": seed_rep,
        "n_train": cfg.n_train,
        "n_test": cfg.n_test,
        "A": A,
        "alphas": [float(a) for a in alphas],
        "ci_level": cfg.ci_level,
        "z_ci": z_ci,
        "scheme": "Scheme A: fixed full-data CI mechanism; evaluate hits vs true f0",
        "two_stage": "Stage-1 beta=1; Stage-2 cov-only beta=1/alpha",
        "normalization": "x minmax->[0,1] then [-1,1]; y zscore on full pool; CI in raw",
        "inducing_points": int(cfg.inducing_points),
        "alpha_grid": "15-point uniform in (0,1] (or linspace if alpha_min>0)",
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "saved_plot": bool(cfg.save_plot),
    }
    with open(os.path.join(out_dir, f"meta_rep{rep_id:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"[rep {rep_id:03d}] saved: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Scheme A: full-data CI mechanism + pointwise hits vs f0 (A=15 alphas)")
    p.add_argument("--rep-id", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--alpha-min", type=float, default=0.0,
                   help="If <=0 uses grid {1/15,...,1}; else linspace(alpha_min, alpha_max, 15).")
    p.add_argument("--alpha-max", type=float, default=1.0)
    p.add_argument("--no-plot", action="store_true", help="Disable saving per-rep diagnostic plot.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_rep(
        rep_id=int(args.rep_id),
        out_dir=str(args.out_dir),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        no_plot=bool(args.no_plot),
    )
