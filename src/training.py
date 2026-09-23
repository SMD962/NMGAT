"""Shared A/B training engine: only the supervised-label mask differs."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm.auto import tqdm
from .model import MassNet
from .data import PreparedData, prepare_data, make_default_offsets, standardize_by_idx, build_edge_index

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((a - b) ** 2))


def add_feature_noise(
    x: torch.Tensor,
    *,
    continuous_feature_idx: Sequence[int],
    noise_std: float,
    noise_prob: float = 1.0,
    node_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    if noise_std <= 0.0 or noise_prob <= 0.0 or len(continuous_feature_idx) == 0:
        return x
    if noise_prob < 1.0:
        if torch.rand((), device=x.device).item() >= noise_prob:
            return x

    feat_t = torch.tensor(continuous_feature_idx, dtype=torch.long, device=x.device)
    noise = torch.zeros((x.shape[0], len(continuous_feature_idx)), dtype=x.dtype, device=x.device)

    if node_idx is None:
        noise.normal_(mean=0.0, std=float(noise_std))
    else:
        node_t = torch.tensor(node_idx, dtype=torch.long, device=x.device)
        subset_noise = torch.randn(
            (len(node_idx), len(continuous_feature_idx)),
            dtype=x.dtype,
            device=x.device,
        ) * float(noise_std)
        noise[node_t] = subset_noise

    x_noisy = x.clone()
    x_noisy[:, feat_t] = x_noisy[:, feat_t] + noise
    return x_noisy


def save_predictions_csv(
    out_path: Path,
    indices: Sequence[int],
    coords: Sequence[Tuple[int, int]],
    a_vals: Sequence[int],
    dataset_split: Sequence[str],
    participates_in_loss: Sequence[bool],
    train_uncertainty: Sequence[float],
    pred_residual: Sequence[float],
    eth: Sequence[float],
    eexp: Sequence[float],
    residual: Sequence[float],
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Z",
            "N",
            "A",
            "dataset_split",
            "participates_in_loss",
            "train_uncertainty",
            "pred_residual",
            "pred_binding_energy",
            "actual_binding_energy",
            "theory_binding_energy",
            "actual_residual",
            "pred_minus_actual_residual",
        ])
        for idx in indices:
            z, n = coords[idx]
            a = int(a_vals[idx])
            split = str(dataset_split[idx])
            in_loss = bool(participates_in_loss[idx])
            unc = float(train_uncertainty[idx])
            pred_r = float(pred_residual[idx])
            eth_v = float(eth[idx])
            eexp_v = float(eexp[idx])
            resid_v = float(residual[idx])
            pred_e = eth_v - pred_r
            diff = pred_r - resid_v
            writer.writerow([z, n, a, split, in_loss, unc, pred_r, pred_e, eexp_v, eth_v, resid_v, diff])


@dataclass
class TrainConfig:
    data_path: Path = Path("dataset")
    train_csv: str = "2020_WS4.csv"
    test_csv: str = "mass_testWS4.csv"
    unlabeled_csv: str | None = "unlabeled_WS4.csv"
    exclude_cols: tuple[str, ...] = ("Eexp", "residual", "uncertainty")
    neighbor_offsets: tuple[tuple[int, int], ...] = field(default_factory=make_default_offsets)
    epochs: int = 3500
    eval_every: int = 10
    lr: float = 8e-4
    weight_decay: float = 1e-3
    grad_clip: float = 2.0
    loss: str = "huber"
    huber_delta: float = 0.38
    hidden_dim: int = 64
    heads: int = 2
    dropout: float = 0.2
    concat: bool = False
    norm: str | None = "layer_norm"
    ema_start_epoch: int = 100
    ema_decay: float = 0.99
    feature_noise_std: float = 0.03
    feature_noise_prob: float = 1.0
    uncertainty_trim_ratio: float = 0.05
    noise_scope: str = "train_unlabeled"
    force_discrete_cols: tuple[str, ...] = ("Z", "N", "A")
    discrete_int_tol: float = 1e-6
    save_dir: Path = Path("artifacts")
    plot_dir: Path = Path("plots")
    device: str = "auto"


def resolve_noise_node_idx(data: PreparedData, noise_scope: str) -> list[int] | None:
    if noise_scope == "train":
        return list(data.train_idx)
    if noise_scope == "train_unlabeled":
        return list(data.train_idx) + list(data.unlabeled_idx)
    if noise_scope == "all":
        return None
    raise ValueError("noise_scope must be 'train', 'train_unlabeled', or 'all'")


def train_one_seed(data: PreparedData, cfg: TrainConfig, seed: int) -> dict[str, Any]:
    device = data.x.device
    set_seed(seed)

    train_idx = list(data.train_idx)
    loss_train_idx = list(data.loss_train_idx)
    test_idx = list(data.test_idx)

    x_std, mean, std = standardize_by_idx(data.x, train_idx)

    train_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    loss_train_t = torch.tensor(loss_train_idx, dtype=torch.long, device=device)
    test_t = torch.tensor(test_idx, dtype=torch.long, device=device)

    edge_index = data.edge_index
    noise_node_idx = resolve_noise_node_idx(data, cfg.noise_scope)

    model = MassNet(
        in_dim=x_std.shape[1],
        hidden_dim=cfg.hidden_dim,
        heads=cfg.heads,
        dropout=cfg.dropout,
        concat=cfg.concat,
        norm=cfg.norm,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 400, T_mult=2, eta_min=1e-5)
    if cfg.loss.lower() == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif cfg.loss.lower() == "huber":
        criterion = nn.SmoothL1Loss(beta=cfg.huber_delta)
    else:
        raise ValueError("loss must be 'huber' or 'mse'")

    ema_model = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay),
        use_buffers=False,
    ).to(device)
    ema_started = False
    ema_start = max(1, int(cfg.ema_start_epoch))

    history: list[dict[str, float]] = []

    pbar = tqdm(range(1, cfg.epochs + 1), desc=f"seed={seed}", leave=False)
    for epoch in pbar:
        model.train()
        opt.zero_grad(set_to_none=True)

        noisy_x = add_feature_noise(
            x_std,
            continuous_feature_idx=data.continuous_feature_idx,
            noise_std=cfg.feature_noise_std,
            noise_prob=cfg.feature_noise_prob,
            node_idx=noise_node_idx,
        )
        pred = model(noisy_x, edge_index)
        loss_val = criterion(pred[loss_train_t], data.residual[loss_train_t])
        loss_val.backward()

        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

        opt.step()
        sch.step()

        if epoch >= ema_start:
            ema_model.update_parameters(model)
            ema_started = True

        if epoch == 1 or epoch % cfg.eval_every == 0:
            model.eval()
            with torch.inference_mode():
                pred_eval = model(x_std, edge_index)
                e_pred = data.eth - pred_eval
                train_rmsd = rmse(e_pred[train_t], data.eexp[train_t])
                train_rmsd_loss_subset = rmse(e_pred[loss_train_t], data.eexp[loss_train_t])
                test_rmsd = rmse(e_pred[test_t], data.eexp[test_t])

            row = {
                "epoch": int(epoch),
                "lr": float(opt.param_groups[0]["lr"]),
                "train_rmsd": float(train_rmsd),
                "train_rmsd_loss_subset": float(train_rmsd_loss_subset),
                "test_rmsd": float(test_rmsd),
                "train_loss": float(loss_val.detach().cpu()),
            }
            history.append(row)
            pbar.set_postfix(
                {
                    "train": f"{float(train_rmsd):.4f}",
                    "train95": f"{float(train_rmsd_loss_subset):.4f}",
                    "test": f"{float(test_rmsd):.4f}",
                }
            )

    final_model: nn.Module = ema_model if ema_started else model

    seed_dir = Path(cfg.save_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), seed_dir / "final.pt")

    final_model.eval()
    with torch.inference_mode():
        pred_full = final_model(x_std, edge_index).detach().cpu().numpy()

    eth_cpu = data.eth.detach().cpu().numpy()
    eexp_cpu = data.eexp.detach().cpu().numpy()
    residual_cpu = data.residual.detach().cpu().numpy()

    e_pred_full = eth_cpu - pred_full
    final_train_rmsd = float(np.sqrt(np.mean((e_pred_full[train_idx] - eexp_cpu[train_idx]) ** 2)))
    final_train_rmsd_loss_subset = float(
        np.sqrt(np.mean((e_pred_full[loss_train_idx] - eexp_cpu[loss_train_idx]) ** 2))
    )
    final_test_rmsd = float(np.sqrt(np.mean((e_pred_full[test_idx] - eexp_cpu[test_idx]) ** 2)))

    if history:
        last_row = history[-1]
        last_train_rmsd = float(last_row["train_rmsd"])
        last_train_rmsd_loss_subset = float(last_row["train_rmsd_loss_subset"])
        last_test_rmsd = float(last_row["test_rmsd"])
    else:
        last_train_rmsd = last_train_rmsd_loss_subset = last_test_rmsd = float("nan")

    save_predictions_csv(
        seed_dir / "pred_all.csv",
        list(range(len(data.coords))),
        data.coords,
        data.a_vals,
        data.dataset_split,
        data.participates_in_loss,
        data.train_uncertainty,
        pred_full,
        eth_cpu,
        eexp_cpu,
        residual_cpu,
    )

    return {
        "seed": seed,
        "train_size": len(train_idx),
        "loss_train_size": len(loss_train_idx),
        "masked_train_size": len(data.masked_train_idx),
        "test_size": len(test_idx),
        "final_train_rmsd": final_train_rmsd,
        "final_train_rmsd_loss_subset": final_train_rmsd_loss_subset,
        "final_test_rmsd": final_test_rmsd,
        "last_train_rmsd": last_train_rmsd,
        "last_train_rmsd_loss_subset": last_train_rmsd_loss_subset,
        "last_test_rmsd": last_test_rmsd,
        "ema_started": ema_started,
        "ema_start_epoch": int(cfg.ema_start_epoch),
        "ema_decay": float(cfg.ema_decay),
        "history": history,
        "feature_cols": data.feature_cols,
        "discrete_feature_cols": data.discrete_feature_cols,
        "continuous_feature_cols": data.continuous_feature_cols,
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
        "checkpoint": str(seed_dir / "final.pt"),
        "pred_all": pred_full,
    }


def save_history_csv(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
