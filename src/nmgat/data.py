"""Graph construction, training masks, and training-only input normalization."""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Any
import numpy as np
import pandas as pd
import torch

def make_default_offsets(radius: int = 2, include_diagonal: bool = True) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for dz in range(-radius, radius + 1):
        for dn in range(-radius, radius + 1):
            if dz == 0 and dn == 0:
                continue
            if not include_diagonal and (abs(dz) + abs(dn) != 1):
                continue
            offsets.append((dz, dn))
    return tuple(dict.fromkeys(offsets))


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])


def infer_feature_cols(df_train: pd.DataFrame, feature_cols: Sequence[str] | None, exclude_cols: Sequence[str]) -> list[str]:
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df_train.columns]
        if missing:
            raise ValueError(f"feature_cols missing in train: {missing}")
        return list(feature_cols)
    exclude = set(exclude_cols)
    cols = [c for c in df_train.columns if c not in exclude and pd.api.types.is_numeric_dtype(df_train[c])]
    if not cols:
        raise RuntimeError("No numeric feature columns found. Provide feature_cols explicitly.")
    return cols


def infer_discrete_feature_mask(
    df_train: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    force_discrete_cols: Sequence[str] = (),
    int_tol: float = 1e-6,
) -> list[bool]:
    forced = set(force_discrete_cols)
    mask: list[bool] = []
    for col in feature_cols:
        if col in forced:
            mask.append(True)
            continue

        vals = df_train[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            mask.append(False)
            continue

        v = vals[finite]
        is_integer_like = np.isclose(v, np.round(v), atol=float(int_tol), rtol=0.0).all()
        mask.append(bool(is_integer_like))
    return mask


def validate_numeric(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    arr = df[list(cols)].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} has {bad} non-finite feature values")


def split_train_by_uncertainty(values: Sequence[float], trim_ratio: float) -> tuple[list[int], list[int]]:
    ratio = float(trim_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("uncertainty_trim_ratio must be in [0.0, 1.0)")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("uncertainty values must be one-dimensional")
    if len(arr) == 0:
        return [], []
    if not np.isfinite(arr).all():
        raise ValueError("train uncertainty contains non-finite values")

    trim_count = math.ceil(len(arr) * ratio)
    if trim_count >= len(arr):
        raise ValueError(
            f"uncertainty_trim_ratio={ratio} masks all {len(arr)} training samples; lower the ratio"
        )

    order = np.argsort(-arr, kind="mergesort")
    masked_idx = sorted(order[:trim_count].tolist())
    masked_set = set(masked_idx)
    loss_idx = [idx for idx in range(len(arr)) if idx not in masked_set]
    return loss_idx, masked_idx


def build_edge_index(coords: Sequence[Tuple[int, int]], offsets: Sequence[Tuple[int, int]]) -> torch.Tensor:
    coord_to_idx = {c: i for i, c in enumerate(coords)}
    edges: set[tuple[int, int]] = set()
    for i, (z, n) in enumerate(coords):
        for dz, dn in offsets:
            j = coord_to_idx.get((z + dz, n + dn))
            if j is not None:
                edges.add((i, j))
                edges.add((j, i))
    if not edges:
        raise ValueError("No edges were created. Check neighbor offsets.")
    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()


def standardize_by_idx(x: torch.Tensor, idx: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_t = torch.tensor(idx, dtype=torch.long, device=x.device)
    mean = x[idx_t].mean(dim=0)
    std = x[idx_t].std(dim=0, unbiased=False)
    std = torch.clamp(std, min=1e-8)
    return (x - mean) / std, mean, std


@dataclass
class PreparedData:
    coords: list[tuple[int, int]]
    a_vals: list[int]
    edge_index: torch.Tensor
    x: torch.Tensor
    eth: torch.Tensor
    eexp: torch.Tensor
    residual: torch.Tensor
    train_idx: list[int]
    loss_train_idx: list[int]
    masked_train_idx: list[int]
    test_idx: list[int]
    unlabeled_idx: list[int]
    dataset_split: list[str]
    participates_in_loss: list[bool]
    train_uncertainty: list[float]
    feature_cols: list[str]
    discrete_feature_idx: list[int]
    continuous_feature_idx: list[int]
    discrete_feature_cols: list[str]
    continuous_feature_cols: list[str]

    def to(self, device: torch.device) -> "PreparedData":
        return PreparedData(
            coords=self.coords,
            a_vals=self.a_vals,
            edge_index=self.edge_index.to(device),
            x=self.x.to(device),
            eth=self.eth.to(device),
            eexp=self.eexp.to(device),
            residual=self.residual.to(device),
            train_idx=self.train_idx,
            loss_train_idx=self.loss_train_idx,
            masked_train_idx=self.masked_train_idx,
            test_idx=self.test_idx,
            unlabeled_idx=self.unlabeled_idx,
            dataset_split=self.dataset_split,
            participates_in_loss=self.participates_in_loss,
            train_uncertainty=self.train_uncertainty,
            feature_cols=self.feature_cols,
            discrete_feature_idx=self.discrete_feature_idx,
            continuous_feature_idx=self.continuous_feature_idx,
            discrete_feature_cols=self.discrete_feature_cols,
            continuous_feature_cols=self.continuous_feature_cols,
        )


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return drop_unnamed(pd.read_csv(path))


def prepare_data(
    cfg: Any,
    *,
    z_col: str = "Z",
    n_col: str = "N",
    a_col: str = "A",
    eth_col: str = "Eth",
    eexp_col: str = "Eexp",
    residual_col: str = "residual",
    feature_cols: Sequence[str] | None = None,
) -> PreparedData:
    uncertainty_col = "uncertainty"
    data_path = Path(cfg.data_path)
    df_train = load_df(data_path / cfg.train_csv)
    df_test = load_df(data_path / cfg.test_csv)
    df_unl = None
    if cfg.unlabeled_csv is not None:
        df_unl = load_df(data_path / cfg.unlabeled_csv)

    def enforce(df: pd.DataFrame, name: str, require_target: bool) -> None:
        for c in (z_col, n_col):
            if c not in df.columns:
                raise ValueError(f"{name} missing column '{c}'")
            df[c] = df[c].astype(int)
        if a_col in df.columns:
            df[a_col] = df[a_col].astype(int)
        else:
            df[a_col] = (df[z_col] + df[n_col]).astype(int)
        if eth_col not in df.columns:
            raise ValueError(f"{name} missing column '{eth_col}'")
        df[eth_col] = df[eth_col].astype(float)
        if require_target:
            for c in (eexp_col, residual_col):
                if c not in df.columns:
                    raise ValueError(f"{name} missing column '{c}'")
                df[c] = df[c].astype(float)

    enforce(df_train, "train", True)
    enforce(df_test, "test", True)
    if df_unl is not None:
        enforce(df_unl, "unlabeled", False)
        df_unl[eexp_col] = np.nan
        df_unl[residual_col] = np.nan

    if uncertainty_col not in df_train.columns:
        raise ValueError(f"train missing column '{uncertainty_col}'")
    train_uncertainty_series = pd.to_numeric(df_train[uncertainty_col], errors="coerce")
    train_uncertainty_arr = train_uncertainty_series.to_numpy(dtype=float)
    if not np.isfinite(train_uncertainty_arr).all():
        raise ValueError("train uncertainty must be finite numeric values")
    df_train[uncertainty_col] = train_uncertainty_arr
    loss_train_idx, masked_train_idx = split_train_by_uncertainty(train_uncertainty_arr, cfg.uncertainty_trim_ratio)

    feat_cols = infer_feature_cols(df_train, feature_cols, cfg.exclude_cols)
    discrete_mask = infer_discrete_feature_mask(
        df_train,
        feat_cols,
        force_discrete_cols=cfg.force_discrete_cols,
        int_tol=cfg.discrete_int_tol,
    )
    discrete_feature_idx = [i for i, is_discrete in enumerate(discrete_mask) if is_discrete]
    continuous_feature_idx = [i for i, is_discrete in enumerate(discrete_mask) if not is_discrete]
    discrete_feature_cols = [feat_cols[i] for i in discrete_feature_idx]
    continuous_feature_cols = [feat_cols[i] for i in continuous_feature_idx]

    validate_numeric(df_train, feat_cols, "train")
    validate_numeric(df_test, feat_cols, "test")
    if df_unl is not None:
        validate_numeric(df_unl, feat_cols, "unlabeled")

    df_unl_f = None
    if df_unl is not None:
        labeled_keys = set(zip(df_train[z_col].tolist(), df_train[n_col].tolist())) | set(
            zip(df_test[z_col].tolist(), df_test[n_col].tolist())
        )
        keep_mask = [((int(z), int(n)) not in labeled_keys) for z, n in zip(df_unl[z_col], df_unl[n_col])]
        df_unl_f = df_unl.loc[keep_mask].reset_index(drop=True)

    frames = [df_train, df_test] + ([df_unl_f] if df_unl_f is not None else [])
    df_all = pd.concat(frames, ignore_index=True)

    coords = list(zip(df_all[z_col].astype(int).tolist(), df_all[n_col].astype(int).tolist()))
    a_vals = df_all[a_col].astype(int).tolist()
    edge_index = build_edge_index(coords, cfg.neighbor_offsets)

    x = torch.tensor(df_all[feat_cols].to_numpy(dtype=float), dtype=torch.float32)
    eth = torch.tensor(df_all[eth_col].to_numpy(dtype=float), dtype=torch.float32)
    eexp = torch.tensor(df_all[eexp_col].to_numpy(dtype=float), dtype=torch.float32)
    residual = torch.tensor(df_all[residual_col].to_numpy(dtype=float), dtype=torch.float32)

    n_tr, n_te = len(df_train), len(df_test)
    n_unl = len(df_unl_f) if df_unl_f is not None else 0
    train_idx = list(range(0, n_tr))
    test_idx = list(range(n_tr, n_tr + n_te))
    unl_idx = list(range(n_tr + n_te, n_tr + n_te + n_unl)) if n_unl > 0 else []
    masked_train_set = set(masked_train_idx)
    dataset_split = (["train"] * n_tr) + (["test"] * n_te) + (["unlabeled"] * n_unl)
    participates_in_loss = [idx not in masked_train_set for idx in train_idx] + ([False] * (n_te + n_unl))
    train_uncertainty = train_uncertainty_arr.tolist() + ([float("nan")] * (n_te + n_unl))

    return PreparedData(
        coords=coords,
        a_vals=a_vals,
        edge_index=edge_index,
        x=x,
        eth=eth,
        eexp=eexp,
        residual=residual,
        train_idx=train_idx,
        loss_train_idx=loss_train_idx,
        masked_train_idx=masked_train_idx,
        test_idx=test_idx,
        unlabeled_idx=unl_idx,
        dataset_split=dataset_split,
        participates_in_loss=participates_in_loss,
        train_uncertainty=train_uncertainty,
        feature_cols=feat_cols,
        discrete_feature_idx=discrete_feature_idx,
        continuous_feature_idx=continuous_feature_idx,
        discrete_feature_cols=discrete_feature_cols,
        continuous_feature_cols=continuous_feature_cols,
    )
