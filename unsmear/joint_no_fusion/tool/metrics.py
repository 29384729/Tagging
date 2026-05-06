"""ROC, early-stopping, and reconstruction metric helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from preprocessing import wrap_dphi_np


def gap_recovery(model_fpr: float, baseline_fpr: float, teacher_fpr: float) -> float:
    """Compute how much of the HLT-to-teacher FPR gap the model recovers."""
    denom = float(baseline_fpr) - float(teacher_fpr)
    if abs(denom) < 1e-12:
        return float("nan")
    return (float(baseline_fpr) - float(model_fpr)) / denom


@torch.no_grad()
def predict_joint_reco(model, loader, device=None) -> np.ndarray:
    """Collect reconstructed features from a joint model."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    reco_rows = []
    for batch in loader:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        reco, _logits = model(x, m)
        reco_rows.append(reco.detach().cpu().numpy())
    return np.concatenate(reco_rows, axis=0)


def metric_dict(res_1d: np.ndarray) -> dict[str, float]:
    """Summarize one-dimensional residuals."""
    arr = np.asarray(res_1d, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "mae": float(np.mean(np.abs(arr))),
        "p50_abs": float(np.percentile(np.abs(arr), 50)),
        "p90_abs": float(np.percentile(np.abs(arr), 90)),
        "p95_abs": float(np.percentile(np.abs(arr), 95)),
    }


def maybe_wrap_residual(name: str, feat_idx: int, residual: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Wrap dPhi residuals in physical space before computing metrics."""
    if str(name) == "dPhi" or int(feat_idx) == 1:
        sc = float(scale)
        return wrap_dphi_np(np.asarray(residual) * sc) / sc
    return np.asarray(residual)


def _auc_scores(
    labels: Sequence[float] | np.ndarray,
    preds: Sequence[float] | np.ndarray,
    sample_weight: Optional[Sequence[float] | np.ndarray] = None,
    *,
    use_sample_weight: bool,
) -> tuple[float, float]:
    labels_np = np.asarray(labels)
    preds_np = np.asarray(preds)
    auc = float(roc_auc_score(labels_np, preds_np))
    auc_weighted = auc
    if bool(use_sample_weight) and sample_weight is not None:
        auc_weighted = float(roc_auc_score(labels_np, preds_np, sample_weight=np.asarray(sample_weight, dtype=np.float64)))
    return auc, auc_weighted

def compute_roc(
    y: np.ndarray,
    p: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    roc_kwargs = {}
    auc_weighted = float(roc_auc_score(y, p))
    if sample_weight is not None:
        roc_kwargs["sample_weight"] = sample_weight
        auc_weighted = float(roc_auc_score(y, p, sample_weight=sample_weight))
    fpr, tpr, _ = roc_curve(y, p, **roc_kwargs)
    auc = float(roc_auc_score(y, p))
    return fpr, tpr, auc, auc_weighted

def fpr_at_target_tpr(tpr: np.ndarray, fpr: np.ndarray, target_tpr: float) -> float:
    """Interpolate the FPR at a target TPR."""
    tpr = np.asarray(tpr, dtype=np.float64)
    fpr = np.asarray(fpr, dtype=np.float64)
    order = np.argsort(tpr)
    tpr_sorted = tpr[order]
    fpr_sorted = fpr[order]
    tpr_unique, unique_idx = np.unique(tpr_sorted, return_index=True)
    fpr_unique = fpr_sorted[unique_idx]
    return float(np.interp(float(target_tpr), tpr_unique, fpr_unique))

def resolve_early_stop_metric_name(metric_name: str) -> str:
    """Validate and normalize the early-stopping metric name."""
    normalized = str(metric_name).strip()
    if normalized not in {"val_auc", "val_auc_weighted"}:
        raise ValueError(f"Unsupported early_stop_metric: {metric_name}")
    return normalized

def select_early_stop_score(
    metric_name: str,
    *,
    val_auc: float,
    val_auc_weighted: float,
) -> float:
    """Return the validation score used for early stopping."""
    normalized = resolve_early_stop_metric_name(metric_name)
    if normalized == "val_auc":
        return float(val_auc)
    return float(val_auc_weighted)
