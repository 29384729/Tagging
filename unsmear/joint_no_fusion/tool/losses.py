"""Classification, distillation, and unsmear regression losses."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from preprocessing import wrap_dphi_torch


def masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute SmoothL1 over valid tokens only."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def masked_smooth_l1_wrap_dphi(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Use a wrap-aware residual for the dPhi dimension."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def _weighted_mean(values: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Optionally apply a weighted mean over the batch dimension."""
    if weight is None:
        return values.mean()
    w = weight.to(dtype=values.dtype)
    return (values * w).sum() / w.sum().clamp_min(1e-12)

def _batch_weight_total(weight: Optional[torch.Tensor], batch_size: int) -> float:
    if weight is None:
        return float(batch_size)
    return max(float(weight.detach().sum().item()), 1e-12)

def _maybe_sample_weight(
    sample_weight: Optional[torch.Tensor],
    enabled: bool,
) -> Optional[torch.Tensor]:
    return sample_weight if bool(enabled) else None

def _loss_denominator(
    sample_weight: Optional[torch.Tensor],
    batch_size: int,
    *,
    use_sample_weight: bool,
) -> float:
    if bool(use_sample_weight):
        return _batch_weight_total(sample_weight, batch_size)
    return float(batch_size)

def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-event BCE followed by an event-weighted mean."""
    per_event = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _weighted_mean(per_event, sample_weight)

def _per_jet_masked_smooth_l1(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked SmoothL1 for each jet."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum(dim=(1, 2))
    den = mask.to(pred.dtype).sum(dim=1).clamp_min(1.0)
    return num / den

def _per_jet_masked_smooth_l1_per_feature(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked SmoothL1 for each jet and feature."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, tgt, reduction="none")
    num = (diff * m).sum(dim=1)
    den = mask.to(pred.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return num / den

def _per_jet_masked_smooth_l1_wrap_dphi(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return wrap-aware masked SmoothL1 for each jet."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum(dim=(1, 2))
    den = mask.to(pred.dtype).sum(dim=1).clamp_min(1.0)
    return num / den

def _per_jet_masked_smooth_l1_wrap_dphi_per_feature(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return wrap-aware masked SmoothL1 for each jet and feature."""
    diff = pred - tgt
    scale = (
        dphi_scale
        if isinstance(dphi_scale, torch.Tensor)
        else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    )
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum(dim=1)
    den = mask.to(pred.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    return num / den

def _resolve_feature_loss_weights(
    feat_names: Sequence[str],
    feature_loss_weights: Optional[Sequence[float] | np.ndarray],
) -> np.ndarray:
    """Validate and normalize per-feature regression weights."""
    if feature_loss_weights is None:
        return np.ones(len(feat_names), dtype=np.float32)
    arr = np.asarray(feature_loss_weights, dtype=np.float32).reshape(-1)
    if arr.shape[0] != len(feat_names):
        raise ValueError(
            f"Expected {len(feat_names)} feature weights, got {arr.shape[0]} for features {list(feat_names)}"
        )
    if np.any(arr < 0.0):
        raise ValueError("Feature loss weights must be non-negative.")
    return arr

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    T: float,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    s_soft = torch.sigmoid(student_logits / float(T))
    t_soft = torch.sigmoid(teacher_logits / float(T))
    per_event = F.binary_cross_entropy(s_soft, t_soft, reduction="none")
    return _weighted_mean(per_event, sample_weight) * (float(T) ** 2)

def attn_loss(
    s_attn: torch.Tensor,
    t_attn: torch.Tensor,
    s_mask: torch.Tensor,
    t_mask: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Align the shape of the pooling-attention distribution."""
    eps = 1e-8
    s_valid = s_attn * s_mask.float()
    t_valid = t_attn * t_mask.float()
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    per_event = (s_ent - t_ent) ** 2 + (s_valid.max(dim=1)[0] - t_valid.max(dim=1)[0]) ** 2
    return _weighted_mean(per_event, sample_weight)

def regression_loss_terms(
    mu: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    *,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    sample_weight: Optional[torch.Tensor] = None,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    phys_consistency_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Unsmear regression loss terms for the joint model."""
    idx_map = {n: i for i, n in enumerate(list(feat_names))}
    dphi_idx = idx_map.get("dPhi", None)
    deta_idx = idx_map.get("dEta", None)
    dr_idx = idx_map.get("dR", None)
    feat_std_arr = np.asarray(feat_stds, dtype=np.float32)
    feat_mean_arr = (
        np.zeros(len(feat_names), dtype=np.float32)
        if feat_means is None
        else np.asarray(feat_means, dtype=np.float32)
    )
    dphi_scale = float(feat_std_arr[int(dphi_idx)]) if dphi_idx is not None else 1.0
    feature_weight_arr = _resolve_feature_loss_weights(feat_names, feature_loss_weights)
    feature_weight_tensor = torch.as_tensor(feature_weight_arr, device=mu.device, dtype=mu.dtype)
    feat_std_tensor = torch.as_tensor(feat_std_arr, device=mu.device, dtype=mu.dtype)
    feat_mean_tensor = torch.as_tensor(feat_mean_arr, device=mu.device, dtype=mu.dtype)
    phys_weight = float(phys_consistency_weight)
    if phys_weight < 0.0:
        raise ValueError("phys_consistency_weight must be non-negative.")

    if dphi_idx is not None:
        base_per_jet = _per_jet_masked_smooth_l1_wrap_dphi(
            mu,
            y,
            m,
            dphi_idx=int(dphi_idx),
            dphi_scale=dphi_scale,
        )
        base_per_jet_by_feature = _per_jet_masked_smooth_l1_wrap_dphi_per_feature(
            mu,
            y,
            m,
            dphi_idx=int(dphi_idx),
            dphi_scale=dphi_scale,
        )
    else:
        base_per_jet = _per_jet_masked_smooth_l1(mu, y, m)
        base_per_jet_by_feature = _per_jet_masked_smooth_l1_per_feature(mu, y, m)
    base_unweighted = _weighted_mean(base_per_jet, sample_weight)

    feature_losses: dict[str, torch.Tensor] = {}
    weighted_feature_losses: dict[str, torch.Tensor] = {}
    for feat_idx, feat_name in enumerate(list(feat_names)):
        feat_loss = _weighted_mean(base_per_jet_by_feature[:, feat_idx], sample_weight)
        feature_losses[str(feat_name)] = feat_loss
        weighted_feature_losses[str(feat_name)] = feat_loss * feature_weight_tensor[feat_idx]
    base = sum(weighted_feature_losses.values(), torch.zeros((), device=mu.device, dtype=mu.dtype))

    cons_raw = torch.zeros((), device=mu.device, dtype=mu.dtype)
    if (dr_idx is not None) and (deta_idx is not None) and (dphi_idx is not None):
        deta_raw = mu[..., int(deta_idx)] * feat_std_tensor[int(deta_idx)] + feat_mean_tensor[int(deta_idx)]
        # Wrap dPhi in physical space so equivalent angles across the pi boundary are not treated as large residuals.
        dphi_raw = wrap_dphi_torch(
            mu[..., int(dphi_idx)] * feat_std_tensor[int(dphi_idx)] + feat_mean_tensor[int(dphi_idx)]
        )
        dR_pred_raw = mu[..., int(dr_idx)] * feat_std_tensor[int(dr_idx)] + feat_mean_tensor[int(dr_idx)]
        dR_cons_raw = torch.sqrt(deta_raw**2 + dphi_raw**2 + 1e-12)
        cons_per_jet = _per_jet_masked_smooth_l1(
            dR_pred_raw.unsqueeze(-1),
            dR_cons_raw.unsqueeze(-1),
            m,
        )
        cons_raw = _weighted_mean(cons_per_jet, sample_weight)
    cons = cons_raw * phys_weight

    return {
        "total": base + cons,
        "base": base,
        "base_unweighted": base_unweighted,
        "phys": cons,
        "dr_cons_raw": cons_raw,
        "feature_losses": feature_losses,
        "weighted_feature_losses": weighted_feature_losses,
        "feature_loss_weights": {
            str(feat_name): float(feature_weight_arr[idx]) for idx, feat_name in enumerate(list(feat_names))
        },
    }
