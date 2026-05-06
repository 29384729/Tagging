"""Masked regression, Gaussian NLL, classification, KD, and attention losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from preprocessing import wrap_dphi_torch


def masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute SmoothL1 on `[B,S,D]` only where `mask=True`."""
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
    """SmoothL1 with a wrap-aware residual on the dPhi dimension.

    If dPhi is trained in standardized space, pass `dphi_scale=std(dPhi)`
    so wrapping happens in angle space: `wrap((pred-tgt)*std)/std`.
    """
    diff = pred - tgt
    scale = dphi_scale if isinstance(dphi_scale, torch.Tensor) else torch.tensor(float(dphi_scale), device=pred.device, dtype=pred.dtype)
    diff_phi = wrap_dphi_torch(diff[..., int(dphi_idx)] * scale) / scale
    diff = diff.clone()
    diff[..., int(dphi_idx)] = diff_phi
    # smooth_l1 on residual
    per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    m = mask.to(pred.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE on `[B,S,D]` only where `mask=True`."""
    m = mask.to(pred.dtype).unsqueeze(-1)
    diff = (pred - tgt) ** 2
    num = (diff * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def masked_gaussian_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    active_dim_mask: Optional[torch.Tensor] = None,
    log_var_clip: float = 6.0,
) -> torch.Tensor:
    """Diagonal Gaussian NLL for heteroscedastic regression.

    Args:
      mu, log_var, tgt: [B,S,D]
      mask: [B,S] bool
      active_dim_mask: [D] bool/0-1 mask controlling which dimensions use uncertainty;
        disabled dimensions are treated as `log_var=0` (equivalent to fixed variance).
    """
    log_var = torch.clamp(log_var, min=-float(log_var_clip), max=float(log_var_clip))
    if active_dim_mask is not None:
        adm = active_dim_mask.to(dtype=mu.dtype, device=mu.device).view(1, 1, -1)
        log_var = log_var * adm  # Disabled dimensions => 0

    diff2 = (tgt - mu) ** 2
    inv_var = torch.exp(-log_var)
    per = 0.5 * (diff2 * inv_var + log_var)
    m = mask.to(mu.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def masked_gaussian_nll_wrap_dphi(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    dphi_idx: int,
    dphi_scale: float | torch.Tensor = 1.0,
    active_dim_mask: Optional[torch.Tensor] = None,
    log_var_clip: float = 6.0,
) -> torch.Tensor:
    """Diagonal Gaussian NLL with a wrap-aware residual on dPhi.

    Same note as above: in standardized space, pass `dphi_scale=std(dPhi)`.
    """
    log_var = torch.clamp(log_var, min=-float(log_var_clip), max=float(log_var_clip))
    if active_dim_mask is not None:
        adm = active_dim_mask.to(dtype=mu.dtype, device=mu.device).view(1, 1, -1)
        log_var = log_var * adm

    err = tgt - mu
    scale = dphi_scale if isinstance(dphi_scale, torch.Tensor) else torch.tensor(float(dphi_scale), device=mu.device, dtype=mu.dtype)
    err_phi = wrap_dphi_torch(err[..., int(dphi_idx)] * scale) / scale
    err = err.clone()
    err[..., int(dphi_idx)] = err_phi

    diff2 = err ** 2
    inv_var = torch.exp(-log_var)
    per = 0.5 * (diff2 * inv_var + log_var)
    m = mask.to(mu.dtype).unsqueeze(-1)
    num = (per * m).sum()
    den = m.sum().clamp_min(1.0)
    return num / den

def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    s_soft = torch.sigmoid(student_logits / float(T))
    t_soft = torch.sigmoid(teacher_logits / float(T))
    return F.binary_cross_entropy(s_soft, t_soft) * (float(T) ** 2)

def attn_loss(s_attn: torch.Tensor, t_attn: torch.Tensor, s_mask: torch.Tensor, t_mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    s_valid = s_attn * s_mask.float()
    t_valid = t_attn * t_mask.float()
    s_ent = -(s_valid * torch.log(s_valid + eps)).sum(dim=1)
    t_ent = -(t_valid * torch.log(t_valid + eps)).sum(dim=1)
    return F.mse_loss(s_ent, t_ent) + F.mse_loss(s_valid.max(dim=1)[0], t_valid.max(dim=1)[0])
