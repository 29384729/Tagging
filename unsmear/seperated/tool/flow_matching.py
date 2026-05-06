"""Flow matching bridge and sampling utilities."""

from __future__ import annotations

from typing import Tuple

import torch

from preprocessing import wrap_dphi_torch


def fm_make_bridge(
    x_post: torch.Tensor,
    x_pre: torch.Tensor,
    t: torch.Tensor,
    *,
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flow Matching bridge:
      x_t = (1-t)*x_post + t*x_pre
      v*  = x_pre - x_post

    If `dphi_idx` is provided, that dimension is treated as an angular variable
    in standardized space. We then build the bridge along the shortest wrapped
    path in angle space rather than by direct linear interpolation in the
    standardized coordinate.

    Args:
      x_post, x_pre: [B,S,D]
      t: [B] in [0,1]
    Returns:
      x_t: [B,S,D]
      v:   [B,S,D]
    """
    tt = t.view(-1, 1, 1).to(dtype=x_post.dtype, device=x_post.device)
    x_t = (1.0 - tt) * x_post + tt * x_pre
    v = x_pre - x_post

    if dphi_idx is not None:
        idx = int(dphi_idx)
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x_post.device, dtype=x_post.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x_post.device, dtype=x_post.dtype)
        )
        phi_post = x_post[..., idx] * scale + mean
        phi_pre = x_pre[..., idx] * scale + mean
        delta_phi = wrap_dphi_torch(phi_pre - phi_post)
        phi_t = wrap_dphi_torch(phi_post + tt.squeeze(-1) * delta_phi)

        x_t = x_t.clone()
        v = v.clone()
        x_t[..., idx] = (phi_t - mean) / scale
        v[..., idx] = delta_phi / scale

    return x_t, v

@torch.no_grad()
def fm_sample_euler(
    model,
    *,
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 20,
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Euler integration from t=0 -> 1:
      x_{k+1} = x_k + (1/steps) * v_theta(x_k, t_k; cond)
    """
    x = x0
    B = x.shape[0]
    mean = None
    scale = None
    if dphi_idx is not None:
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x.device, dtype=x.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x.device, dtype=x.dtype)
        )
    for k in range(int(steps)):
        t = torch.full((B,), float(k) / float(max(1, steps)), device=x.device, dtype=x.dtype)
        v = model(x, cond, mask, t)
        dt = 1.0 / float(max(1, steps))
        x = x + dt * v
        if dphi_idx is not None:
            x = x.clone()
            x[..., int(dphi_idx)] = (wrap_dphi_torch(x[..., int(dphi_idx)] * scale + mean) - mean) / scale
        # Keep padding tokens at 0.
        x = x * mask.to(x.dtype).unsqueeze(-1)
    return x

@torch.no_grad()
def fm_sample_heun(
    model,
    *,
    x0: torch.Tensor,
    cond: torch.Tensor,
    mask: torch.Tensor,
    steps: int = 20,
    dphi_idx: int | None = None,
    dphi_mean: float | torch.Tensor = 0.0,
    dphi_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Heun's method (RK2) integration from t=0 -> 1.

    This is often much more stable than Euler for the same number of steps.
    """
    x = x0
    B = x.shape[0]
    dt = 1.0 / float(max(1, steps))
    mean = None
    scale = None
    if dphi_idx is not None:
        mean = (
            dphi_mean
            if isinstance(dphi_mean, torch.Tensor)
            else torch.tensor(float(dphi_mean), device=x.device, dtype=x.dtype)
        )
        scale = (
            dphi_scale
            if isinstance(dphi_scale, torch.Tensor)
            else torch.tensor(float(dphi_scale), device=x.device, dtype=x.dtype)
        )
    for k in range(int(steps)):
        t0 = float(k) / float(max(1, steps))
        t1 = float(k + 1) / float(max(1, steps))
        t_vec0 = torch.full((B,), t0, device=x.device, dtype=x.dtype)
        t_vec1 = torch.full((B,), t1, device=x.device, dtype=x.dtype)
        v0 = model(x, cond, mask, t_vec0)
        x_euler = x + dt * v0
        if dphi_idx is not None:
            x_euler = x_euler.clone()
            x_euler[..., int(dphi_idx)] = (
                wrap_dphi_torch(x_euler[..., int(dphi_idx)] * scale + mean) - mean
            ) / scale
        x_euler = x_euler * mask.to(x.dtype).unsqueeze(-1)
        v1 = model(x_euler, cond, mask, t_vec1)
        x = x + 0.5 * dt * (v0 + v1)
        if dphi_idx is not None:
            x = x.clone()
            x[..., int(dphi_idx)] = (wrap_dphi_torch(x[..., int(dphi_idx)] * scale + mean) - mean) / scale
        x = x * mask.to(x.dtype).unsqueeze(-1)
    return x



# -----------------------------
# Downstream tagger utilities
# -----------------------------
