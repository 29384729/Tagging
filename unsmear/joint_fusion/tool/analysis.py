"""Fusion diagnostics for joint unsmearing models."""

from __future__ import annotations

import pandas as pd
import torch


def _resolve_device(model, device=None):
    if device is not None:
        return device
    return next(model.parameters()).device


@torch.no_grad()
def collect_fusion_ratio_rows(model, loader, *, device=None) -> pd.DataFrame:
    """Plot gate / alpha diagnostics for each repeat."""
    model.eval()
    device = _resolve_device(model, device)
    rows: list[dict[str, float]] = []
    eps = 1e-12
    alpha_raw_value = float("nan")
    if hasattr(model, "cls_alpha"):
        alpha_raw_value = float(model.cls_alpha.detach().cpu().item())

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].detach().cpu().numpy()
        weights = batch["weight"].detach().cpu().numpy()
        _reco, logits, aux = model(x, mask, return_aux=True)

        z_main = aux["z_main"]
        z_delta = aux["z_delta"]
        gate = aux["gate"]
        alpha = aux["alpha"].to(dtype=z_main.dtype)
        fusion_vec = alpha * gate * z_delta
        fusion_norm = torch.linalg.vector_norm(fusion_vec, dim=-1)
        base_norm = torch.linalg.vector_norm(z_main, dim=-1)
        delta_norm = torch.linalg.vector_norm(z_delta, dim=-1)
        ratio = fusion_norm / base_norm.clamp_min(eps)

        logits_np = logits.squeeze(-1).detach().cpu().numpy()
        ratio_np = ratio.detach().cpu().numpy()
        fusion_norm_np = fusion_norm.detach().cpu().numpy()
        base_norm_np = base_norm.detach().cpu().numpy()
        delta_norm_np = delta_norm.detach().cpu().numpy()
        alpha_effective = float(alpha.detach().cpu().item())

        for i, value in enumerate(ratio_np):
            rows.append({
                "batch_idx": int(batch_idx),
                "row_idx": int(i),
                "label": float(labels[i]),
                "weight": float(weights[i]),
                "logit": float(logits_np[i]),
                "ratio": float(value),
                "fusion_ratio": float(value),
                "fusion_norm": float(fusion_norm_np[i]),
                "base_norm": float(base_norm_np[i]),
                "delta_norm": float(delta_norm_np[i]),
                "gate_mean": float(gate[i].detach().mean().cpu().item()),
                "gate_std": float(gate[i].detach().std(unbiased=False).cpu().item()),
                "alpha_raw": alpha_raw_value,
                "alpha_effective": alpha_effective,
                "alpha": alpha_effective,
            })

    return pd.DataFrame(rows)
