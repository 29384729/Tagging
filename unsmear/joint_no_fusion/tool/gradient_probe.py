"""Gradient conflict diagnostics for KD and joint training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from io_utils import _write_csv_rows, ensure_dir
from losses import (
    _maybe_sample_weight,
    attn_loss,
    kd_loss,
    regression_loss_terms,
    weighted_bce_with_logits,
)


def _flatten_grad_list(grads: list[Optional[torch.Tensor]], params: list[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten a list of gradients into a single vector."""
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros_like(p, memory_format=torch.contiguous_format).reshape(-1))
        else:
            parts.append(g.detach().reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts, dim=0)

def get_shared_grad_groups(model) -> dict[str, dict[str, Any]]:
    """Return parameter groups for different parts of the shared trunk."""
    groups: dict[str, dict[str, Any]] = {}

    input_proj_params = list(model.input_proj.parameters()) if hasattr(model, "input_proj") else []
    trunk = None
    trunk_name = None
    if hasattr(model, "encoder"):
        trunk = model.encoder
        trunk_name = "encoder"
    elif hasattr(model, "transformer"):
        trunk = model.transformer
        trunk_name = "transformer"

    if trunk is None or not hasattr(trunk, "layers"):
        raise ValueError("Model does not expose encoder/transformer layers for gradient probing.")

    layers = list(trunk.layers)
    if not layers:
        raise ValueError("Shared trunk has no layers.")

    middle_idx = 1 if len(layers) > 1 else 0
    last_idx = len(layers) - 1

    all_params = list(input_proj_params)
    for layer in layers:
        all_params.extend(list(layer.parameters()))

    groups["shared_all"] = {
        "params": all_params,
        "module": f"input_proj + {trunk_name}.layers[*]",
    }
    groups["input_proj"] = {
        "params": input_proj_params,
        "module": "input_proj",
    }
    groups["layer_1"] = {
        "params": list(layers[middle_idx].parameters()),
        "module": f"{trunk_name}.layers.{middle_idx}",
    }
    groups["layer_last"] = {
        "params": list(layers[last_idx].parameters()),
        "module": f"{trunk_name}.layers.{last_idx}",
    }
    return groups

def _grad_norm(vec: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vec).item())

def _grad_cosine(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    na = torch.linalg.vector_norm(vec_a)
    nb = torch.linalg.vector_norm(vec_b)
    if float(na.item()) < 1e-12 or float(nb.item()) < 1e-12:
        return float("nan")
    return float(torch.dot(vec_a, vec_b).item() / (na.item() * nb.item()))

def gradient_probe_from_losses(
    model,
    loss_map: dict[str, Optional[torch.Tensor]],
) -> dict[str, Any]:
    """Measure shared-trunk gradient norms and cosines from several loss tensors."""
    groups = get_shared_grad_groups(model)
    active_losses = {k: v for k, v in loss_map.items() if v is not None}

    grad_vectors: dict[str, dict[str, torch.Tensor]] = {}
    norm_rows = []
    for loss_name, loss_tensor in active_losses.items():
        grad_vectors[loss_name] = {}
        for group_name, info in groups.items():
            params = list(info["params"])
            if not params:
                vec = torch.zeros(1, device=loss_tensor.device, dtype=loss_tensor.dtype)
            else:
                grads = torch.autograd.grad(
                    loss_tensor,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                vec = _flatten_grad_list(list(grads), params)
            grad_vectors[loss_name][group_name] = vec
            norm_rows.append(
                {
                    "group": group_name,
                    "group_module": info["module"],
                    "loss_component": loss_name,
                    "grad_norm": _grad_norm(vec),
                }
            )

    cosine_rows = []
    loss_names = list(active_losses.keys())
    for i in range(len(loss_names)):
        for j in range(i + 1, len(loss_names)):
            a = loss_names[i]
            b = loss_names[j]
            pair_name = f"{a}_vs_{b}"
            for group_name, info in groups.items():
                cosine_rows.append(
                    {
                        "group": group_name,
                        "group_module": info["module"],
                        "pair": pair_name,
                        "cosine": _grad_cosine(
                            grad_vectors[a][group_name],
                            grad_vectors[b][group_name],
                        ),
                    }
                )

    return {
        "norm_rows": norm_rows,
        "cosine_rows": cosine_rows,
        "group_modules": {k: v["module"] for k, v in groups.items()},
    }

def feature_gradient_probe_from_regression_terms(
    model,
    reg_terms: dict[str, Any],
) -> dict[str, Any]:
    """Build feature-level gradient probes from per-feature regression losses."""
    feature_losses = {
        str(name): loss for name, loss in dict(reg_terms.get("feature_losses", {})).items() if loss is not None
    }
    feature_weights = {
        str(name): float(weight) for name, weight in dict(reg_terms.get("feature_loss_weights", {})).items()
    }
    diag = gradient_probe_from_losses(model, feature_losses)
    diag["scalar_rows"] = [
        {
            "loss_component": str(name),
            "scalar_loss": float(loss.item()),
            "feature_weight": float(feature_weights.get(str(name), 1.0)),
        }
        for name, loss in feature_losses.items()
    ]
    diag["norm_rows"] = [
        {
            **row,
            "feature_weight": float(feature_weights.get(str(row["loss_component"]), 1.0)),
        }
        for row in diag.get("norm_rows", [])
    ]
    return diag

def make_even_interval_batch_indices(total_batches: int, sample_count: int) -> list[int]:
    total = int(total_batches)
    count = int(sample_count)
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))

    raw = np.linspace(0, total - 1, num=count)
    picked = np.clip(np.round(raw).astype(int), 0, total - 1).tolist()

    out: list[int] = []
    seen: set[int] = set()
    for idx in picked:
        ii = int(idx)
        if ii not in seen:
            out.append(ii)
            seen.add(ii)

    if len(out) < count:
        for idx in range(total):
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
            if len(out) >= count:
                break
        out = sorted(out[:count])
    return out

def _gradient_probe_output_paths(prefix: str | Path) -> dict[str, Path]:
    p = Path(prefix)
    ensure_dir(p.parent)
    return {
        "scalar": p.parent / f"{p.name}_scalar_losses.csv",
        "norm": p.parent / f"{p.name}_grad_norms.csv",
        "cos": p.parent / f"{p.name}_grad_cosines.csv",
        "feature_scalar": p.parent / f"{p.name}_feature_scalar_losses.csv",
        "feature_norm": p.parent / f"{p.name}_feature_grad_norms.csv",
        "feature_cos": p.parent / f"{p.name}_feature_grad_cosines.csv",
        "meta": p.parent / f"{p.name}_meta.json",
    }

def _append_gradient_probe_rows(
    *,
    scalar_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    diag: dict[str, Any],
    model_name: str,
    split: str,
    epoch: int,
    batch_idx: int,
    sample_idx: int,
    total_batches: int,
):
    base_row = {
        "model": str(model_name),
        "split": str(split),
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "sample_idx": int(sample_idx),
        "total_batches": int(total_batches),
        "batch_fraction": float((int(batch_idx) + 1) / max(1, int(total_batches))),
    }
    scalar_row_list = diag.get("scalar_rows", None)
    if scalar_row_list is not None:
        for row in scalar_row_list:
            scalar_rows.append(
                {
                    **base_row,
                    **row,
                    "loss_component": str(row["loss_component"]),
                    "scalar_loss": float(row["scalar_loss"]),
                }
            )
    else:
        for loss_name, loss_value in diag.get("scalar_losses", {}).items():
            scalar_rows.append(
                {
                    **base_row,
                    "loss_component": str(loss_name),
                    "scalar_loss": float(loss_value),
                }
            )
    for row in diag.get("norm_rows", []):
        norm_rows.append(
            {
                **base_row,
                **row,
                "grad_norm": float(row["grad_norm"]),
            }
        )
    for row in diag.get("cosine_rows", []):
        cosine_rows.append(
            {
                **base_row,
                **row,
                "cosine": float(row["cosine"]),
            }
        )

def _format_epoch_mean_grad_norm_summary(
    norm_rows: Sequence[dict[str, Any]],
    *,
    epoch: int,
    split: str,
    loss_order: Sequence[str],
    loss_weights: Optional[dict[str, float]] = None,
    group: str = "shared_all",
    label: Optional[str] = None,
) -> str:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in norm_rows:
        if int(row.get("epoch", -1)) != int(epoch):
            continue
        if str(row.get("split", "")) != str(split):
            continue
        if str(row.get("group", "")) != str(group):
            continue
        loss_name = str(row.get("loss_component", ""))
        if loss_name not in loss_order:
            continue
        weight = 1.0 if loss_weights is None else float(loss_weights.get(loss_name, 1.0))
        sums[loss_name] = sums.get(loss_name, 0.0) + float(row["grad_norm"]) * weight
        counts[loss_name] = counts.get(loss_name, 0) + 1

    parts = []
    for loss_name in loss_order:
        if counts.get(loss_name, 0) <= 0:
            continue
        parts.append(f"{loss_name}={sums[loss_name] / max(counts[loss_name], 1):.4f}")
    if not parts:
        return ""
    summary_label = str(label) if label is not None else str(split)
    return f"{summary_label}_{group}_mean_grad_norm[{', '.join(parts)}]"

def save_gradient_probe_tables(
    output_prefix: str | Path,
    *,
    scalar_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    cosine_rows: list[dict[str, Any]],
    feature_scalar_rows: Optional[list[dict[str, Any]]] = None,
    feature_norm_rows: Optional[list[dict[str, Any]]] = None,
    feature_cosine_rows: Optional[list[dict[str, Any]]] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Path]:
    paths = _gradient_probe_output_paths(output_prefix)
    scalar_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "loss_component",
        "scalar_loss",
    ]
    norm_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "group",
        "group_module",
        "loss_component",
        "grad_norm",
    ]
    cos_fields = [
        "model",
        "split",
        "epoch",
        "batch_idx",
        "sample_idx",
        "total_batches",
        "batch_fraction",
        "group",
        "group_module",
        "pair",
        "cosine",
    ]
    feature_scalar_rows = list(feature_scalar_rows or [])
    feature_norm_rows = list(feature_norm_rows or [])
    feature_cosine_rows = list(feature_cosine_rows or [])

    def _fieldnames(base_fields: list[str], rows: list[dict[str, Any]]) -> list[str]:
        extra = []
        extra_seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in base_fields and key not in extra_seen:
                    extra.append(str(key))
                    extra_seen.add(str(key))
        return list(base_fields) + extra

    scalar_fields = _fieldnames(scalar_fields, list(scalar_rows) + feature_scalar_rows)
    norm_fields = _fieldnames(norm_fields, list(norm_rows) + feature_norm_rows)
    cos_fields = _fieldnames(cos_fields, list(cosine_rows) + feature_cosine_rows)
    _write_csv_rows(paths["scalar"], scalar_fields, list(scalar_rows))
    _write_csv_rows(paths["norm"], norm_fields, list(norm_rows))
    _write_csv_rows(paths["cos"], cos_fields, list(cosine_rows))
    if feature_scalar_rows:
        _write_csv_rows(paths["feature_scalar"], scalar_fields, feature_scalar_rows)
    if feature_norm_rows:
        _write_csv_rows(paths["feature_norm"], norm_fields, feature_norm_rows)
    if feature_cosine_rows:
        _write_csv_rows(paths["feature_cos"], cos_fields, feature_cosine_rows)
    meta = dict(extra_meta or {})
    meta["output_prefix"] = str(Path(output_prefix))
    paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return paths

def collect_loader_gradient_probes(
    *,
    loader,
    sample_count: int,
    probe_fn: Callable[[dict[str, Any]], dict[str, Any]],
    model_name: str,
    split: str,
    epoch: int,
) -> dict[str, list[dict[str, Any]]]:
    total_batches = len(loader)
    picked = make_even_interval_batch_indices(total_batches, int(sample_count))
    picked_set = set(picked)
    picked_rank = {idx: rank for rank, idx in enumerate(picked)}
    scalar_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    feature_scalar_rows: list[dict[str, Any]] = []
    feature_norm_rows: list[dict[str, Any]] = []
    feature_cosine_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx not in picked_set:
            continue
        diag = probe_fn(batch)
        _append_gradient_probe_rows(
            scalar_rows=scalar_rows,
            norm_rows=norm_rows,
            cosine_rows=cosine_rows,
            diag=diag,
            model_name=model_name,
            split=split,
            epoch=int(epoch),
            batch_idx=int(batch_idx),
            sample_idx=int(picked_rank[batch_idx]),
            total_batches=int(total_batches),
        )
        feature_diag = diag.get("feature_probe", None)
        if feature_diag is not None:
            _append_gradient_probe_rows(
                scalar_rows=feature_scalar_rows,
                norm_rows=feature_norm_rows,
                cosine_rows=feature_cosine_rows,
                diag=feature_diag,
                model_name=model_name,
                split=split,
                epoch=int(epoch),
                batch_idx=int(batch_idx),
                sample_idx=int(picked_rank[batch_idx]),
                total_batches=int(total_batches),
            )
    return {
        "scalar_rows": scalar_rows,
        "norm_rows": norm_rows,
        "cosine_rows": cosine_rows,
        "feature_scalar_rows": feature_scalar_rows,
        "feature_norm_rows": feature_norm_rows,
        "feature_cosine_rows": feature_cosine_rows,
    }

@torch.no_grad()
def clone_batch_to_cpu(batch: dict[str, Any]) -> dict[str, Any]:
    """Clone a batch to CPU for fixed gradient probing."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out

def probe_hlt_kd_gradients(
    student,
    teacher,
    batch: dict[str, Any],
    *,
    device,
    kd_temperature: float,
    kd_alpha: float,
    kd_alpha_attn: float = 0.0,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, Any]:
    """Collect shared-trunk gradient diagnostics for the HLT+KD student."""
    student.eval()
    teacher.eval()

    x_hlt = batch["hlt"].to(device)
    x_off = batch["off"].to(device)
    m_hlt = batch["mask_hlt"].to(device)
    m_off = batch["mask_off"].to(device)
    y = batch["label"].to(device)
    w = batch["weight"].to(device)

    if float(kd_alpha_attn) > 0.0:
        with torch.no_grad():
            teacher_logits, teacher_attn = teacher(x_off, m_off, return_attention=True)
            teacher_logits = teacher_logits.squeeze(-1)
        student_logits, student_attn = student(x_hlt, m_hlt, return_attention=True)
        student_logits = student_logits.squeeze(-1)
    else:
        with torch.no_grad():
            teacher_logits = teacher(x_off, m_off).squeeze(-1)
        student_logits = student(x_hlt, m_hlt).squeeze(-1)

    aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
    hard_loss = weighted_bce_with_logits(student_logits, y, sample_weight=w)
    kd_loss_val = kd_loss(student_logits, teacher_logits, float(kd_temperature), sample_weight=aux_weight)
    attn_loss_val: Optional[torch.Tensor] = None
    if float(kd_alpha_attn) > 0.0:
        attn_loss_val = attn_loss(student_attn, teacher_attn, m_hlt, m_off, sample_weight=aux_weight)
    total_loss = (
        (1.0 - float(kd_alpha)) * hard_loss
        + float(kd_alpha) * kd_loss_val
        + float(kd_alpha_attn) * (torch.zeros_like(hard_loss) if attn_loss_val is None else attn_loss_val)
    )

    out = gradient_probe_from_losses(
        student,
        {
            "hard": hard_loss,
            "kd": kd_loss_val,
            "attn": attn_loss_val,
            "total": total_loss,
        },
    )
    out["scalar_losses"] = {
        "hard": float(hard_loss.item()),
        "kd": float(kd_loss_val.item()),
        "attn": float("nan") if attn_loss_val is None else float(attn_loss_val.item()),
        "total": float(total_loss.item()),
    }
    return out

def probe_joint_gradients(
    model,
    batch: dict[str, Any],
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    joint_unsmear_weight: float = 1.0,
    joint_cls_weight: float = 1.0,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, Any]:
    """Collect shared-trunk gradient diagnostics for a joint model."""
    model.eval()
    if teacher is not None:
        teacher.eval()

    x = batch["x"].to(device)
    y_uns = batch["y_unsmear"].to(device)
    m = batch["mask"].to(device)
    y_cls = batch["label"].to(device)
    w = batch["weight"].to(device)

    kd_attn_enabled = bool(use_kd) and (teacher is not None) and (float(kd_alpha_attn) > 0.0)
    if kd_attn_enabled:
        reco, logits, student_attn = model(x, m, return_attention=True)
    else:
        reco, logits = model(x, m)
    aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
    reg_terms = regression_loss_terms(
        reco,
        y_uns,
        m,
        feat_names=feat_names,
        feat_means=feat_means,
        feat_stds=feat_stds,
        sample_weight=aux_weight,
        feature_loss_weights=feature_loss_weights,
        phys_consistency_weight=joint_phys_weight,
    )
    hard_loss = weighted_bce_with_logits(logits.squeeze(-1), y_cls, sample_weight=w)

    kd_loss_val: Optional[torch.Tensor] = None
    attn_loss_val: Optional[torch.Tensor] = None
    cls_loss = hard_loss
    if bool(use_kd) and (teacher is not None):
        with torch.no_grad():
            if kd_attn_enabled:
                teacher_logits, teacher_attn = teacher(y_uns, m, return_attention=True)
                teacher_logits = teacher_logits.squeeze(-1)
            else:
                teacher_logits = teacher(y_uns, m).squeeze(-1)
        kd_loss_val = kd_loss(
            logits.squeeze(-1),
            teacher_logits,
            float(kd_temperature),
            sample_weight=aux_weight,
        )
        if kd_attn_enabled:
            attn_loss_val = attn_loss(student_attn, teacher_attn, m, m, sample_weight=aux_weight)
        cls_loss = (
            (1.0 - float(kd_alpha)) * hard_loss
            + float(kd_alpha) * kd_loss_val
            + float(kd_alpha_attn) * (torch.zeros_like(hard_loss) if attn_loss_val is None else attn_loss_val)
        )

    total_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss
    out = gradient_probe_from_losses(
        model,
        {
            "unsmear": reg_terms["total"],
            "phys": reg_terms["phys"],
            "hard": hard_loss,
            "kd": kd_loss_val,
            "attn": attn_loss_val,
            "total": total_loss,
        },
    )
    out["scalar_losses"] = {
        "unsmear": float(reg_terms["total"].item()),
        "phys": float(reg_terms["phys"].item()),
        "hard": float(hard_loss.item()),
        "kd": float("nan") if kd_loss_val is None else float(kd_loss_val.item()),
        "attn": float("nan") if attn_loss_val is None else float(attn_loss_val.item()),
        "total": float(total_loss.item()),
    }
    out["feature_probe"] = feature_gradient_probe_from_regression_terms(model, reg_terms)
    return out
