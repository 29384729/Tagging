"""Filesystem, checkpoint, and artifact helpers for no-fusion joint experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch


def set_repeat_seed(seed_value: int) -> None:
    """Seed numpy and torch for one repeat."""
    np.random.seed(int(seed_value))
    torch.manual_seed(int(seed_value))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_value))


def find_existing_repeat_dir(repeat_root: str | Path, repeat_seed: int) -> str | None:
    """Find an existing repeat directory that ends with the requested seed suffix."""
    root = Path(repeat_root)
    target_suffix = f"_seed_{int(repeat_seed)}"
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(target_suffix))
    return str(candidates[0]) if candidates else None


def repeat_artifact_paths(repeat_dir: str | Path) -> dict[str, Any]:
    """Return the standard checkpoint, metrics, and prediction paths for a repeat."""
    repeat_dir = Path(repeat_dir)
    repeat_ckpt_dir = repeat_dir / "ckpts"
    repeat_metrics_dir = repeat_dir / "metrics"
    repeat_pred_dir = repeat_dir / "predictions"
    return {
        "repeat_dir": str(repeat_dir),
        "repeat_ckpt_dir": str(repeat_ckpt_dir),
        "repeat_metrics_dir": str(repeat_metrics_dir),
        "repeat_pred_dir": str(repeat_pred_dir),
        "epoch_metrics_paths": {
            "teacher_off": str(repeat_metrics_dir / "teacher_off_epoch_metrics.csv"),
            "student_hlt": str(repeat_metrics_dir / "student_hlt_epoch_metrics.csv"),
            "hlt_kd": str(repeat_metrics_dir / "hlt_kd_epoch_metrics.csv"),
            "joint_no_kd": str(repeat_metrics_dir / "joint_no_kd_epoch_metrics.csv"),
            "joint_with_kd": str(repeat_metrics_dir / "joint_with_kd_epoch_metrics.csv"),
        },
        "ckpt_paths": {
            "Teacher(OFF_FULL)": str(repeat_ckpt_dir / "teacher_offline.pt"),
            "Student(HLT)": str(repeat_ckpt_dir / "student_hlt.pt"),
            "Student(HLT)+KD": str(repeat_ckpt_dir / "student_hlt_kd.pt"),
            "JointSharedEncoder(HLT,no_kd)": str(repeat_ckpt_dir / "joint_sharedencoder_no_kd.pt"),
            "JointSharedEncoder(HLT,with_kd)": str(repeat_ckpt_dir / "joint_sharedencoder_with_kd.pt"),
        },
        "prediction_paths": {
            "Teacher(OFF_FULL)": str(repeat_pred_dir / "teacher_offline_test_preds.npz"),
            "Student(HLT)": str(repeat_pred_dir / "student_hlt_test_preds.npz"),
            "Student(HLT)+KD": str(repeat_pred_dir / "student_hlt_kd_test_preds.npz"),
            "JointSharedEncoder(HLT,no_kd)": str(repeat_pred_dir / "joint_sharedencoder_no_kd_test_preds.npz"),
            "JointSharedEncoder(HLT,with_kd)": str(repeat_pred_dir / "joint_sharedencoder_with_kd_test_preds.npz"),
        },
    }


def load_prediction_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load a prediction bundle saved by save_prediction_bundle."""
    data = np.load(path)
    return {
        "preds": np.asarray(data["preds"], dtype=np.float32),
        "labels": np.asarray(data["labels"], dtype=np.float32),
        "weights": np.asarray(data["weights"], dtype=np.float32),
    }


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_config(config: dict, path: str | Path) -> Path:
    """Save a config dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return p

def save_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a model checkpoint (state_dict + optional metadata)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, p.as_posix())
    return p

def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a model checkpoint into `model` and return the full payload."""
    def _remap_shared_backbone_state_keys(
        target_model: torch.nn.Module,
        state_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if not hasattr(target_model, "token_encoder") or not hasattr(target_model, "cls_head"):
            return state_dict, False

        exact_map: dict[str, str] = {"pool_query": "cls_head.pool_query"}
        if hasattr(target_model, "unsmear_decoder"):
            prefix_map = [
                ("input_proj.", "token_encoder.input_proj."),
                ("pos_embed.", "token_encoder.pos_embed."),
                ("encoder.", "token_encoder.transformer."),
                ("pool_attn.", "cls_head.pool_attn."),
                ("cls_norm.", "cls_head.norm."),
                ("classifier.", "cls_head.classifier."),
            ]
        else:
            prefix_map = [
                ("input_proj.", "token_encoder.input_proj."),
                ("transformer.", "token_encoder.transformer."),
                ("pool_attn.", "cls_head.pool_attn."),
                ("norm.", "cls_head.norm."),
                ("classifier.", "cls_head.classifier."),
            ]

        remapped: dict[str, Any] = {}
        changed = False
        for key, value in state_dict.items():
            new_key = exact_map.get(key, key)
            for old_prefix, new_prefix in prefix_map:
                if new_key.startswith(old_prefix):
                    new_key = f"{new_prefix}{new_key[len(old_prefix):]}"
                    break
            remapped[new_key] = value
            changed = changed or (new_key != key)
        return remapped, changed

    p = Path(path)
    payload = torch.load(p.as_posix(), map_location=map_location)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    try:
        model.load_state_dict(state, strict=bool(strict))
    except RuntimeError as exc:
        remapped_state, changed = _remap_shared_backbone_state_keys(model, state)
        if not changed:
            raise
        try:
            model.load_state_dict(remapped_state, strict=bool(strict))
            state = remapped_state
            print(f"Applied checkpoint key remapping for: {p}")
        except RuntimeError:
            raise exc
    return payload if isinstance(payload, dict) else {"state_dict": state}

def save_rows_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    """Save a list of dict rows as CSV."""
    p = Path(path)
    ensure_dir(p.parent)
    rows = list(rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    _write_csv_rows(p, fieldnames, rows)
    return p

def save_prediction_bundle(
    path: str | Path,
    *,
    preds: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> Path:
    """Save predictions / labels / weights into a compressed NPZ bundle."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(
        p,
        preds=np.asarray(preds, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )
    return p

def _write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]):
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def save_epoch_metrics_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        fieldnames = ["epoch"]
        _write_csv_rows(p, fieldnames, [])
        return p

    preferred = [
        "model",
        "epoch",
        "early_stop_metric",
        "best_stop_score",
        "train_loss",
        "train_auc",
        "train_auc_weighted",
        "train_total",
        "train_hard",
        "train_kd",
        "train_attn",
        "train_joint",
        "train_uns",
        "train_phys",
        "train_cls",
        "train_gate_mean",
        "train_gate_std",
        "val_loss",
        "val_total",
        "val_hard",
        "val_kd",
        "val_attn",
        "val_joint",
        "val_uns",
        "val_phys",
        "val_cls",
        "val_gate_mean",
        "val_gate_std",
        "val_auc",
        "val_auc_weighted",
        "alpha",
        "best_auc",
        "best_auc_weighted",
        "no_imp",
        "is_best",
        "stopped_after_epoch",
    ]
    present = []
    row_keys = set()
    for row in rows:
        row_keys.update(row.keys())
    for key in preferred:
        if key in row_keys:
            present.append(key)
    for key in sorted(row_keys):
        if key not in present:
            present.append(key)
    _write_csv_rows(p, present, rows)
    return p
