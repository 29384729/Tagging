"""Technical analysis helpers for the no-fusion joint notebook."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract_teacher_embedding(model, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher logits and pooled embedding for a batch."""
    out = model(x, mask, return_aux=True)
    if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError("Teacher model must return (logits, aux) when return_aux=True.")
    logits, aux = out
    return logits.squeeze(-1), aux["z"]


def per_sample_embedding_mse(candidate_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    """Compute per-sample mean squared embedding distance."""
    return torch.mean((candidate_z - target_z) ** 2, dim=-1)


def _resolve_device(model, device=None):
    if device is not None:
        return device
    return next(model.parameters()).device


@torch.no_grad()
def collect_embedding_distance_rows(teacher, joint_no_kd, joint_with_kd, loader, *, device=None) -> pd.DataFrame:
    """Compare HLT and joint reconstructions by teacher embedding distance to the offline target."""
    teacher.eval()
    joint_no_kd.eval()
    joint_with_kd.eval()
    device = _resolve_device(teacher, device)
    rows: list[dict[str, float]] = []
    sample_offset = 0
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_uns = batch["y_unsmear"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].detach().cpu().numpy()
        weights = batch["weight"].detach().cpu().numpy()
        teacher_logits, target_z = extract_teacher_embedding(teacher, y_uns, mask)
        hlt_logits, hlt_z = extract_teacher_embedding(teacher, x, mask)
        reco_no_kd, logits_no_kd = joint_no_kd(x, mask)
        reco_with_kd, logits_with_kd = joint_with_kd(x, mask)
        _no_kd_teacher_logits, no_kd_z = extract_teacher_embedding(teacher, reco_no_kd, mask)
        _with_kd_teacher_logits, with_kd_z = extract_teacher_embedding(teacher, reco_with_kd, mask)
        variants = {
            "hlt_input": (hlt_logits, per_sample_embedding_mse(hlt_z, target_z)),
            "joint_no_kd_reco": (logits_no_kd.squeeze(-1), per_sample_embedding_mse(no_kd_z, target_z)),
            "joint_with_kd_reco": (logits_with_kd.squeeze(-1), per_sample_embedding_mse(with_kd_z, target_z)),
        }
        for method, (method_logits, distance) in variants.items():
            for i in range(x.shape[0]):
                rows.append({
                    "sample_index": int(sample_offset + i),
                    "batch_idx": int(batch_idx),
                    "row_idx": int(i),
                    "method": str(method),
                    "label": float(labels[i]),
                    "weight": float(weights[i]),
                    "teacher_logit": float(teacher_logits[i].detach().cpu().item()),
                    "method_logit": float(method_logits[i].detach().cpu().item()),
                    "teacher_embedding_distance": float(distance[i].detach().cpu().item()),
                })
        sample_offset += int(x.shape[0])
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_case_rows(teacher, hlt, joint_no_kd, loader, *, device=None) -> pd.DataFrame:
    """Collect case-study rows where models disagree with the offline teacher."""
    teacher.eval()
    hlt.eval()
    joint_no_kd.eval()
    device = _resolve_device(teacher, device)
    rows: list[dict[str, float]] = []
    sample_offset = 0
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_uns = batch["y_unsmear"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].detach().cpu().numpy()
        weights = batch["weight"].detach().cpu().numpy()
        teacher_logits, target_z = extract_teacher_embedding(teacher, y_uns, mask)
        hlt_logits, _hlt_model_z = extract_teacher_embedding(hlt, x, mask)
        _hlt_teacher_logits, hlt_teacher_z = extract_teacher_embedding(teacher, x, mask)
        reco_no_kd, joint_logits = joint_no_kd(x, mask)
        _joint_teacher_logits, joint_z = extract_teacher_embedding(teacher, reco_no_kd, mask)
        hlt_distance = per_sample_embedding_mse(hlt_teacher_z, target_z)
        joint_distance = per_sample_embedding_mse(joint_z, target_z)
        teacher_pred = (torch.sigmoid(teacher_logits) >= 0.5).detach().cpu().numpy()
        hlt_pred = (torch.sigmoid(hlt_logits) >= 0.5).detach().cpu().numpy()
        joint_pred = (torch.sigmoid(joint_logits.squeeze(-1)) >= 0.5).detach().cpu().numpy()
        for i in range(x.shape[0]):
            label_bool = bool(labels[i] >= 0.5)
            if bool(teacher_pred[i]) != label_bool:
                continue
            hlt_correct = bool(hlt_pred[i]) == label_bool
            joint_correct = bool(joint_pred[i]) == label_bool
            if (not hlt_correct) and joint_correct:
                case_group = "hlt_wrong_joint_correct"
            elif hlt_correct and (not joint_correct):
                case_group = "hlt_correct_joint_wrong"
            elif (not hlt_correct) and (not joint_correct):
                case_group = "both_wrong"
            else:
                case_group = "both_correct"
            hlt_dist = float(hlt_distance[i].detach().cpu().item())
            joint_dist = float(joint_distance[i].detach().cpu().item())
            rows.append({
                "sample_index": int(sample_offset + i),
                "batch_idx": int(batch_idx),
                "row_idx": int(i),
                "label": float(labels[i]),
                "weight": float(weights[i]),
                "case_group": case_group,
                "teacher_logit": float(teacher_logits[i].detach().cpu().item()),
                "hlt_logit": float(hlt_logits[i].detach().cpu().item()),
                "joint_no_kd_logit": float(joint_logits.squeeze(-1)[i].detach().cpu().item()),
                "hlt_teacher_embedding_distance": hlt_dist,
                "joint_no_kd_teacher_embedding_distance": joint_dist,
                "distance_improvement_vs_hlt": hlt_dist - joint_dist,
            })
        sample_offset += int(x.shape[0])
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_distance_logit_rows(teacher, hlt, joint_no_kd, joint_with_kd, loader, *, device=None) -> pd.DataFrame:
    """Collect embedding distance and logit-gap rows for HLT and joint variants."""
    teacher.eval()
    hlt.eval()
    joint_no_kd.eval()
    joint_with_kd.eval()
    device = _resolve_device(teacher, device)
    rows: list[dict[str, float]] = []
    sample_offset = 0
    for batch_idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        y_uns = batch["y_unsmear"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].detach().cpu().numpy()
        weights = batch["weight"].detach().cpu().numpy()
        teacher_logits, target_z = extract_teacher_embedding(teacher, y_uns, mask)
        hlt_logits, _hlt_model_z = extract_teacher_embedding(hlt, x, mask)
        _hlt_teacher_logits, hlt_teacher_z = extract_teacher_embedding(teacher, x, mask)
        reco_no_kd, joint_no_kd_logits = joint_no_kd(x, mask)
        reco_with_kd, joint_with_kd_logits = joint_with_kd(x, mask)
        variants = {
            "hlt_input": (hlt_logits, hlt_teacher_z),
            "joint_no_kd_reco": (joint_no_kd_logits.squeeze(-1), extract_teacher_embedding(teacher, reco_no_kd, mask)[1]),
            "joint_with_kd_reco": (joint_with_kd_logits.squeeze(-1), extract_teacher_embedding(teacher, reco_with_kd, mask)[1]),
        }
        for method, (logits, z) in variants.items():
            distance = per_sample_embedding_mse(z, target_z)
            logit_gap_signed = logits - teacher_logits
            logit_gap_abs = torch.abs(logit_gap_signed)
            prob_gap_signed = torch.sigmoid(logits) - torch.sigmoid(teacher_logits)
            prob_gap_abs = torch.abs(prob_gap_signed)
            for i in range(x.shape[0]):
                rows.append({
                    "sample_index": int(sample_offset + i),
                    "batch_idx": int(batch_idx),
                    "row_idx": int(i),
                    "method": str(method),
                    "label": float(labels[i]),
                    "weight": float(weights[i]),
                    "teacher_logit": float(teacher_logits[i].detach().cpu().item()),
                    "method_logit": float(logits[i].detach().cpu().item()),
                    "teacher_embedding_distance": float(distance[i].detach().cpu().item()),
                    "embedding_mse": float(distance[i].detach().cpu().item()),
                    "logit_gap_signed": float(logit_gap_signed[i].detach().cpu().item()),
                    "logit_gap_abs": float(logit_gap_abs[i].detach().cpu().item()),
                    "prob_gap_signed": float(prob_gap_signed[i].detach().cpu().item()),
                    "prob_gap_abs": float(prob_gap_abs[i].detach().cpu().item()),
                })
        sample_offset += int(x.shape[0])
    return pd.DataFrame(rows)


def corr_safe(series_a, series_b, method: str = "pearson") -> float:
    """Return a finite correlation value when both series have variation."""
    a = pd.Series(series_a).replace([np.inf, -np.inf], np.nan).dropna()
    b = pd.Series(series_b).replace([np.inf, -np.inf], np.nan).dropna()
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return float("nan")
    if a.loc[common].nunique() < 2 or b.loc[common].nunique() < 2:
        return float("nan")
    return float(a.loc[common].corr(b.loc[common], method=method))


def build_binned_curve(df_method: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """Build a binned embedding-distance versus logit-gap summary curve."""
    df = df_method.copy()
    distance_col = "teacher_embedding_distance" if "teacher_embedding_distance" in df.columns else "embedding_mse"
    df["distance_bin"] = pd.qcut(df[distance_col], q=int(n_bins), duplicates="drop")
    grouped = df.groupby("distance_bin", observed=True)
    return grouped.agg(
        distance_bin_mean=(distance_col, "mean"),
        abs_gap_mean=("logit_gap_abs", "mean"),
        count=("logit_gap_abs", "size"),
    ).reset_index(drop=True)
