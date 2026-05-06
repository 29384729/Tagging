"""HLT smearing, feature engineering, and feature-space conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


def wrap_dphi_np(dphi: np.ndarray) -> np.ndarray:
    """Wrap an angular difference into (-pi, pi]."""
    return np.arctan2(np.sin(dphi), np.cos(dphi))

def wrap_dphi_torch(dphi: torch.Tensor) -> torch.Tensor:
    """Wrap an angular difference into (-pi, pi] for torch loss computation."""
    return torch.atan2(torch.sin(dphi), torch.cos(dphi))

@dataclass
class HLTEffectsCfg:
    """HLT effects config for the pure unsmear setup."""

    pt_threshold_offline: float = 0.5
    pt_threshold_hlt: float = 0.5
    pt_resolution: float = 0.10
    eta_resolution: float = 0.03
    phi_resolution: float = 0.03

def apply_hlt_effects_pair(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build pure unsmear pairs `(pre_smear, post_smear)`.

    Args:
      const: [N,S,4] (pt,eta,phi,E)
      mask:  [N,S] bool
    Returns:
      pre_smear_const:  [N,S,4] after thresholding, before smearing
      post_smear_const: [N,S,4] after thresholding and smearing
      post_mask:        [N,S] bool mask used for training/evaluation
    """
    rs = np.random.RandomState(int(seed))
    hlt = const.copy()
    hlt_mask = mask.copy()

    # Apply the offline threshold first so offline/HLT start from the same token set.
    pt_thr_off = float(cfg.pt_threshold_offline)
    hlt_mask = hlt_mask & (hlt[:, :, 0] >= pt_thr_off)
    hlt[~hlt_mask] = 0.0

    # Apply the HLT threshold on top; in the pure setup we usually keep it equal to offline.
    pt_thr_hlt = float(cfg.pt_threshold_hlt)
    below = (hlt[:, :, 0] < pt_thr_hlt) & hlt_mask
    hlt_mask[below] = False
    hlt[~hlt_mask] = 0.0

    # Snapshot before smearing.
    pre = hlt.copy()

    # Apply smearing only.
    valid = hlt_mask.copy()
    n_jets, max_part, _ = hlt.shape
    pt_noise = rs.normal(1.0, float(cfg.pt_resolution), size=(n_jets, max_part))
    pt_noise = np.clip(pt_noise, 0.5, 1.5)
    hlt[:, :, 0] = np.where(valid, hlt[:, :, 0] * pt_noise, 0.0)
    eta_noise = rs.normal(0.0, float(cfg.eta_resolution), size=(n_jets, max_part))
    hlt[:, :, 1] = np.where(valid, np.clip(hlt[:, :, 1] + eta_noise, -5.0, 5.0), 0.0)
    phi_noise = rs.normal(0.0, float(cfg.phi_resolution), size=(n_jets, max_part))
    new_phi = hlt[:, :, 2] + phi_noise
    hlt[:, :, 2] = np.where(valid, np.arctan2(np.sin(new_phi), np.cos(new_phi)), 0.0)
    # Recompute E using a massless approximation after smearing.
    hlt[:, :, 3] = np.where(valid, hlt[:, :, 0] * np.cosh(np.clip(hlt[:, :, 1], -5.0, 5.0)), 0.0)

    pre = np.nan_to_num(pre, nan=0.0, posinf=0.0, neginf=0.0)
    hlt = np.nan_to_num(hlt, nan=0.0, posinf=0.0, neginf=0.0)
    pre[~hlt_mask] = 0.0
    hlt[~hlt_mask] = 0.0
    return pre.astype(np.float32), hlt.astype(np.float32), hlt_mask.astype(bool)

def compute_jet_axis(const: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the jet axis from the summed token four-vectors."""
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5.0, 5.0)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    m = mask.astype(np.float32)
    jet_px = (px * m).sum(axis=1, keepdims=True)
    jet_py = (py * m).sum(axis=1, keepdims=True)
    jet_pz = (pz * m).sum(axis=1, keepdims=True)
    jet_E = (E * m).sum(axis=1, keepdims=True)

    jet_pt = np.sqrt(jet_px**2 + jet_py**2) + 1e-8
    jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2) + 1e-8
    jet_eta = 0.5 * np.log(
        np.clip((jet_p + jet_pz) / (jet_p - jet_pz + 1e-8), 1e-8, 1e8)
    )
    jet_phi = np.arctan2(jet_py, jet_px)
    return {
        "jet_px": jet_px,
        "jet_py": jet_py,
        "jet_pz": jet_pz,
        "jet_E": jet_E,
        "jet_pt": jet_pt,
        "jet_eta": jet_eta,
        "jet_phi": jet_phi,
    }

def compute_features_with_axis(
    const: np.ndarray, mask: np.ndarray, axis: Dict[str, np.ndarray], *, kind: str = "7d"
) -> np.ndarray:
    """Compute engineered features with an externally provided axis.

    kind:
      - 3d: dEta, dPhi, log_pt
      - 4d: dEta, dPhi, log_pt, log_E
      - 7d: dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR
    """
    pt = np.maximum(const[:, :, 0], 1e-8)
    eta = np.clip(const[:, :, 1], -5.0, 5.0)
    phi = const[:, :, 2]
    E = np.maximum(const[:, :, 3], 1e-8)

    jet_eta = axis["jet_eta"]
    jet_phi = axis["jet_phi"]
    jet_pt = axis["jet_pt"]
    jet_E = axis["jet_E"]

    dEta = eta - jet_eta
    dPhi = wrap_dphi_np(phi - jet_phi)
    log_pt = np.log(pt + 1e-8)
    log_E = np.log(E + 1e-8)
    log_pt_rel = np.log(pt / jet_pt + 1e-8)
    log_E_rel = np.log(E / (jet_E + 1e-8) + 1e-8)
    dR = np.sqrt(dEta**2 + dPhi**2)

    k = str(kind).lower()
    if k == "3d":
        feats = np.stack([dEta, dPhi, log_pt], axis=-1)
    elif k == "4d":
        feats = np.stack([dEta, dPhi, log_pt, log_E], axis=-1)
    elif k == "7d":
        feats = np.stack([dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR], axis=-1)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")
    feats = np.clip(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    feats[~mask] = 0.0
    return feats.astype(np.float32)

def get_stats(feat: np.ndarray, mask: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = np.zeros(feat.shape[-1], dtype=np.float64)
    stds = np.zeros(feat.shape[-1], dtype=np.float64)
    for i in range(feat.shape[-1]):
        vals = feat[idx][:, :, i][mask[idx]]
        means[i] = float(np.nanmean(vals))
        stds[i] = float(np.nanstd(vals) + 1e-8)
    return means.astype(np.float32), stds.astype(np.float32)

def standardize(
    feat: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray, *, clip: float = 10.0
) -> np.ndarray:
    out = (feat - means[None, None, :]) / stds[None, None, :]
    out = np.clip(out, -float(clip), float(clip))
    out = np.nan_to_num(out, 0.0)
    out[~mask] = 0.0
    return out.astype(np.float32)

def build_unsmear_epoch_arrays(
    const: np.ndarray,
    mask: np.ndarray,
    cfg: HLTEffectsCfg,
    *,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int = 42,
    clip: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regenerate unsmear training inputs for one epoch.

    Notes:
    - The target still uses features from the unsmeared view in its own axis frame.
    - The input is re-smeared and the post-smear axis is recomputed, so the jet axis can change each epoch.
    - All returned arrays are already standardized with the given means/stds and can be fed directly into `UnsmearJetDataset`.
    """
    pre_const, post_const, post_mask = apply_hlt_effects_pair(const, mask, cfg, seed=int(seed))

    pre_const = np.asarray(pre_const, dtype=np.float32)
    post_const = np.asarray(post_const, dtype=np.float32)
    post_mask = np.asarray(post_mask, dtype=bool)

    pre_const[~post_mask] = 0.0
    post_const[~post_mask] = 0.0

    axis_pre = compute_jet_axis(pre_const, post_mask)
    axis_post = compute_jet_axis(post_const, post_mask)
    feat_pre = compute_features_with_axis(pre_const, post_mask, axis_pre, kind=feature_kind)
    feat_post = compute_features_with_axis(post_const, post_mask, axis_post, kind=feature_kind)

    x_std = standardize(feat_post, post_mask, means, stds, clip=float(clip))
    y_std = standardize(feat_pre, post_mask, means, stds, clip=float(clip))
    return x_std.astype(np.float32), y_std.astype(np.float32), post_mask.astype(bool)

def get_feat_names(kind: str) -> list[str]:
    k = str(kind).lower()
    if k == "3d":
        return ["dEta", "dPhi", "log_pt"]
    if k == "4d":
        return ["dEta", "dPhi", "log_pt", "log_E"]
    if k == "7d":
        return ["dEta", "dPhi", "log_pt", "log_E", "log_pt_rel", "log_E_rel", "dR"]
    raise ValueError(f"Unknown feature kind: {kind}")

def feats_to_7d(
    feat: np.ndarray,
    mask: np.ndarray,
    axis: Dict[str, np.ndarray],
    *,
    kind: str,
) -> np.ndarray:
    """Expand 3D/4D features into 7D engineered features in raw feature space.

    Mainly used when unsmear predicts only 3D/4D features but the downstream model expects 7D inputs.

    Args:
      feat: [N,S,Dk] raw feature values (not standardized)
      mask: [N,S] bool
      axis: jet-axis dict (usually the post-smear axis; contains `jet_eta/jet_phi/jet_pt/jet_E`)
      kind: '3d'/'4d'/'7d', describing the semantics of the input features
    Returns:
      out: [N,S,7] raw 7D engineered features
    """
    k = str(kind).lower()
    if k == "7d":
        out = np.asarray(feat, dtype=np.float32)
        out[~mask] = 0.0
        return out

    feat = np.asarray(feat, dtype=np.float32)
    dEta = feat[..., 0]
    dPhi = feat[..., 1]
    log_pt = feat[..., 2]

    jet_eta = np.asarray(axis["jet_eta"], dtype=np.float32)  # [N,1]
    jet_phi = np.asarray(axis["jet_phi"], dtype=np.float32)  # [N,1]
    jet_pt = np.asarray(axis["jet_pt"], dtype=np.float32)    # [N,1]
    jet_E = np.asarray(axis["jet_E"], dtype=np.float32)      # [N,1]

    pt = np.exp(np.clip(log_pt, -20.0, 20.0))
    dR = np.sqrt(dEta**2 + dPhi**2)
    log_pt_rel = np.log(pt / (jet_pt + 1e-8) + 1e-8)

    if k == "4d":
        log_E = feat[..., 3]
        E = np.exp(np.clip(log_E, -20.0, 20.0))
    elif k == "3d":
        # 3D features do not include energy, so fill E with a massless approximation using absolute eta.
        eta = dEta + jet_eta
        E = pt * np.cosh(np.clip(eta, -5.0, 5.0))
        log_E = np.log(E + 1e-8)
    else:
        raise ValueError(f"Unknown feature kind: {kind}")

    log_E_rel = np.log(E / (jet_E + 1e-8) + 1e-8)

    out = np.stack([dEta, dPhi, log_pt, log_E, log_pt_rel, log_E_rel, dR], axis=-1)
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)
    out[~mask] = 0.0
    return out.astype(np.float32)
