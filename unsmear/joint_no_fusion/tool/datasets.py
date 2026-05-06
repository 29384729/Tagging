"""Dataset and DataLoader helpers for baseline and joint training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from preprocessing import HLTEffectsCfg, build_epoch_train_arrays


def _repeat_train_rows(arr: np.ndarray, repeats: int) -> np.ndarray:
    arr_np = np.asarray(arr)
    if int(repeats) <= 1:
        return arr_np
    return np.repeat(arr_np, int(repeats), axis=0)


class JetDataset(Dataset):
    def __init__(
        self,
        feat_off: np.ndarray,
        feat_hlt: np.ndarray,
        labels: np.ndarray,
        masks_off: np.ndarray,
        masks_hlt: np.ndarray,
        weights: np.ndarray,
    ):
        self.off = torch.tensor(feat_off, dtype=torch.float32)
        self.hlt = torch.tensor(feat_hlt, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.mask_off = torch.tensor(masks_off, dtype=torch.bool)
        self.mask_hlt = torch.tensor(masks_hlt, dtype=torch.bool)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return {
            "off": self.off[i],
            "hlt": self.hlt[i],
            "mask_off": self.mask_off[i],
            "mask_hlt": self.mask_hlt[i],
            "label": self.labels[i],
            "weight": self.weights[i],
        }

class JointJetDataset(Dataset):
    """Joint-training dataset for the shared-encoder model."""

    def __init__(
        self,
        x_hlt: np.ndarray,
        y_off: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ):
        self.x = torch.tensor(x_hlt, dtype=torch.float32)
        self.y_unsmear = torch.tensor(y_off, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, i):
        return {
            "x": self.x[i],
            "y_unsmear": self.y_unsmear[i],
            "mask": self.mask[i],
            "label": self.labels[i],
            "weight": self.weights[i],
        }

def make_epoch_hlt_train_loader(
    *,
    epoch: int,
    batch_size: int,
    feat_off_train: np.ndarray,
    off_mask_train: np.ndarray,
    labels_train: np.ndarray,
    weights_train: np.ndarray,
    train_const_raw: np.ndarray,
    train_mask_raw: np.ndarray,
    cfg: HLTEffectsCfg,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int,
    fixed_feat_hlt_train: np.ndarray,
    fixed_hlt_mask_train: np.ndarray,
    seed_stride: int = 1,
    resmear_each_epoch: bool = False,
    clip: float = 10.0,
) -> DataLoader:
    """Build the HLT training loader for the current epoch."""
    x_ep, _y_ep, m_ep = build_epoch_train_arrays(
        train_const_raw,
        train_mask_raw,
        cfg,
        feature_kind=feature_kind,
        means=means,
        stds=stds,
        seed=seed,
        epoch=epoch,
        fixed_x=fixed_feat_hlt_train,
        fixed_y=feat_off_train,
        fixed_mask=fixed_hlt_mask_train,
        seed_stride=seed_stride,
        resmear_each_epoch=resmear_each_epoch,
        clip=clip,
    )
    ds = JetDataset(feat_off_train, x_ep, labels_train, off_mask_train, m_ep, weights_train)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)

def make_epoch_joint_train_loader(
    *,
    epoch: int,
    batch_size: int,
    labels_train: np.ndarray,
    weights_train: np.ndarray,
    train_const_raw: np.ndarray,
    train_mask_raw: np.ndarray,
    cfg: HLTEffectsCfg,
    feature_kind: str,
    means: np.ndarray,
    stds: np.ndarray,
    seed: int,
    fixed_x_train: np.ndarray,
    fixed_y_train: np.ndarray,
    fixed_mask_train: np.ndarray,
    seed_stride: int = 1,
    resmear_each_epoch: bool = False,
    clip: float = 10.0,
) -> DataLoader:
    """Build the joint-training loader for the current epoch."""
    x_ep, y_ep, m_ep = build_epoch_train_arrays(
        train_const_raw,
        train_mask_raw,
        cfg,
        feature_kind=feature_kind,
        means=means,
        stds=stds,
        seed=seed,
        epoch=epoch,
        fixed_x=fixed_x_train,
        fixed_y=fixed_y_train,
        fixed_mask=fixed_mask_train,
        seed_stride=seed_stride,
        resmear_each_epoch=resmear_each_epoch,
        clip=clip,
    )
    ds = JointJetDataset(x_ep, y_ep, m_ep, labels_train, weights_train)
    return DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
