"""Dataset classes for separated unsmear and downstream tagger training."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class UnsmearJetDataset(Dataset):
    """Jet-level dataset: predict per-token target features from smeared inputs."""

    def __init__(
        self,
        x_post: np.ndarray,
        y_pre: np.ndarray,
        mask: np.ndarray,
    ):
        self.x_post = torch.tensor(x_post, dtype=torch.float32)
        self.y_pre = torch.tensor(y_pre, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.x_post.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x_post[i], "y": self.y_pre[i], "mask": self.mask[i]}

class JetDataset(Dataset):
    def __init__(self, feat_off: np.ndarray, feat_hlt: np.ndarray, labels: np.ndarray, masks_off: np.ndarray, masks_hlt: np.ndarray, weights: np.ndarray):
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
