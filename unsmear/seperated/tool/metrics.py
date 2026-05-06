"""ROC and small analysis metric helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_roc(y: np.ndarray, p: np.ndarray):
    fpr, tpr, _ = roc_curve(y, p)
    auc = float(roc_auc_score(y, p))
    return fpr, tpr, auc
