"""Downstream tagger training and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from losses import attn_loss, kd_loss


@dataclass
class TrainCfg:
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 3
    patience: int = 15
    grad_clip: float = 1.0
    min_delta: float = 0.0
    min_epochs: int = 0

def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def train_standard(model, loader, opt, device, feat_key: str, mask_key: str):
    model.train()
    preds, labs = [], []
    total_loss = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, m).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))

@torch.no_grad()
def evaluate(model, loader, device, feat_key: str, mask_key: str):
    model.eval()
    preds, labs = [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        logits = model(x, m).squeeze(-1)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
    return float(roc_auc_score(labs, preds)), np.asarray(preds), np.asarray(labs)

def train_kd(student, teacher, loader, opt, device, cfg: dict):
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"]["alpha_attn"])

    preds, labs = [], []
    total_loss = 0.0

    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
            t_logits = t_logits.squeeze(-1)

        opt.zero_grad(set_to_none=True)
        s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
        s_logits = s_logits.squeeze(-1)

        loss_kd = kd_loss(s_logits, t_logits, T)
        loss_hard = F.binary_cross_entropy_with_logits(s_logits, y, weight=w)
        loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off)

        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.item()) * int(y.shape[0])
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())

    return total_loss / max(1, len(preds)), float(roc_auc_score(labs, preds))
