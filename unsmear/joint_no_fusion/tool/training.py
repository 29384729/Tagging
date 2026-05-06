"""Training, loading, and evaluation loops for baseline, KD, and no-fusion joint models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from gradient_probe import (
    _append_gradient_probe_rows,
    _format_epoch_mean_grad_norm_summary,
    _gradient_probe_output_paths,
    collect_loader_gradient_probes,
    feature_gradient_probe_from_regression_terms,
    gradient_probe_from_losses,
    make_even_interval_batch_indices,
    probe_hlt_kd_gradients,
    probe_joint_gradients,
    save_gradient_probe_tables,
)
from io_utils import load_checkpoint, save_checkpoint, save_epoch_metrics_table
from losses import (
    _batch_weight_total,
    _loss_denominator,
    _maybe_sample_weight,
    attn_loss,
    kd_loss,
    regression_loss_terms,
    weighted_bce_with_logits,
)
from metrics import _auc_scores, resolve_early_stop_metric_name, select_early_stop_score


def get_scheduler(opt, warmup: int, total: int):
    def lr_lambda(ep):
        if ep < int(warmup):
            return float(ep + 1) / float(max(1, warmup))
        return 0.5 * (1.0 + np.cos(np.pi * (ep - warmup) / float(max(1, total - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def train_standard(
    model,
    loader,
    opt,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
    model.train()
    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x, m).squeeze(-1)
        loss = weighted_bce_with_logits(logits, y, sample_weight=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        den = _batch_weight_total(w, int(y.shape[0]))
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return total_loss / max(total_den, 1e-12), auc, auc_weighted

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
    model.eval()
    preds, labs, weights = [], [], []
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        logits = model(x, m).squeeze(-1)
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return (auc, auc_weighted, preds_np, labs_np, weights_np)

@torch.no_grad()
def eval_standard_model(
    model,
    loader,
    device,
    feat_key: str,
    mask_key: str,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float | np.ndarray]:
    """Evaluate a standard classifier and return loss/AUC summaries."""
    model.eval()
    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x = batch[feat_key].to(device)
        m = batch[mask_key].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)
        logits = model(x, m).squeeze(-1)
        loss = weighted_bce_with_logits(logits, y, sample_weight=w)
        den = _batch_weight_total(w, int(y.shape[0]))
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "loss": total_loss / max(total_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
        "preds": preds_np,
        "labels": labs_np,
        "weights": weights_np,
    }

def train_kd(
    student,
    teacher,
    loader,
    opt,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
):
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            if a_attn > 0.0:
                t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                t_logits = t_logits.squeeze(-1)
            else:
                t_logits = teacher(x_off, m_off).squeeze(-1)

        opt.zero_grad(set_to_none=True)
        if a_attn > 0.0:
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * den
        total_den += den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return total_loss / max(total_den, 1e-12), auc, auc_weighted

def train_kd_detailed(
    student,
    teacher,
    loader,
    opt,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float]:
    """Train one KD epoch and return total/hard/kd/attn terms."""
    student.train()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    total_mix_den = 0.0
    total_hard_den = 0.0
    total_aux_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        with torch.no_grad():
            if a_attn > 0.0:
                t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                t_logits = t_logits.squeeze(-1)
            else:
                t_logits = teacher(x_off, m_off).squeeze(-1)

        opt.zero_grad(set_to_none=True)
        if a_attn > 0.0:
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        hard_den = _batch_weight_total(w, int(y.shape[0]))
        aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * mix_den
        total_hard += float(loss_hard.item()) * hard_den
        total_kd += float(loss_kd.item()) * aux_den
        total_attn += float(loss_a.item()) * aux_den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_hard_den += hard_den
        total_aux_den += aux_den
    auc, auc_weighted = _auc_scores(
        labs,
        preds,
        np.asarray(weights, dtype=np.float64),
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "total": total_loss / max(total_mix_den, 1e-12),
        "hard": total_hard / max(total_hard_den, 1e-12),
        "kd": total_kd / max(total_aux_den, 1e-12),
        "attn": total_attn / max(total_aux_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
    }

@torch.no_grad()
def eval_kd_student(
    student,
    teacher,
    loader,
    device,
    cfg: dict,
    *,
    use_sample_weight_for_all_losses: bool = True,
) -> dict[str, float | np.ndarray]:
    """Evaluate a KD student and return total/hard/kd/attn terms."""
    student.eval()
    teacher.eval()
    T = float(cfg["kd"]["temperature"])
    a_kd = float(cfg["kd"]["alpha_kd"])
    a_attn = float(cfg["kd"].get("alpha_attn", 0.0))

    preds, labs, weights = [], [], []
    total_loss = 0.0
    total_hard = 0.0
    total_kd = 0.0
    total_attn = 0.0
    total_mix_den = 0.0
    total_hard_den = 0.0
    total_aux_den = 0.0
    for batch in loader:
        x_hlt = batch["hlt"].to(device)
        x_off = batch["off"].to(device)
        m_hlt = batch["mask_hlt"].to(device)
        m_off = batch["mask_off"].to(device)
        y = batch["label"].to(device)
        w = batch["weight"].to(device)

        if a_attn > 0.0:
            t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
            t_logits = t_logits.squeeze(-1)
            s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
            s_logits = s_logits.squeeze(-1)
        else:
            t_logits = teacher(x_off, m_off).squeeze(-1)
            s_logits = student(x_hlt, m_hlt).squeeze(-1)
        aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
        loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
        loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
        loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
        if a_attn > 0.0:
            loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
        loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a

        hard_den = _batch_weight_total(w, int(y.shape[0]))
        aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        total_loss += float(loss.item()) * mix_den
        total_hard += float(loss_hard.item()) * hard_den
        total_kd += float(loss_kd.item()) * aux_den
        total_attn += float(loss_a.item()) * aux_den
        preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
        labs.extend(y.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_hard_den += hard_den
        total_aux_den += aux_den

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    return {
        "total": total_loss / max(total_mix_den, 1e-12),
        "hard": total_hard / max(total_hard_den, 1e-12),
        "kd": total_kd / max(total_aux_den, 1e-12),
        "attn": total_attn / max(total_aux_den, 1e-12),
        "auc": auc,
        "auc_weighted": auc_weighted,
        "preds": preds_np,
        "labels": labs_np,
        "weights": weights_np,
    }

def make_opt(
    model,
    *,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
):
    """Create the optimizer and scheduler."""
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    sch = get_scheduler(opt, int(warmup_epochs), int(epochs))
    return opt, sch

def train_or_load_standard_model(
    name: str,
    model,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    feat_key: str,
    mask_key: str,
    allow_load: bool,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load a standard classification model."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        return model

    opt, sch = make_opt(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    metrics_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        loss, train_auc, train_auc_weighted = train_standard(
            model,
            epoch_train_loader,
            opt,
            device,
            feat_key,
            mask_key,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        sch.step()
        val_res = eval_standard_model(
            model,
            val_loader,
            device,
            feat_key,
            mask_key,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        val_loss = float(val_res["loss"])
        val_auc = float(val_res["auc"])
        val_auc_weighted = float(val_res["auc_weighted"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_auc_weighted),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = float(val_auc)
            best_auc_weighted = float(val_auc_weighted)
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_loss": float(loss),
                "train_auc": float(train_auc),
                "train_auc_weighted": float(train_auc_weighted),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_auc_weighted),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_loss={loss:.5f} "
                f"train_auc={train_auc:.5f} train_auc_w={train_auc_weighted:.5f} "
                f"val_loss={val_loss:.5f} val_auc={val_auc:.5f} val_auc_w={val_auc_weighted:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return model

def train_or_load_kd_standard_model(
    name: str,
    student,
    teacher,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    allow_load: bool,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    kd_temperature: float,
    kd_alpha: float,
    kd_alpha_attn: float,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    grad_probe_cfg: Optional[dict[str, Any]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load the KD baseline classifier."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    probe_cfg = dict(grad_probe_cfg or {})
    probe_prefix = probe_cfg.get("output_prefix", None)
    probe_name = str(probe_cfg.get("model_name", name))
    train_probe_batches = int(probe_cfg.get("train_batches_per_epoch", 0))
    val_probe_batches = int(probe_cfg.get("val_batches_per_epoch", 0))
    probe_enabled = probe_prefix is not None and (train_probe_batches > 0 or val_probe_batches > 0)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(student, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        if probe_enabled:
            probe_paths = _gradient_probe_output_paths(probe_prefix)
            if not (probe_paths["scalar"].is_file() and probe_paths["norm"].is_file() and probe_paths["cos"].is_file()):
                print(f"[{name}] Gradient probe tables not found for the loaded checkpoint. Rerun training with loading disabled to regenerate them.")
        return student

    opt, sch = make_opt(
        student,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    kd_cfg = {
        "kd": {
            "temperature": float(kd_temperature),
            "alpha_kd": float(kd_alpha),
            "alpha_attn": float(kd_alpha_attn),
        }
    }
    probe_scalar_rows: list[dict[str, Any]] = []
    probe_norm_rows: list[dict[str, Any]] = []
    probe_cosine_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        train_probe_idx = make_even_interval_batch_indices(len(epoch_train_loader), train_probe_batches) if probe_enabled else []
        train_probe_set = set(train_probe_idx)
        train_probe_rank = {idx: rank for rank, idx in enumerate(train_probe_idx)}
        student.train()
        teacher.eval()
        T = float(kd_cfg["kd"]["temperature"])
        a_kd = float(kd_cfg["kd"]["alpha_kd"])
        a_attn = float(kd_cfg["kd"].get("alpha_attn", 0.0))
        preds, labs, weights = [], [], []
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0
        total_attn = 0.0
        total_mix_den = 0.0
        total_hard_den = 0.0
        total_aux_den = 0.0
        for batch_idx, batch in enumerate(epoch_train_loader):
            x_hlt = batch["hlt"].to(device)
            x_off = batch["off"].to(device)
            m_hlt = batch["mask_hlt"].to(device)
            m_off = batch["mask_off"].to(device)
            y = batch["label"].to(device)
            w = batch["weight"].to(device)

            with torch.no_grad():
                if a_attn > 0.0:
                    t_logits, t_attn = teacher(x_off, m_off, return_attention=True)
                    t_logits = t_logits.squeeze(-1)
                else:
                    t_logits = teacher(x_off, m_off).squeeze(-1)

            opt.zero_grad(set_to_none=True)
            if a_attn > 0.0:
                s_logits, s_attn = student(x_hlt, m_hlt, return_attention=True)
                s_logits = s_logits.squeeze(-1)
            else:
                s_logits = student(x_hlt, m_hlt).squeeze(-1)
            aux_weight = _maybe_sample_weight(w, use_sample_weight_for_all_losses)
            loss_kd = kd_loss(s_logits, t_logits, T, sample_weight=aux_weight)
            loss_hard = weighted_bce_with_logits(s_logits, y, sample_weight=w)
            loss_a = torch.zeros((), device=device, dtype=loss_hard.dtype)
            if a_attn > 0.0:
                loss_a = attn_loss(s_attn, t_attn, m_hlt, m_off, sample_weight=aux_weight)
            loss = a_kd * loss_kd + (1.0 - a_kd) * loss_hard + a_attn * loss_a

            if batch_idx in train_probe_set:
                diag = gradient_probe_from_losses(
                    student,
                    {
                        "hard": loss_hard,
                        "kd": loss_kd,
                        "attn": loss_a if a_attn > 0.0 else None,
                        "total": loss,
                    },
                )
                diag["scalar_losses"] = {
                    "hard": float(loss_hard.item()),
                    "kd": float(loss_kd.item()),
                    "attn": float(loss_a.item()) if a_attn > 0.0 else float("nan"),
                    "total": float(loss.item()),
                }
                _append_gradient_probe_rows(
                    scalar_rows=probe_scalar_rows,
                    norm_rows=probe_norm_rows,
                    cosine_rows=probe_cosine_rows,
                    diag=diag,
                    model_name=probe_name,
                    split="train",
                    epoch=int(ep),
                    batch_idx=int(batch_idx),
                    sample_idx=int(train_probe_rank[batch_idx]),
                    total_batches=int(len(epoch_train_loader)),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            hard_den = _batch_weight_total(w, int(y.shape[0]))
            aux_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            mix_den = _loss_denominator(w, int(y.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            total_loss += float(loss.item()) * mix_den
            total_hard += float(loss_hard.item()) * hard_den
            total_kd += float(loss_kd.item()) * aux_den
            total_attn += float(loss_a.item()) * aux_den
            preds.extend(torch.sigmoid(s_logits).detach().cpu().numpy().flatten())
            labs.extend(y.detach().cpu().numpy().flatten())
            weights.extend(w.detach().cpu().numpy().flatten())
            total_mix_den += mix_den
            total_hard_den += hard_den
            total_aux_den += aux_den
        train_auc, train_auc_weighted = _auc_scores(
            labs,
            preds,
            np.asarray(weights, dtype=np.float64),
            use_sample_weight=use_sample_weight_for_all_losses,
        )
        train_res = {
            "total": total_loss / max(total_mix_den, 1e-12),
            "hard": total_hard / max(total_hard_den, 1e-12),
            "kd": total_kd / max(total_aux_den, 1e-12),
            "attn": total_attn / max(total_aux_den, 1e-12),
            "auc": float(train_auc),
            "auc_weighted": float(train_auc_weighted),
        }
        sch.step()
        val_res = eval_kd_student(
            student,
            teacher,
            val_loader,
            device,
            kd_cfg,
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        if probe_enabled and val_probe_batches > 0:
            val_probe_rows = collect_loader_gradient_probes(
                loader=val_loader,
                sample_count=val_probe_batches,
                probe_fn=lambda batch: probe_hlt_kd_gradients(
                    student,
                    teacher,
                    batch,
                    device=device,
                    kd_temperature=float(kd_temperature),
                    kd_alpha=float(kd_alpha),
                    kd_alpha_attn=float(kd_alpha_attn),
                    use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
                ),
                model_name=probe_name,
                split="val",
                epoch=int(ep),
            )
            probe_scalar_rows.extend(val_probe_rows["scalar_rows"])
            probe_norm_rows.extend(val_probe_rows["norm_rows"])
            probe_cosine_rows.extend(val_probe_rows["cosine_rows"])
        kd_grad_loss_weights = {
            "hard": float(1.0 - kd_alpha),
            "kd": float(kd_alpha),
            "attn": float(kd_alpha_attn),
        }
        train_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="train",
            loss_order=["hard", "kd", "attn"],
            loss_weights=kd_grad_loss_weights,
            label="train",
        )
        val_grad_norm_summary = _format_epoch_mean_grad_norm_summary(
            probe_norm_rows,
            epoch=int(ep),
            split="val",
            loss_order=["hard", "kd", "attn"],
            loss_weights=kd_grad_loss_weights,
            label="val",
        )
        grad_norm_suffix = ""
        grad_norm_parts = [part for part in [train_grad_norm_summary, val_grad_norm_summary] if part]
        if grad_norm_parts:
            grad_norm_suffix = " " + " ".join(grad_norm_parts)
        val_auc = float(val_res["auc"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_res["auc_weighted"]),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = float(val_auc)
            best_auc_weighted = float(val_res["auc_weighted"])
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_total": float(train_res["total"]),
                "train_hard": float(train_res["hard"]),
                "train_kd": float(train_res["kd"]),
                "train_attn": float(train_res["attn"]),
                "train_auc": float(train_res["auc"]),
                "train_auc_weighted": float(train_res["auc_weighted"]),
                "val_total": float(val_res["total"]),
                "val_hard": float(val_res["hard"]),
                "val_kd": float(val_res["kd"]),
                "val_attn": float(val_res["attn"]),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_res["auc_weighted"]),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_total={train_res['total']:.5f} "
                f"train_hard={train_res['hard']:.5f} train_kd={train_res['kd']:.5f} train_attn={train_res['attn']:.5f} "
                f"train_auc={train_res['auc']:.5f} train_auc_w={train_res['auc_weighted']:.5f} "
                f"val_total={val_res['total']:.5f} val_hard={val_res['hard']:.5f} "
                f"val_kd={val_res['kd']:.5f} val_attn={val_res['attn']:.5f} "
                f"val_auc={val_auc:.5f} val_auc_w={val_res['auc_weighted']:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
                f"{grad_norm_suffix}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        student.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    if probe_enabled:
        save_gradient_probe_tables(
            probe_prefix,
            scalar_rows=probe_scalar_rows,
            norm_rows=probe_norm_rows,
            cosine_rows=probe_cosine_rows,
            extra_meta={
                "model_name": probe_name,
                "train_batches_per_epoch": int(train_probe_batches),
                "val_batches_per_epoch": int(val_probe_batches),
                "epochs": int(completed_epochs),
                "kind": "hlt_kd",
                "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
            },
        )
    save_checkpoint(
        student,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "kd_enabled": True,
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return student

@torch.no_grad()
def eval_joint_model(
    model,
    loader,
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    use_sample_weight_for_all_losses: bool = True,
):
    """Evaluate a joint model."""
    model.eval()
    if teacher is not None:
        teacher.eval()
    kd_enabled = bool(use_kd) and (teacher is not None)

    sums = {
        "joint_total": 0.0,
        "unsmear_total": 0.0,
        "phys_total": 0.0,
        "cls_hard_total": 0.0,
        "cls_kd_total": 0.0,
        "cls_attn_total": 0.0,
        "cls_total": 0.0,
        "gate_mean_total": 0.0,
        "gate_std_total": 0.0,
    }
    preds, labs, weights = [], [], []
    total_mix_den = 0.0
    total_aux_den = 0.0
    total_hard_den = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y_uns = batch["y_unsmear"].to(device)
        m = batch["mask"].to(device)
        y_cls = batch["label"].to(device)
        w = batch["weight"].to(device)

        kd_attn_enabled = kd_enabled and (float(kd_alpha_attn) > 0.0)
        if kd_attn_enabled:
            reco, logits, s_attn, cls_aux = model(x, m, return_attention=True, return_aux=True)
        else:
            reco, logits, cls_aux = model(x, m, return_aux=True)
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

        kd_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
        attn_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
        cls_loss = hard_loss
        if kd_enabled:
            if kd_attn_enabled:
                teacher_logits, t_attn = teacher(y_uns, m, return_attention=True)
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
                attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=aux_weight)
            cls_loss = (
                (1.0 - float(kd_alpha)) * hard_loss
                + float(kd_alpha) * kd_loss_val
                + float(kd_alpha_attn) * attn_loss_val
            )

        joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss

        mix_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        aux_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
        hard_den = _batch_weight_total(w, int(x.shape[0]))
        sums["joint_total"] += float(joint_loss.item()) * mix_den
        sums["unsmear_total"] += float(reg_terms["total"].item()) * aux_den
        sums["phys_total"] += float(reg_terms["phys"].item()) * aux_den
        sums["cls_hard_total"] += float(hard_loss.item()) * hard_den
        sums["cls_kd_total"] += float(kd_loss_val.item()) * aux_den
        sums["cls_attn_total"] += float(attn_loss_val.item()) * aux_den
        sums["cls_total"] += float(cls_loss.item()) * mix_den
        sums["gate_mean_total"] += float(cls_aux["gate_mean"].item()) * mix_den
        sums["gate_std_total"] += float(cls_aux["gate_std"].item()) * mix_den
        preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().flatten())
        labs.extend(y_cls.detach().cpu().numpy().flatten())
        weights.extend(w.detach().cpu().numpy().flatten())
        total_mix_den += mix_den
        total_aux_den += aux_den
        total_hard_den += hard_den

    preds_np = np.asarray(preds)
    labs_np = np.asarray(labs)
    weights_np = np.asarray(weights, dtype=np.float64)
    auc, auc_weighted = _auc_scores(
        labs_np,
        preds_np,
        weights_np,
        use_sample_weight=use_sample_weight_for_all_losses,
    )
    out = {
        "joint_total": sums["joint_total"] / max(total_mix_den, 1e-12),
        "unsmear_total": sums["unsmear_total"] / max(total_aux_den, 1e-12),
        "phys_total": sums["phys_total"] / max(total_aux_den, 1e-12),
        "cls_hard_total": sums["cls_hard_total"] / max(total_hard_den, 1e-12),
        "cls_kd_total": sums["cls_kd_total"] / max(total_aux_den, 1e-12),
        "cls_attn_total": sums["cls_attn_total"] / max(total_aux_den, 1e-12),
        "cls_total": sums["cls_total"] / max(total_mix_den, 1e-12),
        "gate_mean": sums["gate_mean_total"] / max(total_mix_den, 1e-12),
        "gate_std": sums["gate_std_total"] / max(total_mix_den, 1e-12),
        "alpha": float(model.get_fusion_alpha().detach().item()) if hasattr(model, "get_fusion_alpha") else 0.0,
    }
    out["auc"] = auc
    out["auc_weighted"] = auc_weighted
    out["preds"] = preds_np
    out["labels"] = labs_np
    out["weights"] = weights_np
    return out

def train_or_load_joint_model(
    name: str,
    model,
    ckpt_path: str | Path,
    train_loader,
    val_loader,
    *,
    device,
    feat_names: list[str],
    feat_means: Optional[np.ndarray] = None,
    feat_stds: np.ndarray,
    feature_loss_weights: Optional[Sequence[float] | np.ndarray] = None,
    joint_phys_weight: float = 0.0,
    joint_unsmear_weight: float,
    joint_cls_weight: float,
    lr: float,
    weight_decay: float,
    warmup_epochs: int,
    epochs: int,
    patience: int,
    early_stop_metric: str = "val_auc",
    use_sample_weight_for_all_losses: bool = True,
    teacher=None,
    use_kd: bool = False,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.0,
    kd_alpha_attn: float = 0.0,
    allow_load: bool = False,
    train_loader_factory: Optional[Callable[[int], DataLoader]] = None,
    grad_probe_cfg: Optional[dict[str, Any]] = None,
    epoch_metrics_path: str | Path | None = None,
):
    """Train or load a joint model."""
    early_stop_metric = resolve_early_stop_metric_name(early_stop_metric)
    probe_cfg = dict(grad_probe_cfg or {})
    probe_prefix = probe_cfg.get("output_prefix", None)
    probe_name = str(probe_cfg.get("model_name", name))
    train_probe_batches = int(probe_cfg.get("train_batches_per_epoch", 0))
    val_probe_batches = int(probe_cfg.get("val_batches_per_epoch", 0))
    probe_enabled = probe_prefix is not None and (train_probe_batches > 0 or val_probe_batches > 0)
    if bool(allow_load) and Path(ckpt_path).is_file():
        load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint: {ckpt_path}")
        if epoch_metrics_path is not None and not Path(epoch_metrics_path).is_file():
            print(f"[{name}] Epoch-metrics table not found for the loaded checkpoint. Rerun training with loading disabled to regenerate it.")
        if probe_enabled:
            probe_paths = _gradient_probe_output_paths(probe_prefix)
            if not (probe_paths["scalar"].is_file() and probe_paths["norm"].is_file() and probe_paths["cos"].is_file()):
                print(f"[{name}] Gradient probe tables not found for the loaded checkpoint. Rerun training with loading disabled to regenerate them.")
        return model

    kd_enabled = bool(use_kd) and (teacher is not None)
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt, sch = make_opt(
        model,
        lr=float(lr),
        weight_decay=float(weight_decay),
        warmup_epochs=int(warmup_epochs),
        epochs=int(epochs),
    )
    best_auc, best_auc_weighted, best_stop_score, best_state, no_imp = 0.0, 0.0, float("-inf"), None, 0
    metrics_rows: list[dict[str, Any]] = []
    probe_scalar_rows: list[dict[str, Any]] = []
    probe_norm_rows: list[dict[str, Any]] = []
    probe_cosine_rows: list[dict[str, Any]] = []
    probe_feature_scalar_rows: list[dict[str, Any]] = []
    probe_feature_norm_rows: list[dict[str, Any]] = []
    probe_feature_cosine_rows: list[dict[str, Any]] = []
    completed_epochs = 0
    for ep in range(1, int(epochs) + 1):
        model.train()
        epoch_train_loader = train_loader_factory(ep) if train_loader_factory is not None else train_loader
        train_probe_idx = make_even_interval_batch_indices(len(epoch_train_loader), train_probe_batches) if probe_enabled else []
        train_probe_set = set(train_probe_idx)
        train_probe_rank = {idx: rank for rank, idx in enumerate(train_probe_idx)}
        train_preds, train_labs, train_weights = [], [], []
        tot_joint, tot_uns, tot_phys, tot_cls, tot_hard, tot_kd, tot_attn = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        tot_gate_mean, tot_gate_std = 0.0, 0.0
        total_mix_den, total_aux_den, total_hard_den = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(epoch_train_loader):
            x = batch["x"].to(device)
            y_uns = batch["y_unsmear"].to(device)
            m = batch["mask"].to(device)
            y_cls = batch["label"].to(device)
            w = batch["weight"].to(device)

            opt.zero_grad(set_to_none=True)
            kd_attn_enabled = kd_enabled and (float(kd_alpha_attn) > 0.0)
            if kd_attn_enabled:
                reco, logits, s_attn, cls_aux = model(x, m, return_attention=True, return_aux=True)
            else:
                reco, logits, cls_aux = model(x, m, return_aux=True)
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

            kd_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
            attn_loss_val = torch.zeros((), device=device, dtype=hard_loss.dtype)
            cls_loss = hard_loss
            if kd_enabled:
                with torch.no_grad():
                    if kd_attn_enabled:
                        teacher_logits, t_attn = teacher(y_uns, m, return_attention=True)
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
                    attn_loss_val = attn_loss(s_attn, t_attn, m, m, sample_weight=aux_weight)
                cls_loss = (
                    (1.0 - float(kd_alpha)) * hard_loss
                    + float(kd_alpha) * kd_loss_val
                    + float(kd_alpha_attn) * attn_loss_val
                )

            joint_loss = float(joint_unsmear_weight) * reg_terms["total"] + float(joint_cls_weight) * cls_loss
            if batch_idx in train_probe_set:
                diag = gradient_probe_from_losses(
                    model,
                    {
                        "unsmear": reg_terms["total"],
                        "phys": reg_terms["phys"],
                        "hard": hard_loss,
                        "kd": kd_loss_val if kd_enabled else None,
                        "attn": attn_loss_val if kd_attn_enabled else None,
                        "total": joint_loss,
                    },
                )
                diag["scalar_losses"] = {
                    "unsmear": float(reg_terms["total"].item()),
                    "phys": float(reg_terms["phys"].item()),
                    "hard": float(hard_loss.item()),
                    "kd": float(kd_loss_val.item()) if kd_enabled else float("nan"),
                    "attn": float(attn_loss_val.item()) if kd_attn_enabled else float("nan"),
                    "total": float(joint_loss.item()),
                }
                diag["feature_probe"] = feature_gradient_probe_from_regression_terms(model, reg_terms)
                _append_gradient_probe_rows(
                    scalar_rows=probe_scalar_rows,
                    norm_rows=probe_norm_rows,
                    cosine_rows=probe_cosine_rows,
                    diag=diag,
                    model_name=probe_name,
                    split="train",
                    epoch=int(ep),
                    batch_idx=int(batch_idx),
                    sample_idx=int(train_probe_rank[batch_idx]),
                    total_batches=len(epoch_train_loader),
                )
                feature_diag = diag.get("feature_probe", None)
                if feature_diag is not None:
                    _append_gradient_probe_rows(
                        scalar_rows=probe_feature_scalar_rows,
                        norm_rows=probe_feature_norm_rows,
                        cosine_rows=probe_feature_cosine_rows,
                        diag=feature_diag,
                        model_name=probe_name,
                        split="train",
                        epoch=int(ep),
                        batch_idx=int(batch_idx),
                        sample_idx=int(train_probe_rank[batch_idx]),
                        total_batches=len(epoch_train_loader),
                    )
            joint_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            mix_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            aux_den = _loss_denominator(w, int(x.shape[0]), use_sample_weight=use_sample_weight_for_all_losses)
            hard_den = _batch_weight_total(w, int(x.shape[0]))
            tot_joint += float(joint_loss.item()) * mix_den
            tot_uns += float(reg_terms["total"].item()) * aux_den
            tot_phys += float(reg_terms["phys"].item()) * aux_den
            tot_cls += float(cls_loss.item()) * mix_den
            tot_hard += float(hard_loss.item()) * hard_den
            tot_kd += float(kd_loss_val.item()) * aux_den
            tot_attn += float(attn_loss_val.item()) * aux_den
            tot_gate_mean += float(cls_aux["gate_mean"].item()) * mix_den
            tot_gate_std += float(cls_aux["gate_std"].item()) * mix_den
            train_preds.extend(torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy().flatten())
            train_labs.extend(y_cls.detach().cpu().numpy().flatten())
            train_weights.extend(w.detach().cpu().numpy().flatten())
            total_mix_den += mix_den
            total_aux_den += aux_den
            total_hard_den += hard_den

        sch.step()
        val_res = eval_joint_model(
            model,
            val_loader,
            device=device,
            feat_names=feat_names,
            feat_means=feat_means,
            feat_stds=feat_stds,
            feature_loss_weights=feature_loss_weights,
            joint_phys_weight=joint_phys_weight,
            joint_unsmear_weight=float(joint_unsmear_weight),
            joint_cls_weight=float(joint_cls_weight),
            teacher=teacher,
            use_kd=kd_enabled,
            kd_temperature=float(kd_temperature),
            kd_alpha=float(kd_alpha),
            kd_alpha_attn=float(kd_alpha_attn),
            use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
        )
        if probe_enabled and val_probe_batches > 0:
            val_probe_rows = collect_loader_gradient_probes(
                loader=val_loader,
                sample_count=val_probe_batches,
                probe_fn=lambda batch: probe_joint_gradients(
                    model,
                    batch,
                    device=device,
                    feat_names=feat_names,
                    feat_means=feat_means,
                    feat_stds=feat_stds,
                    feature_loss_weights=feature_loss_weights,
                    joint_phys_weight=joint_phys_weight,
                    teacher=teacher,
                    use_kd=kd_enabled,
                    kd_temperature=float(kd_temperature),
                    kd_alpha=float(kd_alpha),
                    kd_alpha_attn=float(kd_alpha_attn),
                    joint_unsmear_weight=float(joint_unsmear_weight),
                    joint_cls_weight=float(joint_cls_weight),
                    use_sample_weight_for_all_losses=use_sample_weight_for_all_losses,
                ),
                model_name=probe_name,
                split="val",
                epoch=int(ep),
            )
            probe_scalar_rows.extend(val_probe_rows["scalar_rows"])
            probe_norm_rows.extend(val_probe_rows["norm_rows"])
            probe_cosine_rows.extend(val_probe_rows["cosine_rows"])
            probe_feature_scalar_rows.extend(val_probe_rows["feature_scalar_rows"])
            probe_feature_norm_rows.extend(val_probe_rows["feature_norm_rows"])
            probe_feature_cosine_rows.extend(val_probe_rows["feature_cosine_rows"])
        train_joint = tot_joint / max(total_mix_den, 1e-12)
        train_uns = tot_uns / max(total_aux_den, 1e-12)
        train_phys = tot_phys / max(total_aux_den, 1e-12)
        train_cls = tot_cls / max(total_mix_den, 1e-12)
        train_hard = tot_hard / max(total_hard_den, 1e-12)
        train_kd = tot_kd / max(total_aux_den, 1e-12)
        train_attn = tot_attn / max(total_aux_den, 1e-12)
        train_gate_mean = tot_gate_mean / max(total_mix_den, 1e-12)
        train_gate_std = tot_gate_std / max(total_mix_den, 1e-12)
        train_auc, train_auc_weighted = _auc_scores(
            train_labs,
            train_preds,
            np.asarray(train_weights, dtype=np.float64),
            use_sample_weight=use_sample_weight_for_all_losses,
        )
        alpha_value = float(model.get_fusion_alpha().detach().item()) if hasattr(model, "get_fusion_alpha") else 0.0
        val_auc = float(val_res["auc"])
        stop_score = select_early_stop_score(
            early_stop_metric,
            val_auc=float(val_auc),
            val_auc_weighted=float(val_res["auc_weighted"]),
        )
        improved = bool(stop_score > best_stop_score + 1e-4)
        if stop_score > best_stop_score + 1e-4:
            best_auc = val_auc
            best_auc_weighted = float(val_res["auc_weighted"])
            best_stop_score = float(stop_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        metrics_rows.append(
            {
                "model": str(name),
                "epoch": int(ep),
                "early_stop_metric": str(early_stop_metric),
                "best_stop_score": float(best_stop_score),
                "train_joint": float(train_joint),
                "train_uns": float(train_uns),
                "train_phys": float(train_phys),
                "train_cls": float(train_cls),
                "train_hard": float(train_hard),
                "train_kd": float(train_kd),
                "train_attn": float(train_attn),
                "train_gate_mean": float(train_gate_mean),
                "train_gate_std": float(train_gate_std),
                "train_auc": float(train_auc),
                "train_auc_weighted": float(train_auc_weighted),
                "val_joint": float(val_res["joint_total"]),
                "val_uns": float(val_res["unsmear_total"]),
                "val_phys": float(val_res["phys_total"]),
                "val_cls": float(val_res["cls_total"]),
                "val_hard": float(val_res["cls_hard_total"]),
                "val_kd": float(val_res["cls_kd_total"]),
                "val_attn": float(val_res["cls_attn_total"]),
                "val_gate_mean": float(val_res["gate_mean"]),
                "val_gate_std": float(val_res["gate_std"]),
                "val_auc": float(val_auc),
                "val_auc_weighted": float(val_res["auc_weighted"]),
                "alpha": float(alpha_value),
                "best_auc": float(best_auc),
                "best_auc_weighted": float(best_auc_weighted),
                "no_imp": int(no_imp),
                "is_best": int(improved),
            }
        )
        completed_epochs = int(ep)
        if ep == 1 or ep % 2 == 0:
            print(
                f"[{name}] ep={ep:03d} train_joint={train_joint:.5f} train_uns={train_uns:.5f} train_phys={train_phys:.5f} "
                f"train_cls={train_cls:.5f} train_hard={train_hard:.5f} train_kd={train_kd:.5f} train_attn={train_attn:.5f} "
                f"train_gate={train_gate_mean:.4f}+/-{train_gate_std:.4f} alpha={alpha_value:.4f} "
                f"train_auc={train_auc:.5f} train_auc_w={train_auc_weighted:.5f} "
                f"val_joint={val_res['joint_total']:.5f} val_uns={val_res['unsmear_total']:.5f} val_phys={val_res['phys_total']:.5f} "
                f"val_cls={val_res['cls_total']:.5f} val_hard={val_res['cls_hard_total']:.5f} "
                f"val_kd={val_res['cls_kd_total']:.5f} val_attn={val_res['cls_attn_total']:.5f} "
                f"val_gate={val_res['gate_mean']:.4f}+/-{val_res['gate_std']:.4f} "
                f"val_auc={val_auc:.5f} val_auc_w={val_res['auc_weighted']:.5f} "
                f"monitor={early_stop_metric} best_monitor={best_stop_score:.5f} no_imp={no_imp}"
            )
        if no_imp >= int(patience):
            print(f"[{name}] Early stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if metrics_rows:
        metrics_rows[-1]["stopped_after_epoch"] = int(completed_epochs)
    if epoch_metrics_path is not None:
        save_epoch_metrics_table(epoch_metrics_path, metrics_rows)
    if probe_enabled:
        save_gradient_probe_tables(
            probe_prefix,
            scalar_rows=probe_scalar_rows,
            norm_rows=probe_norm_rows,
            cosine_rows=probe_cosine_rows,
            feature_scalar_rows=probe_feature_scalar_rows,
            feature_norm_rows=probe_feature_norm_rows,
            feature_cosine_rows=probe_feature_cosine_rows,
            extra_meta={
                "model_name": str(probe_name),
                "feature_names": list(feat_names),
                "feature_loss_weights": [float(x) for x in np.asarray(feature_loss_weights if feature_loss_weights is not None else np.ones(len(feat_names)), dtype=np.float64).tolist()],
                "loss_weights": {
                    "unsmear": float(joint_unsmear_weight),
                    "phys": float(joint_phys_weight),
                    "hard": float(joint_cls_weight),
                    "kd": float(joint_cls_weight) * float(kd_alpha) if kd_enabled else 0.0,
                    "attn": float(joint_cls_weight) * float(kd_alpha_attn) if kd_enabled else 0.0,
                    "total": 1.0,
                },
            },
        )
    save_checkpoint(
        model,
        ckpt_path,
        extra={
            "best_val_auc": float(best_auc),
            "best_val_auc_weighted": float(best_auc_weighted),
            "early_stop_metric": str(early_stop_metric),
            "best_stop_score": float(best_stop_score),
            "kd_enabled": bool(kd_enabled),
            "joint_phys_weight": float(joint_phys_weight),
            "cls_use_delta_fusion": bool(getattr(model, "cls_use_delta_fusion", False)),
            "cls_detach_delta_for_cls": bool(getattr(model, "cls_detach_delta_for_cls", False)),
            "cls_alpha": float(model.get_fusion_alpha().detach().item()) if hasattr(model, "get_fusion_alpha") else 0.0,
            "use_sample_weight_for_all_losses": bool(use_sample_weight_for_all_losses),
        },
    )
    print(f"Saved checkpoint: {ckpt_path}")
    return model
