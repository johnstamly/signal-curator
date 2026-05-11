"""Training utilities: data augmentation, weighted-BCE training loop, batched inference."""
from __future__ import annotations
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from signal_curator.model import MicroConv1D


def augment(labeled_buffer: List[Dict]) -> List[Dict]:
    """Expand each entry with 3 cheap variants (4x total).

    Variants: identity, +10 sample roll, -10 sample roll, additive Gaussian
    noise at 2% of signal std. These preserve the visual GOOD/BAD verdict
    while exposing the model to small temporal and amplitude perturbations.

    Each input dict: {"signal": np.ndarray (L,), "label": int 0|1}.
    """
    aug = []
    for entry in labeled_buffer:
        sig = entry["signal"].astype(np.float32)
        label = entry["label"]
        noise = np.random.normal(0, 0.02 * (sig.std() + 1e-8), sig.shape).astype(np.float32)
        aug.append({"signal": sig.copy(), "label": label})
        aug.append({"signal": np.roll(sig, 10).astype(np.float32), "label": label})
        aug.append({"signal": np.roll(sig, -10).astype(np.float32), "label": label})
        aug.append({"signal": (sig + noise), "label": label})
    return aug


def augment_and_train(
    labeled_buffer: List[Dict],
    epochs: int = 300,
    input_len: int = 500,
    patience: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
) -> Tuple[MicroConv1D, Dict]:
    """Train MicroConv1D from scratch on an augmented labelled buffer.

    Pipeline:
        1. Stratify a 20% validation split at the pre-augmentation level
        2. 4x augment only the training split (no val leakage)
        3. Train with sqrt-weighted BCE for class imbalance, gradient clipping
        4. Early stop on val loss with configurable patience
        5. Restore best-val model before returning

    labeled_buffer: list of {"signal": np.ndarray (L,), "label": int 0|1}.
    Returns (model, metrics) where metrics keys are val_loss, val_acc,
    train_losses, val_losses, epochs_ran, early_stopped, final_lr.
    """
    if len(labeled_buffer) < 2:
        raise ValueError(
            f"augment_and_train requires at least 2 samples, got {len(labeled_buffer)}"
        )

    pos_buf = [e for e in labeled_buffer if e["label"] == 1]
    neg_buf = [e for e in labeled_buffer if e["label"] == 0]
    random.shuffle(pos_buf)
    random.shuffle(neg_buf)

    n_val_pos = max(1, int(0.2 * len(pos_buf)))
    n_val_neg = max(1, int(0.2 * len(neg_buf)))

    val_buf = pos_buf[:n_val_pos] + neg_buf[:n_val_neg]
    train_buf = pos_buf[n_val_pos:] + neg_buf[n_val_neg:]

    aug = augment(train_buf)
    np.random.shuffle(aug)

    def _to_tensor(buf: list) -> tuple:
        sigs = np.array([e["signal"] for e in buf], dtype=np.float32)
        lbls = np.array([e["label"] for e in buf], dtype=np.float32)
        mean = sigs.mean(axis=1, keepdims=True)
        std = sigs.std(axis=1, keepdims=True) + 1e-8
        sigs = (sigs - mean) / std
        return torch.tensor(sigs[:, None, :]), torch.tensor(lbls[:, None])

    X_train, y_train = _to_tensor(aug)
    X_val, y_val = _to_tensor(val_buf)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = MicroConv1D(input_len=input_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_pos = sum(1 for e in train_buf if e["label"] == 1)
    n_neg = sum(1 for e in train_buf if e["label"] == 0)
    w_pos = np.sqrt(len(train_buf) / (2 * max(n_pos, 1)))
    w_neg = np.sqrt(len(train_buf) / (2 * max(n_neg, 1)))

    def _weighted_bce(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = torch.where(target == 1, w_pos, w_neg)
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        return (bce * weights).mean()

    criterion = _weighted_bce
    val_criterion = nn.BCELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for _epoch in range(epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * yb.shape[0]
            epoch_n += yb.shape[0]
        train_losses.append(epoch_loss / max(epoch_n, 1))

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                v_loss += val_criterion(preds, yb).item() * yb.shape[0]
                v_n += yb.shape[0]
        val_loss = v_loss / max(v_n, 1)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb)
            correct += ((preds > 0.5) == yb.bool()).sum().item()
            total += yb.shape[0]

    metrics = {
        "val_loss": best_val_loss,
        "val_acc": correct / total if total > 0 else 0.0,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_ran": len(train_losses),
        "early_stopped": len(train_losses) < epochs,
        "final_lr": optimizer.param_groups[0]["lr"],
    }
    return model, metrics


def run_inference(model: MicroConv1D, signals: np.ndarray) -> np.ndarray:
    """Batched inference over all signals.

    signals: (N, L) float32 array, where L matches the model's input_len.
    Returns: (N,) float32 array of P(GOOD) values in [0, 1].
    """
    model.eval()
    mean = signals.mean(axis=1, keepdims=True)
    std = signals.std(axis=1, keepdims=True) + 1e-8
    signals_norm = ((signals - mean) / std).astype(np.float32)

    X = torch.tensor(signals_norm[:, None, :])
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            batch = X[i: i + 64]
            probs = model(batch).squeeze(1).numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)
