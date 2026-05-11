"""Streamlit labelling app -- the main user interface for signal-curator.

Launch:
    streamlit run -m signal_curator.app

Or via the CLI:
    signal-curator demo
    signal-curator run --data /path/to/your.h5 --db /path/to/labels.db

Configuration is read from environment variables; see config.py for the full
list. Sensible defaults work out-of-the-box when launched via the CLI.
"""
from __future__ import annotations
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from signal_curator.config import Config
from signal_curator.dataset import make_adapter
from signal_curator.labels_db import (
    get_all_labels, get_labeled_ids, get_stats, init_db,
    insert_label, signal_id, update_label,
)
from signal_curator.model import MicroConv1D
from signal_curator.session import SignalKey
from signal_curator.trainer import augment_and_train, run_inference

_AUTO_ACCEPT = 0.90
_AUTO_REJECT = 0.10
_UNCERTAIN_LOW = 0.40
_UNCERTAIN_HIGH = 0.60

CONFIG: Config = Config.from_env()
init_db(str(CONFIG.db_path))

st.set_page_config(page_title="Signal Curator", layout="wide")
st.title("Signal Curator -- active learning labelling")


# -------------------------------------------------------------------------
# Queue management
# -------------------------------------------------------------------------

def _rebuild_queue(adapter, freq: int) -> None:
    labeled_ids = get_labeled_ids(str(CONFIG.db_path), freq)
    cache: dict[SignalKey, float] = st.session_state.get("inference_cache", {})

    if cache:
        uncertain = [
            (k, p) for k, p in cache.items()
            if _UNCERTAIN_LOW <= p <= _UNCERTAIN_HIGH
            and signal_id(k.panel, k.freq_khz, k.actuator,
                          k.receiver, k.cycle, k.rep_idx) not in labeled_ids
        ]
        uncertain.sort(key=lambda x: abs(x[1] - 0.5))
        queue = [k for k, _ in uncertain]
    else:
        all_keys = adapter.get_signal_keys(freq)
        all_keys = [
            k for k in all_keys
            if signal_id(k.panel, k.freq_khz, k.actuator,
                         k.receiver, k.cycle, k.rep_idx) not in labeled_ids
        ]
        random.shuffle(all_keys)
        queue = all_keys

    st.session_state.queue = queue
    st.session_state.current_idx = 0


def _run_inference_full(adapter, freq: int, model: MicroConv1D) -> dict:
    """Run inference on all unlabeled signals at the selected frequency."""
    labeled_ids = get_labeled_ids(str(CONFIG.db_path), freq)
    keys = adapter.get_signal_keys(freq)
    keys = [
        k for k in keys
        if signal_id(k.panel, k.freq_khz, k.actuator,
                     k.receiver, k.cycle, k.rep_idx) not in labeled_ids
    ]
    if not keys:
        return {}
    signals = np.stack([adapter.get_signal(k, CONFIG.window) for k in keys])
    probs = run_inference(model, signals)
    return {k: float(p) for k, p in zip(keys, probs)}


# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------

def _on_label(key: SignalKey, label: str) -> None:
    sid = signal_id(key.panel, key.freq_khz, key.actuator,
                    key.receiver, key.cycle, key.rep_idx)
    inserted = insert_label(
        str(CONFIG.db_path), sid, key.panel, key.freq_khz,
        key.actuator, key.receiver, key.cycle, key.rep_idx, label,
    )
    if inserted:
        st.session_state.current_idx += 1


def _on_relabel(key: SignalKey, new_label: str) -> None:
    sid = signal_id(key.panel, key.freq_khz, key.actuator,
                    key.receiver, key.cycle, key.rep_idx)
    update_label(str(CONFIG.db_path), sid, new_label)
    st.session_state.setdefault("relabeled_ids", {})[sid] = new_label


# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------

def _do_train(adapter, freq: int) -> None:
    all_labels = get_all_labels(str(CONFIG.db_path), freq)
    goods = [r for r in all_labels if r["label"] == "GOOD"]
    bads = [r for r in all_labels if r["label"] == "BAD"]
    if len(goods) < 2 or len(bads) < 2:
        st.warning("Label at least 2 GOOD and 2 BAD signals before training.")
        return

    labeled_buffer: list[dict] = []
    labeled_keys: list[SignalKey] = []
    for r in all_labels:
        key = SignalKey(r["panel"], r["freq_khz"], r["actuator"],
                        r["receiver"], r["cycle"], r["rep_idx"])
        try:
            sig = adapter.get_signal(key, CONFIG.window)
        except KeyError:
            continue
        labeled_buffer.append({
            "signal": sig,
            "label": 1 if r["label"] == "GOOD" else 0,
        })
        labeled_keys.append(key)

    with st.spinner("Training model..."):
        model, metrics = augment_and_train(
            labeled_buffer,
            epochs=st.session_state.get("train_epochs", 150),
            patience=st.session_state.get("train_patience", 30),
            input_len=CONFIG.input_len,
        )

    weights_path = CONFIG.weights_dir / f"model_{freq}kHz_weights.pt"
    torch.save(model.state_dict(), weights_path)

    cache = _run_inference_full(adapter, freq, model)

    st.session_state.model = model
    st.session_state.last_metrics = metrics
    st.session_state.inference_cache = cache
    st.session_state.model_weights_path = str(weights_path)
    _rebuild_queue(adapter, freq)

    # Validation / misclassified review data
    signals_arr = np.stack([e["signal"] for e in labeled_buffer])
    true_labels = np.array([e["label"] for e in labeled_buffer])
    probs_labeled = run_inference(model, signals_arr)
    pred_labels = (probs_labeled > 0.5).astype(int)
    tp = int(((pred_labels == 1) & (true_labels == 1)).sum())
    tn = int(((pred_labels == 0) & (true_labels == 0)).sum())
    fp = int(((pred_labels == 1) & (true_labels == 0)).sum())
    fn = int(((pred_labels == 0) & (true_labels == 1)).sum())
    acc = (tp + tn) / len(true_labels)

    fp_mask = (pred_labels == 1) & (true_labels == 0)
    fn_mask = (pred_labels == 0) & (true_labels == 1)
    misclassified = []
    for i in np.where(fp_mask | fn_mask)[0]:
        kind = "FP" if fp_mask[i] else "FN"
        misclassified.append({
            "key": labeled_keys[i],
            "signal": signals_arr[i],
            "true_label": "GOOD" if true_labels[i] == 1 else "BAD",
            "pred_prob": float(probs_labeled[i]),
            "kind": kind,
        })

    meta = []
    for r in all_labels:
        meta.append(f"{r['panel']} act{r['actuator']} recv{r['receiver']} "
                    f"c{r['cycle']} r{r['rep_idx']}")
    st.session_state.relabeled_ids = {}
    st.session_state.validation_results = {
        "freq": freq, "signals_arr": signals_arr, "true_labels": true_labels,
        "probs_labeled": probs_labeled, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "meta": meta, "misclassified": misclassified,
        "train_losses": metrics.get("train_losses", []),
        "val_losses": metrics.get("val_losses", []),
        "val_acc": metrics["val_acc"],
        "epochs_ran": metrics.get("epochs_ran", 0),
        "early_stopped": metrics.get("early_stopped", False),
        "final_lr": metrics.get("final_lr", 0.0),
    }
    st.success(f"Training done -- val acc: {metrics['val_acc']:.1%} | labeled acc: {acc:.1%}")
    st.rerun()


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------

with st.sidebar:
    if not CONFIG.data_path.exists():
        st.error(f"Data file not found: {CONFIG.data_path}")
        st.info("Set SIGNAL_CURATOR_DATA, or run `signal-curator demo`.")
        st.stop()

    if ("adapter" not in st.session_state
            or st.session_state.get("_data_path") != str(CONFIG.data_path)):
        with st.spinner("Loading dataset..."):
            st.session_state.adapter = make_adapter(CONFIG.adapter, str(CONFIG.data_path))
        st.session_state._data_path = str(CONFIG.data_path)

    adapter = st.session_state.adapter
    available_freqs = adapter.frequencies
    if not available_freqs:
        st.error("No signals found in the dataset.")
        st.stop()
    freq = st.selectbox("Frequency (kHz)", available_freqs)

    if st.session_state.get("_session_freq") != freq:
        st.session_state._session_freq = freq
        st.session_state.inference_cache = {}
        weights_path = CONFIG.weights_dir / f"model_{freq}kHz_weights.pt"
        model: MicroConv1D | None = None
        if weights_path.exists():
            model = MicroConv1D(input_len=CONFIG.input_len)
            model.load_state_dict(torch.load(weights_path, weights_only=True))
            with st.spinner("Restoring inference cache..."):
                st.session_state.inference_cache = _run_inference_full(adapter, freq, model)
            st.info("Model restored from saved weights.")
        st.session_state.model = model
        _rebuild_queue(adapter, freq)

    st.markdown("---")
    st.markdown("**Training config**")
    st.session_state.train_epochs = st.slider("Max epochs", 50, 300,
                                              st.session_state.get("train_epochs", 150), 10)
    st.session_state.train_patience = st.slider("Early-stop patience", 10, 60,
                                                st.session_state.get("train_patience", 30), 5)

    st.markdown("---")
    st.markdown("**Labelling progress**")
    stats = get_stats(str(CONFIG.db_path), freq)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total", stats["total"])
    c2.metric("GOOD", stats["good"])
    c3.metric("BAD", stats["bad"])

    st.markdown("---")
    if st.session_state.get("model") is None:
        st.markdown("Model: **not trained yet**")
    else:
        metrics = st.session_state.get("last_metrics", {})
        st.markdown(f"Model val acc: **{metrics.get('val_acc', 0.0):.1%}**")
        st.progress(metrics.get("val_acc", 0.0))

    cache = st.session_state.get("inference_cache", {})
    manual_ids = get_labeled_ids(str(CONFIG.db_path), freq)
    auto_only = {
        k: p for k, p in cache.items()
        if signal_id(k.panel, k.freq_khz, k.actuator,
                     k.receiver, k.cycle, k.rep_idx) not in manual_ids
    }
    auto_good = sum(1 for p in auto_only.values() if p > _AUTO_ACCEPT)
    auto_bad = sum(1 for p in auto_only.values() if p < _AUTO_REJECT)
    uncertain_count = sum(1 for p in auto_only.values()
                          if _UNCERTAIN_LOW <= p <= _UNCERTAIN_HIGH)
    st.markdown("---")
    st.markdown("**Queue**")
    st.markdown(f"Auto-accepted (GOOD): **{auto_good}**")
    st.markdown(f"Auto-rejected (BAD): **{auto_bad}**")
    st.markdown(f"Uncertain (needs label): **{uncertain_count}**")

    if st.button("Train & Filter Dataset"):
        _do_train(adapter, freq)

    recent = get_all_labels(str(CONFIG.db_path), freq)
    if recent:
        st.markdown("---")
        st.markdown("**Recent labels**")
        for r in reversed(recent[-3:]):
            st.markdown(
                f"{r['label']} -- {r['panel']} act{r['actuator']} "
                f"recv{r['receiver']} c{r['cycle']} r{r['rep_idx']}"
            )


# -------------------------------------------------------------------------
# Validation results (persisted across st.rerun)
# -------------------------------------------------------------------------

def _render_validation() -> None:
    vr = st.session_state.get("validation_results")
    if vr is None:
        return
    freq = vr["freq"]
    signals_arr = vr["signals_arr"]
    true_labels = vr["true_labels"]
    probs_labeled = vr["probs_labeled"]
    tp, tn, fp, fn, acc = vr["tp"], vr["tn"], vr["fp"], vr["fn"], vr["acc"]
    misclassified = vr["misclassified"]
    train_losses = vr["train_losses"]
    val_losses = vr["val_losses"]
    val_acc = vr["val_acc"]
    epochs_ran = vr.get("epochs_ran", len(train_losses))
    early_stopped = vr.get("early_stopped", False)
    final_lr = vr.get("final_lr", 0.0)
    meta = vr["meta"]

    stop_msg = f"Early stopped at epoch {epochs_ran}" if early_stopped \
        else f"Ran all {epochs_ran} epochs"
    st.success(f"Training done -- val acc: {val_acc:.1%} | labeled acc: {acc:.1%}")
    st.caption(f"{stop_msg} | final LR: {final_lr:.2e}")
    st.markdown("### Classifier validation")

    col_hist, col_info = st.columns([2, 1])
    with col_hist:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        good_probs = probs_labeled[true_labels == 1]
        bad_probs = probs_labeled[true_labels == 0]
        ax.hist(good_probs, bins=20, range=(0, 1), color="#38a169", alpha=0.8,
                label=f"GOOD (n={len(good_probs)})")
        ax.hist(bad_probs, bins=20, range=(0, 1), color="#e53e3e", alpha=0.8,
                label=f"BAD (n={len(bad_probs)})")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("P(GOOD)")
        ax.set_ylabel("Count")
        ax.set_title("Prediction distribution by true label")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    with col_info:
        st.markdown(f"**Accuracy:** {acc:.1%} ({tp + tn}/{len(true_labels)})")
        st.markdown(f"**True positive:** {tp}")
        st.markdown(f"**True negative:** {tn}")
        st.markdown(f"**False positive:** {fp}")
        st.markdown(f"**False negative:** {fn}")

    if train_losses and val_losses:
        fig, ax = plt.subplots(figsize=(8, 3))
        eprange = range(1, len(train_losses) + 1)
        ax.plot(eprange, train_losses, color="#63b3ed", linewidth=1.2, label="Train loss")
        ax.plot(eprange, val_losses, color="#e53e3e", linewidth=1.2, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & validation loss")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    if misclassified:
        st.markdown(f"### Review misclassified signals ({len(misclassified)})")
        st.markdown("Click 'Relabel' on signals where the model's disagreement "
                    "exposes a labelling mistake. *FP/FN re-inspection is the "
                    "operator-self-correction step.*")
        relabeled = st.session_state.get("relabeled_ids", {})
        for i, m in enumerate(misclassified):
            key = m["key"]
            sid = signal_id(key.panel, key.freq_khz, key.actuator,
                            key.receiver, key.cycle, key.rep_idx)
            col_plot, col_meta, col_btn = st.columns([3, 1, 1])
            with col_plot:
                fig_m, ax_m = plt.subplots(figsize=(6, 1.5))
                ax_m.plot(m["signal"], color="#ecc94b", linewidth=0.7)
                st.pyplot(fig_m)
                plt.close(fig_m)
            with col_meta:
                st.markdown(f"P={m['pred_prob']:.2f} ({m['kind']})")
                st.markdown(f"true: {m['true_label']}")
            with col_btn:
                if sid in relabeled:
                    st.success(f"Relabelled {relabeled[sid]}")
                else:
                    new_label = "GOOD" if m["kind"] == "FP" else "BAD"
                    st.button(
                        f"Relabel {new_label}",
                        key=f"relabel_{i}",
                        on_click=_on_relabel,
                        args=(key, new_label),
                    )


# -------------------------------------------------------------------------
# Main display
# -------------------------------------------------------------------------

_render_validation()

queue: list[SignalKey] = st.session_state.get("queue", [])
idx: int = st.session_state.get("current_idx", 0)

if not queue or idx >= len(queue):
    st.info(
        "No more signals in the queue. "
        "Hit **Train & Filter Dataset** to refresh the uncertainty queue, "
        "or switch frequency."
    )
else:
    key = queue[idx]
    try:
        signal = adapter.get_signal(key, CONFIG.window)
    except KeyError:
        st.warning(f"Signal data not available for {key}")
        signal = None

    p_val = st.session_state.get("inference_cache", {}).get(key)
    meta = (f"**{key.panel}** -- act={key.actuator} -- recv={key.receiver} "
            f"-- cycle={key.cycle} -- rep={key.rep_idx}")
    if p_val is not None and _UNCERTAIN_LOW <= p_val <= _UNCERTAIN_HIGH:
        meta += f"   :orange[P = {p_val:.2f} (uncertain)]"
    st.markdown(meta)

    if signal is not None:
        fig, ax = plt.subplots(figsize=(10, 2.5))
        x_samples = np.arange(CONFIG.window[0], CONFIG.window[0] + len(signal))
        ax.plot(x_samples, signal, color="#3182ce", linewidth=0.9)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    col1, col2, col3 = st.columns([3, 3, 1])
    with col1:
        st.button("GOOD (valid wave)", use_container_width=True,
                  on_click=_on_label, args=(key, "GOOD"))
    with col2:
        st.button("BAD (sensor error)", use_container_width=True,
                  on_click=_on_label, args=(key, "BAD"))
    with col3:
        if st.button("Skip"):
            st.session_state.queue.append(queue[idx])
            st.session_state.current_idx += 1
            st.rerun()
