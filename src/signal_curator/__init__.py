"""signal-curator: human-in-the-loop active learning labeling app for 1D signal datasets."""
from __future__ import annotations

__version__ = "0.1.1"

from signal_curator.config import Config
from signal_curator.dataset import DatasetAdapter, GenericAdapter, make_adapter
from signal_curator.model import MicroConv1D
from signal_curator.session import LabelingSession, SignalKey
from signal_curator.trainer import augment, augment_and_train, run_inference
from signal_curator import labels_db

__all__ = [
    "__version__",
    "Config",
    "DatasetAdapter",
    "GenericAdapter",
    "LabelingSession",
    "MicroConv1D",
    "SignalKey",
    "augment",
    "augment_and_train",
    "labels_db",
    "make_adapter",
    "run_inference",
]
