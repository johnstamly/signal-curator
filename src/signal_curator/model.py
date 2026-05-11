"""MicroConv1D: small 1D CNN binary classifier for signal quality assessment."""
from __future__ import annotations
import torch
import torch.nn as nn


class MicroConv1D(nn.Module):
    """Microscopic 1D conv classifier for binary signal-quality scoring.

    Architecture: 3 strided-conv blocks with channel taper 1->8->12->4,
    followed by a 16-unit linear hidden layer and a sigmoid output. Roughly
    2.5k parameters total -- intentionally small for the data regimes that
    arise from operator-driven labelling (hundreds to a few thousand labels).

    Input:  (batch, 1, L)  --- L determined by signal-window length.
    Output: (batch, 1)     --- probability of the positive class (GOOD=1).
    """

    def __init__(self, input_len: int = 500) -> None:
        super().__init__()
        l1 = input_len // 4
        l2 = l1 // 4
        l3 = l2 // 2
        flat_size = 4 * l3
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, padding=7),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(4),
            nn.Dropout(p=0.15),
            nn.Conv1d(8, 12, kernel_size=11, padding=5),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(4),
            nn.Dropout(p=0.15),
            nn.Conv1d(12, 4, kernel_size=7, padding=3),
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2),
            nn.Dropout(p=0.15),
            nn.Flatten(),
            nn.Linear(flat_size, 16),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.4),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
