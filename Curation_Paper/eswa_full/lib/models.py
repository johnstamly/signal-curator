"""Registry of 1D binary classifiers for guided-wave signals.

Every model maps (B, 1, L) -> p(GOOD) of shape (B,), ending in a sigmoid
(Task 5 trains with nn.BCELoss).  ``width`` in ``cfg`` scales channel / hidden
counts; its default of 1.0 sits inside the parameter band of each architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.model import MicroConv1D

ARCHS = ("microconv1d", "cnn_wide", "resnet1d", "bigru", "transformer")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _ch(base: int, width: float, mult: int = 1) -> int:
    return max(mult, int(round(base * width / mult)) * mult)


class _Squeeze(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0])


# ---------------------------------------------------------------- microconv1d
class MicroConv1DReLU(MicroConv1D):
    """MicroConv1D with every LeakyReLU(0.01) replaced by ReLU (same params)."""

    def __init__(self, input_len: int = 500) -> None:
        super().__init__(input_len)
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.LeakyReLU):
                self.net[i] = nn.ReLU()


class _MicroWrap(nn.Module):
    def __init__(self, inner: MicroConv1D) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).reshape(x.shape[0])


def _microconv1d(input_len: int, activation: str = "leaky_relu", **_) -> nn.Module:
    if activation == "relu":
        return _MicroWrap(MicroConv1DReLU(input_len))
    if activation in ("leaky_relu", "leaky"):
        return _MicroWrap(MicroConv1D(input_len))
    raise ValueError(f"unknown activation {activation!r}")


# ------------------------------------------------------------------ cnn_wide
def _cnn_wide(input_len: int, width: float = 1.0, **_) -> nn.Module:
    chans = [_ch(c, width) for c in (16, 32, 48, 64)]
    layers: list[nn.Module] = []
    cin = 1
    for c in chans:
        layers += [nn.Conv1d(cin, c, kernel_size=5, padding=2),
                   nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2)]
        cin = c
    layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(),
               nn.Linear(cin, 32), nn.ReLU(), nn.Dropout(0.3),
               nn.Linear(32, 1), nn.Sigmoid(), _Squeeze()]
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ resnet1d
class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = (nn.Identity() if stride == 1 and cin == cout else
                      nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(cout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.short(x))


class _ResNet1D(nn.Module):
    def __init__(self, width: float = 1.0) -> None:
        super().__init__()
        c1, c2, c3 = (_ch(c, width) for c in (32, 64, 128))
        self.stem = nn.Sequential(nn.Conv1d(1, c1, 7, padding=3, bias=False),
                                  nn.BatchNorm1d(c1), nn.ReLU())
        self.stages = nn.Sequential(_Block(c1, c1, 1), _Block(c1, c2, 2), _Block(c2, c3, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(c3, 1), nn.Sigmoid(), _Squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def _resnet1d(input_len: int, width: float = 1.0, **_) -> nn.Module:
    return _ResNet1D(width)


# --------------------------------------------------------------------- bigru
class _BiGRU(nn.Module):
    def __init__(self, width: float = 1.0) -> None:
        super().__init__()
        c1, c2, h = _ch(16, width), _ch(32, width), _ch(48, width)
        self.front = nn.Sequential(
            nn.Conv1d(1, c1, 7, stride=2, padding=3), nn.BatchNorm1d(c1), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.BatchNorm1d(c2), nn.ReLU())
        self.gru = nn.GRU(c2, h, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * h, 1), nn.Sigmoid(), _Squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.front(x).transpose(1, 2)        # (B, T<=125, C)
        out, _ = self.gru(z)
        return self.head(out.mean(dim=1))


def _bigru(input_len: int, width: float = 1.0, **_) -> nn.Module:
    return _BiGRU(width)


# --------------------------------------------------------------- transformer
class _PatchTransformer(nn.Module):
    def __init__(self, input_len: int, width: float = 1.0, patch: int = 10,
                 layers: int = 2, nhead: int = 4) -> None:
        super().__init__()
        d = _ch(40, width, mult=nhead)
        self.patch_embed = nn.Conv1d(1, d, kernel_size=patch, stride=patch)
        n_tok = input_len // patch
        self.pos = nn.Parameter(torch.zeros(1, n_tok, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=2 * d + 64, dropout=0.1,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 1), nn.Sigmoid(), _Squeeze())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(x).transpose(1, 2)  # (B, N, d)
        z = z[:, : self.pos.shape[1]] + self.pos
        return self.head(self.norm(self.enc(z)).mean(dim=1))


def _transformer(input_len: int, width: float = 1.0, **kw) -> nn.Module:
    return _PatchTransformer(input_len, width, **kw)


_BUILDERS = {"microconv1d": _microconv1d, "cnn_wide": _cnn_wide, "resnet1d": _resnet1d,
             "bigru": _bigru, "transformer": _transformer}


def build_model(name: str, input_len: int = 500, **cfg) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"unknown architecture {name!r}; choose from {ARCHS}")
    return _BUILDERS[name](input_len, **cfg)
