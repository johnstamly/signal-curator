import pytest
import torch
import torch.nn as nn

from Curation_Paper.eswa_full.lib.models import ARCHS, build_model, n_params

BANDS = {
    "microconv1d": (2529, 2529),
    "cnn_wide": (20_000, 40_000),
    "resnet1d": (70_000, 130_000),
    "bigru": (10_000, 60_000),
    "transformer": (30_000, 80_000),
}


def test_archs_registry():
    assert ARCHS == ("microconv1d", "cnn_wide", "resnet1d", "bigru", "transformer")


@pytest.mark.parametrize("name", ARCHS)
def test_param_band(name):
    lo, hi = BANDS[name]
    assert lo <= n_params(build_model(name, 500)) <= hi


@pytest.mark.parametrize("L", [250, 500, 1000])
@pytest.mark.parametrize("name", ARCHS)
def test_output_shape_and_range(name, L):
    m = build_model(name, L).eval()
    x = torch.randn(7, 1, L)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (7,)
    assert torch.isfinite(y).all()
    assert ((y >= 0) & (y <= 1)).all()


@pytest.mark.parametrize("name", ARCHS)
def test_eval_deterministic(name):
    m = build_model(name, 500).eval()
    x = torch.randn(7, 1, 500)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


@pytest.mark.parametrize("name", ARCHS)
def test_trainable_with_bce(name):
    m = build_model(name, 500).train()
    y = m(torch.randn(8, 1, 500))
    loss = nn.BCELoss()(y, torch.ones(8))
    loss.backward()
    assert torch.isfinite(loss)


def test_microconv_relu_variant():
    base = build_model("microconv1d", 500)
    relu = build_model("microconv1d", 500, activation="relu")
    assert n_params(relu) == n_params(base) == 2529
    assert not any(isinstance(m, nn.LeakyReLU) for m in relu.modules())
    assert any(isinstance(m, nn.LeakyReLU) for m in base.modules())
    assert any(isinstance(m, nn.ReLU) for m in relu.modules())
    assert relu(torch.randn(3, 1, 500)).shape == (3,)


@pytest.mark.parametrize("name", ["cnn_wide", "resnet1d", "bigru", "transformer"])
def test_width_changes_size(name):
    small = n_params(build_model(name, 500, width=0.5))
    big = n_params(build_model(name, 500, width=1.5))
    assert small < n_params(build_model(name, 500)) < big


def test_unknown_name():
    with pytest.raises(ValueError):
        build_model("nope")
