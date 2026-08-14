"""Token Radius Attention wiring through the wan2_1 attention dispatcher.

Exercises the gated branch added to ``attention()`` in
``scope.core.pipelines.wan2_1.modules.attention``, plus the budget /
radius math of the capability module itself.
"""

import pytest
import torch

from scope.core.pipelines.wan2_1.modules import attention as attention_module
from scope.core.pipelines.wan2_1.modules.token_radius import (
    TokenRadiusAttention,
    TokenRadiusConfig,
    configure_token_radius,
)


def _qkv(length, keys, heads=4, channels=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    make = lambda n: torch.randn(2, n, heads, channels, generator=gen)  # noqa: E731
    return make(length), make(keys), make(keys)


def teardown_function(_):
    # Restore the dispatcher's shared config for other tests.
    configure_token_radius(TokenRadiusConfig())


def test_dispatcher_is_dense_when_disabled():
    q, k, v = _qkv(8, 8)
    configure_token_radius(TokenRadiusConfig(enabled=False))

    out = attention_module.attention(q, k, v)

    assert out.shape == q.shape
    assert out.dtype == q.dtype


def test_dispatcher_routes_through_token_radius_when_enabled(monkeypatch):
    q, k, v = _qkv(8, 8)
    configure_token_radius(TokenRadiusConfig(enabled=True))

    calls = []
    monkeypatch.setattr(
        attention_module,
        "token_radius_attention",
        lambda *a: calls.append(a) or None,
    )
    attention_module.attention(q, k, v)

    assert calls == [(q, k, v)]


def test_token_radius_output_shape_and_dtype():
    q, k, v = _qkv(64, 1024, seed=1)
    tra = TokenRadiusAttention(TokenRadiusConfig(enabled=True))

    out = tra(q, k, v)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()


def test_token_radius_mask_respects_radius():
    tra = TokenRadiusAttention(TokenRadiusConfig(enabled=True))
    tra._radius = torch.tensor([1.0, 4.0])

    mask = tra.build_mask(num_queries=2, num_keys=8, device=torch.device("cpu"))

    # Each row keeps ceil(radius) most-recent keys.
    assert mask.sum(dim=1).tolist() == [7, 4]
    # Masked entries are the temporally distant keys, not recent ones.
    assert not bool(mask[0, -1])
    assert not bool(mask[1, -4])


def test_higher_entropy_query_gets_larger_budget():
    tra = TokenRadiusAttention(TokenRadiusConfig(enabled=True))
    num_keys = 4096
    # One low-entropy (concentrated) and one high-entropy (diffuse) query
    # in the same group, so the spread-based normalization has a signal.
    entropy = torch.tensor([[[0.0, 5.0, 0.0, 5.0]]])

    budget = tra.token_budget(entropy, num_keys)

    assert budget[0, 0, 0].item() == pytest.approx(0.02 * num_keys, rel=1e-3)
    assert budget[0, 0, 1].item() == pytest.approx(0.19 * num_keys, rel=1e-3)


def test_uniform_entropy_falls_back_to_midpoint_budget():
    tra = TokenRadiusAttention(TokenRadiusConfig(enabled=True))
    entropy = torch.full((1, 1, 4), 3.0)

    budget = tra.token_budget(entropy, 4096)

    midpoint = (0.02 * 0.19) ** 0.5 * 4096
    assert budget[0, 0, 0].item() == pytest.approx(midpoint, rel=1e-3)


def test_short_sequences_fall_back_to_none():
    q, k, v = _qkv(8, 8)
    tra = TokenRadiusAttention(TokenRadiusConfig(enabled=True))

    assert tra(q, k, v) is None


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        TokenRadiusConfig(min_density=0.5, max_density=0.2)
    with pytest.raises(ValueError):
        TokenRadiusConfig(decay=1.5)
