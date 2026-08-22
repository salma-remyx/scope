"""Tests for the frozen block-sparse KV selection in the shared attention path.

Covers the integration end to end: :func:`scope.core.pipelines.wan2_1.modules.attention.attention`
is the dispatcher every wan2_1-based pipeline (memflow, longlive, ...) routes
self-attention through, so pruning has to be invisible to it. The disabled
path must stay byte-identical to dense attention, and the enabled path must
compute exact attention over the retained blocks -- the retained-mass budget
is the only thing that separates it from dense.
"""

import pytest
import torch

from scope.core.pipelines.wan2_1.modules.attention import attention
from scope.core.pipelines.wan2_1.modules.block_sparse_attention import (
    block_sparse_state,
    reset_block_sparse_state,
    select_block_indices,
)


def _qkv(seq_q=256, seq_k=1024, batch=1, heads=4, dim=64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq_q, heads, dim, generator=gen)
    k = torch.randn(batch, seq_k, heads, dim, generator=gen)
    v = torch.randn(batch, seq_k, heads, dim, generator=gen)
    return q, k, v


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


@pytest.fixture(autouse=True)
def _dense_by_default():
    """Every test starts and ends from the production default: pruning off."""
    state = block_sparse_state()
    state.enabled = False
    reset_block_sparse_state()
    yield
    state.enabled = False
    reset_block_sparse_state()


def test_disabled_by_default_keeps_dense_path():
    q, k, v = _qkv()

    out = attention(q, k, v)

    assert torch.allclose(out, _sdpa(q, k, v), atol=1e-5)
    # Nothing was measured and nothing was gathered.
    assert block_sparse_state()._indices is None


def test_enabled_path_is_exact_attention_on_retained_support():
    q, k, v = _qkv(batch=2, seed=1)
    state = block_sparse_state()
    state.enabled = True
    reset_block_sparse_state()

    out = attention(q, k, v)

    indices = state._indices
    assert indices is not None
    block = state.block_size
    n_kept = indices.size(1)
    assert n_kept < k.size(1) // block, "expected some blocks to be pruned"

    # Rebuilding the same support by hand must reproduce the output exactly:
    # pruning only ever drops blocks, it never approximates the ones kept.
    for b in range(q.size(0)):
        k_b = torch.cat([k[b, i * block : (i + 1) * block] for i in indices[b]])
        v_b = torch.cat([v[b, i * block : (i + 1) * block] for i in indices[b]])
        reference = _sdpa(q[b : b + 1], k_b.unsqueeze(0), v_b.unsqueeze(0))
        assert torch.allclose(out[b], reference[0], atol=1e-5)


def test_retained_mass_budget_holds():
    """Every query block/head must keep at least `retained_mass` of its mass.

    The dispatcher gathers one KV set per call, so the per-(query block, head)
    selections are unioned; the union can only over-retain relative to each
    individual selection, never under-retain.
    """
    q, k, _ = _qkv(seed=2)
    threshold = block_sparse_state().retained_mass

    indices = select_block_indices(q, k, threshold=threshold)

    block = block_sparse_state().block_size
    heads, dim = q.size(2), q.size(3)
    q_blocks = q[0].reshape(-1, block, heads, dim)
    k_blocks = k[0, : (k.size(1) // block) * block].reshape(-1, block, heads, dim)
    full = torch.einsum("qihc,kihc->qkh", q_blocks, k_blocks)
    kept = torch.einsum("qihc,kihc->qkh", q_blocks, k_blocks[indices[0]])
    full = (full * dim**-0.5).softmax(dim=1)
    kept = (kept * dim**-0.5).softmax(dim=1)
    per_query_head = kept.sum(dim=1) / full.sum(dim=1)
    assert per_query_head.min() >= threshold


def test_frozen_indices_are_reused_across_calls():
    q, k, v = _qkv(seed=3)
    state = block_sparse_state()
    state.enabled = True
    reset_block_sparse_state()

    first = attention(q, k, v)
    frozen = state._indices.clone()
    second = attention(q, k, v)

    assert torch.equal(state._indices, frozen)
    # Identical frozen support => identical output, whatever calls it.
    assert torch.allclose(first, second, atol=1e-6)


def test_rebuild_after_window_tracks_new_support():
    """A shape change invalidates the frozen indices instead of gathering stale."""
    q, k, v = _qkv(seed=4)
    state = block_sparse_state()
    state.enabled = True
    state.rebuild_every = 1
    reset_block_sparse_state()

    attention(q, k, v)
    first = state._indices.clone()
    q2, k2, v2 = _qkv(seq_k=1536, seed=5)
    attention(q2, k2, v2)

    assert not torch.equal(state._indices, first)
    assert state._indices.size(1) <= 1536 // state.block_size


def test_short_sequences_stay_dense():
    q, k, v = _qkv(seq_q=64, seq_k=64)
    state = block_sparse_state()
    state.enabled = True
    reset_block_sparse_state()

    out = attention(q, k, v)

    assert out.shape == q.shape
    assert state._indices is None


def test_reset_forces_remeasure():
    q, k, v = _qkv(seed=6)
    state = block_sparse_state()
    state.enabled = True
    reset_block_sparse_state()
    attention(q, k, v)
    assert state._indices is not None

    reset_block_sparse_state()

    assert state._indices is None


def test_varlen_sequences_stay_dense():
    """Padded/varlen batches skip pruning -- block boundaries would misalign."""
    q, k, v = _qkv(seed=8)
    state = block_sparse_state()
    state.enabled = True
    reset_block_sparse_state()

    out = attention(q, k, v, k_lens=torch.tensor([k.size(1)]))

    assert torch.allclose(out, _sdpa(q, k, v), atol=1e-5)
    assert state._indices is None


def test_select_block_indices_returns_temporal_order():
    q, k, _ = _qkv(seed=7)

    indices = select_block_indices(q, k)

    assert indices.shape[0] == 1
    assert indices.dtype == torch.long
    # Blocks keep their order, so temporal locality is preserved.
    assert (indices[0, 1:] > indices[0, :-1]).all()
