"""Tests for the LongLive Hybrid Noise Init (Diff-VF HNI) block.

Covers three things:

1. *Wiring* -- the block is registered in the live ``ALL_BLOCKS`` chain
   between ``auto_prepare_latents`` and ``denoise`` (the call-site edit).
2. *The HNI insight* -- a shared global noise makes the low-frequency band
   identical across chunks (long-range coherence) while the high-frequency
   band stays per-chunk (motion diversity), without changing the noise scale.
3. *Block behaviour* -- disabled is an exact no-op; enabled applies the hybrid
   transform, persists a deterministic global noise, and reuses it across
   chunks.
"""

from types import SimpleNamespace

import torch

from scope.core.pipelines.longlive.blocks.hybrid_noise_init import (
    HybridNoiseInitBlock,
    hybrid_noise,
    low_pass,
)
from scope.core.pipelines.longlive.modular_blocks import ALL_BLOCKS

RATIO = 0.25


# --------------------------------------------------------------------------
# Wiring (imports the existing modular_blocks module; exercises the edit)
# --------------------------------------------------------------------------


def test_block_is_wired_between_prepare_latents_and_denoise():
    keys = list(ALL_BLOCKS.keys())
    assert "hybrid_noise_init" in keys
    assert ALL_BLOCKS["hybrid_noise_init"] is HybridNoiseInitBlock
    # Must run after noise/latents are prepared and before they are denoised.
    assert keys.index("auto_prepare_latents") < keys.index("hybrid_noise_init")
    assert keys.index("hybrid_noise_init") < keys.index("denoise")


# --------------------------------------------------------------------------
# Core insight (pure helpers, no diffusers block machinery)
# --------------------------------------------------------------------------


def test_hybrid_noise_shares_global_low_frequency_across_chunks():
    torch.manual_seed(0)
    shape = (1, 3, 16, 24, 24)
    global_noise = torch.randn(shape)
    chunk_a = torch.randn(shape)
    chunk_b = torch.randn(shape)

    hybrid_a = hybrid_noise(chunk_a, global_noise, RATIO)
    hybrid_b = hybrid_noise(chunk_b, global_noise, RATIO)

    # Low-frequency structure is drawn from the shared global noise, so it is
    # near-identical across chunks -> long-range coherence.
    assert torch.allclose(
        low_pass(hybrid_a, RATIO), low_pass(hybrid_b, RATIO), atol=1e-4
    )
    # Raw chunks do NOT share low-frequency structure -- sanity check the above
    # is a real effect, not a trivial one.
    assert not torch.allclose(
        low_pass(chunk_a, RATIO), low_pass(chunk_b, RATIO), atol=1e-4
    )

    # High-frequency detail stays per-chunk, so the full latents still differ
    # -> motion diversity is preserved.
    assert not torch.allclose(hybrid_a, hybrid_b)

    # The orthonormal band split preserves noise scale exactly (Parseval).
    assert 0.8 < (hybrid_a.std() / chunk_a.std()).item() < 1.2


def test_low_pass_is_idempotent_projection():
    torch.manual_seed(1)
    latents = torch.randn(1, 3, 16, 24, 24)
    once = low_pass(latents, RATIO)
    twice = low_pass(once, RATIO)
    assert torch.allclose(once, twice, atol=1e-4)


# --------------------------------------------------------------------------
# Block integration (real __call__, lightweight fakes)
# --------------------------------------------------------------------------


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _FakeGenerator:
    def __init__(self):
        self.model = _FakeModel()


def _make_components(strength, ratio=RATIO):
    return SimpleNamespace(
        config=SimpleNamespace(
            hybrid_noise_strength=strength,
            hybrid_noise_low_freq_ratio=ratio,
        ),
        generator=_FakeGenerator(),
    )


def _identity_block_state(state):
    return state


def _noop_set_block_state(state, block_state):
    pass


def _make_block():
    """A real HybridNoiseInitBlock with block-state bypassed for a unit test.

    Mirrors how test_audio_packets uses object.__new__ / SimpleNamespace to
    exercise logic without standing up the full diffusers pipeline state.
    """
    block = HybridNoiseInitBlock()
    block.get_block_state = _identity_block_state
    block.set_block_state = _noop_set_block_state
    return block


def _make_state(seed=42):
    return SimpleNamespace(
        latents=torch.randn(1, 3, 16, 16, 16),
        base_seed=seed,
        global_noise=None,
    )


def test_block_is_identity_when_disabled():
    block = _make_block()
    components = _make_components(strength=0.0)
    state = _make_state()
    original = state.latents.clone()

    block(components, state)

    # strength == 0 -> baseline latents returned untouched.
    assert torch.equal(state.latents, original)
    assert state.global_noise is None


def test_block_applies_hybrid_and_persists_global_noise():
    block = _make_block()
    components = _make_components(strength=1.0)
    state = _make_state(seed=42)
    original = state.latents.clone()

    block(components, state)

    # Latents changed (hybrid noise applied)...
    assert not torch.equal(state.latents, original)
    # ...and the persistent global noise was stored...
    assert state.global_noise is not None
    # ...seeded only from base_seed (deterministic, identical across chunks).
    rng = torch.Generator(device="cpu").manual_seed(42)
    expected = torch.randn(
        state.global_noise.shape, generator=rng, device="cpu", dtype=torch.float32
    )
    assert torch.equal(state.global_noise, expected)


def test_block_reuses_global_noise_across_chunks():
    block = _make_block()
    components = _make_components(strength=1.0)
    state = _make_state()

    block(components, state)
    first_chunk_noise = state.global_noise

    # Second chunk: must reuse the same persistent global noise, not re-init.
    state.latents = torch.randn(1, 3, 16, 16, 16)
    block(components, state)

    assert state.global_noise is first_chunk_noise
