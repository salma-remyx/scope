"""Hybrid Noise Initialization for long-range video coherence.

Adapts the *Hybrid Noise Initialization* (HNI) strategy from Diff-VF
(Wang et al., "Diff-VF: Training-free High-quality Long Video Generation via
Diffusion Model", arXiv:2608.05976) to the LongLive streaming pipeline.

Diff-VF HNI constrains the *global semantics* of a long video by initialising
each generation window with a **hybrid** noise: a low-frequency component
shared across the whole video (stable global colour / composition) combined
with a high-frequency component that is fresh per window (per-window motion
diversity). It is training-free and model-agnostic -- it only reshapes the
initial noise, never the base model.

Adaptation (Mode 2 -- auxiliary components substituted):

Diff-VF ships HNI for *window-scheduled* long-video generation over a known
total length. LongLive instead generates the video as a stream of non-
overlapping chunks driven by a causal KV cache. The core HNI mechanism -- a
persistent global low-frequency noise blended with each chunk's local
high-frequency noise -- is kept at full fidelity; the window scheduler is
substituted with the LongLive chunk stream, persisting the single global noise
tensor in pipeline state across chunks.

Diff-VF's other two strategies (Weighted Window Sampling and Temporal Extended
Sampling) are window-overlap mechanisms with no analogue in the non-overlapping
causal stream, so they are intentionally out of scope here.
"""

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import (
    ComponentSpec,
    InputParam,
    OutputParam,
)


def low_pass(latents: torch.Tensor, ratio: float) -> torch.Tensor:
    """Idempotent frequency-domain spatial low-pass of ``[B, T, C, H, W]``.

    Keeps the central ``ratio`` fraction of the 2D spectrum (the low
    frequencies) per ``(frame, channel)`` slice via an FFT mask. Because the
    mask is an exact projection, ``low_pass(low_pass(x)) == low_pass(x)``, so
    the low-frequency band of the hybrid noise is shared *exactly* across
    chunks -- this is the frequency-domain operation Diff-VF HNI uses. Input is
    promoted to float32 for the FFT (torch.fft does not support bfloat16) and
    cast back.
    """
    batch, frames, channels, height, width = latents.shape
    orig_dtype = latents.dtype
    flat = latents.to(torch.float32).reshape(batch * frames, channels, height, width)
    freq = torch.fft.fftshift(torch.fft.fft2(flat, norm="ortho"), dim=(-2, -1))
    cy, cx = height // 2, width // 2
    kh = min(max(1, int(round(height * ratio / 2))), max(1, height // 2))
    kw = min(max(1, int(round(width * ratio / 2))), max(1, width // 2))
    low = torch.zeros_like(freq)
    band = (slice(None), slice(None), slice(cy - kh, cy + kh), slice(cx - kw, cx + kw))
    low[band] = freq[band]
    shifted = torch.fft.ifftshift(low, dim=(-2, -1))
    out = torch.fft.ifft2(shifted, norm="ortho").real
    return out.reshape(batch, frames, channels, height, width).to(orig_dtype)


def hybrid_noise(
    latents: torch.Tensor, global_noise: torch.Tensor, ratio: float = 0.25
) -> torch.Tensor:
    """Diff-VF Hybrid Noise Initialization for one chunk.

    Combines the *global low-frequency* structure (shared across every chunk ->
    long-range coherence) with the *local high-frequency* detail of ``latents``
    (fresh per chunk -> motion diversity)::

        hybrid = low_pass(global_noise) + (latents - low_pass(latents))

    Because ``low_pass`` is an idempotent projection, ``low_pass(hybrid) ==
    low_pass(global_noise)`` for every chunk -- the global structure is shared
    exactly -- while the high-frequency residual still varies per chunk. The
    orthonormal FFT band split also preserves the input variance exactly
    (Parseval), so the diffusion scheduler keeps seeing noise of the expected
    scale.
    """
    global_low = low_pass(global_noise, ratio)
    local_high = latents - low_pass(latents, ratio)
    return global_low + local_high


class HybridNoiseInitBlock(ModularPipelineBlocks):
    """Wire HNI into the LongLive block chain.

    Sits between ``auto_prepare_latents`` (which produces the chunk's noisy
    latents) and ``denoise``. It replaces the chunk's pure noise with the HNI
    hybrid noise, persisting a single global noise tensor across chunks so the
    low-frequency structure is shared over the whole video.

    Two tunables, read off the pipeline config (absent -> safe defaults):

    - ``hybrid_noise_strength`` (float, default ``0.0``): blend factor.
      ``0.0`` reproduces the baseline exactly (HNI off); ``1.0`` fully replaces
      the low-frequency band with the shared global structure. Opt-in so
      existing generations are unchanged until a user turns it on.
    - ``hybrid_noise_low_freq_ratio`` (float, default ``0.25``): low-pass
      aggressiveness -- a larger ratio shares more low-frequency content
      globally (stronger coherence, less per-chunk variation).
    """

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("generator", torch.nn.Module)]

    @property
    def expected_configs(self) -> list:
        # Tunables are read defensively off the pipeline config via getattr,
        # so they are optional and need not be declared against model.yaml.
        return []

    @property
    def description(self) -> str:
        return (
            "Hybrid Noise Init (Diff-VF HNI): blend a persistent global "
            "low-frequency noise with each chunk's local high-frequency "
            "noise for long-range temporal coherence."
        )

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=torch.Tensor,
                description="Noisy latents prepared for this chunk",
            ),
            InputParam(
                "base_seed",
                type_hint=int,
                description="Base seed; seeds the persistent global noise",
            ),
            InputParam(
                "global_noise",
                type_hint=torch.Tensor,
                description="Persistent global noise shared across chunks",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Hybrid noisy latents to denoise",
            ),
            OutputParam(
                "global_noise",
                type_hint=torch.Tensor,
                description="Persistent global noise shared across chunks",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        latents = block_state.latents

        strength = float(getattr(components.config, "hybrid_noise_strength", 0.0))
        # strength == 0 -> HNI off: return the baseline latents untouched.
        if strength <= 0.0:
            self.set_block_state(state, block_state)
            return components, state

        ratio = float(getattr(components.config, "hybrid_noise_low_freq_ratio", 0.25))
        generator_param = next(components.generator.model.parameters())
        device = generator_param.device
        dtype = generator_param.dtype

        # Initialise the persistent global noise once per session: seeded only
        # from base_seed (no chunk offset) so it is identical across chunks ->
        # shared global structure.
        global_noise = getattr(block_state, "global_noise", None)
        if global_noise is None or global_noise.shape != latents.shape:
            base_seed = getattr(block_state, "base_seed", None)
            if base_seed is None:
                base_seed = 42
            rng = torch.Generator(device=device).manual_seed(int(base_seed))
            global_noise = torch.randn(
                latents.shape, generator=rng, device=device, dtype=dtype
            )
            block_state.global_noise = global_noise

        hybrid = hybrid_noise(latents, global_noise, ratio)
        blended = (1.0 - strength) * latents + strength * hybrid
        block_state.latents = blended.to(dtype)

        self.set_block_state(state, block_state)
        return components, state
