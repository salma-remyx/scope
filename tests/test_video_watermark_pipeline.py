"""Integration tests for the video-watermark provenance pipeline.

These exercise the registry wiring (the call-site edit in
``scope.core.pipelines.registry``) plus the blind embed/extract round-trip
that is the pipeline's core value for AI-content provenance.
"""

import torch

from scope.core.pipelines.interface import Pipeline
from scope.core.pipelines.video_watermark.pipeline import (
    VideoWatermarkPipeline,
    extract_watermark,
)
from scope.core.pipelines.video_watermark.schema import VideoWatermarkConfig


def _frames(value: float = 128.0, count: int = 2, height: int = 96, width: int = 96):
    """Build the realtime frame contract: a list of ``[1, H, W, C]`` [0,255] tensors."""
    return [torch.full((1, height, width, 3), value) for _ in range(count)]


def test_config_identity_and_pipeline_base():
    assert VideoWatermarkConfig.pipeline_id == "video-watermark"
    assert VideoWatermarkPipeline.get_config_class() is VideoWatermarkConfig
    assert issubclass(VideoWatermarkPipeline, Pipeline)


def test_pipeline_is_registered():
    # Lazy import: importing the registry module triggers full builtin-pipeline
    # discovery, which we only want during this test (mirrors the pattern in
    # test_pipeline_registry.py to keep collection lightweight).
    from scope.core.pipelines.registry import PipelineRegistry

    assert PipelineRegistry.is_registered("video-watermark")
    assert PipelineRegistry.get("video-watermark") is VideoWatermarkPipeline


def test_embed_then_extract_roundtrips_message():
    message = "scope"
    pipe = VideoWatermarkPipeline(
        watermark_message=message,
        watermark_strength=0.05,
        device=torch.device("cpu"),
    )

    out = pipe(video=_frames())
    watermarked = out["video"]

    assert watermarked.shape == (2, 96, 96, 3)
    assert torch.all((watermarked >= 0) & (watermarked <= 1))

    recovered = extract_watermark(
        watermarked, num_bits=pipe.num_bits, key=pipe.watermark_key
    )
    assert recovered == message


def test_pipeline_extract_method_round_trips():
    pipe = VideoWatermarkPipeline(watermark_message="scope", device=torch.device("cpu"))
    out = pipe(video=_frames())
    assert pipe.extract(out["video"]) == "scope"


def test_wrong_key_does_not_recover_message():
    pipe = VideoWatermarkPipeline(
        watermark_message="scope",
        watermark_key=1337,
        watermark_strength=0.05,
        device=torch.device("cpu"),
    )
    watermarked = pipe(video=_frames())["video"]

    # A different key yields independent carriers -> random bits -> garbage.
    wrong = extract_watermark(watermarked, num_bits=pipe.num_bits, key=9999)
    assert wrong != "scope"


def test_watermark_is_imperceptible_at_default_strength():
    original = (
        torch.stack([f.squeeze(0) for f in _frames(value=200.0)], dim=0).float() / 255.0
    )
    pipe = VideoWatermarkPipeline(device=torch.device("cpu"))  # strength 0.03

    out = pipe(video=_frames(value=200.0))["video"]

    # Mean perturbation tracks the (small) embedding strength; peaks from the
    # carrier's Gaussian tails are bounded well below a visible step.
    mean_delta = (out - original).abs().mean().item()
    max_delta = (out - original).abs().max().item()
    assert mean_delta < 0.05
    assert max_delta < 0.3


def test_zero_strength_is_a_passthrough():
    frames = _frames()
    original = torch.stack([f.squeeze(0) for f in frames], dim=0).float() / 255.0

    pipe = VideoWatermarkPipeline(
        watermark_message="scope",
        watermark_strength=0.0,
        device=torch.device("cpu"),
    )
    out = pipe(video=frames)["video"]

    assert torch.allclose(out, original.clamp(0, 1))
