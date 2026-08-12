"""Integration tests for the GEDepth ground-embedding pipeline.

These exercise the registry wiring in the (non-new)
``scope.core.pipelines.registry`` module — the tuple added there must
surface a registered ``gedepth`` pipeline — and the analytic
ground-embedding geometry that is GEDepth's core contribution.
"""

import torch

from scope.core.pipelines import GEDepthPipeline
from scope.core.pipelines.gedepth.ground_embedding import ground_distance_map
from scope.core.pipelines.gedepth.schema import GEDepthConfig
from scope.core.pipelines.registry import PipelineRegistry


def _frames(num_frames: int = 2, height: int = 16, width: int = 24):
    """List of [1, H, W, C] uint8 tensors in the pipeline's THWC contract."""
    return [
        torch.randint(0, 255, (1, height, width, 3), dtype=torch.uint8)
        for _ in range(num_frames)
    ]


def test_gedepth_is_registered():
    # Importing the registry module runs _register_pipelines(); the gedepth
    # entry added to pipeline_configs must produce a registered pipeline.
    assert PipelineRegistry.is_registered("gedepth")
    assert PipelineRegistry.get("gedepth").__name__ == "GEDepthPipeline"
    assert PipelineRegistry.get_config_class("gedepth") is GEDepthConfig


def test_ground_embedding_shape_range_and_dtype():
    pipeline = GEDepthPipeline(GEDepthConfig(), device=torch.device("cpu"))
    output = pipeline(video=_frames())["video"]

    assert output.shape == (2, 16, 24, 3)
    assert output.dtype == torch.float32
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0


def test_prior_is_identical_across_frames_for_static_camera():
    # The ground embedding is a content-independent geometric prior: for a
    # static camera it must be identical frame-to-frame regardless of input.
    pipeline = GEDepthPipeline(GEDepthConfig(), device=torch.device("cpu"))
    output = pipeline(video=_frames(num_frames=3))["video"]

    assert torch.equal(output[0], output[1])
    assert torch.equal(output[1], output[2])


def test_top_of_image_is_further_than_bottom():
    # With a downward pitch the bottom rows look at nearby ground (small
    # distance) and the top rows look toward the horizon/sky (far). Higher
    # value == further, so the top half's mean must exceed the bottom half's.
    pipeline = GEDepthPipeline(GEDepthConfig(), device=torch.device("cpu"))
    frame = pipeline(video=_frames(num_frames=1, height=32, width=32))["video"][0]

    top_half = frame[:16].mean()
    bottom_half = frame[16:].mean()
    assert top_half > bottom_half


def test_sky_rows_are_at_the_far_end():
    # Rows above the horizon never strike the ground and must clamp to the
    # far end of the [0, 1] range (1.0 by default).
    distance = ground_distance_map(
        height=16, width=8, camera_height=1.5, pitch_deg=0.0, vertical_fov_deg=60.0
    )
    assert torch.isinf(distance[0]).all()  # top row points above the horizon

    pipeline = GEDepthPipeline(GEDepthConfig(), device=torch.device("cpu"))
    frame = pipeline(video=_frames(num_frames=1, height=16, width=8))["video"][0]
    assert torch.allclose(frame[0], torch.ones_like(frame[0]))


def test_invert_flips_near_far_convention():
    base = GEDepthPipeline(GEDepthConfig(), device=torch.device("cpu"))
    inverted = GEDepthPipeline(GEDepthConfig(invert=True), device=torch.device("cpu"))

    frames = _frames(num_frames=1, height=16, width=16)
    a = base(video=frames)["video"]
    b = inverted(video=frames)["video"]

    assert torch.allclose(a, 1.0 - b, atol=1e-6)


def test_pitch_changes_the_prior():
    # Looking further down (larger pitch) brings the near-ground region up
    # the image, so the depth prior must change — geometry is wired through.
    low_pitch = GEDepthPipeline(
        GEDepthConfig(pitch_deg=2.0), device=torch.device("cpu")
    )(video=_frames(num_frames=1, height=32, width=32))["video"]
    high_pitch = GEDepthPipeline(
        GEDepthConfig(pitch_deg=30.0), device=torch.device("cpu")
    )(video=_frames(num_frames=1, height=32, width=32))["video"]

    assert not torch.allclose(low_pitch, high_pitch, atol=1e-6)
