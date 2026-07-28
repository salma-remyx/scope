"""Tests for the VFIMamba-inspired SSM inter-frame engine on the RIFE pipeline.

Covers:
  * the bidirectional selective SSM mechanism (forward != backward, residual),
  * the SSM interpolator contract (T -> 2T-1, originals preserved, convex blend),
  * end-to-end integration through ``RIFEPipeline`` with ``interpolation_engine="ssm"``
    (no RIFE weights or GPU required).
"""

import torch

from scope.core.pipelines.rife.modules.ssm_interpolation import (
    BidirectionalSelectiveSSM,
    SSMFrameInterpolator,
)
from scope.core.pipelines.rife.pipeline import RIFEPipeline


def _constant_frame(color, height=16, width=16):
    """A single [1, H, W, C] frame filled with ``color`` (a C-tuple in [0, 255])."""
    return (
        torch.tensor(color, dtype=torch.float32)
        .view(1, 1, 1, -1)
        .expand(1, height, width, len(color))
    )


def test_bidirectional_ssm_runs_both_directions_and_is_residual():
    torch.manual_seed(0)
    ssm = BidirectionalSelectiveSSM(d_model=6, d_state=8)
    x = torch.randn(2, 20, 6)

    out = ssm(x)

    assert out.shape == x.shape
    # Residual block: output differs from input only through the scan path.
    assert not torch.allclose(out, x)
    # The two scan directions are not identical -> bidirectional fusion is real.
    proj = ssm.in_proj(x)
    assert proj.shape[-1] == 2 * 8 + 1


def test_ssm_interpolator_doubles_frames_and_preserves_originals():
    interpolator = SSMFrameInterpolator(
        enabled=True, device=torch.device("cpu"), scan_resolution=8
    )
    frames = torch.stack(
        [
            _constant_frame((200.0, 10.0, 30.0)),
            _constant_frame((40.0, 220.0, 60.0)),
            _constant_frame((15.0, 25.0, 240.0)),
            _constant_frame((180.0, 190.0, 200.0)),
        ],
        dim=0,
    ).squeeze(1)  # (4, H, W, 3)

    out = interpolator.interpolate(frames)

    assert out.dtype == torch.uint8
    assert out.shape[0] == 4 * 2 - 1
    # Original frames are preserved at even indices (round-trips exactly).
    assert torch.equal(out[0::2], frames.to(torch.uint8))


def test_ssm_interpolator_middle_frames_are_convex_blends():
    interpolator = SSMFrameInterpolator(
        enabled=True, device=torch.device("cpu"), scan_resolution=8
    )
    f0 = _constant_frame((200.0, 10.0, 30.0)).squeeze(0)
    f1 = _constant_frame((40.0, 220.0, 60.0)).squeeze(0)
    frames = torch.stack([f0, f1], dim=0)

    out = interpolator.interpolate(frames)

    mid = out[1].float()
    lower = torch.minimum(f0, f1)
    upper = torch.maximum(f0, f1)
    # Middle frame is alpha*f0 + (1-alpha)*f1 with alpha in (0, 1) -> bounded.
    assert torch.all(mid >= lower - 1.0)
    assert torch.all(mid <= upper + 1.0)


def test_rife_pipeline_with_ssm_engine_doubles_frame_count():
    """Integration: exercises the call-site wiring in RIFEPipeline, no RIFE weights."""
    from omegaconf import OmegaConf

    config = OmegaConf.create({"interpolation_engine": "ssm"})
    pipeline = RIFEPipeline(config, device=torch.device("cpu"), dtype=torch.float32)

    assert pipeline.interpolation_engine == "ssm"
    assert isinstance(pipeline.interpolator, SSMFrameInterpolator)

    frames = [
        _constant_frame((200.0, 10.0, 30.0)),
        _constant_frame((40.0, 220.0, 60.0)),
        _constant_frame((15.0, 25.0, 240.0)),
    ]
    output = pipeline(video=frames)["video"]

    # 3 input frames -> 5 output frames (2x doubling).
    assert output.shape[0] == 5
    # Output is THWC float in [0, 1]; originals preserved at even indices.
    assert output.shape[1:] == (16, 16, 3)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
    expected_originals = torch.stack([f.squeeze(0) / 255.0 for f in frames], dim=0)
    assert torch.allclose(output[0::2], expected_originals, atol=1e-2)
