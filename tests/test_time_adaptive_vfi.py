"""Tests for the time-adaptive VFI pipeline and its registry wiring.

The registry test imports the non-new ``scope.core.pipelines.registry``
module and exercises the call-site edit that registers the pipeline; the
remaining tests cover the pure scheduling and trajectory logic (no GPU or
RIFE weights required).
"""

import torch

from scope.core.pipelines.registry import PipelineRegistry
from scope.core.pipelines.time_adaptive_vfi import time_adaptive
from scope.core.pipelines.time_adaptive_vfi.pipeline import TimeAdaptiveVFIPipeline


def test_registry_wires_time_adaptive_vfi():
    """The call-site edit registers the new pipeline in the registry."""
    assert "time-adaptive-vfi" in PipelineRegistry.list_pipelines()
    assert PipelineRegistry.is_registered("time-adaptive-vfi")
    resolved = PipelineRegistry.get("time-adaptive-vfi")
    assert resolved is TimeAdaptiveVFIPipeline
    assert resolved.get_config_class().pipeline_id == "time-adaptive-vfi"


def test_config_exposes_time_adaptive_fields():
    config = TimeAdaptiveVFIPipeline.get_config_class()
    assert config.pipeline_id == "time-adaptive-vfi"
    defaults = config().model_dump()
    assert defaults["num_intermediate_frames"] == 1
    assert defaults["interpolation_schedule"] == "uniform"
    assert defaults["resshift_shift_ratio"] == 1.25


def test_uniform_schedule():
    assert time_adaptive.uniform_schedule(0) == []
    assert time_adaptive.uniform_schedule(1) == [0.5]
    assert time_adaptive.uniform_schedule(3) == [0.25, 0.5, 0.75]
    # monotonically increasing, strictly inside (0, 1)
    times = time_adaptive.uniform_schedule(5)
    assert times == sorted(times)
    assert all(0.0 < t < 1.0 for t in times)


def test_resshift_schedule_is_valid_and_nonuniform():
    times = time_adaptive.resshift_schedule(3, shift_ratio=1.5)
    assert len(times) == 3
    assert times == sorted(times)
    assert all(0.0 < t < 1.0 for t in times)
    # gaps follow a geometric progression -> not all equal
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert gaps[0] != gaps[-1]
    # distinct from uniform for the same count
    assert times != time_adaptive.uniform_schedule(3)


def test_resshift_schedule_ratio_one_is_uniform():
    # geometric ratio 1.0 -> equal weights -> uniform spacing (within float tol)
    resshift = time_adaptive.resshift_schedule(4, shift_ratio=1.0)
    uniform = time_adaptive.uniform_schedule(4)
    assert len(resshift) == len(uniform)
    for r, u in zip(resshift, uniform, strict=True):
        assert abs(r - u) < 1e-9


def test_resshift_schedule_rejects_nonpositive_ratio():
    try:
        time_adaptive.resshift_schedule(2, shift_ratio=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive shift_ratio")


def test_densify_sequence_with_fake_synthesizer():
    # Fake synthesizer: linear blend at time t (no RIFE model needed).
    def synth(frame0, frame1, t):
        blended = frame0.float() * (1.0 - t) + frame1.float() * t
        return blended.clamp(0, 255).to(torch.uint8)

    frames = torch.zeros((3, 4, 4, 3), dtype=torch.uint8)
    frames[0] = 0
    frames[1] = 100
    frames[2] = 200

    out = time_adaptive.densify_sequence(frames, synth, num_intermediate=2)
    # 3 frames + 2 gaps * 2 intermediates = 7
    assert out.shape == (7, 4, 4, 3)
    # First intermediate at uniform t=1/3 between 0 and 100 ~= 33.
    assert out[1][0, 0, 0].item() == int(100 * (1.0 / 3.0))
    # Endpoints are preserved exactly.
    assert out[0][0, 0, 0].item() == 0
    assert out[-1][0, 0, 0].item() == 200


def test_densify_sequence_default_reproduces_frame_doubling_count():
    def synth(frame0, frame1, t):
        return frame0

    frames = torch.zeros((5, 2, 2, 3), dtype=torch.uint8)
    # 1 intermediate + uniform == RIFE doubling: 2T - 1 frames.
    out = time_adaptive.densify_sequence(frames, synth, num_intermediate=1)
    assert out.shape == (9, 2, 2, 3)


def test_densify_sequence_passes_through_short_input():
    def synth(frame0, frame1, t):
        return frame0

    single = torch.zeros((1, 4, 4, 3), dtype=torch.uint8)
    out = time_adaptive.densify_sequence(single, synth, num_intermediate=3)
    assert out.shape == (1, 4, 4, 3)


def test_interpolate_pair_validates_endpoint_shape():
    bad = torch.zeros((4, 4), dtype=torch.uint8)
    try:
        time_adaptive.interpolate_pair(bad, bad, lambda f0, f1, t: f0, [0.5])
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-[H,W,C] endpoints")


def test_rife_synthesizer_requires_loaded_model():
    # Without a loaded RIFE model the arbitrary-t synthesizer must fail loudly
    # rather than silently producing wrong output.

    class FakeInterpolator:
        model = None
        device = torch.device("cpu")

    synthesizer = time_adaptive.RIFETimestepSynthesizer(FakeInterpolator())
    frame = torch.zeros((4, 4, 3), dtype=torch.uint8)
    try:
        synthesizer(frame, frame, 0.5)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when RIFE model is not loaded")
