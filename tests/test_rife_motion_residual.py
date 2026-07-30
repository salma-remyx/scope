"""Tests for the MoMo-inspired disentangled-motion residual estimator.

The integration tests exercise ``RIFEInterpolator`` -- the interpolation engine
that ``RIFEPipeline.__call__`` delegates to -- with a stubbed model so no RIFE
weights are required. They verify the wiring added for the disentangled-motion
decomposition (``inference_motion`` exposes RIFE's flow/mask, and the
interpolator stores the per-frame residual confidence on
``last_motion_residual``).
"""

import torch

from scope.core.pipelines.rife.modules import interpolation as rife_interpolation
from scope.core.pipelines.rife.modules.motion_residual import (
    backward_warp,
    disentangle_motion,
    estimate_motion_residual_confidence,
    residual_ambiguity,
)


class _FakeRIFEModel:
    """Stand-in for RIFE HDv3 that returns zero flow + even blend mask.

    Zero flow means each warped view is the source frame itself, so the
    residual proxy reduces to ``mean(|img0 - img1|)`` -- easy to assert.
    """

    def inference_motion(self, img0, img1, scale=1.0, **_):
        merged = (img0 + img1) / 2.0
        flow = torch.zeros(
            img0.shape[0],
            4,
            img0.shape[2],
            img0.shape[3],
            device=img0.device,
            dtype=img0.dtype,
        )
        mask = torch.full(
            (img0.shape[0], 1, img0.shape[2], img0.shape[3]),
            0.5,
            device=img0.device,
            dtype=img0.dtype,
        )
        return merged, flow, mask


def _make_interpolator(fake_model):
    """Build a RIFEInterpolator without loading weights (object.__new__ shortcut)."""
    interp = object.__new__(rife_interpolation.RIFEInterpolator)
    interp.enabled = True
    interp.device = torch.device("cpu")
    interp.model = fake_model
    interp.model_path = None
    interp.last_motion_residual = []
    return interp


def _frames(*values):
    """Build a [T, 32, 32, 3] uint8 clip from per-frame constant values."""
    return torch.stack([torch.full((32, 32, 3), v, dtype=torch.uint8) for v in values])


def test_interpolator_reports_high_confidence_for_static_content(monkeypatch):
    """Identical consecutive frames -> the flow-warped views agree -> confidence ~1."""
    monkeypatch.setattr(rife_interpolation, "RIFE_AVAILABLE", True)
    interp = _make_interpolator(_FakeRIFEModel())

    result = interp.interpolate(_frames(128, 128))

    assert result.shape[0] == 3  # 2 input frames -> 2*2-1 = 3 output frames
    assert result.dtype == torch.uint8
    assert len(interp.last_motion_residual) == 1
    stats = interp.last_motion_residual[0]
    assert stats["residual_energy"] < 1e-5
    assert stats["confidence"] > 0.99


def test_interpolator_reports_low_confidence_for_disagreeing_frames(monkeypatch):
    """Maximally different frames -> warped views disagree -> confidence ~0."""
    monkeypatch.setattr(rife_interpolation, "RIFE_AVAILABLE", True)
    interp = _make_interpolator(_FakeRIFEModel())

    interp.interpolate(_frames(0, 255))

    stats = interp.last_motion_residual[0]
    assert stats["residual_energy"] > 0.99
    assert stats["confidence"] < 1e-2


def test_residual_confidence_is_monotonic_in_frame_disagreement(monkeypatch):
    """More disagreement between the warped views -> lower confidence."""
    monkeypatch.setattr(rife_interpolation, "RIFE_AVAILABLE", True)
    interp = _make_interpolator(_FakeRIFEModel())

    interp.interpolate(_frames(0, 64))
    mid = interp.last_motion_residual[0]["confidence"]
    interp.interpolate(_frames(0, 255))
    far = interp.last_motion_residual[0]["confidence"]

    assert mid > far


def test_backward_warp_is_identity_for_zero_flow():
    frame = torch.arange(3 * 8 * 8, dtype=torch.float32).view(1, 3, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)

    warped = backward_warp(frame, flow)

    assert torch.allclose(warped, frame)


def test_residual_ambiguity_and_confidence_math():
    img0 = torch.zeros(1, 3, 8, 8)
    img1 = torch.ones(1, 3, 8, 8)
    flow = torch.zeros(1, 4, 8, 8)
    mask = torch.full((1, 1, 8, 8), 0.5)

    warped0, warped1, blend = disentangle_motion(img0, img1, flow, mask)

    assert torch.allclose(warped0, img0)  # zero flow -> identity warp
    assert torch.allclose(warped1, img1)
    assert torch.allclose(blend, torch.full_like(img0, 0.5))

    ambiguity = residual_ambiguity(warped0, warped1)
    assert ambiguity.shape == (1, 1, 8, 8)
    assert torch.allclose(ambiguity, torch.ones(1, 1, 8, 8))

    stats = estimate_motion_residual_confidence(img0, img1, flow, mask)
    assert len(stats) == 1
    assert abs(stats[0]["residual_energy"] - 1.0) < 1e-6
    assert abs(stats[0]["confidence"] - 0.0) < 1e-6

    # Identical frames -> no residual -> full confidence.
    identical = estimate_motion_residual_confidence(img0, img0, flow, mask)
    assert abs(identical[0]["residual_energy"] - 0.0) < 1e-6
    assert abs(identical[0]["confidence"] - 1.0) < 1e-6
