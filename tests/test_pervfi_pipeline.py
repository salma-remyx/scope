"""Integration tests for the PerVFI asymmetric-blending interpolation pipeline.

These exercise the wiring (registration through the existing
:class:`PipelineRegistry`, config parity with the sibling RIFE pipeline) and
the core asymmetric-blend behaviour on synthetic frames. They import from
non-new modules (``scope.core.pipelines.registry``, ``base_schema``,
``rife.schema``) to prove the integration, not just self-test the new file.
"""

import numpy as np
import torch

from scope.core.pipelines.base_schema import BasePipelineConfig, UsageType
from scope.core.pipelines.pervfi.asymmetric_blend import (
    asymmetric_blend,
    estimate_flow,
)
from scope.core.pipelines.pervfi.pipeline import PerVFIPipeline
from scope.core.pipelines.pervfi.schema import PerVFIConfig
from scope.core.pipelines.registry import PipelineRegistry
from scope.core.pipelines.rife.schema import RIFEConfig


def _textured(shift: int = 0, size: int = 40) -> torch.Tensor:
    """A textured frame (so flow is well-defined everywhere) translated by shift."""
    rng = np.random.default_rng(7)
    base = rng.random((3, size - 12, size - 12)).astype(np.float32)
    canvas = np.zeros((3, size, size), dtype=np.float32)
    s = 6 + shift
    canvas[:, s : s + base.shape[1], s : s + base.shape[2]] = base
    return torch.from_numpy(canvas)


def test_pervfi_is_registered_through_existing_registry():
    """The pipeline must be discoverable via the existing PipelineRegistry."""
    assert PipelineRegistry.is_registered("pervfi")
    pipeline_cls = PipelineRegistry.get("pervfi")
    assert pipeline_cls is PerVFIPipeline


def test_pervfi_config_shape_matches_rife_postprocessor_contract():
    """PerVFI is a quality-oriented alternative to RIFE: same contract shape."""
    assert issubclass(PerVFIConfig, BasePipelineConfig)
    assert PerVFIConfig.pipeline_id == "pervfi"
    # Same usage / video-mode / no-prompt contract as RIFE.
    assert PerVFIConfig.usage == RIFEConfig.usage == [UsageType.POSTPROCESSOR]
    assert PerVFIConfig.produces_video is True
    assert PerVFIConfig.supports_prompts is False
    assert "video" in PerVFIConfig.get_supported_modes()
    # Weight-free adapted port: no artifacts are declared.
    assert PerVFIConfig.artifacts == []


def test_flow_estimator_recovers_translation():
    """The parameter-free flow proxy recovers a known translation."""
    img0 = _textured(shift=0)[None]  # NCHW
    img1 = _textured(shift=4)[None]
    flow = estimate_flow(img0, img1, levels=3, iters=4, window=7)  # img0 -> img1
    # Content moved right/down by (4, 4) pixels; check the textured interior.
    interior = flow[:, :, 12:28, 12:28]
    dx = interior[:, 0].median().item()
    dy = interior[:, 1].median().item()
    assert abs(dx - 4.0) < 1.5
    assert abs(dy - 4.0) < 1.5


def test_asymmetric_blend_recovers_endpoints():
    """At t=0/1 the asymmetric blend favors the corresponding endpoint.

    Asserted relatively (correct-endpoint error < cross-endpoint error) so the
    property is robust to the flow proxy's sub-pixel noise, plus a sane absolute
    bound that the correct endpoint is recovered closely.
    """
    img0 = _textured(shift=0)[None]
    img1 = _textured(shift=4)[None]
    f01 = estimate_flow(img0, img1, levels=3, iters=4)
    f10 = estimate_flow(img1, img0, levels=3, iters=4)
    sl = (slice(None), slice(None), slice(14, 26), slice(14, 26))

    at0 = asymmetric_blend(img0, img1, f01, f10, t=0.0)
    at1 = asymmetric_blend(img0, img1, f01, f10, t=1.0)

    err0_to_0 = torch.abs(at0[sl] - img0[sl]).mean().item()
    err0_to_1 = torch.abs(at0[sl] - img1[sl]).mean().item()
    err1_to_1 = torch.abs(at1[sl] - img1[sl]).mean().item()
    err1_to_0 = torch.abs(at1[sl] - img0[sl]).mean().item()

    # The blend favors the correct endpoint and recovers it reasonably well.
    assert err0_to_0 < err0_to_1 and err0_to_0 < 0.25
    assert err1_to_1 < err1_to_0 and err1_to_1 < 0.25


def test_pipeline_doubles_frame_rate():
    """The integrated pipeline turns T frames into 2T-1 interpolated frames."""
    config = type("C", (), {"height": 40, "width": 40})()
    pipeline = PerVFIPipeline(config, device=torch.device("cpu"), dtype=torch.float32)

    frames = []
    for shift in range(4):
        chwt = _textured(shift=shift)  # 3,40,40
        thwc = chwt.permute(1, 2, 0).unsqueeze(0) * 255.0  # 1,40,40,3 in [0,255]
        frames.append(thwc)

    out = pipeline(video=frames)["video"]
    assert out.ndim == 4  # THWC
    assert out.shape[0] == 2 * len(frames) - 1
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
