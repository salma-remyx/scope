"""Integration tests for the metric-depth pipeline (UniDepth, adapted).

Exercises the wiring (registry + config) and the ``rescale_to_metric``
recovery core without loading model weights or requiring a GPU.
"""

import torch

from scope.core.pipelines.base_schema import BasePipelineConfig
from scope.core.pipelines.metric_depth.pipeline import (
    MetricDepthPipeline,
    rescale_to_metric,
)
from scope.core.pipelines.metric_depth.schema import MetricDepthConfig
from scope.core.pipelines.registry import PipelineRegistry


def test_get_config_class_returns_metric_config():
    """The pipeline exposes its config class through the node contract."""
    assert MetricDepthPipeline.get_config_class() is MetricDepthConfig
    assert issubclass(MetricDepthConfig, BasePipelineConfig)


def test_config_metadata():
    """Config carries the registry id and preprocessor usage."""
    config = MetricDepthConfig()
    assert config.pipeline_id == "metric-depth"
    assert config.pipeline_id == MetricDepthConfig.pipeline_id
    from scope.core.pipelines.base_schema import UsageType

    assert UsageType.PREPROCESSOR in config.usage


def test_registry_round_trip():
    """The pipeline registers cleanly via the same call built-ins use.

    Mirrors the ``PipelineRegistry.register`` invocation that
    ``_register_pipelines`` performs, so a passing round-trip proves the
    wiring (id derivation, config-class lookup) is sound. GPU-gated built-ins
    are skipped on CPU CI, so we register explicitly and restore prior state
    (the real built-in may be pre-registered under a GPU runner).
    """
    pipeline_id = MetricDepthConfig.pipeline_id
    original = PipelineRegistry.get(pipeline_id)
    PipelineRegistry.register(pipeline_id, MetricDepthPipeline)
    try:
        assert PipelineRegistry.is_registered(pipeline_id)
        assert PipelineRegistry.get(pipeline_id) is MetricDepthPipeline
        assert PipelineRegistry.get_config_class(pipeline_id) is MetricDepthConfig
    finally:
        if original is not None:
            PipelineRegistry.register(pipeline_id, original)
        else:
            PipelineRegistry.unregister(pipeline_id)

    assert PipelineRegistry.get(pipeline_id) is original


def test_rescale_to_metric_is_monotonic_and_median_anchored():
    """Farther relative depth -> deeper metric depth; median anchor holds."""
    relative = torch.linspace(0.0, 1.0, 101)
    metric = rescale_to_metric(
        relative,
        median_depth_m=25.0,
        depth_min_m=1.0,
        depth_max_m=50.0,
    )

    # Monotonic non-decreasing (relative order preserved through affine map).
    assert torch.all(metric[1:] >= metric[:-1])
    # Robust median anchor recovers the absolute scale.
    assert abs(float(metric.median()) - 25.0) < 1e-4
    # Depth is strictly positive.
    assert torch.all(metric > 0)


def test_rescale_to_metric_is_camera_aware():
    """Pinhole correction Z ∝ f: a longer focal yields proportionally deeper."""
    relative = torch.linspace(0.4, 0.6, 64)  # maps well above the eps floor
    common = {
        "median_depth_m": 25.0,
        "depth_min_m": 0.5,
        "depth_max_m": 50.0,
    }
    base = rescale_to_metric(
        relative, focal_length_px=800.0, reference_focal_px=800.0, **common
    )
    longer = rescale_to_metric(
        relative, focal_length_px=1600.0, reference_focal_px=800.0, **common
    )

    assert torch.allclose(longer, base * 2.0, atol=1e-4)


def test_rescale_to_metric_preserves_shape_and_range():
    """Output shape matches input; without a focal the median anchor holds."""
    relative = torch.rand((3, 16, 16))
    metric = rescale_to_metric(relative, median_depth_m=10.0)

    assert metric.shape == relative.shape
    assert abs(float(metric.median()) - 10.0) < 1e-4


def test_rescale_to_metric_rejects_invalid_range():
    """A non-increasing metric range is a configuration error."""
    raised = False
    try:
        rescale_to_metric(torch.rand((4, 4)), depth_min_m=10.0, depth_max_m=10.0)
    except ValueError:
        raised = True
    assert raised
