"""Integration tests for the surface-normals-from-depth node.

These tests import the existing ``scope.core.pipelines.registry`` (a non-new
module) so the wiring edit -- the ``surface_normals`` entry added to the
registry's ``pipeline_configs`` list -- is exercised via auto-registration,
and they assert the Metric3Dv2-inspired geometry behaves correctly.
"""

import torch

# Importing the registry runs ``_initialize_registry`` and registers every
# config-driven pipeline whose VRAM requirements are met. surface-normals has
# no learned model (estimated_vram_gb unset) so it registers without a GPU.
from scope.core.pipelines.registry import PipelineRegistry
from scope.core.pipelines.surface_normals.geometry import (
    CANONICAL_FOCAL_PX,
    canonical_camera_transform,
    depth_to_surface_normals,
)
from scope.core.pipelines.surface_normals.pipeline import SurfaceNormalsPipeline
from scope.core.pipelines.surface_normals.schema import SurfaceNormalsConfig


def test_canonical_transform_rescales_depth():
    depth = torch.full((4, 4), 2.0)

    out = canonical_camera_transform(depth, focal_length=500.0, canonical_focal=1000.0)

    assert torch.allclose(out, torch.full((4, 4), 4.0))


def test_frontal_plane_normal_points_at_viewer():
    depth = torch.full((8, 8), 5.0)

    normals = depth_to_surface_normals(depth, focal_length=1000.0)

    assert normals.shape == (8, 8, 3)
    # Every pixel (borders included) is a unit vector aligned with +Z.
    assert torch.allclose(normals.norm(dim=-1), torch.ones((8, 8)), atol=1e-5)
    assert torch.all(normals[..., 2] > 0.99)


def test_tilted_plane_normal_tilts_in_y():
    # A plane slanted around the X axis: depth grows with the row index and is
    # constant across columns (a (10, 10) map).
    rows = torch.arange(10, dtype=torch.float32)
    depth = (1000.0 + rows.unsqueeze(1)).expand(10, 10).contiguous()

    normals = depth_to_surface_normals(depth, focal_length=1000.0)

    center = normals[5, 5]
    assert torch.allclose(center.norm(), torch.tensor(1.0), atol=1e-5)
    # Slanted surface -> the normal tilts off the Z axis into Y.
    assert abs(center[1].item()) > 0.3
    assert abs(center[2].item()) < 0.95


def test_node_is_registered_via_wiring_edit():
    # Exercises the registry wiring edit: surface-normals auto-registers.
    assert PipelineRegistry.is_registered("surface-normals")
    pipeline_class = PipelineRegistry.get("surface-normals")
    assert pipeline_class is SurfaceNormalsPipeline
    assert pipeline_class.get_config_class() is SurfaceNormalsConfig
    assert "normals" not in SurfaceNormalsConfig.outputs
    assert SurfaceNormalsConfig.outputs == ["video", "depth"]


def test_pipeline_produces_normals_and_depth_from_depth_input():
    pipeline = SurfaceNormalsPipeline(focal_length=1000.0, canonical_focal=1000.0)

    # Depth as THWC (T=2, H=8, W=8, C=3) like video-depth-anything output.
    depth = torch.full((2, 8, 8, 3), 0.5)
    out = pipeline(video=depth)

    assert set(out) == {"video", "depth"}
    normals = out["video"]
    assert normals.shape == (2, 8, 8, 3)
    # normal-map RGB encoding lives in [0, 1].
    assert normals.min().item() >= 0.0
    assert normals.max().item() <= 1.0
    # Canonical depth preserved (focal == canonical_focal) and broadcast to 3ch.
    assert out["depth"].shape == (2, 8, 8, 3)
    assert torch.allclose(out["depth"], torch.full((2, 8, 8, 3), 0.5))


def test_pipeline_canonical_focal_rescales_depth_output():
    pipeline = SurfaceNormalsPipeline(
        focal_length=500.0, canonical_focal=CANONICAL_FOCAL_PX
    )

    depth = torch.full((1, 4, 4, 3), 1.0)
    out = pipeline(video=depth)

    # canonical_focal / focal_length = 1000 / 500 = 2.0
    assert torch.allclose(out["depth"], torch.full((1, 4, 4, 3), 2.0))
