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
    normals_to_rgb,
)
from scope.core.pipelines.surface_normals.pipeline import SurfaceNormalsPipeline
from scope.core.pipelines.surface_normals.schema import SurfaceNormalsConfig


def test_canonical_transform_rescales_depth():
    depth = torch.full((4, 4), 2.0)

    out = canonical_camera_transform(depth, focal_length=500.0, canonical_focal=1000.0)

    assert torch.allclose(out, torch.full((4, 4), 4.0))

    # The inverse performs Metric3Dv2's "align and decode" un-alignment:
    # d_target = d_canonical * (f_target / f_canonical).
    back = canonical_camera_transform(
        depth, focal_length=500.0, canonical_focal=1000.0, inverse=True
    )
    assert torch.allclose(back, torch.full((4, 4), 1.0))


def test_frontal_plane_normal_points_at_viewer():
    depth = torch.full((8, 8), 5.0)

    normals = depth_to_surface_normals(depth, focal_length=1000.0)

    assert normals.shape == (8, 8, 3)
    # Every pixel (borders included) is a unit vector aligned with -Z --
    # facing back toward the camera in the right-handed camera frame
    # (+Z into the scene), per Metric3Dv2's evaluation protocol.
    assert torch.allclose(normals.norm(dim=-1), torch.ones((8, 8)), atol=1e-5)
    assert torch.all(normals[..., 2] < -0.99)


def test_default_intrinsics_follow_canonical_rule():
    # Metric3Dv2 Section 3.2: without explicit intrinsics the geometry uses
    # f_can = 1000 px and principal point cx = cy = 0.5 * max(W, H) --
    # deterministically, not a per-axis center heuristic. On a non-square
    # image this differs from ((W-1)/2, (H-1)/2).
    h, w = 16, 64
    focal, radius = 100.0, 5.0
    cx = cy = 0.5 * max(w, h)
    u, v = torch.meshgrid(
        torch.arange(w, dtype=torch.float64),
        torch.arange(h, dtype=torch.float64),
        indexing="xy",
    )
    # Sphere centered on the camera -- normals are sensitive to the
    # principal point used for back-projection.
    depth = (
        radius * focal / torch.sqrt(focal**2 + (u - cx) ** 2 + (v - cy) ** 2)
    ).float()

    default = depth_to_surface_normals(depth)
    explicit_canonical = depth_to_surface_normals(
        depth, focal_length=CANONICAL_FOCAL_PX, principal_point=(cx, cy)
    )
    naive_heuristic = depth_to_surface_normals(
        depth,
        focal_length=CANONICAL_FOCAL_PX,
        principal_point=((w - 1) / 2.0, (h - 1) / 2.0),
    )

    # The defaults are exactly the canonical rule...
    assert torch.equal(default, explicit_canonical)
    # ...and not the per-axis center heuristic.
    assert not torch.allclose(default, naive_heuristic)


def test_tilted_plane_normal_matches_ground_truth_parity():
    # Analytic plane Z = alpha * Y + Z0 in the camera frame (x right,
    # y down, +Z into the scene). Its visible-surface normal faces the
    # camera: n = (0, alpha, -1) / sqrt(alpha^2 + 1).
    alpha, z0, focal = 0.5, 5.0, 1000.0
    h = w = 32
    cy = 0.5 * max(w, h)  # the canonical principal point
    v = torch.arange(h, dtype=torch.float32)
    depth = (z0 / (1.0 - alpha * (v - cy) / focal)).unsqueeze(1).expand(h, w)

    normals = depth_to_surface_normals(depth, focal_length=focal)

    expected = torch.tensor([0.0, alpha, -1.0])
    expected = expected / expected.norm()
    dot = (normals[2:-2, 2:-2] * expected).sum(dim=-1)
    # Dot-product parity with ground truth, no global sign flip needed.
    assert torch.all(dot > 0.9)


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
    # Un-aligned depth preserved (focal == canonical_focal) and broadcast to 3ch.
    assert out["depth"].shape == (2, 8, 8, 3)
    assert torch.allclose(out["depth"], torch.full((2, 8, 8, 3), 0.5))


def test_pipeline_unaligns_canonical_depth_before_normals():
    # Out-of-domain target camera: the target intrinsics (f = 500) differ
    # from the canonical model camera (f_can = 1000). The pipeline must
    # un-align the canonical-space prediction by f_target / f_can before
    # normal extraction (Metric3Dv2 Section 3.3).
    pipeline = SurfaceNormalsPipeline(
        focal_length=500.0, canonical_focal=CANONICAL_FOCAL_PX
    )

    depth = torch.full((1, 4, 4, 3), 1.0)
    out = pipeline(video=depth)

    # d_target = d_can * (500 / 1000) = 0.5
    assert torch.allclose(out["depth"], torch.full((1, 4, 4, 3), 0.5))

    # Normals match an explicit reference: un-align first, then extract.
    rows = torch.arange(8, dtype=torch.float32)
    canonical = (1000.0 + rows.unsqueeze(1)).expand(8, 8).contiguous()
    out = pipeline(video=canonical)

    reference_metric = canonical_camera_transform(
        canonical, 500.0, CANONICAL_FOCAL_PX, inverse=True
    )
    reference_normals = normals_to_rgb(
        depth_to_surface_normals(reference_metric, focal_length=500.0)
    )
    assert torch.allclose(out["video"][0], reference_normals, atol=1e-5)
