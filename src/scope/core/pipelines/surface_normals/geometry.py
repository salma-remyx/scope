"""Surface-normal and canonical-camera-space geometry for monocular depth.

Adapted from Metric3Dv2 (Yin et al., CVPR 2024) -- a geometric foundation
model that recovers metric depth and surface normals from a single image by
normalizing diverse camera intrinsics into a canonical camera space.
Paper: https://arxiv.org/abs/2404.15506

Scope of this adaptation (Mode 2 -- adapted port):

* KEPT AT FULL FIDELITY -- the canonical camera space transformation
  (Metric3Dv2's core cross-camera generalization mechanism) and the
  depth -> surface-normal back-projection (the geometric relationship the
  paper learns to predict directly from RGB).
* SUBSTITUTED -- Metric3Dv2's trained metric-depth / normal heads are
  replaced by deterministic geometry applied to an existing monocular
  depth estimate (e.g. this repo's ``video-depth-anything`` node). The
  canonical-space depth is camera-invariant (focal-length ambiguity
  removed) but is not absolutely metric without the pretrained checkpoint;
  surface normals, being a local geometric property, are recovered exactly
  from the depth plus intrinsics.

These functions are pure tensor math (no learned weights, no GPU required)
so they unit-test deterministically.
"""

from __future__ import annotations

import torch

# Canonical focal length (pixels) Metric3Dv2 normalizes intrinsics toward.
# The absolute value only sets the overall depth scale; cross-camera
# consistency comes from mapping every input to this common reference.
CANONICAL_FOCAL_PX: float = 1000.0


def canonical_camera_transform(
    depth: torch.Tensor,
    focal_length: float,
    canonical_focal: float = CANONICAL_FOCAL_PX,
    inverse: bool = False,
) -> torch.Tensor:
    """Rescale ``depth`` between Metric3Dv2's canonical and target camera spaces.

    Cameras with different focal lengths observe the same scene at different
    depth scales. Mapping every depth to a common canonical focal length
    removes that camera-specific scale so a downstream interpretation is
    consistent across sources.

    With ``inverse=False`` (default) this maps target-camera depth into the
    canonical space, scaling by ``canonical_focal / focal_length``. With
    ``inverse=True`` it performs the paper's "align and decode" un-alignment
    (Section 3.3): the model's canonical-space prediction is scaled back to
    the target camera's metric space by ``focal_length / canonical_focal``
    before any downstream geometry (point clouds, normals) is computed.

    Args:
        depth: Depth map (any shape); values interpreted as metric or
            relative depth.
        focal_length: Target camera focal length in pixels.
        canonical_focal: Canonical focal length in pixels.
        inverse: If True, un-align canonical depth back to the target camera.

    Returns:
        Depth rescaled into the requested camera space (same shape as input).
    """
    scale = float(canonical_focal) / float(focal_length)
    if inverse:
        scale = 1.0 / scale
    return depth * scale


def _replicate_pad2d(plane: torch.Tensor) -> torch.Tensor:
    """Pad an ``(H, W)`` map by one on every side, replicating the edges."""
    padded = torch.cat([plane[:, :1], plane, plane[:, -1:]], dim=1)
    return torch.cat([padded[:1, :], padded, padded[-1:, :]], dim=0)


def depth_to_surface_normals(
    depth: torch.Tensor,
    focal_length: float = CANONICAL_FOCAL_PX,
    principal_point: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Estimate per-pixel surface normals from a depth map.

    Back-projects each pixel into 3D with a pinhole camera of ``focal_length``
    (after un-aligning to the target camera space), builds two in-plane
    tangent vectors from neighbouring 3D points, and returns their cross
    product as the surface normal facing the camera. This is the geometric
    relationship Metric3Dv2 learns to predict from RGB; here it is computed
    deterministically from an existing depth map.

    When explicit intrinsics are omitted, the deterministic canonical rule
    from Metric3Dv2 Section 3.2 is used: focal length ``CANONICAL_FOCAL_PX``
    (1000 px) and the principal point centered at exactly half the maximum
    image dimension, ``cx = cy = 0.5 * max(W, H)``.

    The camera frame is the standard right-handed CV frame (x right, y down,
    +Z into the scene). Normals are computed as ``cross(tangent_y,
    tangent_x)`` so a fronto-parallel plane yields normals aligned with
    ``-Z`` -- pointing back toward the camera, as Metric3Dv2's evaluation
    protocol expects for visible surfaces.

    Args:
        depth: Depth map of shape ``(H, W)`` with strictly positive values.
        focal_length: Focal length in pixels used for back-projection.
        principal_point: Optional ``(cx, cy)`` in pixels. Defaults to the
            canonical rule ``0.5 * max(W, H)`` for both axes.

    Returns:
        Unit surface normals of shape ``(H, W, 3)``, facing the camera.
    """
    if depth.ndim != 2:
        raise ValueError(
            "depth_to_surface_normals expects a (H, W) map, "
            f"got shape {tuple(depth.shape)}"
        )

    device, dtype = depth.device, depth.dtype
    h, w = depth.shape
    focal = float(focal_length)
    if principal_point is None:
        # Metric3Dv2 Section 3.2: deterministic canonical principal point.
        cx = cy = 0.5 * max(w, h)
    else:
        cx, cy = (float(p) for p in principal_point)

    vy, vx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    # Pinhole back-projection: pixel (u, v) -> 3D point.
    z = depth.to(dtype=dtype)
    x = (vx - cx) * z / focal
    y = (vy - cy) * z / focal

    # Central differences over the replicate-padded (H+2, W+2) coordinate
    # maps. We slice the inner (H, W) region for the "center" pixel and offset
    # slices for its +/-1 neighbours, so every output pixel (borders included)
    # has a well-defined, non-degenerate pair of tangents.
    xpad, ypad, zpad = _replicate_pad2d(x), _replicate_pad2d(y), _replicate_pad2d(z)
    rows = slice(1, h + 1)
    cols = slice(1, w + 1)
    tangent_x = torch.stack(
        [
            xpad[rows, 2 : w + 2] - xpad[rows, 0:w],
            ypad[rows, 2 : w + 2] - ypad[rows, 0:w],
            zpad[rows, 2 : w + 2] - zpad[rows, 0:w],
        ],
        dim=-1,
    )
    tangent_y = torch.stack(
        [
            xpad[2 : h + 2, cols] - xpad[0:h, cols],
            ypad[2 : h + 2, cols] - ypad[0:h, cols],
            zpad[2 : h + 2, cols] - zpad[0:h, cols],
        ],
        dim=-1,
    )

    # Normalize the tangents before the cross product. The raw tangent
    # magnitude scales with (depth / focal)^2, which is tiny for typical
    # monocular depth; normalizing first keeps the cross product O(1) and the
    # final unit normal well-conditioned regardless of the depth scale.
    tangent_x = tangent_x / (tangent_x.norm(dim=-1, keepdim=True) + 1e-12)
    tangent_y = tangent_y / (tangent_y.norm(dim=-1, keepdim=True) + 1e-12)

    # Invert the cross-product order relative to cross(tangent_x, tangent_y):
    # in the right-handed camera frame (+Z into the scene) that order yields
    # normals pointing away from the camera -- the "Z-axis backflip". This
    # order makes visible-surface normals face the camera (-Z).
    normals = torch.cross(tangent_y, tangent_x, dim=-1)
    return normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """Map surface normals in ``[-1, 1]`` to an RGB preview in ``[0, 1]``.

    Normals face the camera (-Z in the camera frame); the preview flips Z so
    the conventional normal-map encoding ``(n + 1) / 2`` renders a
    fronto-parallel surface blue ("blue faces the viewer"). Accepts any shape
    whose last dimension is 3.
    """
    flip_z = torch.tensor([1.0, 1.0, -1.0], device=normals.device, dtype=normals.dtype)
    return (normals * flip_z + 1.0) / 2.0
