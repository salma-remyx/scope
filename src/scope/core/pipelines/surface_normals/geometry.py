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
) -> torch.Tensor:
    """Rescale ``depth`` into Metric3Dv2's canonical camera space.

    Cameras with different focal lengths observe the same scene at different
    depth scales. Mapping every depth to a common canonical focal length
    removes that camera-specific scale so a downstream interpretation is
    consistent across sources. The depth is scaled by
    ``canonical_focal / focal_length``.

    Args:
        depth: Depth map (any shape); values interpreted as metric or
            relative depth.
        focal_length: Source focal length in pixels.
        canonical_focal: Canonical focal length in pixels.

    Returns:
        Depth rescaled into canonical camera space (same shape as input).
    """
    scale = float(canonical_focal) / float(focal_length)
    return depth * scale


def _replicate_pad2d(plane: torch.Tensor) -> torch.Tensor:
    """Pad an ``(H, W)`` map by one on every side, replicating the edges."""
    padded = torch.cat([plane[:, :1], plane, plane[:, -1:]], dim=1)
    return torch.cat([padded[:1, :], padded, padded[-1:, :]], dim=0)


def depth_to_surface_normals(
    depth: torch.Tensor,
    focal_length: float = CANONICAL_FOCAL_PX,
) -> torch.Tensor:
    """Estimate per-pixel surface normals from a depth map.

    Back-projects each pixel into 3D with a pinhole camera of ``focal_length``
    (after normalizing to the canonical camera space), builds two in-plane
    tangent vectors from neighbouring 3D points, and returns their cross
    product as the outward surface normal. This is the geometric
    relationship Metric3Dv2 learns to predict from RGB; here it is computed
    deterministically from an existing depth map.

    A fronto-parallel plane yields normals aligned with ``+Z`` -- the
    standard normal-map convention ("blue" faces the viewer).

    Args:
        depth: Depth map of shape ``(H, W)`` with strictly positive values.
        focal_length: Focal length in pixels used for back-projection.

    Returns:
        Unit surface normals of shape ``(H, W, 3)``.
    """
    if depth.ndim != 2:
        raise ValueError(
            "depth_to_surface_normals expects a (H, W) map, "
            f"got shape {tuple(depth.shape)}"
        )

    device, dtype = depth.device, depth.dtype
    h, w = depth.shape
    focal = float(focal_length)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

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

    normals = torch.cross(tangent_x, tangent_y, dim=-1)
    return normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)


def normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """Map surface normals in ``[-1, 1]`` to an RGB preview in ``[0, 1]``.

    Uses the conventional normal-map encoding ``(n + 1) / 2`` so a
    fronto-parallel surface renders blue. Accepts any shape whose last
    dimension is 3.
    """
    return (normals + 1.0) / 2.0
