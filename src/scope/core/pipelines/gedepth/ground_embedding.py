"""Ground embedding — camera-parameter-decoupled ground-plane depth prior.

Adapted from *GEDepth: Ground Embedding for Monocular Depth Estimation*
(https://arxiv.org/abs/2309.09975). The paper's core contribution is a
**ground embedding** (GE) module that decouples camera parameters
(intrinsics + extrinsics) from learned depth features by injecting a
per-pixel geometric prior — the distance from the camera to the ground
plane along each pixel's viewing ray. Conditioning on this prior is what
lets the learned decoder produce metric, camera-generalizing depth.

This module implements that geometric construction at full fidelity and
parameter-free: given camera height, pitch and vertical field of view it
returns the per-pixel ground-plane distance map. The learned HRNet
encoder + decoder that consume the embedding in the original paper (and
the pretrained checkpoint they require) are intentionally out of scope —
the camera-decoupled geometry is the contribution delivered here.
"""

from __future__ import annotations

import math

import torch

# Rays at or above the horizon never strike the ground; anything below
# this depression angle is treated as sky (infinite distance).
_HORIZON_EPS = 1e-6


def ground_distance_map(
    height: int,
    width: int,
    camera_height: float,
    pitch_deg: float,
    vertical_fov_deg: float,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Per-pixel ground-plane distance prior (the GEDepth ground embedding).

    Models a pinhole camera mounted ``camera_height`` meters above a flat
    ground plane and pitched ``pitch_deg`` degrees below the horizon. For
    every pixel it returns the Euclidean ground distance to where that
    pixel's viewing ray strikes the ground. Pixels whose ray lies at or
    above the horizon (the sky region) are mapped to ``+inf``.

    The result depends only on the camera parameters and image geometry,
    not on pixel content — the prior is identical across frames for a
    static camera, which is exactly the decoupling property GEDepth
    exploits.

    Args:
        height: Image height ``H`` in pixels.
        width: Image width ``W`` in pixels.
        camera_height: Camera height above the ground plane, in meters.
        pitch_deg: Camera pitch below the horizon, in degrees (positive
            looks down toward the ground).
        vertical_fov_deg: Vertical field of view, in degrees.
        device: Device to place the output tensor on.

    Returns:
        ``[H, W]`` float32 tensor of ground distances in meters. Sky
        pixels are ``+inf``; lower image rows (nearby ground) get smaller
        distances.
    """
    # Focal length in pixels derived from the vertical field of view, and
    # principal point assumed at the image center (cx, cy) = (W/2, H/2).
    focal_length = 0.5 * height / math.tan(math.radians(vertical_fov_deg) / 2.0)
    cy = height / 2.0
    theta = math.radians(pitch_deg)

    # Viewing ray depression angle below the horizon for each image row.
    rows = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(-1)
    phi = theta + torch.atan((rows - cy) / focal_length)  # [H, 1]

    # Ground distance = camera_height / tan(depression). Rows at/above the
    # horizon (phi <= 0) never hit the ground -> +inf.
    safe_phi = phi.clamp(min=_HORIZON_EPS)
    distance = camera_height / torch.tan(safe_phi)
    distance = torch.where(phi > _HORIZON_EPS, distance, torch.tensor(float("inf")))

    return distance.expand(height, width).contiguous()


def ground_embedding(
    distance: torch.Tensor,
    *,
    max_distance: float = 80.0,
    invert: bool = False,
) -> torch.Tensor:
    """Normalize a ground-distance map into a [0, 1] depth-like embedding.

    Distances are clamped to ``max_distance`` (sky pixels at ``+inf`` map
    to the far end) and linearly scaled so the output is a stable depth
    prior. By default higher values mean *further* from the camera,
    matching the :class:`~scope.core.pipelines.video_depth_anything`
    convention; set ``invert`` to flip.

    GEDepth feeds a log-compressed form of this distance into its decoder
    to handle the large dynamic range; we expose the linearly normalized
    form as the depth-map output, which is the more useful surface for a
    visualization / preprocessor node.

    Args:
        distance: ``[..., H, W]`` ground-distance tensor in meters.
        max_distance: Distance clamp (meters) mapping to the far end.
        invert: If True, near pixels take higher values.

    Returns:
        ``[..., H, W]`` float tensor in [0, 1].
    """
    clamped = torch.clamp(distance, max=float(max_distance))
    normalized = clamped / float(max_distance)
    if invert:
        normalized = 1.0 - normalized
    return normalized.clamp(0.0, 1.0)
