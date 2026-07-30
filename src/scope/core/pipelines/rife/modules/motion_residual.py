"""Disentangled-motion residual estimation for RIFE interpolation.

Adapted from the core insight of *Disentangled Motion Modeling for Video Frame
Interpolation* (MoMo, Hu et al., arXiv:2406.17256): an interpolated frame can be
*disentangled* into

  1. a **motion** component produced by optical-flow warping of the two source
     frames, and
  2. an **appearance residual** -- the part of the true intermediate frame that
     flow-based warping cannot explain (disocclusions, large motions, fine
     texture), which MoMo models with a lightweight diffusion.

RIFE HDv3 (v4.x) outputs *only* the motion component -- its merged frame is the
pure flow-warp + blend reconstruction and the refinement network was removed.
That makes it the clean baseline MoMo improves upon, and makes the residual the
quantity of interest.

This module is **Mode 3 (inspired experiment)**: we do not port MoMo's diffusion
residual generator (it needs a dedicated checkpoint and a distillation trainer
that this real-time pipeline cannot host). Instead we measure, parameter-free,
*where that residual would be large*. The proxy is the disagreement between the
two flow-warped views of the source frames: where ``warp(img0, flow0t)`` and
``warp(img1, flow1t)`` disagree, the motion-warp is ambiguous and the appearance
residual MoMo targets is expected to be large. Aggregated per interpolated
frame, that disagreement becomes an interpolation-confidence / motion-complexity
signal usable for adaptive frame-rate decisions, quality gating, or telemetry.

All operations are parameter-free (no learned weights): they reuse the optical
flow and blend mask RIFE already computes, adding only a backward warp and a
subtraction per interpolated frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def backward_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``frame`` by a 2-channel optical flow, matching RIFE's warp.

    Args:
        frame: ``[B, C, H, W]`` source frames.
        flow: ``[B, 2, H, W]`` optical flow; channel 0 is the horizontal (width)
            displacement, channel 1 the vertical (height) displacement, in
            pixel units.

    Returns:
        ``[B, C, H, W]`` warped frames. Zero flow returns the input unchanged,
        identical to RIFE's ``warplayer.warp`` (same normalization,
        ``align_corners=True``, border padding) but device-safe for CPU.
    """
    _, _, h, w = frame.shape
    device = frame.device
    dtype = frame.dtype
    horizontal = (
        torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        .view(1, 1, 1, w)
        .expand(frame.shape[0], 1, h, w)
    )
    vertical = (
        torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        .view(1, 1, h, 1)
        .expand(frame.shape[0], 1, h, w)
    )
    base_grid = torch.cat([horizontal, vertical], dim=1)

    flow_norm = torch.cat(
        [
            flow[:, 0:1] / ((w - 1.0) / 2.0),
            flow[:, 1:2] / ((h - 1.0) / 2.0),
        ],
        dim=1,
    )
    grid = (base_grid + flow_norm).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=frame,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def disentangle_motion(
    img0: torch.Tensor,
    img1: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct the motion component of the intermediate frame.

    This is exactly RIFE HDv3's output: ``mask * warp(img0, flow0t) +
    (1 - mask) * warp(img1, flow1t)``. Exposing the two warped views separately is
    what lets us estimate the appearance residual.

    Args:
        img0, img1: ``[B, C, H, W]`` source frames (any shared value range).
        flow: ``[B, 4, H, W]`` RIFE bidirectional flow; ``[:, :2]`` is the
            img0->mid flow, ``[:, 2:4]`` the img1->mid flow.
        mask: ``[B, 1, H, W]`` blend mask in ``[0, 1]``.

    Returns:
        ``(warped0, warped1, warp_blend)`` each ``[B, C, H, W]``.
    """
    warped0 = backward_warp(img0, flow[:, :2])
    warped1 = backward_warp(img1, flow[:, 2:4])
    warp_blend = warped0 * mask + warped1 * (1.0 - mask)
    return warped0, warped1, warp_blend


def residual_ambiguity(warped0: torch.Tensor, warped1: torch.Tensor) -> torch.Tensor:
    """Per-pixel disagreement between the two warped views (residual proxy).

    Where the two flow-warped views of the source frames disagree, the
    motion-warp reconstruction is ambiguous -- the appearance residual MoMo
    models is expected to be large there. For frames in ``[0, 1]`` the result is
    in ``[0, 1]``.

    Args:
        warped0, warped1: ``[B, C, H, W]`` warped views (shared value range).

    Returns:
        ``[B, 1, H, W]`` per-pixel mean disagreement across channels.
    """
    return warped0.sub(warped1).abs().mean(dim=1, keepdim=True)


def estimate_motion_residual_confidence(
    img0: torch.Tensor,
    img1: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Estimate per-frame interpolation confidence via disentangled motion.

    Decomposes each interpolated frame into its motion component (flow-warp +
    blend) and measures the warp-view disagreement as a parameter-free proxy for
    MoMo's appearance residual. The disagreement is summarized as a per-frame
    ``residual_energy`` in ``[0, 1]`` (0 = the two warped views agree perfectly,
    i.e. flow fully explains the frame; 1 = they maximally disagree) and a
    ``confidence`` of ``1 - residual_energy`` (high where flow-based
    interpolation is trustworthy, low where a generative residual would help).

    Args:
        img0, img1: ``[B, C, H, W]`` source frames in ``[0, 1]``.
        flow: ``[B, 4, H, W]`` RIFE bidirectional flow.
        mask: ``[B, 1, H, W]`` blend mask in ``[0, 1]``.

    Returns:
        One ``{"residual_energy": float, "confidence": float}`` dict per batch
        element.
    """
    img0 = img0.float()
    img1 = img1.float()
    warped0, warped1, _ = disentangle_motion(img0, img1, flow.float(), mask.float())
    ambiguity = residual_ambiguity(warped0, warped1)
    residual_energy = ambiguity.flatten(start_dim=1).mean(dim=1).clamp(0.0, 1.0)
    confidence = (1.0 - residual_energy).clamp(0.0, 1.0)
    return [
        {
            "residual_energy": float(residual_energy[b].item()),
            "confidence": float(confidence[b].item()),
        }
        for b in range(img0.shape[0])
    ]
