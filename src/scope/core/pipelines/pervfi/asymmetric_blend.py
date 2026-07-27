"""Asymmetric blending for video frame interpolation.

Adapted (Mode-2) port of the *core mechanism* of PerVFI -- "Perception-Oriented
Video Frame Interpolation via Asymmetric Blending", Zhang et al., CVPR 2024
(https://arxiv.org/abs/2404.06692v1, Apache-2.0).

What is kept at full fidelity
-----------------------------
PerVFI's named contribution -- the **asymmetric, occlusion-aware blend** of the
two endpoint frames toward the interpolation time ``t``. Conventional
(interpolation-time-symmetric) blending averages two imperfectly motion-aligned
warps, and that average is exactly what shows up as blur and ghosting in
misaligned / occluded regions. PerVFI instead warps each endpoint toward ``t``
with the forward flow scaled by ``t`` and the backward flow scaled by ``(1 - t)``
and combines them with a per-pixel *reliability* weight, so the unreliable
(occluded) side is suppressed rather than averaged in. That asymmetric,
reliability-weighted combination is implemented here verbatim
(``asymmetric_blend``).

What is substituted (Mode-2 adaptations)
----------------------------------------
PerVFI's full method ships a learned flow network (RAFT/GMA/GMFlow), a learned
``Softmetric`` reliability network, a ``cupy``-based forward soft-splat warp,
and a conditional normalizing-flow "perception" generator trained with
perceptual + adversarial losses. None of those can be hosted or verified here
(no CUDA ``cupy``, no trained weights, and the perception generator is a
training-time framework). They are replaced with parameter-free, torch-native
equivalents -- exactly the "learned estimator -> parameter-free proxy" move
the adaptation mode allows:

  * learned flow estimator       -> parameter-free pyramidal Lucas-Kanade
                                    (``estimate_flow``);
  * learned ``Softmetric``       -> backward-flow round-trip consistency
                                    occlusion/reliability proxy
                                    (``reliability``);
  * ``cupy`` forward soft-splat  -> ``grid_sample`` backward warp
                                    (``backward_warp``);
  * normalizing-flow perception  -> CUT. The deterministic asymmetric blend is
    generator + perceptual loss     the inference-time deliverable; the
                                    perception-oriented *training* framework is
                                    out of scope for this integration.

So this module is a weight-free, CPU-runnable realization of PerVFI's core
insight rather than a reproduction of its reported numbers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "backward_warp",
    "estimate_flow",
    "reliability",
    "asymmetric_blend",
    "interpolate_pair",
]

# Cached identity sampling grids keyed by (device, H, W). Flow values are added
# in normalized coordinates at sampling time, so the base grid is reusable.
_GRID_CACHE: dict[tuple[str, int, int], torch.Tensor] = {}


def backward_warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample ``img`` at ``location + flow`` -- the standard grid_sample warp.

    This is the torch-native substitute for PerVFI's ``cupy`` forward
    soft-splatting. ``flow[:, 0]`` is the x-displacement (dx) and ``flow[:, 1]``
    the y-displacement (dy), both in pixels; the result satisfies
    ``out[y, x] = img[y + dy, x + dx]``.

    Args:
        img: NCHW tensor.
        flow: N,2,H,W tensor of (dx, dy) pixel displacements.

    Returns:
        NCHW warped tensor.
    """
    n, _, h, w = img.shape
    key = (str(flow.device), h, w)
    if key not in _GRID_CACHE:
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=flow.device),
            torch.linspace(-1.0, 1.0, w, device=flow.device),
            indexing="ij",
        )
        base = torch.stack((xs, ys), dim=0).unsqueeze(0)  # 1,2,H,W (x, y)
        _GRID_CACHE[key] = base
    base = _GRID_CACHE[key].expand(n, -1, -1, -1)
    fx = flow[:, 0:1] / ((w - 1) / 2.0)
    fy = flow[:, 1:2] / ((h - 1) / 2.0)
    grid = (base + torch.cat([fx, fy], dim=1)).permute(0, 2, 3, 1)  # N,H,W,2
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _sobel(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial gradients (Ix, Iy) of an NCHW image via a 3x3 Sobel filter."""
    kx = torch.tensor(
        [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        device=img.device,
        dtype=img.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    n, c, h, w = img.shape
    flat = img.reshape(n * c, 1, h, w)
    pad = F.pad(flat, (1, 1, 1, 1), mode="replicate")
    ix = F.conv2d(pad, kx).reshape(n, c, h, w) / 8.0
    iy = F.conv2d(pad, ky).reshape(n, c, h, w) / 8.0
    return ix, iy


def _box(x: torch.Tensor, window: int) -> torch.Tensor:
    """Windowed box average with replicate padding (odd ``window``)."""
    if window <= 1:
        return x
    p = window // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="replicate"), window, stride=1)


@torch.no_grad()
def estimate_flow(
    img0: torch.Tensor,
    img1: torch.Tensor,
    levels: int = 4,
    iters: int = 3,
    window: int = 7,
) -> torch.Tensor:
    """Parameter-free coarse-to-fine Lucas-Kanade flow from ``img0`` to ``img1``.

    Substitutes PerVFI's learned RAFT/GMA/GMFlow flow network. Returns a flow
    field ``f01`` (N,2,H,W, in (dx,dy) pixels) such that
    ``img0[y, x] ~ img1[y + dy, x + dx]``.

    Args:
        img0, img1: NCHW float tensors in roughly the same intensity range.
        levels: pyramid levels (coarse-to-fine).
        iters: LK refinement iterations per level.
        window: local integration window for the structure tensor.

    Returns:
        N,2,H,W flow (dx, dy) in pixels.
    """
    pyr0 = [img0]
    pyr1 = [img1]
    for _ in range(levels - 1):
        pyr0.append(F.avg_pool2d(pyr0[-1], 2))
        pyr1.append(F.avg_pool2d(pyr1[-1], 2))

    n = img0.shape[0]
    h_c, w_c = pyr0[-1].shape[-2:]
    flow = torch.zeros(n, 2, h_c, w_c, device=img0.device, dtype=img0.dtype)

    for lv in range(levels - 1, -1, -1):
        i0, i1 = pyr0[lv], pyr1[lv]
        if lv != levels - 1:
            flow = F.interpolate(flow, size=i0.shape[-2:], mode="bilinear") * 2.0
        for _ in range(iters):
            i1w = backward_warp(i1, flow)
            it = i1w - i0
            ix, iy = _sobel(i1w)
            ixt = _box((ix * it).sum(1, keepdim=True), window)
            iyt = _box((iy * it).sum(1, keepdim=True), window)
            sxx = _box((ix * ix).sum(1, keepdim=True), window)
            sxy = _box((ix * iy).sum(1, keepdim=True), window)
            syy = _box((iy * iy).sum(1, keepdim=True), window)
            # Trace-relative Tikhonov regularization of the structure tensor:
            # scale-invariant and unconditionally positive-definite, so the solve
            # is always stable and the update -> 0 in flat regions (b -> 0)
            # without a brittle hard mask on the determinant.
            reg = 1e-3 + 0.05 * (sxx + syy)
            sxx_r = sxx + reg
            syy_r = syy + reg
            det_r = sxx_r * syy_r - sxy * sxy
            inv = 1.0 / det_r
            dux = -(syy_r * ixt - sxy * iyt) * inv
            duy = -(sxx_r * iyt - sxy * ixt) * inv
            flow = flow + torch.cat([dux, duy], dim=1)
    return flow


def reliability(flow01: torch.Tensor, flow10: torch.Tensor) -> torch.Tensor:
    """Per-pixel reliability in [0, 1] from forward/backward flow consistency.

    Substitutes PerVFI's learned ``Softmetric`` network: pixels whose forward
    flow and (forward-warped) backward flow agree -- i.e. a round trip through
    both endpoints returns to ~the same place -- are treated as reliable;
    occluded / dis-occluded pixels, where the round trip is large, are
    down-weighted so they do not get averaged into the blend.

    Args:
        flow01: img0 -> img1 flow (N,2,H,W).
        flow10: img1 -> img0 flow (N,2,H,W), same spatial size.

    Returns:
        N,1,H,W reliability weights in [0, 1].
    """
    flow10_at0 = backward_warp(flow10, flow01)
    round_trip = flow01 + flow10_at0
    err = torch.sqrt((round_trip**2).sum(dim=1, keepdim=True) + 1e-12)
    return torch.exp(-err / 3.0)


def asymmetric_blend(
    img0: torch.Tensor,
    img1: torch.Tensor,
    flow01: torch.Tensor,
    flow10: torch.Tensor,
    t: float = 0.5,
) -> torch.Tensor:
    """PerVFI asymmetric blend producing the frame at interpolation time ``t``.

    Mirrors PerVFI's ``featurePyramid`` scaling: ``img0`` is warped toward ``t``
    using the forward flow scaled by ``t`` with reliability scaled by ``2t``,
    and ``img1`` using the backward flow scaled by ``(1 - t)`` with reliability
    scaled by ``2(1 - t)``. The two warps are combined reliability-weighted --
    *not* averaged -- so at ``t != 0.5`` the blend is asymmetric (favoring the
    nearer endpoint) and the occluded side is suppressed instead of smeared.

    Args:
        img0, img1: NCHW endpoint frames in [0, 1].
        flow01, flow10: img0 -> img1 and img1 -> img0 flows (N,2,H,W).
        t: interpolation time in [0, 1]; 0.5 is the frame-doubling midpoint.

    Returns:
        NCHW interpolated frame in [0, 1].
    """
    # Content of img0 at p moves to p + t*flow01[p] at time t, so the time-t
    # frame samples img0 at p - t*flow01[p] (and symmetrically for img1).
    warped0 = backward_warp(img0, -t * flow01)
    warped1 = backward_warp(img1, -(1.0 - t) * flow10)

    wa = reliability(flow01, flow10) * (2.0 * t) + 1e-6
    wb = reliability(flow10, flow01) * (2.0 * (1.0 - t)) + 1e-6
    return (warped0 * wa + warped1 * wb) / (wa + wb)


@torch.no_grad()
def interpolate_pair(
    img0: torch.Tensor,
    img1: torch.Tensor,
    t: float = 0.5,
    **flow_kwargs,
) -> torch.Tensor:
    """Estimate bidirectional flow and return the asymmetric blend at time ``t``.

    Thin orchestrator over :func:`estimate_flow` + :func:`asymmetric_blend`.
    ``img0`` / ``img1`` are NCHW frames in [0, 1]; ``flow_kwargs`` forward to
    :func:`estimate_flow` (levels, iters, window).
    """
    flow01 = estimate_flow(img0, img1, **flow_kwargs)
    flow10 = estimate_flow(img1, img0, **flow_kwargs)
    return asymmetric_blend(img0, img1, flow01, flow10, t=t)
