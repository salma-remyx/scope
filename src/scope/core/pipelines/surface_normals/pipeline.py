"""Surface-normals-from-depth pipeline node.

Applies Metric3Dv2's canonical-camera-space geometry (see :mod:`geometry`)
to a monocular depth estimate. Feed it the output of ``video-depth-anything``
(a depth map encoded as THWC frames) and it returns a surface-normal preview
plus the un-aligned target-camera metric depth map.

Adapted from Metric3Dv2 (https://arxiv.org/abs/2404.15506); see
:mod:`geometry` for the scope of the adaptation. The node is model-free, so
it registers and runs on CPU.
"""

from __future__ import annotations

import logging

import torch

from ..interface import Pipeline, Requirements
from .geometry import (
    CANONICAL_FOCAL_PX,
    canonical_camera_transform,
    depth_to_surface_normals,
    normals_to_rgb,
)
from .schema import SurfaceNormalsConfig

logger = logging.getLogger(__name__)


class SurfaceNormalsPipeline(Pipeline):
    """Surface normal estimation from a depth map.

    Consumes a depth prediction on its ``video`` input (THWC frames, e.g. the
    output of ``video-depth-anything``). Following Metric3Dv2's "align and
    decode" data flow (Section 3.3), the prediction is treated as depth in
    canonical camera space and is first un-aligned back to the target
    camera's metric space -- scaled by ``focal_length / canonical_focal`` --
    before any geometry runs. Emits a surface-normal preview on ``video``
    plus the un-aligned target-camera depth map on ``depth``.
    """

    @classmethod
    def get_config_class(cls) -> type[SurfaceNormalsConfig]:
        return SurfaceNormalsConfig

    def __init__(
        self,
        focal_length: float = CANONICAL_FOCAL_PX,
        canonical_focal: float = CANONICAL_FOCAL_PX,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        # Absorb unused schema defaults (height, width, base_seed, ...)
        # forwarded by the pipeline manager's config-driven load path.
        del kwargs
        self.focal_length = float(focal_length)
        self.canonical_focal = float(canonical_focal)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> dict:
        """Process a canonical-space depth prediction into surface normals.

        Args:
            video: Depth prediction in canonical camera space as THWC frames
                (e.g. ``video-depth-anything`` output), a channel-less
                ``(T, H, W)`` depth tensor, a single ``(H, W)`` map, or a
                list of per-frame maps.

        Returns:
            ``{"video": normals_thwc, "depth": metric_depth_thwc}`` where
            ``video`` is a normal-map preview in ``[0, 1]`` and ``depth`` is
            the un-aligned target-camera depth (the canonical-space input
            scaled by ``focal_length / canonical_focal``).
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input depth cannot be None for SurfaceNormalsPipeline")

        depth_thw = _coerce_depth_to_thw(video).to(device=self.device, dtype=self.dtype)

        normals_frames = []
        metric_depth_frames = []
        for frame in depth_thw:
            # Metric3Dv2 "align and decode" (Section 3.3): the prediction is
            # in canonical camera space, so un-align it to the target camera
            # *before* any downstream geometry (normal extraction) runs.
            metric = canonical_camera_transform(
                frame, self.focal_length, self.canonical_focal, inverse=True
            )
            normals = depth_to_surface_normals(metric, focal_length=self.focal_length)
            normals_frames.append(normals)
            metric_depth_frames.append(metric)

        normals = torch.stack(normals_frames, dim=0)  # (T, H, W, 3)
        metric_depth = torch.stack(metric_depth_frames, dim=0)  # (T, H, W)
        metric_depth_thwc = metric_depth.unsqueeze(-1).repeat(1, 1, 1, 3)
        return {"video": normals_to_rgb(normals), "depth": metric_depth_thwc}


def _coerce_depth_to_thw(video: object) -> torch.Tensor:
    """Coerce pipeline depth input into a ``(T, H, W)`` float tensor.

    Accepts a THWC tensor (e.g. ``video-depth-anything``'s output, with depth
    replicated across the channel axis), a channel-less ``(T, H, W)`` depth
    tensor, a single ``(H, W)`` map, or a list of per-frame maps.
    """
    if isinstance(video, (list, tuple)):
        frames = []
        for frame in video:
            tensor = (
                frame if isinstance(frame, torch.Tensor) else torch.as_tensor(frame)
            )
            frames.append(tensor.float().squeeze())
        video = torch.stack(frames, dim=0)
    elif isinstance(video, torch.Tensor):
        video = video.float()
    else:
        video = torch.as_tensor(video, dtype=torch.float32)

    if video.ndim == 2:
        video = video.unsqueeze(0)
    while video.ndim > 3:
        video = video.mean(dim=-1)
    return video
