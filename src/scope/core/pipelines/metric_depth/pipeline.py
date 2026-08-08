"""Metric depth pipeline — camera-aware monocular metric depth from RGB.

Adapted from UniDepth: Universal Monocular Metric Depth Estimation
(Piccinelli et al., CVPR 2024; arXiv:2403.18913). UniDepth predicts
camera-agnostic, zero-shot metric depth from a single image by jointly
recovering metric scale and camera intrinsics, decoupling absolute depth
from the training domain's camera/scale.

This pipeline keeps that core framing — turning an affine-invariant
relative-depth prediction into metric depth under a camera model — while
substituting two of UniDepth's learned components with target-native
equivalents (Mode 2 adapted port):

  * UniDepth's learned metric + camera head is replaced by the analytical
    :func:`rescale_to_metric` recovery, anchored on a robust median prior
    with an optional pinhole focal correction.
  * UniDepth's trained RGB backbone is replaced by the repo's existing
    Video-Depth-Anything relative-depth model (``metric=False``) — the
    relative-only baseline this extends.

The result drops into the same per-frame frames-in / depth-out contract as
``video_depth_anything`` but recovers camera-aware metric depth (metres)
upstream of the stream-visualized [0, 1] output.
"""

import logging
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, Requirements
from ..video_depth_anything import VideoDepthAnythingPipeline
from .schema import MetricDepthConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def rescale_to_metric(
    relative_depth: torch.Tensor,
    *,
    median_depth_m: float = 5.0,
    depth_min_m: float = 0.5,
    depth_max_m: float = 100.0,
    focal_length_px: float | None = None,
    reference_focal_px: float | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Recover camera-aware metric depth (metres) from a relative-depth map.

    Parameter-free proxy for UniDepth's learned metric + camera recovery.
    UniDepth decouples absolute metric scale from an affine-invariant depth
    prediction; here that learned recovery is replaced by an analytical one:

      1. Map the [0, 1] relative depth (1 = farthest) into a provisional
         metric range ``[depth_min_m, depth_max_m]``.
      2. Re-anchor on the median — a robust statistic invariant to a few
         outliers — so the scene's central depth equals ``median_depth_m``.
         This supplies the absolute scale a learned head would predict.
      3. Apply the pinhole camera correction ``Z ∝ f`` when
         ``focal_length_px`` is supplied (relative to ``reference_focal_px``),
         so the same relative prediction maps to camera-consistent metric
         depth across lenses.

    Args:
        relative_depth: float tensor of shape ``(..., H, W)`` in ``[0, 1]``;
            larger values are farther from the camera.
        median_depth_m: Robust absolute-scale anchor in metres.
        depth_min_m: Provisional near bound in metres.
        depth_max_m: Provisional far bound in metres.
        focal_length_px: Optional camera focal length in pixels. When given,
            the output is rescaled relative to ``reference_focal_px`` so depth
            is consistent with the supplied intrinsics (camera-aware).
        reference_focal_px: Focal length (px) at which ``median_depth_m`` is
            the anchor; defaults to ``focal_length_px`` (no correction).
        eps: Numerical floor keeping depth strictly positive.

    Returns:
        Float tensor of the same shape as ``relative_depth`` holding metric
        depth in metres.
    """
    if depth_max_m <= depth_min_m:
        raise ValueError("depth_max_m must be greater than depth_min_m")

    depth = relative_depth.float().clamp(0.0, 1.0)

    # (1) provisional metric map: near -> shallow, far -> deep.
    z = depth_min_m + (depth_max_m - depth_min_m) * depth

    # (2) robust median anchor — recovers the absolute metric scale.
    if z.numel() > 0:
        z = z + (median_depth_m - float(torch.median(z)))

    # (3) camera-aware pinhole correction Z ∝ f.
    if focal_length_px is not None and focal_length_px > 0:
        ref = reference_focal_px if reference_focal_px else focal_length_px
        if ref > 0:
            z = z * (focal_length_px / ref)

    return z.clamp(min=eps)


class MetricDepthPipeline(Pipeline):
    """Monocular metric depth estimation pipeline (UniDepth, adapted)."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return MetricDepthConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Metric Depth pipeline.

        Args:
            config: Pipeline configuration.
            device: Target device (defaults to CUDA if available).
            dtype: Data type for the backbone model weights.
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Reuse the repo's relative-depth backbone as the RGB encoder,
        # standing in for UniDepth's trained metric backbone.
        self.backbone = VideoDepthAnythingPipeline(
            config, device=self.device, dtype=dtype
        )
        self.median_depth_m = float(getattr(config, "median_depth_m", 5.0))
        self.depth_min_m = float(getattr(config, "depth_min_m", 0.5))
        self.depth_max_m = float(getattr(config, "depth_max_m", 100.0))
        self.focal_length_px = getattr(config, "focal_length_px", None)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=4)

    def __call__(self, **kwargs) -> dict:
        """Process video frames and return metric depth maps.

        Args:
            video: Input video frames as a list of tensors (THWC, [0, 255]).

        Returns:
            Metric depth maps as a tensor in THWC format, normalized to
            [0, 1] for the streaming/recording contract (1 = farthest). The
            underlying metric depth in metres is produced by
            :func:`rescale_to_metric`.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for MetricDepthPipeline")

        # Relative depth via the backbone: THWC float in [0, 1], 1 = far.
        relative = self.backbone(video=video)["video"].to(device=self.device)

        # Camera-aware metric recovery (metres) on a single depth channel.
        metric = rescale_to_metric(
            relative[..., 0],
            median_depth_m=self.median_depth_m,
            depth_min_m=self.depth_min_m,
            depth_max_m=self.depth_max_m,
            focal_length_px=self.focal_length_px,
        )

        # Normalize to [0, 1] for the downstream stream/recording contract
        # (matching video_depth_anything) and expand back to 3 channels.
        lo, hi = metric.amin(), metric.amax()
        normalized = (metric - lo) / (hi - lo) if hi > lo else torch.zeros_like(metric)
        return {"video": normalized.unsqueeze(-1).expand(-1, -1, -1, 3)}
