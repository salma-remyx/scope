"""GEDepth ground-embedding pipeline.

Produces a camera-parameter-conditioned ground-plane depth prior as a
video stream — the geometric signal that GEDepth injects into its depth
decoder, exposed here as a standalone depth-map node that slots into the
graph next to ``video_depth_anything``.

Adapted from *GEDepth: Ground Embedding for Monocular Depth Estimation*
(https://arxiv.org/abs/2309.09975). The original paper's learned HRNet
encoder + decoder and its pretrained checkpoint are intentionally out of
scope; this pipeline delivers the paper's core contribution — the
camera-decoupled ground embedding — at full geometric fidelity with no
learned weights.
"""

import logging
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes
from .ground_embedding import ground_distance_map, ground_embedding
from .schema import GEDepthConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class GEDepthPipeline(Pipeline):
    """Camera-parameter-decoupled ground-plane depth prior (GEDepth GE module)."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return GEDepthConfig

    def __init__(self, config, device: torch.device | None = None):
        """Initialize the GEDepth ground-embedding pipeline.

        Args:
            config: Pipeline configuration (camera parameters).
            device: Target device (defaults to CUDA if available).
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.camera_height = float(getattr(config, "camera_height", 1.5))
        self.pitch_deg = float(getattr(config, "pitch_deg", 5.0))
        self.vertical_fov_deg = float(getattr(config, "vertical_fov_deg", 60.0))
        self.max_distance = float(getattr(config, "max_distance", 80.0))
        self.invert = bool(getattr(config, "invert", False))

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Return the ground-embedding depth prior for the input frames.

        Args:
            video: Input frames as a list of THWC tensors. Only the frame
                geometry (H, W) and count (T) are consumed — the ground
                embedding is a purely geometric, content-independent prior
                (identical across frames for a static camera).

        Returns:
            ``{"video": ...}`` with a THWC float tensor in [0, 1] (3
            channels); higher values mean further from the camera,
            matching the video_depth_anything depth convention.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for GEDepthPipeline")

        video = normalize_frame_sizes(video)
        first = video[0]
        _, height, width, _ = first.shape

        distance = ground_distance_map(
            height,
            width,
            camera_height=self.camera_height,
            pitch_deg=self.pitch_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            device=first.device,
        )
        prior = ground_embedding(
            distance, max_distance=self.max_distance, invert=self.invert
        )

        # Broadcast the per-frame prior across the batch as a 3-channel
        # depth map: [H, W] -> [H, W, 3] -> [T, H, W, 3].
        frame = prior.unsqueeze(-1).expand(height, width, 3)
        output = frame.unsqueeze(0).expand(len(video), height, width, 3).contiguous()
        return {"video": output}
