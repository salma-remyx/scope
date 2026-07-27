"""PerVFI frame interpolation pipeline.

Doubles the frame rate of input video using the asymmetric, occlusion-aware
blending mechanism of PerVFI (Perception-Oriented Video Frame Interpolation via
Asymmetric Blending, CVPR 2024, https://arxiv.org/abs/2404.06692v1).

This is a complementary, quality-oriented alternative to the real-time RIFE
interpolator: same video -> interpolated-video I/O contract and config shape as
RIFE, but it trades speed for an occlusion-aware blend that avoids the
blur/ghosting that symmetric blending produces in misaligned regions. The core
mechanism is implemented as a weight-free, CPU-runnable adapted port in
``asymmetric_blend.py``; see that module's docstring for what is kept from the
paper at full fidelity and which auxiliary components are substituted.
"""

import logging
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes, postprocess_chunk, preprocess_chunk
from .asymmetric_blend import interpolate_pair
from .schema import PerVFIConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class PerVFIPipeline(Pipeline):
    """PerVFI asymmetric-blending interpolation pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return PerVFIConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the PerVFI pipeline.

        The asymmetric blend is parameter-free, so -- unlike RIFE -- there is no
        model to load; the pipeline is ready to interpolate immediately.

        Args:
            config: Pipeline configuration.
            device: Target device (defaults to CUDA if available).
            dtype: Data type for processing (kept for RIFE parity; the blend
                itself runs in float32 for numerical stability).
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        logger.info("PerVFI asymmetric-blending interpolator ready (no weights)")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=12)

    def __call__(self, **kwargs) -> dict:
        input = kwargs.get("video")

        if input is None:
            raise ValueError("Input cannot be None for PerVFIPipeline")

        if isinstance(input, list):
            input = normalize_frame_sizes(input)
            # preprocess_chunk returns a BCTHW tensor in [-1, 1].
            input = preprocess_chunk(input, self.device, self.dtype)

        # BCTHW -> THWC in [0, 1] (same normalization path RIFE uses).
        input_thwc = postprocess_chunk(rearrange(input, "B C T H W -> B T C H W"))
        frames = input_thwc  # T, H, W, C in [0, 1]
        t_count = frames.shape[0]
        if t_count < 2:
            return {"video": frames}

        # Permute to TCHW float32 in [0, 1] for the blend.
        tchw = frames.permute(0, 3, 1, 2).float()

        # Interpolate the midpoint of each consecutive pair and interleave:
        # [f0, mid01, f1, mid12, f2, ...] -> 2*T-1 frames.
        out = [tchw[0:1]]
        for i in range(t_count - 1):
            mid = interpolate_pair(tchw[i : i + 1], tchw[i + 1 : i + 2], t=0.5).clamp(
                0.0, 1.0
            )
            out.append(mid)
            out.append(tchw[i + 1 : i + 2])
        interpolated = torch.cat(out, dim=0)  # (2T-1), C, H, W

        # Back to THWC [0, 1] float, matching RIFE's output contract.
        interpolated_thwc = interpolated.permute(0, 2, 3, 1)
        return {"video": interpolated_thwc}
