"""Time-adaptive video frame interpolation pipeline.

Adapted from "Time-adaptive Video Frame Interpolation based on Residual
Diffusion" (arXiv:2504.05402) -- see :mod:`.time_adaptive` for the
contribution mapping. This pipeline reuses the repo's RIFE interpolator as
its flow-based frame synthesizer and wraps it with arbitrary-time,
multi-frame trajectory interpolation (paper contribution 1), optionally
using a ResShift-inspired residual-shift schedule (contribution 2,
structurally retained; learned denoiser substituted by RIFE).
"""

import logging
from typing import TYPE_CHECKING

import torch
from einops import rearrange

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes, postprocess_chunk, preprocess_chunk
from .schema import TimeAdaptiveVFIConfig
from .time_adaptive import (
    RIFETimestepSynthesizer,
    densify_sequence,
    resshift_schedule,
    uniform_schedule,
)

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_SCHEDULES = {"uniform": uniform_schedule, "resshift": resshift_schedule}


class TimeAdaptiveVFIPipeline(Pipeline):
    """Frame interpolation pipeline placing N intermediate frames at
    arbitrary interpolation times between consecutive input frames."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TimeAdaptiveVFIConfig

    def __init__(
        self,
        num_intermediate_frames: int = 1,
        interpolation_schedule: str = "uniform",
        resshift_shift_ratio: float = 1.25,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        """Initialize the time-adaptive VFI pipeline.

        Args:
            num_intermediate_frames: Intermediate frames per input-frame gap.
            interpolation_schedule: 'uniform' or 'resshift' time schedule.
            resshift_shift_ratio: Geometric gap ratio for the resshift schedule.
            device: Target device (defaults to CUDA if available).
            dtype: Data type for preprocessing (unused by RIFE inference).
            **kwargs: Absorbs schema-default fields passed by the loader.
        """
        from ..rife.modules.interpolation import RIFEInterpolator

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        if interpolation_schedule not in _SCHEDULES:
            raise ValueError(
                f"Unknown interpolation_schedule: {interpolation_schedule!r} "
                f"(expected one of {sorted(_SCHEDULES)})."
            )
        self.num_intermediate_frames = max(1, int(num_intermediate_frames))
        self.interpolation_schedule = interpolation_schedule
        self.resshift_shift_ratio = float(resshift_shift_ratio)

        # Reuse the repo's RIFE interpolator as the per-timestep synthesizer.
        logger.info("Loading RIFE HDv3 model for time-adaptive VFI...")
        self.rife_interpolator = RIFEInterpolator(enabled=True, device=self.device)
        self.synthesizer = RIFETimestepSynthesizer(self.rife_interpolator)
        logger.info(
            "Time-adaptive VFI ready (%d intermediate frame(s), %s schedule).",
            self.num_intermediate_frames,
            self.interpolation_schedule,
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=12)

    def __call__(self, **kwargs) -> dict:
        input = kwargs.get("video")

        if input is None:
            raise ValueError("Input cannot be None for TimeAdaptiveVFIPipeline")

        if isinstance(input, list):
            # Normalize frame sizes to handle resolution changes.
            input = normalize_frame_sizes(input)
            # Preprocess: list of frames -> BCTHW tensor in [-1, 1].
            input = preprocess_chunk(input, self.device, self.dtype)

        # Convert from BCTHW to THWC [0, 1] (same contract as RIFEPipeline).
        input_btchw = rearrange(input, "B C T H W -> B T C H W")
        input_thwc = postprocess_chunk(input_btchw)
        input_uint8 = (input_thwc * 255.0).clamp(0, 255).to(torch.uint8)

        schedule = _SCHEDULES[self.interpolation_schedule]
        schedule_kwargs = (
            {"shift_ratio": self.resshift_shift_ratio}
            if self.interpolation_schedule == "resshift"
            else {}
        )
        # Insert N intermediate frames between each consecutive frame pair.
        # num_intermediate_frames=1 + uniform schedule == RIFE frame doubling.
        interpolated = densify_sequence(
            input_uint8,
            self.synthesizer,
            self.num_intermediate_frames,
            schedule=schedule,
            **schedule_kwargs,
        )

        interpolated_float = interpolated.float() / 255.0
        return {"video": interpolated_float}
