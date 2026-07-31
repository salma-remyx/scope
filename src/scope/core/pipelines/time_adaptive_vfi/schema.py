"""Configuration for the time-adaptive VFI pipeline."""

from typing import Literal

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class TimeAdaptiveVFIConfig(BasePipelineConfig):
    """Configuration for the time-adaptive frame interpolation pipeline.

    This pipeline places a configurable number of intermediate frames
    between consecutive input frames at arbitrary interpolation times,
    using RIFE flow synthesis as the per-timestep frame synthesizer. With
    a single intermediate frame and the uniform schedule it reproduces
    RIFE's frame-doubling behaviour; more frames or the resshift schedule
    yield denser, time-adaptive trajectories.

    Adapted from "Time-adaptive Video Frame Interpolation based on Residual
    Diffusion" (arXiv:2504.05402). Model weights reuse the RIFE HDv3
    artifact from Practical-RIFE v4.25: https://github.com/hzwer/Practical-RIFE
    """

    pipeline_id = "time-adaptive-vfi"
    pipeline_name = "Time-Adaptive VFI"
    pipeline_description = (
        "Time-adaptive video frame interpolation. Generates a configurable "
        "number of intermediate frames between consecutive input frames at "
        "arbitrary interpolation times using RIFE flow synthesis."
    )
    docs_url = "https://arxiv.org/abs/2504.05402"
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="daydreamlive/RIFE",
            files=["config.json", "flownet.pkl"],
        ),
    ]
    supports_prompts = False
    modified = True

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    num_intermediate_frames: int = Field(
        default=1,
        ge=1,
        description=(
            "Intermediate frames to synthesise between each consecutive "
            "input-frame pair (1 reproduces RIFE frame doubling)."
        ),
    )
    interpolation_schedule: Literal["uniform", "resshift"] = Field(
        default="uniform",
        description=(
            "Time schedule for placing intermediate frames. 'uniform' spaces "
            "them evenly; 'resshift' uses a ResShift-inspired residual-shift "
            "schedule (geometric gap progression)."
        ),
    )
    resshift_shift_ratio: float = Field(
        default=1.25,
        gt=0.0,
        description=(
            "Geometric ratio between successive time gaps for the resshift "
            "schedule. 1.0 collapses to uniform spacing."
        ),
    )
