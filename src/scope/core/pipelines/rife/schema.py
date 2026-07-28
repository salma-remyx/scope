from typing import Literal

from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class RIFEConfig(BasePipelineConfig):
    """Configuration for RIFE frame interpolation pipeline.

    This pipeline uses RIFE HDv3 (Real-Time Intermediate Flow Estimation)
    to double the frame rate of input video by generating intermediate frames.

    Model weights are from Practical-RIFE v4.25:
    https://github.com/hzwer/Practical-RIFE
    """

    pipeline_id = "rife"
    pipeline_name = "RIFE"
    pipeline_description = (
        "Frame interpolation pipeline using RIFE HDv3 to double the frame rate "
        "of input video by generating intermediate frames."
    )
    docs_url = "https://github.com/hzwer/Practical-RIFE"
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

    # Inter-frame modeling engine. "rife" is the production RIFE HDv3 path;
    # "ssm" swaps in a VFIMamba-inspired bidirectional selective state-space
    # model (no external weights or GPU required). Changing this is a load
    # param: it switches the interpolator, so the stream must be reloaded.
    interpolation_engine: Literal["rife", "ssm"] = Field(
        default="rife",
        description=(
            "Inter-frame modeling engine: 'rife' (RIFE HDv3, default) or 'ssm' "
            "(VFIMamba-inspired bidirectional selective state-space model)."
        ),
        json_schema_extra=ui_field_config(
            is_load_param=True, label="Interpolation engine"
        ),
    )
