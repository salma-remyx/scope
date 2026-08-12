from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class GEDepthConfig(BasePipelineConfig):
    """Configuration for the GEDepth ground-embedding pipeline.

    Emits a camera-parameter-conditioned ground-plane depth prior — the
    geometric signal at the core of GEDepth's ground embedding module.
    No model weights are loaded; the prior is computed analytically from
    the camera parameters below.
    """

    pipeline_id = "gedepth"
    pipeline_name = "GEDepth Ground Embedding"
    pipeline_description = (
        "Camera-parameter-decoupled ground-plane depth prior from the "
        "GEDepth ground embedding module. Analytic (no model weights); "
        "useful as a geometric depth preprocessor or visualization."
    )
    docs_url = "https://arxiv.org/abs/2309.09975"
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    camera_height: float = Field(
        default=1.5,
        gt=0.0,
        description="Camera height above the ground plane, in meters",
    )
    pitch_deg: float = Field(
        default=5.0,
        ge=-45.0,
        le=45.0,
        description="Camera pitch below the horizon, in degrees (positive looks down)",
    )
    vertical_fov_deg: float = Field(
        default=60.0,
        gt=0.0,
        le=180.0,
        description="Vertical field of view, in degrees",
    )
    max_distance: float = Field(
        default=80.0,
        gt=0.0,
        description=(
            "Distance clamp (meters) used to normalize the depth prior to [0, 1]"
        ),
    )
    invert: bool = Field(
        default=False,
        description="If True, near pixels are brighter (flip the near/far convention)",
    )

    modes = {"video": ModeDefaults(default=True)}
