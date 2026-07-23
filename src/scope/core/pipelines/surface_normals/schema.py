from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults


class SurfaceNormalsConfig(BasePipelineConfig):
    """Configuration for the Surface Normals pipeline.

    Estimates per-pixel surface normals (and a target-camera metric depth
    map) from a monocular depth prediction, applying Metric3Dv2's canonical
    camera space un-alignment. Compose after ``video-depth-anything``:
    feed its depth output into this node's ``video`` input.
    """

    pipeline_id = "surface-normals"
    pipeline_name = "Surface Normals"
    pipeline_description = (
        "Surface normal estimation from a depth map via Metric3Dv2-inspired "
        "canonical camera space geometry. Compose after video-depth-anything."
    )
    docs_url = "https://arxiv.org/abs/2404.15506"
    supports_prompts = False
    modified = True
    # No learned model: registers and runs without a GPU.
    outputs = ["video", "depth"]

    modes = {"video": ModeDefaults(default=True)}

    focal_length: float = Field(
        default=1000.0,
        ge=1.0,
        description=(
            "Target camera focal length in pixels; the canonical-space depth "
            "input is un-aligned by focal_length / canonical_focal before "
            "normal extraction"
        ),
    )
    canonical_focal: float = Field(
        default=1000.0,
        ge=1.0,
        description="Metric3Dv2 canonical camera focal length in pixels",
    )
