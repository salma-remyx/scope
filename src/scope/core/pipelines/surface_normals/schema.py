from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults


class SurfaceNormalsConfig(BasePipelineConfig):
    """Configuration for the Surface Normals pipeline.

    Estimates per-pixel surface normals (and a canonical-camera-space depth
    map) from a monocular depth input, applying Metric3Dv2's canonical
    camera space transformation. Compose after ``video-depth-anything``:
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
        description="Source camera focal length in pixels used for depth back-projection",
    )
    canonical_focal: float = Field(
        default=1000.0,
        ge=1.0,
        description="Metric3Dv2 canonical camera focal length in pixels",
    )
