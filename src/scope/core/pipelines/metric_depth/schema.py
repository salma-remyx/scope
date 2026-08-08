from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class MetricDepthConfig(BasePipelineConfig):
    """Configuration for the Metric Depth pipeline (UniDepth, adapted).

    Produces camera-aware monocular metric depth (metres) from RGB video by
    recovering metric scale from a relative-depth backbone. The recovery
    mechanism lives in
    :func:`scope.core.pipelines.metric_depth.pipeline.rescale_to_metric`.
    """

    pipeline_id = "metric-depth"
    pipeline_name = "Metric Depth"
    pipeline_description = (
        "Monocular metric depth estimation from RGB video. Recovers "
        "camera-aware metric depth (metres) from a relative-depth backbone "
        "via an analytical median-anchored, pinhole-corrected recovery. "
        "Adapted from UniDepth (arXiv:2403.18913)."
    )
    docs_url = "https://github.com/lpiccinelli/UniDepth"
    estimated_vram_gb = 1.0
    # Reuses the Video-Depth-Anything relative-depth backbone as its RGB
    # encoder, so it depends on the same checkpoint.
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="daydreamlive/Video-Depth-Anything-Small",
            files=["config.json", "video_depth_anything_vits.pth"],
        ),
    ]
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # Backbone inference (mirrors Video-Depth-Anything).
    fp32: bool = False
    input_size: int | None = 518

    # Metric-recovery parameters (see ``rescale_to_metric``). The median anchor
    # provides the absolute scale a learned metric head would otherwise predict;
    # ``focal_length_px`` optionally makes the recovery camera-aware.
    median_depth_m: float = 5.0
    depth_min_m: float = 0.5
    depth_max_m: float = 100.0
    focal_length_px: float | None = None
