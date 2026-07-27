from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType


class PerVFIConfig(BasePipelineConfig):
    """Configuration for the PerVFI frame interpolation pipeline.

    This is a quality-oriented alternative to RIFE that doubles the frame rate
    of input video using asymmetric, occlusion-aware blending -- the core
    mechanism of PerVFI (Perception-Oriented Video Frame Interpolation via
    Asymmetric Blending, CVPR 2024). The implementation is a weight-free,
    CPU-runnable adapted port; see ``asymmetric_blend.py`` for the precise
    mapping of paper components to parameter-free proxies.

    Reference: https://arxiv.org/abs/2404.06692v1 (Apache-2.0).
    """

    pipeline_id = "pervfi"
    pipeline_name = "PerVFI"
    pipeline_description = (
        "Frame interpolation pipeline using PerVFI's asymmetric, occlusion-aware "
        "blending to double the frame rate of input video -- a quality-oriented "
        "alternative to RIFE."
    )
    docs_url = "https://github.com/Mulns/PerVFI"
    # Weight-free adapted port: no trained artifacts are downloaded (overrides
    # the base default with no annotation so it stays a class attribute, matching
    # how RIFEConfig declares its artifacts).
    artifacts = []
    supports_prompts = False
    modified = True

    usage = [UsageType.POSTPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}
