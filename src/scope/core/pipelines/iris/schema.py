from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, Field, ModeDefaults, UsageType


class IrisConfig(BasePipelineConfig):
    """Configuration for the Iris monocular depth estimation pipeline.

    Iris (CVPR 2026, Cai et al.) is a deterministic diffusion framework that
    injects real-world priors into a diffusion model for monocular depth
    estimation via a two-stage Priors-to-Geometry Deterministic (PGD) schedule.
    It is image-oriented and produces fine-detailed depth maps, complementing
    the lighter/faster ``video-depth-anything`` as a higher-quality depth option.
    """

    pipeline_id = "iris"
    pipeline_name = "Iris Depth"
    pipeline_description = (
        "Monocular depth estimation using the Iris diffusion model. Produces "
        "fine-detailed, high-quality depth maps via a two-stage priors-to-"
        "geometry deterministic schedule. Image-oriented (per-frame)."
    )
    docs_url = "https://github.com/NUST-Machine-Intelligence-Laboratory/Iris"
    estimated_vram_gb = 8.0
    artifacts = [
        HuggingfaceRepoArtifact(
            repo_id="Strike1999/Iris",
            files=[
                "model_index.json",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.safetensors",
                "feature_extractor/preprocessor_config.json",
            ],
        ),
    ]
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    # Iris-specific runtime parameters.
    fp32: bool = Field(
        default=False,
        description="Run inference in float32 instead of float16",
    )
    processing_res: int | None = Field(
        default=None,
        description=(
            "Maximum processing edge resolution; None uses the pipeline default "
            "(768), 0 keeps the input resolution"
        ),
    )
    timesteps: list[int] = Field(
        default=[499, 999],
        description=(
            "Two-stage PGD timestep schedule [t_high, t_low]; the high timestep "
            "stage injects low-frequency real-world priors, the low timestep "
            "stage refines high-frequency geometry"
        ),
    )
