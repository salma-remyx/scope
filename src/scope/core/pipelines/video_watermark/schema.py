from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults

_DEFAULT_MESSAGE = "scope"


class VideoWatermarkConfig(BasePipelineConfig):
    """Configuration for the video-watermark provenance pipeline.

    Embeds an imperceptible, blind-extractable message into every frame for
    AI-content provenance. Uses a parameter-free spread-spectrum watermark
    (no neural network, no model download) so it runs in realtime on CPU or
    GPU and slots into a graph like any other frame->frame node.
    """

    pipeline_id = "video-watermark"
    pipeline_name = "Video Watermark"
    pipeline_description = (
        "Embeds an imperceptible, blind-extractable message into every "
        "frame for AI-content provenance. Parameter-free spread-spectrum "
        "watermark with no model required, optimized for realtime use."
    )
    supports_prompts = False
    modified = True
    requires_models = False

    modes = {"video": ModeDefaults(default=True)}

    watermark_message: str = Field(
        default=_DEFAULT_MESSAGE,
        description=(
            "Provenance message embedded in every frame. Truncated to "
            "num_bits // 8 bytes (capacity)."
        ),
    )
    watermark_strength: float = Field(
        default=0.03,
        ge=0.0,
        le=0.25,
        description=(
            "Embedding strength in [0, 1] pixel units. Higher is more robust "
            "to re-encoding but more visible."
        ),
    )
    watermark_key: int = Field(
        default=1337,
        ge=0,
        description=(
            "Secret seed for the carrier pattern. Recovering the message "
            "requires the same key."
        ),
    )
    num_bits: int = Field(
        default=48,
        ge=8,
        description=(
            "Watermark capacity in bits, rounded down to a multiple of 8 "
            "(capacity = num_bits // 8 bytes)."
        ),
    )
