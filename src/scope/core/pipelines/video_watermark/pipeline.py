"""Blind per-frame video watermarking for AI-content provenance.

Adapted from *Video Seal: Open and Efficient Video Watermarking*
(arXiv:2412.09492). The paper's contribution here is the blind
embed/extract contract for video provenance: an imperceptible message is
added to each frame and later recovered *without* the original frames.

This is a Mode 2 (adapted) port: the paper's *learned neural*
embedder/extractor is substituted with a parameter-free, zero-mean
spread-spectrum signal carried in per-bit horizontal bands. The core
mechanism (blind embed + blind extract of a multi-bit message, low
perceptual strength) is preserved at full fidelity; no model weights are
downloaded and the pipeline runs in realtime on CPU or GPU. The paper's
separate benchmark / attack-robustness evaluation suite is intentionally
out of scope (downstream concern).
"""

import logging
from typing import TYPE_CHECKING

import torch

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes
from .schema import VideoWatermarkConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

_DEFAULT_MESSAGE = "scope"


def _round_bits(num_bits: int) -> int:
    """Round capacity down to a whole-byte (multiple of 8) bit count."""
    return max(8, int(num_bits) // 8 * 8)


def _message_to_bits(message: str, num_bits: int) -> torch.Tensor:
    """Encode a string into a fixed-length antipodal bit tensor ({-1, +1})."""
    capacity_bytes = num_bits // 8
    raw = message.encode("utf-8")[:capacity_bytes].ljust(capacity_bytes, b"\x00")
    bits = torch.tensor(
        [bit for byte in raw for bit in [(byte >> (7 - i)) & 1 for i in range(8)]],
        dtype=torch.float32,
    )
    return bits * 2.0 - 1.0  # {0, 1} -> {-1, +1}


def _bits_to_message(bits: torch.Tensor) -> str:
    """Decode a bit tensor ({-1, +1} or {0, 1}) back into a string."""
    binary = (bits > 0).to(torch.uint8)
    values = binary.tolist()
    out = bytearray()
    for i in range(0, len(values) // 8 * 8, 8):
        byte = 0
        for bit in values[i : i + 8]:
            byte = (byte << 1) | int(bit)
        out.append(byte)
    return bytes(out).rstrip(b"\x00").decode("utf-8", errors="replace")


def _carriers(num_bits: int, height: int, width: int, key: int) -> torch.Tensor:
    """Zero-mean pseudo-random carrier per bit-band, shape ``(num_bits, H, W)``.

    Bit ``i`` is signaled in horizontal rows ``[i*bh:(i+1)*bh]`` so each bit
    owns a disjoint spatial band — that keeps extraction a simple per-band
    correlation and makes the bands independent. Carriers are generated on
    CPU with a seeded generator so embed and extract share an identical
    pattern regardless of the runtime device.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(key))
    carriers = torch.zeros(num_bits, height, width, dtype=torch.float32)
    band_height = max(1, height // num_bits)
    for i in range(num_bits):
        top = i * band_height
        bottom = height if i == num_bits - 1 else min(height, (i + 1) * band_height)
        if bottom <= top:
            continue
        band = torch.randn(bottom - top, width, generator=generator)
        band -= band.mean()  # zero-mean => correlation is blind to image DC
        carriers[i, top:bottom, :] = band
    return carriers


def extract_watermark(video: torch.Tensor, num_bits: int = 48, key: int = 1337) -> str:
    """Blind-extract a spread-spectrum watermark from frames.

    Averages over time and channels, detrends each bit-band (removing local
    image brightness), then correlates against the matching carrier. The
    sign of each correlation is one message bit. No access to the original
    (unwatermarked) frames is required.

    Args:
        video: ``THWC`` (or ``HWC``) float tensor in ``[0, 1]`` or ``[0, 255]``.
        num_bits: Capacity used at embed time (must match).
        key: Secret seed used at embed time (must match).

    Returns:
        The recovered provenance string.
    """
    frames = video.to(device="cpu", dtype=torch.float32)
    if float(frames.max()) > 1.5:  # tolerate [0, 255] input
        frames = frames / 255.0
    if frames.dim() == 4:  # THWC -> HW
        frames = frames.mean(dim=(0, 3))
    elif frames.dim() == 3:  # HWC -> HW
        frames = frames.mean(dim=-1)
    height, width = frames.shape
    num_bits = _round_bits(num_bits)
    carriers = _carriers(num_bits, height, width, key)
    band_height = max(1, height // num_bits)
    bits = torch.zeros(num_bits, dtype=torch.float32)
    for i in range(num_bits):
        top = i * band_height
        bottom = height if i == num_bits - 1 else min(height, (i + 1) * band_height)
        if bottom <= top:
            bits[i] = -1.0
            continue
        band = frames[top:bottom, :]
        band = band - band.mean()  # detrend: drop image brightness, keep signal
        correlation = (band * carriers[i, top:bottom, :]).mean()
        bits[i] = 1.0 if correlation > 0 else -1.0
    return _bits_to_message(bits)


class VideoWatermarkPipeline(Pipeline):
    """Realtime blind spread-spectrum video watermark embedder."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VideoWatermarkConfig

    def __init__(
        self,
        watermark_message: str = _DEFAULT_MESSAGE,
        watermark_strength: float = 0.03,
        watermark_key: int = 1337,
        num_bits: int = 48,
        device: torch.device | None = None,
        **kwargs,
    ):
        """Initialize the watermark pipeline.

        Extra kwargs (height, width, base_seed, ...) come from the config
        schema when the pipeline is loaded by the generic plugin path and
        are accepted but unused by the embedder.
        """
        self.watermark_message = watermark_message
        self.watermark_strength = float(watermark_strength)
        self.watermark_key = int(watermark_key)
        self.num_bits = _round_bits(num_bits)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._carrier_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    def _get_carriers(self, height: int, width: int) -> torch.Tensor:
        cache_key = (self.watermark_key, self.num_bits, height, width)
        carriers = self._carrier_cache.get(cache_key)
        if carriers is None:
            carriers = _carriers(self.num_bits, height, width, self.watermark_key)
            self._carrier_cache[cache_key] = carriers
        return carriers.to(self.device)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Embed the configured watermark into video frames.

        Args:
            video: Input frames as a list of ``[1, H, W, C]`` tensors in
                ``[0, 255]`` (the standard realtime frame contract).

        Returns:
            Dict with ``"video"``: watermarked frames as a ``THWC`` float
            tensor in ``[0, 1]``.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for VideoWatermarkPipeline")

        video = normalize_frame_sizes(video)
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)  # (T,H,W,C)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        _, height, width, _ = frames.shape
        carriers = self._get_carriers(height, width)  # (num_bits, H, W)
        bits = _message_to_bits(self.watermark_message, self.num_bits).to(self.device)

        # Antipodal per-band modulation, summed into one (H, W) pattern and
        # broadcast over channels. Same message every frame (temporal persistence).
        pattern = (carriers * bits.view(-1, 1, 1)).sum(dim=0)
        watermarked = (frames + self.watermark_strength * pattern.unsqueeze(-1)).clamp(
            0, 1
        )
        logger.debug(
            "Embedded %d-bit watermark into %d frame(s) at strength %.3f",
            self.num_bits,
            frames.shape[0],
            self.watermark_strength,
        )
        return {"video": watermarked}

    def extract(self, video: torch.Tensor) -> str:
        """Recover the watermark message from this pipeline's frames."""
        return extract_watermark(video, num_bits=self.num_bits, key=self.watermark_key)
