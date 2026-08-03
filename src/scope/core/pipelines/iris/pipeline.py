"""Iris monocular depth estimation pipeline.

Adapts Iris (CVPR 2026, Cai et al.) into Scope's node/pipeline abstraction as a
higher-quality depth *option* alongside ``video-depth-anything``. Iris is a
deterministic diffusion framework that injects real-world priors into a
diffusion model for monocular depth estimation via a two-stage
Priors-to-Geometry Deterministic (PGD) schedule. It is image-oriented (no
temporal model), so each frame is depth-estimated independently.

The upstream diffusers pipeline is vendored (Apache-2.0) under ``./modules`` and
loaded from the ``Strike1999/Iris`` weights via ``from_pretrained``.

Upstream: https://github.com/NUST-Machine-Intelligence-Laboratory/Iris
Weights:  https://huggingface.co/Strike1999/Iris (Apache-2.0)
"""

import logging
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from scope.core.config import get_models_dir

from ..interface import Pipeline, Requirements
from ..process import normalize_frame_sizes
from .schema import IrisConfig

if TYPE_CHECKING:
    from ..schema import BasePipelineConfig

logger = logging.getLogger(__name__)

# Two-stage Priors-to-Geometry Deterministic schedule from the paper: the high
# timestep stage injects low-frequency real-world priors, the low timestep
# stage refines high-frequency geometry.
_DEFAULT_TIMESTEPS = [499, 999]


def _depth_task_embedding(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Depth task switcher (sin/cos of [1, 0]) for the shared depth/normal UNet.

    Mirrors the upstream reference application: ``[1, 0]`` selects depth
    reconstruction (``[0, 1]`` would select normals).
    """
    task = torch.tensor([1.0, 0.0]).unsqueeze(0).to(device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


class IrisDepthPipeline(Pipeline):
    """Monocular depth estimation via the Iris diffusion model."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return IrisConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the Iris depth pipeline.

        Args:
            config: Pipeline configuration.
            device: Target device (defaults to CUDA if available).
            dtype: Model dtype when not running in fp32 (default: float16).
        """
        # Heavy diffusers import is deferred so importing this module is cheap.
        from .modules.pipeline import IrisPipeline as _IrisDiffusionPipeline

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp32 = getattr(config, "fp32", False)
        self.dtype = torch.float32 if self.fp32 else dtype
        self.processing_res = getattr(config, "processing_res", None)
        self.timesteps = list(getattr(config, "timesteps", None) or _DEFAULT_TIMESTEPS)

        start = time.time()
        logger.info("Loading Iris depth model from Strike1999/Iris...")
        model_path = str(get_models_dir() / "Iris")
        self.pipe = _IrisDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype
        )
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        logger.info(f"Loaded Iris depth model in {time.time() - start:.3f}s")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Process video frames and return depth maps.

        Args:
            video: Input frames as a list of THWC tensors ([1, H, W, C], RGB,
                   uint8 [0, 255] or float [0, 1]).

        Returns:
            Depth maps as a THWC tensor in [0, 1] where higher values indicate
            greater depth (further from camera) -- matching
            ``video-depth-anything`` so the two are drop-in alternatives.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("Input video cannot be None for IrisDepthPipeline")

        video = normalize_frame_sizes(video)

        autocast_ctx = (
            torch.autocast(self.device.type)
            if (self.device.type != "cpu" and not self.fp32)
            else nullcontext()
        )

        depths = []
        with autocast_ctx, torch.no_grad():
            task_emb = _depth_task_embedding(self.device, self.dtype)
            for frame in video:
                frame_np = (
                    frame.cpu().numpy()
                    if isinstance(frame, torch.Tensor)
                    else np.array(frame)
                )
                frame_np = frame_np.squeeze(0)  # (1, H, W, C) -> (H, W, C)

                if frame_np.dtype != np.uint8:
                    frame_np = (
                        (frame_np * 255).astype(np.uint8)
                        if float(frame_np.max()) <= 1.0
                        else frame_np.astype(np.uint8)
                    )

                rgb = (
                    torch.from_numpy(frame_np.astype(np.float32))
                    .permute(2, 0, 1)  # (C, H, W)
                    .unsqueeze(0)  # (1, C, H, W)
                    .to(device=self.device, dtype=self.dtype)
                    / 127.5
                    - 1.0
                )

                output = self.pipe(
                    rgb_in=rgb,
                    task_emb=task_emb,
                    prompt="",
                    timesteps=self.timesteps,
                    processing_res=self.processing_res,
                    match_input_res=True,
                    output_type="np",
                )
                pred = np.asarray(output.images[0])  # (H, W, 3) disparity in [0, 1]

                # Iris is trained in disparity space (near = high). Invert to
                # Scope's depth convention (high = further from camera).
                depths.append(1.0 - pred.mean(axis=-1))

        depth = torch.from_numpy(np.stack(depths, axis=0)).float()  # (T, H, W)
        d_min, d_max = depth.min(), depth.max()
        depth = (
            (depth - d_min) / (d_max - d_min)
            if d_max > d_min
            else torch.zeros_like(depth)
        )
        return {"video": depth.unsqueeze(-1).repeat(1, 1, 1, 3)}  # THWC, 3 channels
