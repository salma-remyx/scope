"""Standalone smoke test for the Iris depth pipeline.

Run manually (needs a GPU and the Strike1999/Iris weights downloaded):

    uv run python -m scope.core.pipelines.iris.test

Drop an input clip at ``assets/input.mp4`` next to this file (or edit the path
below) before running. Not collected by pytest (lives outside ``tests/``).
"""

import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.pipelines.utils import print_statistics

from ..video import load_video
from .pipeline import IrisDepthPipeline


def main():
    """Run the Iris depth pipeline over a short clip and save a depth video."""
    config = OmegaConf.create(
        {
            "fp32": False,
            "processing_res": None,  # pipeline default (768)
            "timesteps": [499, 999],  # two-stage PGD schedule
            "height": 480,
            "width": 832,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = IrisDepthPipeline(config, device=device)

    video_tensor = load_video(
        Path(__file__).parent / "assets" / "input.mp4",
        resize_hw=(config.height, config.width),
        normalize=False,
    )
    num_frames = video_tensor.shape[1]
    video_list = [
        video_tensor[:, i].permute(1, 2, 0).unsqueeze(0)  # CTHW -> [1, H, W, C]
        for i in range(num_frames)
    ]

    latency_measures = []
    fps_measures = []
    depths_list = []

    for i, frame in enumerate(video_list):
        start = time.time()
        output_dict = pipeline(video=[frame])
        depth = output_dict["video"]
        latency = time.time() - start
        fps = depth.shape[0] / latency

        print(
            f"Pipeline processed frame {i + 1}/{num_frames} "
            f"latency={latency:.2f}s fps={fps:.2f}"
        )

        latency_measures.append(latency)
        fps_measures.append(fps)
        depths_list.append(depth.cpu())

    depths = torch.concat(depths_list)
    print(f"Output shape: {depths.shape}")

    output_path = Path(__file__).parent / "output.mp4"
    depths_np = depths.cpu().numpy()
    depths_np = (depths_np - depths_np.min()) / (
        depths_np.max() - depths_np.min() + 1e-8
    )
    export_to_video(depths_np, output_path, fps=30)
    print(f"Saved depth video to {output_path}")

    print_statistics(latency_measures, fps_measures)


if __name__ == "__main__":
    main()
