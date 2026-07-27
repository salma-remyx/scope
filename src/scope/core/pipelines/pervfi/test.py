"""Standalone smoke test for the PerVFI interpolation pipeline.

Unlike RIFE's test (which loads an asset video), this synthesizes a small,
textured translating sequence so it runs anywhere with no media dependency:

    uv run python -m scope.core.pipelines.pervfi.test
"""

import time

import numpy as np
import torch

from scope.core.pipelines.utils import print_statistics

from .pipeline import PerVFIPipeline


def _synthetic_video(num_frames: int = 6, height: int = 64, width: int = 64):
    """A textured square translating across frames (motion for the blend)."""
    rng = np.random.default_rng(0)
    texture = rng.random((1, height - 16, width - 16, 3)).astype(np.float32)
    frames = []
    for i in range(num_frames):
        canvas = np.zeros((1, height, width, 3), dtype=np.float32)
        offset = 4 + i * 4
        oh, ow = height - 16, width - 16
        canvas[:, offset : offset + oh, offset : offset + ow, :] = texture
        frames.append(torch.from_numpy(canvas))
    return frames


def main():
    """Run the PerVFI pipeline on a synthetic sequence and print statistics."""
    config = type("C", (), {"height": 64, "width": 64})()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = PerVFIPipeline(config, device=device)

    video_list = _synthetic_video()
    num_frames = len(video_list)

    start = time.time()
    output_dict = pipeline(video=video_list)
    output = output_dict["video"]
    latency = time.time() - start

    num_output_frames = output.shape[0]
    fps = num_output_frames / latency

    print(f"Input frames: {num_frames}, Output frames: {num_output_frames}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Processing latency: {latency:.2f}s, FPS: {fps:.2f}")

    expected_frames = num_frames * 2 - 1
    if num_output_frames == expected_frames:
        print(
            f"SUCCESS: Frame count increased from {num_frames} to {num_output_frames}"
        )
    else:
        print(
            f"WARNING: Expected {expected_frames} frames, "
            f"got {num_output_frames} frames"
        )

    print_statistics([latency], [fps])


if __name__ == "__main__":
    main()
