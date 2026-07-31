"""Time-adaptive video frame interpolation.

Adapted from "Time-adaptive Video Frame Interpolation based on Residual
Diffusion" (arXiv:2504.05402). The paper's two headline contributions are
mapped onto this repo as follows (Mode 2 adapted port):

* Contribution 1 -- "explicitly handle the interpolation time": delivered
  at full fidelity. RIFE's flow network natively accepts an arbitrary
  interpolation timestep ``t in (0, 1)``, but the stock RIFE pipeline only
  ever queries the midpoint ``t = 0.5`` (frame doubling). This module
  exposes arbitrary-``t`` interpolation so a trajectory of ``N``
  intermediate frames can be synthesised between any two endpoints, not
  just one midpoint frame.

* Contribution 2 -- ResShift residual-diffusion trajectory: the paper's
  learned multi-step diffusion denoiser is substituted by this repo's
  existing RIFE flow-based synthesizer (the repo has no diffusion-based
  VFI training infrastructure). What is retained is ResShift's
  *structural* idea -- a short discrete trajectory that shifts the
  estimate from the source frame toward the target in scheduled steps --
  realised here as the parameter-free :func:`resshift_schedule`.

Substituted auxiliary components (Mode 2):

  - learned diffusion denoiser  -> RIFE flow-based synthesizer at arbitrary t
  - trained interpolation-time re-estimator -> parameter-free schedules
  - animation-domain training data -> inference-time capability only

All frame tensors use the same contract as
:class:`scope.core.pipelines.rife.modules.interpolation.RIFEInterpolator`:
``[T, H, W, C]`` (or ``[H, W, C]`` for a single frame), ``uint8`` in
``[0, 255]``.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def uniform_schedule(num_intermediate: int) -> list[float]:
    """Evenly spaced interpolation times in the open interval (0, 1).

    Args:
        num_intermediate: Number of intermediate frames between two endpoints.

    Returns:
        Sorted interpolation times of length ``num_intermediate``.
        ``num_intermediate == 1`` -> ``[0.5]``; ``3`` -> ``[0.25, 0.5, 0.75]``.
    """
    if num_intermediate < 1:
        return []
    return [(k + 1) / (num_intermediate + 1) for k in range(num_intermediate)]


def resshift_schedule(num_intermediate: int, shift_ratio: float = 1.25) -> list[float]:
    """ResShift-inspired residual-shift time schedule.

    Places intermediate times whose successive gaps follow a geometric
    progression with ratio ``shift_ratio`` -- the discrete analog of
    ResShift's residual-shifting trajectory, which schedules larger shifts
    early and refines later. The result is a non-uniform schedule, in
    contrast to the even spacing of :func:`uniform_schedule`.

    This is a parameter-free proxy for the paper's learned
    interpolation-time estimator: the *trajectory structure* of ResShift is
    retained, the learned denoiser is not (RIFE synthesises each frame).

    Args:
        num_intermediate: Number of intermediate frames between two endpoints.
        shift_ratio: Geometric ratio between successive time gaps. ``> 1``
            grows the gaps over the trajectory; ``< 1`` shrinks them;
            ``1.0`` collapses to :func:`uniform_schedule`. Must be positive.

    Returns:
        Sorted interpolation times in (0, 1), length ``num_intermediate``.
    """
    if num_intermediate < 1:
        return []
    if shift_ratio <= 0:
        raise ValueError("shift_ratio must be positive")
    n_gaps = num_intermediate + 1
    weights = [shift_ratio**k for k in range(n_gaps)]
    total = sum(weights)
    times: list[float] = []
    cumulative = 0.0
    for w in weights[:-1]:  # n_gaps - 1 == num_intermediate interior points
        cumulative += w / total
        times.append(cumulative)
    return times


def interpolate_pair(
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    synthesizer,
    times: list[float],
) -> torch.Tensor:
    """Synthesise intermediate frames between two endpoints at given times.

    Args:
        frame0, frame1: Endpoint frames, shape ``[H, W, C]`` uint8 [0, 255].
        synthesizer: Callable ``(frame0, frame1, t) -> frame`` returning an
            intermediate frame at time ``t in (0, 1)`` (0 == frame0,
            1 == frame1), same shape/dtype as the inputs.
        times: Interpolation times in (0, 1), ascending.

    Returns:
        Tensor ``[len(times), H, W, C]`` uint8 ordered to match ``times``.
    """
    if frame0.ndim != 3 or frame1.ndim != 3:
        raise ValueError("endpoints must be [H, W, C]")
    if not times:
        return frame0.new_zeros((0, *frame0.shape), dtype=frame0.dtype)
    frames = [synthesizer(frame0, frame1, float(t)) for t in times]
    return torch.stack(frames, dim=0)


def densify_sequence(
    frames: torch.Tensor,
    synthesizer,
    num_intermediate: int,
    schedule=uniform_schedule,
    **schedule_kwargs,
) -> torch.Tensor:
    """Insert ``num_intermediate`` frames between each consecutive pair.

    Turns a ``T``-frame sequence into a denser, time-adaptive trajectory of
    ``T + (T - 1) * num_intermediate`` frames. With ``num_intermediate == 1``
    and the uniform schedule this reproduces RIFE's frame-doubling output
    (``2T - 1`` frames); larger values yield slow-motion trajectories, and
    the resshift schedule yields a non-uniform residual-shift trajectory.

    Args:
        frames: ``[T, H, W, C]`` uint8 in [0, 255].
        synthesizer: See :func:`interpolate_pair`.
        num_intermediate: Intermediate frames to insert per gap.
        schedule: Schedule builder ``(n, **kw) -> list[float]``.
        **schedule_kwargs: Forwarded to ``schedule``.

    Returns:
        ``[T + (T - 1) * num_intermediate, H, W, C]`` uint8.
    """
    if frames.ndim != 4:
        raise ValueError("frames must be [T, H, W, C]")
    t = frames.shape[0]
    if t < 2 or num_intermediate < 1:
        return frames
    times = schedule(num_intermediate, **schedule_kwargs)
    out = [frames[0].unsqueeze(0)]
    for i in range(t - 1):
        mids = interpolate_pair(frames[i], frames[i + 1], synthesizer, times)
        out.append(mids)
        out.append(frames[i + 1].unsqueeze(0))
    return torch.cat(out, dim=0)


class RIFETimestepSynthesizer:
    """Adapts a loaded RIFE interpolator to synthesise a frame at arbitrary t.

    RIFE's flow network already accepts an arbitrary timestep via
    ``model.inference(img0, img1, timestep=t)``; this class is the thin
    adapter the stock RIFE pipeline (fixed midpoint) does not expose -- the
    core enabler of the paper's contribution 1.

    The 32-multiple padding contract mirrors
    :meth:`RIFEInterpolator._rife_interpolate`.
    """

    def __init__(self, rife_interpolator):
        self.rife = rife_interpolator
        self.device = rife_interpolator.device

    def __call__(self, frame0: torch.Tensor, frame1: torch.Tensor, t: float):
        """Synthesise one intermediate frame at interpolation time ``t``.

        Args:
            frame0, frame1: ``[H, W, C]`` uint8 [0, 255] endpoints.
            t: Interpolation time in (0, 1).

        Returns:
            ``[H, W, C]`` uint8 [0, 255] intermediate frame.
        """
        model = self.rife.model
        if model is None:
            raise RuntimeError(
                "RIFE model is not loaded; cannot synthesise at arbitrary time."
            )
        h, w, _ = frame0.shape
        i0 = (frame0.permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(self.device)
        i1 = (frame1.permute(2, 0, 1).unsqueeze(0).float() / 255.0).to(self.device)
        # RIFE requires spatial dims to be multiples of 32.
        tmp = 32
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        i0p = F.pad(i0, padding)
        i1p = F.pad(i1, padding)
        with torch.no_grad():
            mid = model.inference(i0p, i1p, timestep=float(t), scale=1.0)
        mid = mid[:, :, :h, :w]
        out = (mid[0].permute(1, 2, 0) * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        return out
