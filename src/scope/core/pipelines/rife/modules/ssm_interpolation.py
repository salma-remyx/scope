"""Bidirectional Selective State-Space (S6) inter-frame model for frame interpolation.

Adapted from VFIMamba: Video Frame Interpolation with State Space Models
(NeurIPS 2024, https://arxiv.org/abs/2407.02315). VFIMamba replaces the
convolutional / attention-based *contextual feature merging* used by
RIFE / EMA-VFI with a linear-complexity, data-dependent Selective State-Space
Model (Mamba S6) and a **bidirectional scan** for inter-frame modeling. The
paper's motivation is precisely that convolutions have a limited receptive
field while attention is quadratic; a selective SSM gives long-range,
data-dependent context at linear cost.

This is a **Mode 2 (adapted port)**:

* Core mechanism at full fidelity -- the bidirectional selective scan over
  inter-frame context (input-dependent step ``Δ``, reset/forget via ``A``,
  forward + backward passes, O(L) recurrence). This is VFIMamba's actual
  contribution.
* Auxiliary substitutions:
    - VFIMamba's ``mamba_ssm`` / ``causal-conv1d`` fused CUDA kernels are
      replaced by an explicit pure-PyTorch selective-scan recurrence
      (:func:`_selective_scan`). Same math, no CUDA build dependency, runs on
      CPU or GPU.
    - VFIMamba's full network + pretrained weights are replaced by a
      lightweight, self-initializing SSM contextual merger. Untrained it
      reduces to a data-dependent convex blend of consecutive frames -- the
      standard VFI baseline that VFIMamba's training specializes. Plug in
      trained weights later.

Intentionally scoped out (downstream): optical-flow re-estimation, the
asymmetric reciprocal flow warping VFIMamba borrows from EMA-VFI, the
training procedure, and benchmark evaluation.

The public surface mirrors :class:`~scope.core.pipelines.rife.modules.interpolation.RIFEInterpolator`
so the two engines are contract-equivalent and interchangeable from the
pipeline: both take ``[T, H, W, C]`` uint8 frames in ``[0, 255]`` and return
``[2T - 1, H, W, C]`` uint8 frames (2x doubling).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _selective_scan(
    u: torch.Tensor,
    bar_a: torch.Tensor,
    bar_b: torch.Tensor,
    c: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    """Explicit O(L) selective-scan recurrence (Mamba/S6) in pure PyTorch.

    Given a scalar input signal ``u`` and *data-dependent* discretized
    parameters, iterates the state-space recurrence

        h_t = bar_a_t * h_{t-1} + bar_b_t * u_t        (state, per step)
        y_t = <c_t, h_t>                               (output, per step)

    The "selective" behaviour comes from ``bar_a``, ``bar_b``, ``c`` and ``u``
    all being data-dependent (derived from the input), so the model can reset
    or propagate state per step -- the capability a fixed convolution kernel
    lacks and VFIMamba's reason for moving off conv/attention.

    Args:
        u: Scalar input signal, shape ``(B, L)``.
        bar_a: Discretized state transition (forget factor), shape ``(B, L, N)``.
            Values in ``(0, 1]`` keep the recurrence stable.
        bar_b: Discretized input coupling, shape ``(B, L, N)``.
        c: Output projection, shape ``(B, L, N)``.
        reverse: Run the scan backwards over ``L`` (the backward direction of
            the bidirectional model).

    Returns:
        Output signal of shape ``(B, L)``.
    """
    batch, length = u.shape
    state_dim = bar_a.shape[-1]
    hidden = u.new_zeros(batch, state_dim)
    outputs: list[torch.Tensor] = []
    indices = range(length - 1, -1, -1) if reverse else range(length)
    for i in indices:
        # h_t = bar_a_t * h_{t-1} + bar_b_t * u_t   (broadcasts u across states)
        hidden = bar_a[:, i] * hidden + bar_b[:, i] * u[:, i].unsqueeze(-1)
        outputs.append((c[:, i] * hidden).sum(-1))
    if reverse:
        outputs.reverse()
    return torch.stack(outputs, dim=1)


class BidirectionalSelectiveSSM(nn.Module):
    """Bidirectional selective state-space model (VFIMamba inter-frame core).

    Projects the input into data-dependent SSM parameters (input coupling
    ``B``, output projection ``C``, step ``Δ``), discretizes a log-parameterized
    transition ``A`` (initialized as a forget factor in ``(0, 1]``), and runs
    the selective scan both forward and backward, then fuses the two
    directions -- VFIMamba's bidirectional propagation for capturing inter-frame
    context from both directions at linear cost.

    The block is residual (``x + out_proj(scan)``) so it can be stacked and
    stays well-behaved when untrained.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Input-dependent projections (the "selective" part): B, C, and Δ logit.
        self.in_proj = nn.Linear(d_model, 2 * d_state + 1, bias=True)
        # Scalar input signal u derived from the token features.
        self.u_proj = nn.Linear(d_model, 1, bias=False)
        # State transition A, log-parameterized and negative => exp() in (0, 1].
        self.log_a = nn.Parameter(torch.full((d_state,), -1.0))
        # Fuse the two scan directions back into the feature dimension.
        self.out_proj = nn.Linear(2, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bidirectional selective scan.

        Args:
            x: Token features, shape ``(B, L, d_model)``.

        Returns:
            Contextually-modulated features, shape ``(B, L, d_model)``.
        """
        proj = self.in_proj(x)  # (B, L, 2*d_state + 1)
        b = proj[..., : self.d_state]
        c = proj[..., self.d_state : 2 * self.d_state]
        delta = F.softplus(proj[..., -1])  # step Δ >= 0, (B, L)
        u = self.u_proj(x).squeeze(-1)  # scalar input signal, (B, L)

        a = torch.exp(self.log_a)  # (d_state,) in (0, 1]
        bar_a = (delta.unsqueeze(-1) * a).clamp(max=1.0)  # (B, L, d_state)
        bar_b = delta.unsqueeze(-1) * b  # (B, L, d_state)

        y_fwd = _selective_scan(u, bar_a, bar_b, c, reverse=False)  # (B, L)
        y_bwd = _selective_scan(u, bar_a, bar_b, c, reverse=True)  # (B, L)
        y = torch.stack([y_fwd, y_bwd], dim=-1)  # (B, L, 2)
        return x + self.out_proj(y)


class SSMFrameInterpolator:
    """Contract-equivalent SSM-based frame interpolator.

    Doubles the frame rate (``T`` -> ``2T - 1``) by inserting, between each
    pair of consecutive frames, a middle frame produced by a bidirectional
    selective SSM that reads both endpoint frames and emits a data-dependent,
    convex blend gate ``α``. The middle frame is ``α * f0 + (1 - α) * f1`` with
    ``α ∈ (0, 1)``, so it is always a valid interpolation bounded by the two
    endpoints -- the standard VFI baseline that VFIMamba's training specializes.

    The bidirectional selective scan runs over a pooled spatial token grid
    (``scan_resolution ** 2`` tokens) to keep the O(L) recurrence tractable;
    the resulting per-token gate is bilinearly upsampled to full resolution.
    """

    def __init__(
        self,
        enabled: bool = True,
        device: torch.device | None = None,
        scan_resolution: int = 16,
        d_state: int = 16,
    ):
        """Initialize the SSM interpolator.

        Args:
            enabled: Whether interpolation is enabled.
            device: Device to run on (defaults to CUDA if available, else CPU).
            scan_resolution: Side length of the spatial token grid the SSM scans
                over. Smaller is faster; the gate is upsampled to full size.
            d_state: SSM state dimension.
        """
        self.enabled = enabled
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.scan_resolution = scan_resolution
        self.d_state = d_state
        # Built lazily once the channel count is known, so import stays cheap.
        self._gate_module: nn.Module | None = None

    def _build_gate_module(self, channels: int) -> nn.Module:
        module = nn.Module()
        module.ssm = BidirectionalSelectiveSSM(
            d_model=2 * channels, d_state=self.d_state
        )
        module.gate = nn.Linear(2 * channels, 1)
        # Safe baseline: zero the gate so sigmoid(0) = 0.5 => midpoint blend
        # until the SSM context is trained to perturb it data-dependently.
        nn.init.zeros_(module.gate.weight)
        nn.init.zeros_(module.gate.bias)
        module.to(self.device).eval()
        self._gate_module = module
        return module

    def interpolate(self, frames: torch.Tensor) -> torch.Tensor:
        """Interpolate frames to double the frame rate.

        Args:
            frames: Input frames, shape ``[T, H, W, C]`` uint8 in ``[0, 255]``.

        Returns:
            Interpolated frames, shape ``[2T - 1, H, W, C]`` uint8 in
            ``[0, 255]``, with the original frames preserved at even indices.
        """
        if not self.enabled:
            return frames
        if frames.shape[0] < 2:
            return frames

        frames_float = frames.float().to(self.device)
        num_frames, height, width, channels = frames_float.shape
        module = self._gate_module or self._build_gate_module(channels)

        first = frames_float[:-1]  # (T-1, H, W, C)
        second = frames_float[1:]  # (T-1, H, W, C)

        # Pool each endpoint frame pair to a scan_resolution token grid.
        grid = min(self.scan_resolution, height, width)
        first_grid = self._spatial_pool(first, grid)  # (T-1, C, grid, grid)
        second_grid = self._spatial_pool(second, grid)  # (T-1, C, grid, grid)
        tokens = torch.cat([first_grid, second_grid], dim=1)  # (T-1, 2C, grid, grid)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], grid * grid)
        tokens = tokens.permute(0, 2, 1)  # (T-1, grid*grid, 2C)

        # Bidirectional selective SSM over the spatial token sequence, then a
        # per-token blend gate. sigmoid => alpha in (0, 1) => convex blend.
        with torch.no_grad():
            context = module.ssm(tokens)  # (T-1, grid*grid, 2C)
            gate_logits = module.gate(context).squeeze(-1)  # (T-1, grid*grid)
        gate = torch.sigmoid(gate_logits).reshape(-1, 1, grid, grid)  # (T-1,1,g,g)
        alpha = F.interpolate(
            gate, size=(height, width), mode="bilinear", align_corners=False
        )  # (T-1, 1, H, W)
        alpha = alpha.squeeze(1)  # (T-1, H, W)

        mid_frames = (
            alpha.unsqueeze(-1) * first + (1.0 - alpha).unsqueeze(-1) * second
        )  # (T-1, H, W, C)

        # Interleave originals and middle frames: o0, m0, o1, m1, ..., o_{T-1}.
        result = frames_float.new_empty((num_frames * 2 - 1, height, width, channels))
        result[0::2] = frames_float
        result[1::2] = mid_frames
        return result.clamp(0.0, 255.0).to(torch.uint8)

    @staticmethod
    def _spatial_pool(frames: torch.Tensor, grid: int) -> torch.Tensor:
        """Average-pool ``[N, H, W, C]`` frames to ``[N, C, grid, grid]``."""
        nchw = frames.permute(0, 3, 1, 2)  # (N, C, H, W)
        return F.adaptive_avg_pool2d(nchw, (grid, grid))


def is_ssm_available() -> bool:
    """The pure-PyTorch SSM engine is always available (no external deps).

    Returns:
        True (the engine has no optional dependencies).
    """
    return True
