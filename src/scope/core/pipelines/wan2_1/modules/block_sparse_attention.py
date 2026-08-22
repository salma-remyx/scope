"""Frozen block-sparse KV selection for the shared attention dispatcher.

Adapted from "LoSA: Near-Lossless Sparse Attention for Training-Free Video
Diffusion Acceleration" (arXiv:2608.12032). Rather than fixing a sparsity
ratio, the method fixes a *retained attention mass* threshold: at one dense
step the exact per-block attention masses are measured, and the smallest
key/value block set that meets the threshold is kept and frozen for the
remaining steps. Fidelity is therefore bounded by construction instead of
degrading as the speedup grows.

Two adaptations were made for this repo, whose realtime pipelines denoise a
handful of steps per generated frame rather than the 50-step offline sweeps
the paper targets:

* the frozen indices are reused for a sliding window of calls
  (``BLOCK_SPARSE_ATTN_REBUILD``) and re-measured afterwards, so the pattern
  tracks content as a streaming session evolves;
* the paper keeps an independent block set per (head, query block). The
  kernels behind :func:`attention` take a single KV set per call, so the
  per-query-block sets are collapsed into their union, ranked by total
  attention mass. The retained-mass guarantee is still measured per head and
  per query block when sizing that union.

The selection is purely a ``k``/``v`` rewrite -- queries are untouched and the
gathered blocks keep their temporal order, so the output contract of
:func:`attention` is unchanged whatever kernel ends up running.
"""

import os

import torch

__all__ = [
    "BlockSparseAttnState",
    "block_sparse_state",
    "reset_block_sparse_state",
    "select_block_indices",
    "block_sparse_gather",
]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class BlockSparseAttnState:
    """Holds the frozen block indices between attention calls.

    Indices are keyed on the call signature (batch, heads, block counts,
    device) so a shape change -- new frame length, new memory-bank size --
    forces a re-measure instead of gathering stale blocks. The rebuild check
    is idempotent, so no lock is needed if two threads race it: the loser
    simply overwrites an equally valid index set.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        retained_mass: float | None = None,
        block_size: int | None = None,
        rebuild_every: int | None = None,
        min_blocks: int | None = None,
    ) -> None:
        self.enabled = (
            _env_bool("ENABLE_BLOCK_SPARSE_ATTN") if enabled is None else enabled
        )
        self.retained_mass = (
            _env_float("BLOCK_SPARSE_ATTN_MASS", 0.99)
            if retained_mass is None
            else retained_mass
        )
        self.block_size = (
            _env_int("BLOCK_SPARSE_ATTN_BLOCK", 128)
            if block_size is None
            else block_size
        )
        self.rebuild_every = (
            _env_int("BLOCK_SPARSE_ATTN_REBUILD", 4)
            if rebuild_every is None
            else rebuild_every
        )
        self.min_blocks = (
            _env_int("BLOCK_SPARSE_ATTN_MIN_BLOCKS", 1)
            if min_blocks is None
            else min_blocks
        )
        self._indices: torch.Tensor | None = None
        self._signature: tuple | None = None
        self.calls_since_rebuild = 0

    def indices_for(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor | None:
        """Return frozen key-block indices for this call, re-measuring if stale.

        ``q`` is ``[B, Lq, H, C]`` and ``k`` is ``[B, Lk, H, C]``, the layout
        the shared ``attention`` dispatcher works in. Returns ``None`` when
        the sequence is too short to split into blocks, leaving the caller on
        the dense path.
        """
        signature = (q.size(0), q.size(2), q.size(1), k.size(1), str(q.device))
        if (
            self._indices is not None
            and signature == self._signature
            and self.calls_since_rebuild < self.rebuild_every
        ):
            self.calls_since_rebuild += 1
            return self._indices

        indices = select_block_indices(
            q,
            k,
            block_size=self.block_size,
            threshold=self.retained_mass,
            min_blocks=self.min_blocks,
        )
        if indices is None:
            # Too few blocks to sparsify -- remember the shape so a hot loop
            # of short calls does not re-attempt the split on every step.
            self._indices = None
            self._signature = signature
            return None
        self.calls_since_rebuild = 0
        self._indices = indices
        self._signature = signature
        return indices

    def reset(self) -> None:
        """Drop the frozen indices, forcing a re-measure on the next call."""
        self._indices = None
        self._signature = None
        self.calls_since_rebuild = 0


def _env_bool(name: str) -> bool:
    return os.getenv(name, "0") == "1"


_STATE = BlockSparseAttnState()


def block_sparse_state() -> BlockSparseAttnState:
    """The process-wide state the shared attention dispatcher reads from."""
    return _STATE


def reset_block_sparse_state() -> None:
    """Clear the frozen indices, e.g. when a session or pipeline reloads."""
    _STATE.reset()


def select_block_indices(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    threshold: float = 0.99,
    min_blocks: int = 1,
) -> torch.Tensor | None:
    """Measure exact block attention masses and keep the smallest sufficient set.

    Both tensors are split into ``block_size`` chunks; the trailing key tokens
    that do not fill a block are never considered for pruning (they are the
    most recent tokens, and always high mass). The measurement is the exact
    softmax over block-level scores, so no estimator is trained and no
    calibration pass is needed.

    Returns a ``[B, n_kept]`` long tensor of key-block indices in temporal
    order, or ``None`` if there is nothing to prune.
    """
    batch, seq_k, heads, dim = k.shape
    seq_q = q.size(1)
    n_key_blocks = seq_k // block_size
    n_query_blocks = seq_q // block_size
    if n_key_blocks <= min_blocks or n_query_blocks < 1:
        return None

    q_blocks = q[:, : n_query_blocks * block_size].reshape(
        batch, n_query_blocks, block_size, heads, dim
    )
    k_blocks = k[:, : n_key_blocks * block_size].reshape(
        batch, n_key_blocks, block_size, heads, dim
    )

    with torch.no_grad():
        # Block-level scores: [B, query blocks, key blocks, heads].
        scores = torch.einsum(
            "bqihc,bkihc->bqkh", q_blocks.float(), k_blocks.float()
        ) * (dim**-0.5)
        probs = scores.softmax(dim=2)

        # Per head and query block, how many top blocks reach the threshold.
        ranked = probs.sort(dim=2, descending=True).values
        retained = ranked.cumsum(dim=2) / probs.sum(dim=2, keepdim=True)
        needed = (retained < threshold).sum(dim=2) + 1
        per_query_kept = int(needed.max().item())
        per_query_kept = max(min(per_query_kept, n_key_blocks), min_blocks)

        # One KV set per call, so collapse the per-query-block selections into
        # their union and rank candidates by total mass across query blocks.
        chosen = probs.mean(dim=3).topk(per_query_kept, dim=2).indices
        union = torch.zeros(batch, n_key_blocks, dtype=torch.bool, device=k.device)
        union.scatter_(1, chosen.reshape(batch, -1), True)
        n_kept = max(int(union.sum(dim=1).max().item()), min_blocks)

        total_mass = probs.mean(dim=3).sum(dim=1)
        indices = total_mass.topk(n_kept, dim=1).indices.sort(dim=1).values

    return indices


def block_sparse_gather(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: BlockSparseAttnState | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Rewrite ``k``/``v`` down to the frozen block support, or return None.

    This is the whole runtime cost of the method on a warm step: one index
    lookup and one gather. The trailing key tokens are re-appended so recent
    frames are always attended to in full.
    """
    if state is None:
        state = _STATE

    if not state.enabled:
        return None

    indices = state.indices_for(q, k)
    if indices is None:
        return None

    batch, seq_k, heads, dim = k.shape
    block_size = state.block_size
    n_kept = indices.size(1)

    n_blocks = seq_k // block_size
    k_blocks = k[:, : n_blocks * block_size].reshape(
        batch, n_blocks, block_size, heads, dim
    )
    v_blocks = v[:, : n_blocks * block_size].reshape(
        batch, n_blocks, block_size, heads, dim
    )
    expand = (batch, n_kept, block_size, heads, dim)
    gather_at = indices[:, :, None, None, None].expand(*expand)
    k_kept = k_blocks.gather(1, gather_at).reshape(
        batch, n_kept * block_size, heads, dim
    )
    v_kept = v_blocks.gather(1, gather_at).reshape(
        batch, n_kept * block_size, heads, dim
    )

    tail = seq_k - (seq_k // block_size) * block_size
    if tail:
        return (
            torch.cat([k_kept, k[:, seq_k - tail :]], dim=1),
            torch.cat([v_kept, v[:, seq_k - tail :]], dim=1),
        )
    return k_kept, v_kept
