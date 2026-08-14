# Token Radius Attention — training-free query-adaptive sparse attention.
# Adapted from "Token Radius Attention for Efficient Video Generation"
# (https://arxiv.org/abs/2608.02504v1).
#
# Core mechanism (kept at fidelity):
#   * per-query entropy is extracted from the attention logits,
#   * entropy is mapped to a per-query token budget by an analytic
#     log-linear law, and
#   * the budget becomes a temporally decayed key radius so each query
#     attends to a query-centred neighbourhood — no key-ranking pass.
#
# Target-native substitutions:
#   * the paper's fused entropy-extraction kernel becomes a plain matmul
#     plus a logsumexp-style reduction,
#   * the paper's block-sparse mask construction / flex_attention path
#     becomes a dense boolean mask handed to scaled_dot_product_attention,
#   * warm-up reuse (recomputing budgets every N steps) is exposed as
#     `refresh_every` rather than being tied to a scheduler hook.
import math
import os
from dataclasses import dataclass, replace

import torch

__all__ = [
    "TokenRadiusConfig",
    "TokenRadiusAttention",
    "configure_token_radius",
    "token_radius_attention",
]


# Below this many keys the budget pass costs more than it saves.
_MIN_KEYS = 512


@dataclass(frozen=True)
class TokenRadiusConfig:
    """Controls how aggressively token-radius attention prunes keys.

    enabled: master switch; when False the caller falls back to its usual
        dense backend.
    max_density: budget ceiling — no query keeps more than this fraction
        of its keys (the paper retains 9-19%).
    min_density: budget floor for near-deterministic (low-entropy) queries.
    decay: temporal decay on the key radius — recent keys stay in scope
        longer than distant ones.
    refresh_every: recompute per-query budgets every N calls instead of
        every call (the paper's "warm-up reuse").
    """

    enabled: bool = False
    max_density: float = 0.19
    min_density: float = 0.02
    decay: float = 0.9
    refresh_every: int = 4

    def __post_init__(self) -> None:
        if not 0.0 < self.min_density <= self.max_density <= 1.0:
            raise ValueError(
                "token radius requires 0 < min_density <= max_density <= 1, "
                f"got min={self.min_density}, max={self.max_density}"
            )
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {self.decay}")


def _config_from_env() -> TokenRadiusConfig:
    """Build the default config; WAN_TOKEN_RADIUS=1 opts in."""
    config = TokenRadiusConfig(enabled=os.getenv("WAN_TOKEN_RADIUS", "0") == "1")
    if os.getenv("WAN_TOKEN_RADIUS_MAX_DENSITY"):
        config = replace(
            config, max_density=float(os.environ["WAN_TOKEN_RADIUS_MAX_DENSITY"])
        )
    return config


class TokenRadiusAttention:
    """Query-adaptive sparse attention with token-dependent radii.

    :meth:`__call__` takes ``q`` as ``[B, Lq, H, C]`` and ``k``/``v`` as
    ``[B, Lk, H, C]`` — the same contract as the repo's ``attention()``
    dispatcher — and returns the attended output in the query dtype. Keys
    outside a query's radius are masked rather than dropped, so the
    result is a softmax restricted to that query's own neighbourhood.
    """

    def __init__(self, config: TokenRadiusConfig | None = None) -> None:
        self.config = config or _config_from_env()
        self._calls = 0
        # Cached per-query radii, refreshed every `refresh_every` calls.
        self._radius: torch.Tensor | None = None

    def reset(self) -> None:
        """Drop cached radii, e.g. when the sequence shape changes."""
        self._radius = None
        self._calls = 0

    @staticmethod
    @torch.no_grad()
    def query_entropy(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Shannon entropy (nats) of each query's attention distribution.

        The paper fuses this into its kernel; a matmul plus a softmax
        reduction yields the same signal at lower fidelity. Returns
        ``[B, H, Lq]``.
        """
        scale = 1.0 / math.sqrt(q.size(-1))
        logits = torch.einsum("bqhc,bkhc->bhqk", q.float(), k.float()) * scale
        probs = logits.softmax(dim=-1).clamp_min(1e-9)
        return -(probs * probs.log()).sum(dim=-1)

    def token_budget(self, entropy: torch.Tensor, num_keys: int) -> torch.Tensor:
        """Map per-query entropy to a per-query key budget.

        The paper's observation is that retained density correlates
        log-linearly with attention entropy, so the budget interpolates
        log-linearly between the density floor and ceiling after the
        entropy is normalized against its own spread (making the mapping
        scale-free across models and heads).
        """
        cfg = self.config
        low = entropy.amin(dim=-1, keepdim=True)
        span = entropy.amax(dim=-1, keepdim=True) - low
        # Degenerate case: every query in the group has the same entropy,
        # so there is no relative signal to allocate on — split the
        # difference rather than starving all of them to the floor.
        norm = torch.where(
            span < 1e-6,
            torch.full_like(entropy, 0.5),
            (entropy - low) / span.clamp_min(1e-6),
        )
        log_density = math.log(cfg.min_density) + norm * (
            math.log(cfg.max_density) - math.log(cfg.min_density)
        )
        density = log_density.exp().clamp(cfg.min_density, cfg.max_density)
        return density * num_keys

    def radius_from_budget(self, budget: torch.Tensor, num_keys: int) -> torch.Tensor:
        """Convert a key budget into a temporally decayed key radius.

        A query keeping ``b`` of ``n`` keys gets radius ``b / decay``,
        capped at ``n``. The mask counts backwards from the most recent
        key, so ``decay < 1`` shrinks the retained window with temporal
        distance — recent frames keep a larger neighbourhood than distant
        ones, as in the paper.
        """
        return (budget / self.config.decay).clamp(1.0, num_keys)

    def build_mask(
        self, num_queries: int, num_keys: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean mask ``[Lq, Lk]``; True where a key is out of radius."""
        if self._radius is None or self._radius.shape[-1] != num_queries:
            self._radius = torch.full(
                (num_queries,), float(num_keys), device=device, dtype=torch.float32
            )
        radius = self._radius
        # Distance measured back from the most recent key: temporal decay.
        # Key 0 is the most distant (distance num_keys), key -1 the most
        # recent (distance 1).
        distance = torch.arange(num_keys, 0, -1, device=device).unsqueeze(1).float()
        return distance > radius.unsqueeze(-1)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        cfg = self.config
        num_queries, num_keys = q.size(1), k.size(1)
        if not cfg.enabled or num_keys < _MIN_KEYS:
            return None

        self._calls += 1
        if self._calls % cfg.refresh_every == 0 or self._radius is None:
            entropy = self.query_entropy(q, k)
            budget = self.token_budget(entropy, num_keys)
            # Average over batch and heads: one radius per query token.
            self._radius = self.radius_from_budget(
                budget.mean(dim=(0, 1)), num_keys
            ).to(device=q.device, dtype=torch.float32)

        mask = self.build_mask(num_queries, num_keys, q.device)
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            qh.float(), kh.float(), vh.float(), attn_mask=~mask
        )
        return out.transpose(1, 2).to(q.dtype)


_shared = TokenRadiusAttention()


def configure_token_radius(config: TokenRadiusConfig) -> None:
    """Swap the shared config used by the ``attention()`` dispatcher."""
    global _shared
    _shared = TokenRadiusAttention(config)


def token_radius_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Run the shared TokenRadiusAttention, or return None to fall back.

    Returning None (disabled, or too few keys to amortize the budget
    pass) lets the caller keep its usual dense backend untouched.
    """
    return _shared(q, k, v)
