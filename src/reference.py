"""Independent FP64 probability calculations for vLLM sampling experiments."""

from __future__ import annotations

import math

import torch


def _as_logits(logits: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("logits must be a non-empty 1D vector")
    if torch.isnan(values).any() or torch.isposinf(values).any():
        raise ValueError("logits may contain finite values and -inf only")
    if torch.isneginf(values).all():
        raise ValueError("at least one logit must be finite")
    return values


def compute_raw_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return the unmodified softmax distribution in float64."""
    return torch.softmax(_as_logits(logits), dim=-1)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by a positive finite temperature in float64."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return _as_logits(logits) / temperature


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Retain the smallest descending-probability prefix reaching ``top_p``.

    Equal-probability tokens at the cutoff can have implementation-specific
    identities; only the retained probability mass is uniquely defined.
    """
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    values = _as_logits(logits)
    sorted_logits, sorted_indices = torch.sort(values, descending=True, stable=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    cutoff = torch.searchsorted(
        cumulative,
        torch.tensor(top_p, dtype=torch.float64),
        right=False,
    )
    retained_count = min(int(cutoff.item()) + 1, values.numel())
    retained = torch.zeros_like(values, dtype=torch.bool)
    retained[sorted_indices[:retained_count]] = True
    return values.masked_fill(~retained, -torch.inf)


def compute_processed_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply temperature, then top-p filtering and renormalize in FP64."""
    scaled = apply_temperature(logits, temperature)
    filtered = apply_top_p(scaled, top_p)
    return torch.softmax(filtered, dim=-1)