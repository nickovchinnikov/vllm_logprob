"""Distribution metrics and multinomial baselines."""

from __future__ import annotations

import torch


def _distribution(values: torch.Tensor, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector")
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not torch.isclose(result.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise ValueError(f"{name} must sum to 1")
    return result


def compare_distributions(
    reference: torch.Tensor, empirical: torch.Tensor
) -> dict[str, float]:
    """Return TV distance, maximum absolute error, and calibration-style ECE."""
    expected = _distribution(reference, "reference")
    observed = _distribution(empirical, "empirical")
    if expected.shape != observed.shape:
        raise ValueError("reference and empirical must have the same shape")

    absolute_error = torch.abs(expected - observed)
    bin_edges = torch.tensor([0.01, 0.05, 0.10, 0.25], dtype=torch.float64)
    bins = torch.bucketize(expected, bin_edges, right=True)
    ece = 0.0
    for bin_id in range(bin_edges.numel() + 1):
        members = bins == bin_id
        if members.any():
            ece += members.sum().item() / expected.numel() * torch.abs(
                expected[members].mean() - observed[members].mean()
            ).item()

    return {
        "tv_distance": 0.5 * absolute_error.sum().item(),
        "max_abs_error": absolute_error.max().item(),
        "calibration_error": ece,
    }


def chi_square_statistic(
    probabilities: torch.Tensor, counts: torch.Tensor
) -> float:
    """Calculate Pearson's statistic, retaining all categories."""
    expected_probs = _distribution(probabilities, "probabilities")
    observed = torch.as_tensor(counts, dtype=torch.float64, device="cpu")
    if observed.shape != expected_probs.shape or torch.any(observed < 0):
        raise ValueError("counts must be nonnegative and match probabilities")
    expected = expected_probs * observed.sum()
    impossible = (expected == 0) & (observed > 0)
    if impossible.any():
        return float("inf")
    included = expected > 0
    return torch.sum((observed[included] - expected[included]).square() / expected[included]).item()


def multinomial_baseline(
    probabilities: torch.Tensor,
    observed_counts: torch.Tensor,
    replicates: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate TV and chi-square null distributions by multinomial simulation."""
    expected = _distribution(probabilities, "probabilities")
    observed = torch.as_tensor(observed_counts, dtype=torch.int64, device="cpu")
    if observed.shape != expected.shape or torch.any(observed < 0):
        raise ValueError("observed_counts must be nonnegative and match probabilities")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    sample_count = int(observed.sum().item())
    if sample_count == 0:
        raise ValueError("observed_counts must contain at least one sample")
    observed_tv = compare_distributions(
        expected, observed.to(torch.float64) / sample_count
    )["tv_distance"]
    observed_chi_square = chi_square_statistic(expected, observed)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tv_values = torch.empty(replicates, dtype=torch.float64)
    chi_square_values = torch.empty(replicates, dtype=torch.float64)
    for index in range(replicates):
        draws = torch.multinomial(expected, sample_count, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=expected.numel())
        frequencies = counts.to(torch.float64) / sample_count
        tv_values[index] = 0.5 * torch.abs(expected - frequencies).sum()
        chi_square_values[index] = chi_square_statistic(expected, counts)

    return {
        "tv_null_q95": torch.quantile(tv_values, 0.95).item(),
        "tv_mc_pvalue": (1 + torch.count_nonzero(tv_values >= observed_tv).item())
        / (replicates + 1),
        "chi_square_mc_pvalue": (
            1 + torch.count_nonzero(chi_square_values >= observed_chi_square).item()
        )
        / (replicates + 1),
    }