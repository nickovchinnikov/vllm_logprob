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
    """Return sampling-fidelity metrics that do not pool token categories."""
    expected = _distribution(reference, "reference")
    observed = _distribution(empirical, "empirical")
    if expected.shape != observed.shape:
        raise ValueError("reference and empirical must have the same shape")

    absolute_error = torch.abs(expected - observed)

    return {
        "tv_distance": 0.5 * absolute_error.sum().item(),
        "max_abs_error": absolute_error.max().item(),
    }


def token_bin_calibration_diagnostic(
    reference: torch.Tensor, empirical: torch.Tensor
) -> float:
    """Return the plan's token-probability-bin diagnostic, not event ECE.

    Token errors can cancel within a probability bin, so this value is kept
    separate from the primary sampling-fidelity metrics.
    """
    expected = _distribution(reference, "reference")
    observed = _distribution(empirical, "empirical")
    if expected.shape != observed.shape:
        raise ValueError("reference and empirical must have the same shape")

    bin_edges = torch.tensor([0.01, 0.05, 0.10, 0.25], dtype=torch.float64)
    bins = torch.bucketize(expected, bin_edges, right=True)
    diagnostic = 0.0
    for bin_id in range(bin_edges.numel() + 1):
        members = bins == bin_id
        if members.any():
            diagnostic += members.sum().item() / expected.numel() * torch.abs(
                expected[members].mean() - observed[members].mean()
            ).item()
    return diagnostic


def distribution_diagnostics(
    before_filter: torch.Tensor, after_filter: torch.Tensor
) -> dict[str, float | int]:
    """Summarize support, entropy, peak mass, and retained pre-filter mass."""
    before = _distribution(before_filter, "before_filter")
    after = _distribution(after_filter, "after_filter")
    if before.shape != after.shape:
        raise ValueError("before_filter and after_filter must have the same shape")
    retained = after > 0

    def entropy(probabilities: torch.Tensor) -> float:
        positive = probabilities[probabilities > 0]
        return -(positive * positive.log()).sum().item()

    return {
        "support_before_filter": int(torch.count_nonzero(before).item()),
        "support_after_filter": int(torch.count_nonzero(after).item()),
        "entropy_before_filter": entropy(before),
        "entropy_after_filter": entropy(after),
        "top_probability_after_filter": after.max().item(),
        "retained_pre_filter_mass": before[retained].sum().item(),
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
    null_samples: tuple[torch.Tensor, torch.Tensor] | None = None,
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

    if null_samples is None:
        tv_values, chi_square_values = multinomial_null_distributions(
            expected, sample_count, replicates, seed
        )
    else:
        tv_values = torch.as_tensor(null_samples[0], dtype=torch.float64, device="cpu")
        chi_square_values = torch.as_tensor(null_samples[1], dtype=torch.float64, device="cpu")
        if tv_values.shape != chi_square_values.shape or tv_values.numel() == 0:
            raise ValueError("null sample vectors must have matching non-empty shapes")
    actual_replicates = tv_values.numel()

    return {
        "tv_null_q95": torch.quantile(tv_values, 0.95).item(),
        "tv_mc_pvalue": (1 + torch.count_nonzero(tv_values >= observed_tv).item())
        / (actual_replicates + 1),
        "chi_square_mc_pvalue": (
            1 + torch.count_nonzero(chi_square_values >= observed_chi_square).item()
        )
        / (actual_replicates + 1),
    }


def multinomial_null_distributions(
    probabilities: torch.Tensor,
    sample_count: int,
    replicates: int = 200,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return simulated TV and Pearson-statistic null samples."""
    expected = _distribution(probabilities, "probabilities")
    if sample_count < 1 or replicates < 1:
        raise ValueError("sample_count and replicates must be positive")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tv_values = torch.empty(replicates, dtype=torch.float64)
    chi_square_values = torch.empty(replicates, dtype=torch.float64)
    for index in range(replicates):
        draws = torch.multinomial(expected, sample_count, replacement=True, generator=generator)
        counts = torch.bincount(draws, minlength=expected.numel())
        frequencies = counts.to(torch.float64) / sample_count
        tv_values[index] = 0.5 * torch.abs(expected - frequencies).sum()
        chi_square_values[index] = chi_square_statistic(expected, counts)
    return tv_values, chi_square_values