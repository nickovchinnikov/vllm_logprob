import torch

from src.statistics import (
    compare_distributions,
    distribution_diagnostics,
    multinomial_baseline,
    multinomial_null_distributions,
    token_bin_calibration_diagnostic,
)


def test_identical_distributions_have_zero_error():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)

    metrics = compare_distributions(probabilities, probabilities)

    assert metrics == {
        "tv_distance": 0.0,
        "max_abs_error": 0.0,
    }


def test_token_bin_calibration_can_hide_error_by_cancellation():
    reference = torch.tensor([0.6, 0.25, 0.1, 0.05], dtype=torch.float64)
    empirical = torch.tensor([0.5, 0.35, 0.1, 0.05], dtype=torch.float64)

    assert abs(compare_distributions(reference, empirical)["tv_distance"] - 0.1) < 1e-12
    assert token_bin_calibration_diagnostic(reference, empirical) == 0.0


def test_distribution_diagnostics_detect_degenerate_top_p_support():
    before = torch.tensor([0.53, 0.30, 0.17], dtype=torch.float64)
    after = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    diagnostics = distribution_diagnostics(before, after)

    assert diagnostics["support_before_filter"] == 3
    assert diagnostics["support_after_filter"] == 1
    assert diagnostics["entropy_after_filter"] == 0.0
    assert diagnostics["top_probability_after_filter"] == 1.0
    assert diagnostics["retained_pre_filter_mass"] == 0.53


def test_multinomial_baseline_is_reproducible_and_bounded():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    counts = torch.tensor([70, 20, 10])

    first = multinomial_baseline(probabilities, counts, replicates=20, seed=7)
    second = multinomial_baseline(probabilities, counts, replicates=20, seed=7)

    assert first == second
    assert 0 < first["tv_mc_pvalue"] <= 1
    assert 0 < first["chi_square_mc_pvalue"] <= 1


def test_numpy_multinomial_null_is_reproducible_across_batch_sizes():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)

    first = multinomial_null_distributions(
        probabilities, sample_count=100, replicates=25, seed=7, batch_size=25
    )
    second = multinomial_null_distributions(
        probabilities, sample_count=100, replicates=25, seed=7, batch_size=5
    )

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_saved_null_samples_reproduce_baseline_metrics():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    counts = torch.tensor([70, 20, 10])
    null_samples = multinomial_null_distributions(
        probabilities, sample_count=100, replicates=20, seed=7
    )

    saved = multinomial_baseline(
        probabilities, counts, null_samples=null_samples
    )
    generated = multinomial_baseline(
        probabilities, counts, replicates=20, seed=7
    )

    assert saved == generated


def test_counts_reject_fractional_and_nonfinite_values():
    probabilities = torch.tensor([0.75, 0.25], dtype=torch.float64)

    for counts in (
        torch.tensor([7.5, 2.5]),
        torch.tensor([float("nan"), 2.0]),
        torch.tensor([-1, 11]),
    ):
        try:
            multinomial_baseline(probabilities, counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"counts {counts} should be rejected")


def test_null_samples_reject_nonfinite_and_negative_values():
    probabilities = torch.tensor([0.75, 0.25], dtype=torch.float64)
    counts = torch.tensor([7, 3])
    invalid = (
        (torch.tensor([float("nan")]), torch.tensor([1.0])),
        (torch.tensor([-0.1]), torch.tensor([1.0])),
        (torch.tensor([0.1]), torch.tensor([float("inf")])),
    )

    for null_samples in invalid:
        try:
            multinomial_baseline(probabilities, counts, null_samples=null_samples)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid null samples should be rejected")


def test_probability_sum_contract_uses_explicit_float64_tolerance():
    valid = torch.tensor([0.5, 0.5000000000004], dtype=torch.float64)
    invalid = torch.tensor([0.5, 0.50000001], dtype=torch.float64)

    assert compare_distributions(valid, valid)["tv_distance"] == 0
    try:
        compare_distributions(invalid / invalid.sum(), invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("unnormalized probability input should be rejected")