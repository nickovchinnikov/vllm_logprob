import torch

from src.statistics import compare_distributions, multinomial_baseline


def test_identical_distributions_have_zero_error():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)

    metrics = compare_distributions(probabilities, probabilities)

    assert metrics == {
        "tv_distance": 0.0,
        "max_abs_error": 0.0,
        "calibration_error": 0.0,
    }


def test_multinomial_baseline_is_reproducible_and_bounded():
    probabilities = torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    counts = torch.tensor([70, 20, 10])

    first = multinomial_baseline(probabilities, counts, replicates=20, seed=7)
    second = multinomial_baseline(probabilities, counts, replicates=20, seed=7)

    assert first == second
    assert 0 < first["tv_mc_pvalue"] <= 1
    assert 0 < first["chi_square_mc_pvalue"] <= 1