import torch
import pytest

from src.reference import (
    apply_temperature,
    apply_top_p,
    compute_processed_probs,
    compute_raw_probs,
)


LOGITS = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float64)


def test_raw_probs_are_normalized_and_nonnegative():
    probabilities = compute_raw_probs(LOGITS)

    assert torch.all(probabilities >= 0)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_identity_temperature_and_top_p_preserve_softmax():
    expected = compute_raw_probs(LOGITS)

    assert torch.allclose(apply_temperature(LOGITS, 1.0), LOGITS)
    assert torch.allclose(apply_top_p(LOGITS, 1.0), LOGITS)
    assert torch.allclose(compute_processed_probs(LOGITS), expected)


def test_top_p_masks_tail_and_renormalizes():
    probabilities = compute_processed_probs(LOGITS, top_p=0.8)

    assert probabilities[-1] == 0
    assert probabilities.sum().item() == 1.0
    assert torch.all(probabilities >= 0)


def test_temperature_changes_concentration_in_expected_direction():
    base = compute_processed_probs(LOGITS)
    cold = compute_processed_probs(LOGITS, temperature=0.7)
    hot = compute_processed_probs(LOGITS, temperature=1.5)

    assert cold.max() > base.max() > hot.max()


def test_invalid_temperature_and_top_p_are_rejected():
    for temperature in (0.0, -1.0, float("inf")):
        try:
            apply_temperature(LOGITS, temperature)
        except ValueError:
            pass
        else:
            raise AssertionError(f"temperature {temperature} should be rejected")

    for top_p in (0.0, -0.1, 1.1, float("nan")):
        try:
            apply_top_p(LOGITS, top_p)
        except ValueError:
            pass
        else:
            raise AssertionError(f"top_p {top_p} should be rejected")


def test_exact_and_adjacent_top_p_boundary():
    probabilities = torch.tensor([0.125, 0.375, 0.5], dtype=torch.float64)
    logits = probabilities.log()

    exact = apply_top_p(logits, 0.5)
    just_below = apply_top_p(logits, 0.5 - 1e-7)
    just_above = apply_top_p(logits, 0.5 + 1e-7)

    assert torch.count_nonzero(torch.isfinite(exact)) == 1
    assert torch.count_nonzero(torch.isfinite(just_below)) == 1
    assert torch.count_nonzero(torch.isfinite(just_above)) == 2


def test_uniform_tied_logits_have_minimal_prefix_support():
    filtered = apply_top_p(torch.zeros(8, dtype=torch.float64), 0.5)
    probabilities = torch.softmax(filtered, dim=-1)

    assert torch.count_nonzero(probabilities) == 4
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_top_p_is_stable_with_a_long_low_probability_tail():
    logits = torch.linspace(0.0, -50.0, 5000, dtype=torch.float64)
    probabilities = compute_processed_probs(logits, top_p=0.95)

    assert 0 < torch.count_nonzero(probabilities) < logits.numel()
    assert torch.isclose(
        probabilities.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12, rtol=1e-10
    )


@pytest.mark.parametrize("top_p", [0.5, 0.5000001, 0.9, 1.0])
def test_reference_support_matches_pinned_vllm_without_cutoff_ties(top_p):
    pytest.importorskip("vllm")
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    logits = torch.tensor([2.31, 1.27, 0.46, -0.83, -2.72], dtype=torch.float64)
    expected_support = torch.isfinite(apply_top_p(logits, top_p))
    actual = apply_top_k_top_p(
        logits.float().unsqueeze(0),
        None,
        torch.tensor([top_p], dtype=torch.float32),
    )[0]

    assert torch.equal(expected_support, torch.isfinite(actual))


def test_reference_and_vllm_uniform_cutoff_have_same_support_size():
    pytest.importorskip("vllm")
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    logits = torch.zeros(1, 8, dtype=torch.float32)
    expected = apply_top_p(logits[0].double(), 0.5)
    actual = apply_top_k_top_p(logits, None, torch.tensor([0.5]))[0]

    assert torch.count_nonzero(torch.isfinite(expected)) == 4
    assert torch.count_nonzero(torch.isfinite(actual)) == 4


def test_reference_support_matches_vllm_at_exact_and_adjacent_cutoffs():
    pytest.importorskip("vllm")
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    logits = torch.tensor([0.125, 0.375, 0.5], dtype=torch.float64).log()
    half = torch.tensor(0.5, dtype=torch.float32)
    probabilities = (
        0.5,
        float(torch.nextafter(half, torch.tensor(0.0, dtype=torch.float32))),
        float(torch.nextafter(half, torch.tensor(1.0, dtype=torch.float32))),
    )
    for top_p in probabilities:
        expected = apply_top_p(logits, top_p)
        actual = apply_top_k_top_p(
            logits.float().unsqueeze(0),
            None,
            torch.tensor([top_p], dtype=torch.float32),
        )[0]
        assert torch.equal(torch.isfinite(expected), torch.isfinite(actual))


@pytest.mark.parametrize("top_p", [0.5, 0.9])
@pytest.mark.parametrize("shape", ["near_uniform", "peaked", "long_tail"])
def test_reference_support_matches_vllm_across_logit_shapes(shape, top_p):
    pytest.importorskip("vllm")
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    if shape == "near_uniform":
        logits = torch.linspace(0.011, -0.027, 128, dtype=torch.float64)
    elif shape == "peaked":
        logits = torch.tensor([8.1, 2.4, 1.3, 0.1, -1.7, -4.2, -9.6])
    else:
        logits = torch.linspace(1.0, -32.0, 1024, dtype=torch.float64)

    expected_support = torch.isfinite(apply_top_p(logits, top_p))
    actual = apply_top_k_top_p(
        logits.float().unsqueeze(0),
        None,
        torch.tensor([top_p], dtype=torch.float32),
    )[0]

    assert torch.equal(expected_support, torch.isfinite(actual))