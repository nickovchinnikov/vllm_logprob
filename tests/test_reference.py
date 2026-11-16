import torch

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