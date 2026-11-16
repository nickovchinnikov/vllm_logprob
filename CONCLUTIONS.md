# Sampling Fidelity in vLLM 0.11.0 — Initial Findings

## Objective

Evaluate whether the probability distributions reported by vLLM are numerically consistent with an independent reference implementation, and whether the native sampler empirically reproduces those distributions.

## Setup

- vLLM 0.11.0, pinned commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`
- CUDA, `forward_native` backend
- Fixed-logit controlled experiments
- 100,000 samples per seed
- Independent FP64 reference implementation
- Temperature, top-p, and input-precision variations
- Separate 10,000-completion public API validation

## Findings

**1. Logprob fidelity**

Raw and processed log probabilities closely match the independent reference implementation. Maximum absolute logprob errors remain below approximately \(5 \times 10^{-6}\) across the tested configurations, with no support mismatches.

**2. Empirical sampling fidelity**

Across 29 controlled seed runs, observed total variation distances remained below their respective 95th-percentile multinomial null baselines. Neither the TV nor chi-square Monte Carlo tests provided robust evidence of a mismatch between the empirical and reference distributions.

**3. Sampling transformations**

Temperature scaling and top-p truncation followed by renormalization behaved consistently with the reference implementation.

**4. Input precision**

FP16 and BF16 quantization changed the input logits, while the sampler operated in FP32. Empirical frequencies remained consistent with the corresponding quantized reference distributions. This validates input-precision sensitivity under an FP32 sampler, not low-precision sampler kernels.

**5. Public API**

A separate 10,000-completion validation through `LLM.generate()` also showed no detectable deviation from its reference comparison.

## Conclusion

No statistically convincing sampling-distribution mismatch was observed in the tested vLLM 0.11.0 native sampling path.

These results do not establish correctness across all backends, sampling configurations, hardware, or the complete inference pipeline. FlashInfer and broader logits distributions remain useful follow-up experiments.

The results suggest that the more fundamental question is not necessarily whether vLLM's sampler implements the wrong probabilities, but which probability distribution a reported logprob is intended to describe: the model's original distribution or the transformed sampling policy.