# Sampling Fidelity Results

- vLLM: `0.11.0`
- Commit: `b8b302cde434df8c9289a2b465406b47ebab1c2d`
- Backend: `forward_native`
- Device: `cuda`
- Samples per seed: `1000`

The plots and table summarize fixed-logit controlled sampling. TV comparisons must be read alongside the multinomial null baseline; a small Monte Carlo p-value alone is not evidence of a practically important mismatch.

| ID | TV (mean +/- SD) | MC TV p-value | Max probability error | ECE | Raw logprob max error | Processed logprob max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | n/a | n/a | n/a | n/a | 2.25848e-06 | n/a |
| E2 | 0.0958798 +/- 0.00835 | 0.4212 | 0.0164049 | 8.14291e-07 | 2.25848e-06 | 2.25848e-06 |
| E3 | 0.0239125 +/- 0.00202 | 0.7877 | 0.00709383 | 2.9727e-07 | 2.25848e-06 | 2.75605e-06 |
| E4 | 0.382534 +/- 0.00766 | 0.2769 | 0.0145842 | 7.50361e-07 | 2.25848e-06 | 1.92406e-06 |
| E5 | 0.0392884 +/- 0.00544 | 0.4577 | 0.0164067 | 8.0658e-07 | 2.25848e-06 | 7.81435e-08 |
| E6 | 0 +/- 0 | 1 | 0 | 0 | 2.25848e-06 | 0 |
| E7 | 0.0958798 +/- 0.00835 | 0.4212 | 0.0164049 | 8.14291e-07 | 2.25848e-06 | 2.25848e-06 |
| E8 | 0.0934239 +/- 0.00776 | 0.4096 | 0.0164441 | 8.60932e-07 | 5.20593e-07 | 5.20593e-07 |
| E9 | 0.0958798 +/- 0.00835 | 0.4212 | 0.0164049 | 8.14291e-07 | 2.25848e-06 | 2.25848e-06 |
| E10 | 0.00386632 +/- 0.00278 | 0.7512 | 0.00386632 | 1.57321e-07 | 2.25848e-06 | 9.4732e-07 |

Figures: `figures/reference_vs_empirical.png`, `figures/temperature_impact.png`, `figures/probability_error.png`, and `figures/tv_by_configuration.png`.

The direct sampler bypasses the public inference API and the outer `Sampler.forward` metadata path. See [`vllm-semantics.md`](../vllm-semantics.md) for interpretation and limitations.
