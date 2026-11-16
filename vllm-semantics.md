# vLLM 0.11.0 Sampling Semantics

## Pinned implementation

- Release: `vllm==0.11.0`
- Git tag: [`v0.11.0`](https://github.com/vllm-project/vllm/tree/v0.11.0)
- Git commit: `b8b302cde434df8c9289a2b465406b47ebab1c2d`
- Main path: [`vllm/v1/sample/sampler.py`](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/sample/sampler.py)
- Filtering/sampling: [`vllm/v1/sample/ops/topk_topp_sampler.py`](https://github.com/vllm-project/vllm/blob/b8b302cde434df8c9289a2b465406b47ebab1c2d/vllm/v1/sample/ops/topk_topp_sampler.py)

## Order and returned values

For `raw_logprobs`, `Sampler.forward` applies `log_softmax(..., dtype=torch.float32)` to the incoming logits before converting logits to FP32 or applying sampling transformations. It then converts sampling logits to FP32, applies token masks and configured processors, and enters `Sampler.sample`.

The random path applies temperature, argmax-invariant processors, then top-k/top-p. The pinned top-p implementation sorts ascending, computes cumulative softmax, masks entries for which `cumulative_probability <= 1 - top_p`, and always retains the final sorted logit. This is equivalent to retaining the shortest descending-probability prefix whose cumulative mass reaches `top_p` when the cutoff is not tied and arithmetic uses the same precision. The FP64 oracle implements that descending-prefix definition independently. Near a numerical boundary, FP32 versus FP64 can alter support; when equal probabilities straddle the cutoff, token identity depends on sort tie-breaking. Tests compare reference and pinned-vLLM support for exact/adjacent FP32 cutoffs and varied distributions, and compare support size rather than token identity for a uniform tied cutoff. Softmax normalization after masking gives the sampling distribution. `processed_logprobs` is the log-softmax of the filtered logits before sampling.

The outer sampler defaults to `raw_logprobs`. `processed_logprobs` and `processed_logits` are produced only when explicitly selected. The direct adapter in this prototype calls `TopKTopPSampler` with `logprobs_mode="processed_logprobs"` and calls the pinned `Sampler.compute_logprobs` method for raw logprobs. It does not claim to test the complete `Sampler.forward` metadata/processor path.

## Greedy behavior

The greedy branch is `argmax`, not categorical sampling. In all-greedy batches, `Sampler.sample` returns the argmax immediately; with `raw_logprobs` selected, requested logprobs still come from the pre-processing raw vector. Therefore the greedy policy's probability distribution is a point mass at argmax, while raw model logprobs describe the model distribution. This prototype checks greedy token identity and does not apply goodness-of-fit tests to it.

## Backend

The `TopKTopPSampler` constructor chooses FlashInfer when available/enabled for CUDA unless intermediate processed logits/logprobs are requested. Selecting `processed_logprobs` disables that optimized path; this prototype asserts that the bound implementation is `forward_native` and records it in run metadata. Native sampling uses FP32 softmax and exponential-race sampling (`probs / exponential_noise`, then argmax). This is a controlled test of that vLLM implementation, not a comparison of native and FlashInfer backends.

## Scope and caveats

The direct adapter isolates temperature, top-p, FP32 sampler arithmetic, and sampling from repeated model inference. It bypasses `SamplingMetadata` and the outer `Sampler.forward`, so it does not exercise penalties, allowed-token masks, custom processors, or public-API logprob truncation/sentinels. Top-p ties at the cutoff can be sensitive to sort ordering; the reference follows the vLLM 0.11.0 lower-tail masking rule, with FP64 arithmetic, while vLLM uses FP32.

Implementation claims were checked against the pinned source files above. Run records must include the installed vLLM version/commit and backend; rerun the semantics review before changing the pin.