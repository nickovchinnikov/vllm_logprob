# vLLM Sampling Distribution Fidelity

A small controlled experiment separating model inference from next-token sampling. It stores one fixed next-token logit vector, computes independent FP64 probabilities, samples that vector with vLLM 0.11.0's native top-k/top-p sampler, and reports reference/logprob/empirical comparisons with a multinomial noise baseline.

## Environment

Use Python 3.11 and an NVIDIA GPU. vLLM is pinned to `0.11.0`; its source commit and sampler semantics are recorded in [vllm-semantics.md](vllm-semantics.md). The vLLM direct sampler is an internal API, so changing the version requires rechecking that note and adapter.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Capture the next-token logits once. The configured SmolLM2 revision is immutable; its resolved revision and token IDs are saved with the logits. FP32 model inference is intentional so E7/E8 can measure nonzero FP16/BF16 input quantization before the sampler converts logits to FP32.

```bash
python -m src.model
```

The saved `logits_fp32.pt` is the fixed input for every configuration. A quick integration pass can reduce the sample count; the configured experiment defaults to 100,000 samples for each of three seeds per random configuration.

```bash
python -m src.experiment --samples 1000
python -m src.experiment
```

Outputs include token IDs and reported vectors under `results/controlled_v2/raw/`, per-seed metrics in `results/controlled_v2/summary.csv`, plots under `results/controlled_v2/figures/`, and a generated report. Configurations E1-E10 are in [configs/experiments.yaml](configs/experiments.yaml). The sample count override is intended only for smoke testing, not final results. Long runs can be split with `--experiments` and continued using `--resume`.

Run the separate public inference-path check after the controlled sampler run:

```bash
python -m src.public_api_validation --samples 10000
```

It uses `LLM.generate` with one prompt and multiple one-token completions, saves sampled-token logprobs and its own Monte Carlo baseline under `results/public_api/`. The public API recomputes logits, so this is an end-to-end validation against the captured FP32 reference, not a same-logits sampler test. Per-seed null samples for the controlled run are saved beside each sample set.

## What is and is not tested

The reference path applies FP64 softmax, temperature, and the vLLM 0.11.0 top-p lower-tail rule. The direct adapter batches identical logits, requests `processed_logprobs`, and asserts the PyTorch-native backend. FP16/BF16 experiments quantize the FP32 model logits, then pass the quantized values back as FP32, matching the outer sampler's explicit FP32 conversion. The run records quantization deltas and actual sampler input dtype. Raw logprobs are calculated with the exact FP32 log-softmax operation identified in the pinned outer sampler. Greedy is checked as argmax, not treated as categorical sampling.

This MVP does not exercise public `LLM.generate`, the full outer sampler metadata/processors, FlashInfer, penalties, speculative decoding, or API-level full-vocabulary logprob serialization. It does not compare the RNG bitstreams across implementations; it compares distributions. `use_fp64_gumbel` is not part of this native exponential-race sampler path.

## Tests

```bash
python -m pytest -q
```

Unit tests cover the FP64 transforms and metrics. The model capture and vLLM sampling steps require the dependencies, model download, and GPU described above.