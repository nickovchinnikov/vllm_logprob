"""Validate one-token sampling through vLLM's public offline LLM API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from src.reference import compute_processed_probs, compute_raw_probs
from src.statistics import (
    compare_distributions,
    distribution_diagnostics,
    multinomial_baseline,
    multinomial_null_distributions,
)


def run(
    model: str,
    revision: str,
    prompt: str,
    logits_path: Path,
    output_dir: Path,
    samples: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> None:
    if samples < 1:
        raise ValueError("samples must be positive")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import vllm
    from vllm import LLM, SamplingParams

    logits = torch.load(logits_path, map_location="cpu", weights_only=True).float()
    raw_reference = compute_raw_probs(logits)
    processed_reference = compute_processed_probs(
        logits, temperature=temperature, top_p=top_p
    )
    before_filter = compute_processed_probs(logits, temperature=temperature, top_p=1.0)

    llm = LLM(
        model=model,
        revision=revision,
        dtype="float32",
        seed=seed,
        max_model_len=128,
        max_num_seqs=128,
        enforce_eager=True,
        compilation_config=0,
        logprobs_mode="raw_logprobs",
        gpu_memory_utilization=0.5,
    )
    params = SamplingParams(
        n=samples,
        max_tokens=1,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        seed=seed,
        logprobs=1,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    request = llm.generate(prompt, params, use_tqdm=False)[0]
    token_ids = torch.tensor(
        [completion.token_ids[0] for completion in request.outputs], dtype=torch.int64
    )
    counts = torch.bincount(token_ids, minlength=logits.numel())
    empirical = counts.to(torch.float64) / token_ids.numel()
    metrics = compare_distributions(processed_reference, empirical)
    null_samples = multinomial_null_distributions(
        processed_reference, token_ids.numel(), replicates=200, seed=seed
    )
    metrics.update(
        multinomial_baseline(
            processed_reference, counts, null_samples=null_samples
        )
    )
    metrics.update(distribution_diagnostics(before_filter, processed_reference))

    reported_sampled_logprobs = []
    sampled_logprob_errors = []
    raw_reference_logprobs = torch.log(raw_reference)
    for token_id, completion in zip(token_ids.tolist(), request.outputs, strict=True):
        token_logprob = completion.logprobs[0][token_id].logprob
        reported_sampled_logprobs.append(token_logprob)
        sampled_logprob_errors.append(
            abs(token_logprob - raw_reference_logprobs[token_id].item())
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_ids": token_ids,
            "reported_raw_logprobs": torch.tensor(reported_sampled_logprobs),
        },
        output_dir / "sampled_tokens.pt",
    )
    torch.save(
        {"tv": null_samples[0], "chi_square": null_samples[1]},
        output_dir / "mc_null.pt",
    )
    result = {
        "validation": "public_offline_llm_generate",
        "model": model,
        "revision": revision,
        "prompt": prompt,
        "logits_file_for_reference_only": str(logits_path),
        "reference_logits_are_recomputed_by_vllm": True,
        "vllm_version": vllm.__version__,
        "logprobs_mode": "raw_logprobs",
        "sampler_backend_env": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
        "v1_multiprocessing_enabled": False,
        "samples": int(token_ids.numel()),
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "reference_comparison": metrics,
        "sampled_raw_logprob_max_abs_error": max(sampled_logprob_errors),
        "sampled_raw_logprob_mean_abs_error": sum(sampled_logprob_errors)
        / len(sampled_logprob_errors),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'result.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--revision", default="12fd25f77366fa6b3b4b768ec3050bf629380bac")
    parser.add_argument("--prompt", default="The sky appears blue because")
    parser.add_argument(
        "--logits", type=Path,
        default=Path("data/model_logits/smollm2-135m-fp32/logits_fp32.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/public_api"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    run(
        args.model,
        args.revision,
        args.prompt,
        args.logits,
        args.output_dir,
        args.samples,
        args.temperature,
        args.top_p,
        args.seed,
    )


if __name__ == "__main__":
    main()