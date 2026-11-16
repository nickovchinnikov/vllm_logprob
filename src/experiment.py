"""Run the configured fixed-logit direct sampler experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import torch
import yaml

from src.reference import compute_processed_probs
from src.reporting import generate_report
from src.statistics import compare_distributions, multinomial_baseline
from src.vllm_adapter import VLLMDirectSampler

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _logprob_comparison(reference: torch.Tensor, reported: torch.Tensor) -> dict[str, float]:
    expected = reference.to(torch.float64)
    actual = reported.to(torch.float64)
    expected_support = torch.isfinite(expected)
    actual_support = torch.isfinite(actual)
    common = expected_support & actual_support
    error = torch.abs(expected[common] - actual[common])
    return {
        "max_abs_logprob_error": error.max().item() if error.numel() else 0.0,
        "mean_abs_logprob_error": error.mean().item() if error.numel() else 0.0,
        "support_mismatch_count": torch.count_nonzero(expected_support != actual_support).item(),
    }


def run(
    logits_path: Path,
    config_path: Path,
    output_dir: Path,
    device: str,
    sample_count: int | None = None,
) -> None:
    logits = torch.load(logits_path, map_location="cpu", weights_only=True).float()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["sampling"]
    count = sample_count or settings["samples_per_seed"]
    adapter = VLLMDirectSampler(device=device)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for experiment in config["experiments"]:
        experiment_id = experiment["id"]
        rounded_logits = logits.to(DTYPES[experiment["logits_dtype"]]).float()
        expected = compute_processed_probs(
            rounded_logits,
            temperature=experiment["temperature"],
            top_p=experiment["top_p"],
        )

        if experiment.get("greedy", False):
            greedy_token, raw_logprobs = adapter.greedy(rounded_logits)
            reference_raw_logprobs = torch.log(compute_processed_probs(rounded_logits))
            rows.append(
                {
                    "experiment": experiment_id,
                    "seed": "",
                    "samples": 1,
                    "greedy_token_id": greedy_token,
                    "reference_argmax_token_id": int(torch.argmax(expected).item()),
                    "greedy_correct": greedy_token == int(torch.argmax(expected).item()),
                    "raw_logprobs": _logprob_comparison(
                        reference_raw_logprobs, raw_logprobs
                    ),
                    "backend": "argmax",
                }
            )
            torch.save(
                {
                    "raw_logprobs": raw_logprobs,
                    "reference_probs": compute_processed_probs(rounded_logits),
                },
                raw_dir / f"{experiment_id}_reported.pt",
            )
            continue

        seed_results = []
        for seed in settings["seeds"]:
            sample_ids, raw_logprobs, processed_logprobs = adapter.sample(
                rounded_logits,
                temperature=experiment["temperature"],
                top_p=experiment["top_p"],
                sample_count=count,
                seed=seed,
                batch_size=settings["batch_size"],
            )
            counts = torch.bincount(sample_ids, minlength=logits.numel())
            empirical = counts.to(torch.float64) / count
            metrics = compare_distributions(expected, empirical)
            metrics.update(
                multinomial_baseline(
                    expected,
                    counts,
                    replicates=settings["mc_replicates"],
                    seed=seed,
                )
            )
            row = {
                "experiment": experiment_id,
                "seed": seed,
                "samples": count,
                "logits_dtype": experiment["logits_dtype"],
                "temperature": experiment["temperature"],
                "top_p": experiment["top_p"],
                "backend": adapter.backend,
                **metrics,
            }
            if not seed_results:
                expected_raw_logprobs = torch.log(compute_processed_probs(rounded_logits))
                reported_raw_logprobs = raw_logprobs
                expected_processed_logprobs = torch.where(
                    expected > 0, expected.log(), torch.full_like(expected, -torch.inf)
                )
                row.update(
                    {
                        "raw_logprobs": _logprob_comparison(
                            expected_raw_logprobs, reported_raw_logprobs
                        ),
                        "processed_logprobs": _logprob_comparison(
                            expected_processed_logprobs, processed_logprobs
                        ),
                    }
                )
                torch.save(
                    {
                        "raw_logprobs": raw_logprobs,
                        "processed_logprobs": processed_logprobs,
                        "reference_probs": expected,
                    },
                    raw_dir / f"{experiment_id}_reported.pt",
                )
            torch.save(sample_ids, raw_dir / f"{experiment_id}_seed{seed}_samples.pt")
            seed_results.append(row)
            rows.append(row)

        (raw_dir / f"{experiment_id}_metrics.json").write_text(
            json.dumps(seed_results, indent=2) + "\n", encoding="utf-8"
        )

    summary_path = output_dir / "summary.csv"
    fields = sorted({key for row in rows for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)
    run_metadata = {
        "vllm_version": "0.11.0",
        "vllm_commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
        "backend": adapter.backend,
        "device": str(adapter.device),
        "gpu_name": torch.cuda.get_device_name(adapter.device)
        if adapter.device.type == "cuda"
        else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "logits_file": str(logits_path),
        "model_metadata": json.loads(
            (logits_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        if (logits_path.parent / "metadata.json").exists()
        else None,
        "samples_per_seed": count,
        "seeds": settings["seeds"],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    generate_report(output_dir)
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=Path("data/model_logits/smollm2-135m/logits_fp32.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    run(args.logits, args.config, args.output_dir, args.device, args.samples)


if __name__ == "__main__":
    main()