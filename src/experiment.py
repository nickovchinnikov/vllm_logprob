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
from src.statistics import (
    compare_distributions,
    distribution_diagnostics,
    multinomial_baseline,
    multinomial_null_distributions,
    token_bin_calibration_diagnostic,
)
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


def _write_summary(rows: list[dict], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {
                key: json.dumps(value) if isinstance(value, dict) else value
                for key, value in row.items()
            }
            writer.writerow(flattened)


def run(
    logits_path: Path,
    config_path: Path,
    output_dir: Path,
    device: str,
    sample_count: int | None = None,
    experiment_ids: set[str] | None = None,
    resume: bool = False,
) -> None:
    logits = torch.load(logits_path, map_location="cpu", weights_only=True).float()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = config["sampling"]
    count = sample_count if sample_count is not None else settings["samples_per_seed"]
    if count < 1:
        raise ValueError("sample count must be positive")
    adapter = VLLMDirectSampler(device=device)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_path = output_dir / "summary.csv"
    if resume and summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    known_ids = {experiment["id"] for experiment in config["experiments"]}
    if experiment_ids is not None and not experiment_ids <= known_ids:
        raise ValueError(f"unknown experiment IDs: {sorted(experiment_ids - known_ids)}")

    for experiment_index, experiment in enumerate(config["experiments"]):
        experiment_id = experiment["id"]
        if experiment_ids is not None and experiment_id not in experiment_ids:
            continue
        rounded_logits = logits.to(DTYPES[experiment["logits_dtype"]]).float()
        quantization_delta = torch.abs(logits - rounded_logits)
        changed_logit_count = int(torch.count_nonzero(quantization_delta).item())
        if experiment.get("require_quantization_change", False) and changed_logit_count == 0:
            raise RuntimeError(
                f"{experiment_id} produced no input quantization change; "
                "capture FP32 model logits before running precision experiments"
            )
        expected = compute_processed_probs(
            rounded_logits,
            temperature=experiment["temperature"],
            top_p=experiment["top_p"],
        )

        if experiment.get("greedy", False):
            greedy_token, raw_logprobs = adapter.greedy(rounded_logits)
            reference_raw_logprobs = torch.log(compute_processed_probs(rounded_logits))
            greedy_row = {
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
            if resume:
                rows = [row for row in rows if row.get("experiment") != experiment_id]
            rows.append(greedy_row)
            torch.save(
                {
                    "raw_logprobs": raw_logprobs,
                    "reference_probs": compute_processed_probs(rounded_logits),
                },
                raw_dir / f"{experiment_id}_reported.pt",
            )
            continue

        before_filter = compute_processed_probs(
            rounded_logits, temperature=experiment["temperature"], top_p=1.0
        )
        diagnostics = distribution_diagnostics(before_filter, expected)
        minimum_support = experiment.get("minimum_support_after_filter", 1)
        if diagnostics["support_after_filter"] < minimum_support:
            raise RuntimeError(
                f"{experiment_id} retained only {diagnostics['support_after_filter']} tokens; "
                f"at least {minimum_support} are required"
            )
        experiment_seeds = experiment.get("seeds", settings["seeds"])
        seed_results = []
        for seed in experiment_seeds:
            mc_seed = (
                settings.get("mc_seed_base", 1000003)
                + experiment_index * 100003
                + int(seed)
            )
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
            null_samples = multinomial_null_distributions(
                expected,
                sample_count=count,
                replicates=settings["mc_replicates"],
                seed=mc_seed,
                batch_size=settings.get("mc_batch_size", 128),
            )
            metrics.update(
                multinomial_baseline(
                    expected, counts, null_samples=null_samples
                )
            )
            metrics["token_bin_calibration_diagnostic"] = (
                token_bin_calibration_diagnostic(expected, empirical)
            )
            metrics.update(diagnostics)
            row = {
                "experiment": experiment_id,
                "seed": seed,
                "mc_seed": mc_seed,
                "samples": count,
                "logits_dtype": experiment["logits_dtype"],
                "temperature": experiment["temperature"],
                "top_p": experiment["top_p"],
                "sampler_input_dtype": adapter.sampler_input_dtype,
                "input_quantization_max_abs": quantization_delta.max().item(),
                "input_quantization_changed_logits": changed_logit_count,
                "backend": adapter.backend,
                **metrics,
            }
            expected_raw_logprobs = torch.log(compute_processed_probs(rounded_logits))
            expected_processed_logprobs = torch.where(
                expected > 0, expected.log(), torch.full_like(expected, -torch.inf)
            )
            row.update(
                {
                    "raw_logprobs": _logprob_comparison(
                        expected_raw_logprobs, raw_logprobs
                    ),
                    "processed_logprobs": _logprob_comparison(
                        expected_processed_logprobs, processed_logprobs
                    ),
                }
            )
            if not seed_results:
                torch.save(
                    {
                        "raw_logprobs": raw_logprobs,
                        "processed_logprobs": processed_logprobs,
                        "reference_probs": expected,
                    },
                    raw_dir / f"{experiment_id}_reported.pt",
                )
            torch.save(sample_ids, raw_dir / f"{experiment_id}_seed{seed}_samples.pt")
            torch.save(
                {"tv": null_samples[0], "chi_square": null_samples[1]},
                raw_dir / f"{experiment_id}_seed{seed}_mc_null.pt",
            )
            seed_results.append(row)
            rows.append(row)
            if resume:
                rows = [
                    existing
                    for existing in rows
                    if not (
                        existing.get("experiment") == experiment_id
                        and str(existing.get("seed", "")) == str(seed)
                    )
                ] + [row]
                _write_summary(rows, summary_path)

        (raw_dir / f"{experiment_id}_metrics.json").write_text(
            json.dumps(seed_results, indent=2) + "\n", encoding="utf-8"
        )

    _write_summary(rows, summary_path)
    run_metadata = {
        "vllm_version": "0.11.0",
        "vllm_commit": "b8b302cde434df8c9289a2b465406b47ebab1c2d",
        "backend": adapter.backend,
        "raw_logprobs_source": "vllm.v1.sample.sampler.Sampler.compute_logprobs",
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
        "mc_replicates": settings["mc_replicates"],
        "mc_seed_base": settings.get("mc_seed_base", 1000003),
        "seeds": {item["id"]: item.get("seeds", settings["seeds"]) for item in config["experiments"]},
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    generate_report(output_dir)
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logits", type=Path, default=Path("data/model_logits/smollm2-135m-fp32/logits_fp32.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled_v2"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--experiments", help="Comma-separated subset, e.g. E2,E3,E4")
    parser.add_argument("--resume", action="store_true", help="Append/replace selected seed rows in the existing summary")
    args = parser.parse_args()
    selected = set(args.experiments.split(",")) if args.experiments else None
    run(
        args.logits,
        args.config,
        args.output_dir,
        args.device,
        args.samples,
        selected,
        args.resume,
    )


if __name__ == "__main__":
    main()