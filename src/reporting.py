"""Generate compact plots and a report from experiment outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_samples(raw_dir: Path, experiment_id: str, seed: str) -> torch.Tensor:
    return torch.load(
        raw_dir / f"{experiment_id}_seed{seed}_samples.pt",
        map_location="cpu",
        weights_only=True,
    )


def generate_report(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_summary(output_dir / "summary.csv")
    sample_rows = [row for row in rows if row.get("seed")]

    baseline = next((row for row in sample_rows if row["experiment"] == "E2"), None)
    if baseline is not None:
        experiment_id, seed = baseline["experiment"], baseline["seed"]
        stored = torch.load(
            raw_dir / f"{experiment_id}_reported.pt",
            map_location="cpu",
            weights_only=True,
        )
        reference = stored["reference_probs"].to(torch.float64)
        counts = torch.bincount(
            _load_samples(raw_dir, experiment_id, seed), minlength=reference.numel()
        )
        empirical = counts.to(torch.float64) / counts.sum()
        top_ids = torch.topk(reference, 20).indices
        positions = torch.arange(top_ids.numel()).numpy()

        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.bar(positions - 0.18, reference[top_ids].numpy(), 0.36, label="Reference")
        axis.bar(positions + 0.18, empirical[top_ids].numpy(), 0.36, label="Empirical")
        axis.set(xlabel="Token rank", ylabel="Probability", title="Reference vs empirical (E2)")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures_dir / "reference_vs_empirical.png", dpi=160)
        plt.close(figure)

        top_ids = torch.topk(reference, 100).indices
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.scatter(reference[top_ids].numpy(), empirical[top_ids].numpy(), s=14)
        limit = max(reference[top_ids].max().item(), empirical[top_ids].max().item())
        axis.plot([0, limit], [0, limit], color="black", linewidth=1)
        axis.set(xlabel="Reference probability", ylabel="Empirical frequency", title="Probability error (E2)")
        figure.tight_layout()
        figure.savefig(figures_dir / "probability_error.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for experiment_id, label in (("E3", "T=0.7"), ("E2", "T=1.0"), ("E4", "T=1.5")):
        path = raw_dir / f"{experiment_id}_reported.pt"
        if path.exists():
            reference = torch.load(path, map_location="cpu", weights_only=True)["reference_probs"]
            axis.plot(
                torch.arange(1, 101),
                torch.sort(reference, descending=True).values[:100].numpy(),
                label=label,
            )
    axis.set(xlabel="Token rank", ylabel="Probability", title="Temperature impact")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "temperature_impact.png", dpi=160)
    plt.close(figure)

    grouped_tv: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        if row.get("tv_distance"):
            grouped_tv[row["experiment"]].append(float(row["tv_distance"]))
    figure, axis = plt.subplots(figsize=(9, 4.5))
    experiment_ids = list(grouped_tv)
    means = [sum(grouped_tv[key]) / len(grouped_tv[key]) for key in experiment_ids]
    deviations = [
        (sum((value - means[index]) ** 2 for value in grouped_tv[key]) / len(grouped_tv[key])) ** 0.5
        for index, key in enumerate(experiment_ids)
    ]
    if experiment_ids:
        axis.bar(experiment_ids, means, yerr=deviations, capsize=3)
    axis.set(xlabel="Experiment", ylabel="TV distance", title="Empirical TV distance by configuration")
    figure.tight_layout()
    figure.savefig(figures_dir / "tv_by_configuration.png", dpi=160)
    plt.close(figure)

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    report_lines = [
        "# Sampling Fidelity Results",
        "",
        f"- vLLM: `{metadata.get('vllm_version', 'not run')}`",
        f"- Commit: `{metadata.get('vllm_commit', 'not recorded')}`",
        f"- Backend: `{metadata.get('backend', 'not run')}`",
        f"- Device: `{metadata.get('device', 'not recorded')}`",
        f"- Samples per seed: `{metadata.get('samples_per_seed', 'not run')}`",
        "",
        "The plots and table summarize fixed-logit controlled sampling. TV comparisons must be read alongside the multinomial null baseline; a small Monte Carlo p-value alone is not evidence of a practically important mismatch.",
        "",
        "| ID | TV (mean +/- SD) | MC TV p-value | Max probability error | ECE | "
        "Raw logprob max error | Processed logprob max error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_experiment_ids = ["E1", *experiment_ids]
    for experiment_id in all_experiment_ids:
        group = [row for row in sample_rows if row["experiment"] == experiment_id]
        if group:
            tv_mean = sum(float(row["tv_distance"]) for row in group) / len(group)
            tv_sd = deviations[experiment_ids.index(experiment_id)]
            p_values = [float(row["tv_mc_pvalue"]) for row in group]
            error_values = [float(row["max_abs_error"]) for row in group]
            calibration_values = [float(row["calibration_error"]) for row in group]
            tv_text = f"{tv_mean:.6g} +/- {tv_sd:.3g}"
            p_text = f"{sum(p_values) / len(p_values):.4g}"
            error_text = f"{sum(error_values) / len(error_values):.6g}"
            calibration_text = f"{sum(calibration_values) / len(calibration_values):.6g}"
        else:
            tv_text = p_text = error_text = calibration_text = "n/a"
        first_row = next((row for row in rows if row["experiment"] == experiment_id), {})
        raw_logprobs = json.loads(first_row["raw_logprobs"]) if first_row.get("raw_logprobs") else {}
        processed_logprobs = (
            json.loads(first_row["processed_logprobs"])
            if first_row.get("processed_logprobs")
            else {}
        )
        raw_error = raw_logprobs.get("max_abs_logprob_error", "n/a")
        processed_error = processed_logprobs.get("max_abs_logprob_error", "n/a")
        raw_text = f"{raw_error:.6g}" if isinstance(raw_error, (float, int)) else raw_error
        processed_text = (
            f"{processed_error:.6g}"
            if isinstance(processed_error, (float, int))
            else processed_error
        )
        report_lines.append(
            f"| {experiment_id} | {tv_text} | {p_text} | {error_text} | "
            f"{calibration_text} | {raw_text} | {processed_text} |"
        )
    report_lines.extend(
        [
            "",
            "Figures: `figures/reference_vs_empirical.png`, `figures/temperature_impact.png`, `figures/probability_error.png`, and `figures/tv_by_configuration.png`.",
            "",
            "The direct sampler bypasses the public inference API and the outer `Sampler.forward` metadata path. See [`vllm-semantics.md`](../vllm-semantics.md) for interpretation and limitations.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")