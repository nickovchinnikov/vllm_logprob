"""Generate compact plots and a report from experiment outputs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.reference import compute_processed_probs
from src.statistics import distribution_diagnostics


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
    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    logits_path = Path(metadata["logits_file"]) if metadata.get("logits_file") else None
    base_logits = (
        torch.load(logits_path, map_location="cpu", weights_only=True).float()
        if logits_path is not None and logits_path.exists()
        else None
    )
    mc_pvalues = [float(row["tv_mc_pvalue"]) for row in sample_rows if row.get("tv_mc_pvalue")]
    tv_exceedances = sum(
        float(row["tv_distance"]) > float(row["tv_null_q95"])
        for row in sample_rows
        if row.get("tv_distance") and row.get("tv_null_q95")
    )
    pvalues_below_threshold = sum(value < 0.05 for value in mc_pvalues)
    evidence_summary = (
        f"Across {len(mc_pvalues)} separate seed-runs, the minimum TV Monte Carlo p-value was "
        f"{min(mc_pvalues):.5f}; {pvalues_below_threshold} were below 0.05 and "
        f"{tv_exceedances} observed TV values exceeded their own null q95. "
        "These are unadjusted exploratory comparisons, not a proof of equal distributions."
        if mc_pvalues
        else "No random sampling runs are recorded."
    )

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

    report_lines = [
        "# Sampling Fidelity Results",
        "",
        f"- vLLM: `{metadata.get('vllm_version', 'not run')}`",
        f"- Commit: `{metadata.get('vllm_commit', 'not recorded')}`",
        f"- Backend: `{metadata.get('backend', 'not run')}`",
        f"- Device: `{metadata.get('device', 'not recorded')}`",
        f"- Samples per seed: `{metadata.get('samples_per_seed', 'not run')}`",
        f"- Monte Carlo replicates per seed-run: `{metadata.get('mc_replicates', 'not recorded')}`",
        "",
        "The table reports each seed separately. Read empirical TV against that seed's multinomial null baseline; p-values are not averaged. The token-bin calibration diagnostic is retained in the CSV but excluded here because token errors can cancel within bins.",
        "",
        evidence_summary,
        "",
        "| ID | Seed | N | TV | Null TV q95 | TV MC p | Chi-square MC p | "
        "Max probability error | Support | Entropy | Raw logprob max error | Processed logprob max error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        raw_logprobs = json.loads(row["raw_logprobs"]) if row.get("raw_logprobs") else {}
        processed_logprobs = (
            json.loads(row["processed_logprobs"])
            if row.get("processed_logprobs")
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
        support = row.get("support_after_filter", "n/a")
        entropy = row.get("entropy_after_filter", "n/a")
        if support not in ("", "n/a"):
            support = str(int(float(support)))
            entropy = f"{float(entropy):.5g}"
        values = [
            row.get("experiment", ""),
            row.get("seed") or "greedy",
            row.get("samples", ""),
            row.get("tv_distance", "n/a"),
            row.get("tv_null_q95", "n/a"),
            row.get("tv_mc_pvalue", "n/a"),
            row.get("chi_square_mc_pvalue", "n/a"),
            row.get("max_abs_error", "n/a"),
            support,
            entropy,
            raw_text,
            processed_text,
        ]
        report_lines.append("| " + " | ".join(str(value) for value in values) + " |")
    if base_logits is not None:
        dtype_by_name = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        top_p_rows = {}
        precision_rows = {}
        for row in rows:
            experiment_id = row.get("experiment", "")
            if row.get("top_p") and float(row["top_p"]) < 1 and experiment_id not in top_p_rows:
                values = base_logits.to(dtype_by_name[row["logits_dtype"]]).float()
                before = compute_processed_probs(
                    values, temperature=float(row["temperature"]), top_p=1.0
                )
                after = compute_processed_probs(
                    values,
                    temperature=float(row["temperature"]),
                    top_p=float(row["top_p"]),
                )
                top_p_rows[experiment_id] = (
                    distribution_diagnostics(before, after),
                    before.max().item(),
                    float(row["top_p"]),
                )
            if experiment_id in {"E7", "E8", "E9"} and experiment_id not in precision_rows:
                precision_rows[experiment_id] = row

        report_lines.extend(
            [
                "",
                "## Top-p Diagnostics",
                "",
                "| ID | top-p | Support before -> after | Entropy before -> after | Top token p before -> after | Retained pre-filter mass |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for experiment_id, (diagnostics, p_before, top_p) in top_p_rows.items():
            report_lines.append(
                f"| {experiment_id} | {top_p:g} | "
                f"{diagnostics['support_before_filter']} -> {diagnostics['support_after_filter']} | "
                f"{diagnostics['entropy_before_filter']:.5g} -> {diagnostics['entropy_after_filter']:.5g} | "
                f"{p_before:.6g} -> {diagnostics['top_probability_after_filter']:.6g} | "
                f"{diagnostics['retained_pre_filter_mass']:.6g} |"
            )

        report_lines.extend(
            [
                "",
                "## Input Precision",
                "",
                "| ID | Quantization dtype -> sampler dtype | Changed logits | Max absolute quantization delta |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for experiment_id, row in precision_rows.items():
            report_lines.append(
                f"| {experiment_id} | {row['logits_dtype']} -> {row['sampler_input_dtype']} | "
                f"{row['input_quantization_changed_logits']} | "
                f"{float(row['input_quantization_max_abs']):.6g} |"
            )
    report_lines.extend(
        [
            "",
            "Per-seed Monte Carlo samples are saved in `raw/*_mc_null.pt`. Figures: `figures/reference_vs_empirical.png`, `figures/temperature_impact.png`, `figures/probability_error.png`, and `figures/tv_by_configuration.png`.",
            "",
            "The direct sampler bypasses the public inference API and the outer `Sampler.forward` metadata path. See [`vllm-semantics.md`](../vllm-semantics.md) for interpretation and limitations.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")