"""Recompute multinomial null baselines without rerunning the vLLM sampler."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml

from src.reporting import generate_report
from src.statistics import multinomial_baseline, multinomial_null_distributions


def _write_summary(rows: list[dict[str, str]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh(
    results_dir: Path,
    config_path: Path,
    replicates: int | None = None,
    seed_base: int | None = None,
    experiment_ids: set[str] | None = None,
) -> None:
    summary_path = results_dir / "summary.csv"
    raw_dir = results_dir / "raw"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_indices = {
        item["id"]: index for index, item in enumerate(config["experiments"])
    }
    metadata_path = results_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_replicates = replicates or int(metadata.get("mc_replicates", 10000))
    actual_seed_base = seed_base if seed_base is not None else int(
        metadata.get("mc_seed_base", 1000003)
    )
    references = {}
    updated_by_experiment: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        experiment_id = row["experiment"]
        if not row.get("seed") or (
            experiment_ids is not None and experiment_id not in experiment_ids
        ):
            continue
        if experiment_id not in experiment_indices:
            raise ValueError(f"experiment {experiment_id} is absent from {config_path}")
        if experiment_id not in references:
            reported = torch.load(
                raw_dir / f"{experiment_id}_reported.pt",
                map_location="cpu",
                weights_only=True,
            )
            references[experiment_id] = reported["reference_probs"].to(torch.float64)
        probabilities = references[experiment_id]
        sample_seed = int(row["seed"])
        sample_count = int(row["samples"])
        mc_seed = actual_seed_base + experiment_indices[experiment_id] * 100003 + sample_seed
        token_ids = torch.load(
            raw_dir / f"{experiment_id}_seed{sample_seed}_samples.pt",
            map_location="cpu",
            weights_only=True,
        )
        counts = torch.bincount(token_ids, minlength=probabilities.numel())
        if counts.sum().item() != sample_count:
            raise ValueError(
                f"{experiment_id} seed {sample_seed}: expected {sample_count} samples, "
                f"found {counts.sum().item()}"
            )

        null_samples = multinomial_null_distributions(
            probabilities,
            sample_count=sample_count,
            replicates=actual_replicates,
            seed=mc_seed,
        )
        metrics = multinomial_baseline(
            probabilities, counts, null_samples=null_samples
        )
        row.update({key: str(value) for key, value in metrics.items()})
        row["mc_seed"] = str(mc_seed)
        torch.save(
            {"tv": null_samples[0], "chi_square": null_samples[1]},
            raw_dir / f"{experiment_id}_seed{sample_seed}_mc_null.pt",
        )
        updated_by_experiment.setdefault(experiment_id, []).append(row)
        _write_summary(rows, summary_path)
        print(f"{experiment_id} seed {sample_seed}: {actual_replicates} null replicates")

    for experiment_id, experiment_rows in updated_by_experiment.items():
        (raw_dir / f"{experiment_id}_metrics.json").write_text(
            json.dumps(experiment_rows, indent=2) + "\n", encoding="utf-8"
        )

    metadata["mc_replicates"] = actual_replicates
    metadata["mc_seed_base"] = actual_seed_base
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    generate_report(results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/controlled_v2"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--experiments", help="Comma-separated subset, e.g. E2,E4,E6")
    args = parser.parse_args()
    if args.replicates is not None and args.replicates < 1:
        parser.error("--replicates must be positive")
    experiment_ids = set(args.experiments.split(",")) if args.experiments else None
    refresh(
        args.results_dir,
        args.config,
        replicates=args.replicates,
        seed_base=args.seed_base,
        experiment_ids=experiment_ids,
    )


if __name__ == "__main__":
    main()