from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


PROJECT_SEEDS = ["seed_2", "seed_18", "seed_33", "seed_45", "seed_255"]
PROJECT_LOSSES = {
    "balanced": ["ce", "focal_loss", "label_smoothing_ce"],
    "imbalanced": [
        "ce",
        "class_balanced_loss",
        "focal_loss",
        "label_smoothing_ce",
        "weighted_ce",
    ],
}


def _loss_label(loss_name: str) -> str:
    return loss_name.replace("_", " ").title().replace("Ce", "CE")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0

    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _softmax_confidence(logits: Sequence[float]) -> tuple[int, float]:
    max_index = max(range(len(logits)), key=lambda index: logits[index])
    max_logit = logits[max_index]
    exp_sum = sum(math.exp(logit - max_logit) for logit in logits)
    return max_index, 1.0 / exp_sum if exp_sum else 0.0


def _logit_columns(fieldnames: Sequence[str] | None, logits_path: Path) -> list[str]:
    if fieldnames is None:
        raise ValueError(f"{logits_path} is empty.")

    columns = sorted(
        [column for column in fieldnames if column.startswith("logit_class_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not columns:
        raise ValueError(f"No logit_class_* columns found in {logits_path}")

    return columns


def compute_epoch_ece_rows(
    logits_path: str | Path,
    n_bins: int = 10,
) -> list[dict[str, float | int | str]]:
    """Compute Expected Calibration Error for each epoch in one test_logits.csv."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    logits_path = Path(logits_path)
    grouped_bins: dict[tuple[str, int], list[dict[str, float | int]]] = {}

    with logits_path.open(newline="") as file:
        reader = csv.DictReader(file)
        logit_columns = _logit_columns(reader.fieldnames, logits_path)

        for record in reader:
            criterion = record["criterion"]
            epoch = int(record["epoch"])
            true_label = int(record["true_label"])
            logits = [float(record[column]) for column in logit_columns]
            predicted_label, confidence = _softmax_confidence(logits)
            bin_index = min(n_bins - 1, int(confidence * n_bins))

            bins = grouped_bins.setdefault(
                (criterion, epoch),
                [
                    {"count": 0, "correct": 0, "confidence_sum": 0.0}
                    for _ in range(n_bins)
                ],
            )
            bins[bin_index]["count"] = int(bins[bin_index]["count"]) + 1
            bins[bin_index]["correct"] = int(bins[bin_index]["correct"]) + int(
                predicted_label == true_label
            )
            bins[bin_index]["confidence_sum"] = (
                float(bins[bin_index]["confidence_sum"]) + confidence
            )

    rows = []
    for criterion, epoch in sorted(grouped_bins, key=lambda item: (item[0], item[1])):
        bins = grouped_bins[(criterion, epoch)]
        total = sum(int(bin_data["count"]) for bin_data in bins)
        ece = 0.0

        for bin_data in bins:
            count = int(bin_data["count"])
            if not count or not total:
                continue

            accuracy = int(bin_data["correct"]) / count
            confidence = float(bin_data["confidence_sum"]) / count
            ece += (count / total) * abs(accuracy - confidence)

        rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "ece": ece,
                "count": total,
            }
        )

    return rows


def collect_split_seed_ece_rows(
    data_root: str | Path,
    split: str,
    losses: Sequence[str],
    seeds: Sequence[str],
    n_bins: int,
) -> list[dict[str, float | int | str]]:
    data_root = Path(data_root)
    rows = []
    missing_files = []

    for loss_name in losses:
        for seed in seeds:
            logits_path = data_root / split / loss_name / seed / "test_logits.csv"
            if not logits_path.exists():
                missing_files.append(logits_path)
                continue

            for row in compute_epoch_ece_rows(logits_path, n_bins=n_bins):
                row["split"] = split
                row["criterion"] = loss_name
                row["seed"] = seed
                rows.append(row)

    if missing_files:
        formatted_missing = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            f"Cannot compute {split} ECE values; missing files:\n{formatted_missing}"
        )

    return rows


def aggregate_ece_rows(
    seed_rows: Sequence[Mapping[str, float | int | str]],
    confidence_z: float = 1.96,
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int], list[Mapping[str, float | int | str]]] = {}
    for row in seed_rows:
        grouped.setdefault((str(row["criterion"]), int(row["epoch"])), []).append(row)

    aggregate_rows = []
    for criterion, epoch in sorted(grouped, key=lambda item: (item[0], item[1])):
        rows = grouped[(criterion, epoch)]
        values = [float(row["ece"]) for row in rows]
        mean = _mean(values)
        std = _sample_std(values)
        ci = confidence_z * std / math.sqrt(len(values)) if values else 0.0

        aggregate_rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "ece_mean": mean,
                "ece_std": std,
                "ece_ci_low": max(0.0, mean - ci),
                "ece_ci_high": mean + ci,
                "count": sum(int(row["count"]) for row in rows),
                "n_seeds": len(values),
            }
        )

    return aggregate_rows


def write_rows_csv(
    rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    fieldnames: Sequence[str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_split_ece(
    aggregate_rows: Sequence[Mapping[str, float | int | str]],
    output_path: str | Path,
    title: str,
    show: bool = False,
) -> None:
    if not aggregate_rows:
        raise ValueError(f"No ECE rows available for {title}.")

    fig, ax = plt.subplots(figsize=(10, 6))
    criteria = sorted({str(row["criterion"]) for row in aggregate_rows})

    for criterion in criteria:
        rows = sorted(
            [row for row in aggregate_rows if str(row["criterion"]) == criterion],
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in rows]
        means = [float(row["ece_mean"]) for row in rows]
        ci_low = [float(row["ece_ci_low"]) for row in rows]
        ci_high = [float(row["ece_ci_high"]) for row in rows]

        ax.plot(epochs, means, marker="o", linewidth=1.8, label=_loss_label(criterion))
        ax.fill_between(epochs, ci_low, ci_high, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def calculate_and_plot_project_ece(
    data_root: str | Path = "training_data",
    output_dir: str | Path = "plots/ece",
    seeds: Sequence[str] = PROJECT_SEEDS,
    n_bins: int = 10,
    confidence_z: float = 1.96,
    show: bool = False,
) -> dict[str, tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]]:
    output_dir = Path(output_dir)
    results = {}

    for split, losses in PROJECT_LOSSES.items():
        seed_rows = collect_split_seed_ece_rows(
            data_root=data_root,
            split=split,
            losses=losses,
            seeds=seeds,
            n_bins=n_bins,
        )
        aggregate_rows = aggregate_ece_rows(seed_rows, confidence_z=confidence_z)

        write_rows_csv(
            aggregate_rows,
            output_dir / f"{split}_ece.csv",
            fieldnames=[
                "criterion",
                "epoch",
                "ece_mean",
                "ece_std",
                "ece_ci_low",
                "ece_ci_high",
                "count",
                "n_seeds",
            ],
        )
        plot_split_ece(
            aggregate_rows,
            output_path=output_dir / f"{split}_ece.png",
            title=f"{split.title()} Loss Functions: Expected Calibration Error",
            show=show,
        )
        results[split] = (seed_rows, aggregate_rows)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate epoch-level Expected Calibration Error from test_logits.csv "
            "files, write one aggregate CSV per split, and plot one ECE curve "
            "comparison for balanced and imbalanced loss functions."
        )
    )
    parser.add_argument(
        "--data-root",
        default="training_data",
        help="Root folder containing balanced/ and imbalanced/ result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/ece",
        help="Directory where ECE CSVs and plots are saved.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of confidence bins for ECE. Defaults to 10.",
    )
    parser.add_argument(
        "--confidence-z",
        type=float,
        default=1.96,
        help="Z value for confidence intervals. Defaults to 1.96 for about 95%%.",
    )
    parser.add_argument("--show", action="store_true", help="Show plots after saving.")
    args = parser.parse_args()

    calculate_and_plot_project_ece(
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
        confidence_z=args.confidence_z,
        show=args.show,
    )


if __name__ == "__main__":
    main()
