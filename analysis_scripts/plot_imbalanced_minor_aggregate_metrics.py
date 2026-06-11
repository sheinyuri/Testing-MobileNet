from __future__ import annotations

import argparse
import csv
import heapq
import math
from pathlib import Path
from typing import Mapping, Sequence

from analysis_scripts.metric_plotting import (
    PROJECT_LOSSES,
    PROJECT_SEEDS,
    _aggregate_plot_metric_columns,
    _metric_columns,
    aggregate_fieldnames,
    aggregate_seed_metrics,
    plot_aggregate_metrics,
    plot_terminal_loss_comparison,
    write_rows_csv,
)


MINOR_CLASSES = (0, 1, 2)


def _normalise_class_counts(
    class_sample_counts: Mapping[int, int] | Sequence[int],
    class_labels: Sequence[int],
) -> dict[int, int]:
    if isinstance(class_sample_counts, Mapping):
        counts = {int(label): int(count) for label, count in class_sample_counts.items()}
    else:
        counts = {
            int(label): int(count)
            for label, count in zip(class_labels, class_sample_counts, strict=True)
        }

    missing = set(class_labels) - set(counts)
    if missing:
        raise ValueError(f"Missing class sample counts for labels: {sorted(missing)}")
    if any(counts[label] < 0 for label in class_labels):
        raise ValueError("Class sample counts must be non-negative.")
    if sum(counts[label] for label in class_labels) == 0:
        raise ValueError("At least one class sample count must be positive.")

    return counts


def _parse_counts(counts_text: str | None, class_labels: Sequence[int]) -> list[int]:
    if counts_text is None:
        return [500] * len(class_labels)

    counts = [int(value.strip()) for value in counts_text.split(",") if value.strip()]
    if len(counts) != len(class_labels):
        raise ValueError(
            f"Expected {len(class_labels)} class counts for labels {list(class_labels)}, "
            f"got {len(counts)}."
        )
    return counts


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_minor_confusion_metrics(
    results_folder: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    minor_classes: Sequence[int] = MINOR_CLASSES,
    confusion_filename: str = "testing_confusion_matrices.csv",
) -> list[dict[str, float | int | str]]:
    results_folder = Path(results_folder)
    confusion_path = results_folder / confusion_filename
    minor_labels = tuple(int(label) for label in minor_classes)
    minor_set = set(minor_labels)
    train_counts = _normalise_class_counts(class_sample_counts, minor_labels)
    total_train_count = sum(train_counts[label] for label in minor_labels)
    grouped_counts: dict[tuple[str, int], dict[tuple[int, int], float]] = {}

    with confusion_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            true_class = int(record["true_class"])
            if true_class not in minor_set:
                continue

            key = (record["criterion"], int(record["epoch"]))
            predicted_class = int(record["predicted_class"])
            count = float(record["count"])
            grouped_counts.setdefault(key, {})[(true_class, predicted_class)] = count

    rows = []
    for criterion, epoch in sorted(grouped_counts, key=lambda item: (item[0], item[1])):
        counts = grouped_counts[(criterion, epoch)]
        total = sum(counts.values())
        true_positive = {
            label: counts.get((label, label), 0.0) for label in minor_labels
        }
        predicted_total = {
            label: sum(counts.get((true_label, label), 0.0) for true_label in minor_labels)
            for label in minor_labels
        }
        actual_total = {
            label: sum(counts.get((label, predicted_label), 0.0) for predicted_label in range(10))
            for label in minor_labels
        }
        precision = {
            label: (
                true_positive[label] / predicted_total[label]
                if predicted_total[label]
                else 0.0
            )
            for label in minor_labels
        }
        recall = {
            label: true_positive[label] / actual_total[label] if actual_total[label] else 0.0
            for label in minor_labels
        }
        f1 = {
            label: (
                2 * precision[label] * recall[label] / (precision[label] + recall[label])
                if precision[label] + recall[label]
                else 0.0
            )
            for label in minor_labels
        }

        rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                "Accuracy": sum(true_positive.values()) / total if total else 0.0,
                "Macro Precision": sum(precision.values()) / len(minor_labels),
                "Macro Recall": sum(recall.values()) / len(minor_labels),
                "Macro F1": sum(f1.values()) / len(minor_labels),
                "Weighted F1": sum(
                    f1[label] * train_counts[label] for label in minor_labels
                )
                / total_train_count,
            }
        )

    return rows


def compute_minor_topk_metrics(
    results_folder: str | Path,
    k: int = 3,
    minor_classes: Sequence[int] = MINOR_CLASSES,
    logits_filename: str = "test_logits.csv",
) -> list[dict[str, float | int | str]]:
    results_folder = Path(results_folder)
    logits_path = results_folder / logits_filename
    minor_labels = tuple(int(label) for label in minor_classes)
    minor_set = set(minor_labels)
    grouped_hits: dict[tuple[str, int], dict[str, object]] = {}

    with logits_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{logits_path} is empty.")

        logit_columns = sorted(
            [column for column in reader.fieldnames if column.startswith("logit_class_")],
            key=lambda column: int(column.rsplit("_", 1)[1]),
        )
        if not logit_columns:
            raise ValueError(f"No logit_class_* columns found in {logits_path}")
        if k < 1 or k > len(logit_columns):
            raise ValueError(f"k must be between 1 and {len(logit_columns)}.")

        for record in reader:
            true_label = int(record["true_label"])
            if true_label not in minor_set:
                continue

            key = (record["criterion"], int(record["epoch"]))
            logits = [
                (float(record[column]), int(column.rsplit("_", 1)[1]))
                for column in logit_columns
            ]
            predicted_topk = {label for _, label in heapq.nlargest(k, logits)}
            hit = true_label in predicted_topk
            grouped = grouped_hits.setdefault(
                key,
                {
                    "hits": 0,
                    "total": 0,
                    "class_hits": {label: 0 for label in minor_labels},
                    "class_totals": {label: 0 for label in minor_labels},
                },
            )
            grouped["hits"] = int(grouped["hits"]) + int(hit)
            grouped["total"] = int(grouped["total"]) + 1
            grouped["class_hits"][true_label] += int(hit)
            grouped["class_totals"][true_label] += 1

    rows = []
    for criterion, epoch in sorted(grouped_hits, key=lambda item: (item[0], item[1])):
        grouped = grouped_hits[(criterion, epoch)]
        class_hits = grouped["class_hits"]
        class_totals = grouped["class_totals"]
        per_class_recall_at_k = [
            class_hits[label] / class_totals[label] if class_totals[label] else 0.0
            for label in minor_labels
        ]

        rows.append(
            {
                "criterion": criterion,
                "epoch": epoch,
                f"Top-{k} Acc": grouped["hits"] / grouped["total"]
                if grouped["total"]
                else 0.0,
                f"Macro Recall@{k}": sum(per_class_recall_at_k) / len(minor_labels),
            }
        )

    return rows


def compute_minor_metrics(
    results_folder: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    k: int = 3,
    minor_classes: Sequence[int] = MINOR_CLASSES,
) -> list[dict[str, float | int | str]]:
    confusion_rows = compute_minor_confusion_metrics(
        results_folder,
        class_sample_counts=class_sample_counts,
        minor_classes=minor_classes,
    )
    topk_rows = compute_minor_topk_metrics(
        results_folder,
        k=k,
        minor_classes=minor_classes,
    )
    topk_by_key = {(row["criterion"], row["epoch"]): row for row in topk_rows}

    merged_rows = []
    for confusion_row in confusion_rows:
        key = (confusion_row["criterion"], confusion_row["epoch"])
        if key in topk_by_key:
            merged_rows.append({**confusion_row, **topk_by_key[key]})

    return sorted(merged_rows, key=lambda row: (str(row["criterion"]), int(row["epoch"])))


def find_missing_imbalanced_files(
    data_root: str | Path,
    losses: Sequence[str],
    seeds: Sequence[str],
) -> list[Path]:
    data_root = Path(data_root)
    missing = []
    for loss_name in losses:
        for seed in seeds:
            seed_folder = data_root / "imbalanced" / loss_name / seed
            for filename in ("testing_confusion_matrices.csv", "test_logits.csv"):
                path = seed_folder / filename
                if not path.exists():
                    missing.append(path)
    return missing


def compute_imbalanced_minor_seed_metrics(
    data_root: str | Path,
    class_sample_counts: Mapping[int, int] | Sequence[int],
    losses: Sequence[str],
    seeds: Sequence[str],
    k: int = 3,
    minor_classes: Sequence[int] = MINOR_CLASSES,
) -> list[dict[str, float | int | str]]:
    missing = find_missing_imbalanced_files(data_root, losses, seeds)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot compute imbalanced minor-class aggregate metrics; missing files:\n"
            f"{formatted}"
        )

    rows = []
    data_root = Path(data_root)
    for loss_name in losses:
        for seed in seeds:
            results_folder = data_root / "imbalanced" / loss_name / seed
            for row in compute_minor_metrics(
                results_folder,
                class_sample_counts=class_sample_counts,
                k=k,
                minor_classes=minor_classes,
            ):
                row["criterion"] = loss_name
                row["seed"] = seed
                row["split"] = "imbalanced_minor"
                rows.append(row)

    return rows


def plot_imbalanced_minor_aggregate_metrics(
    data_root: str | Path = "training_data",
    output_dir: str | Path = "plots/aggregate_metrics/imbalanced_minor",
    minor_class_counts: Mapping[int, int] | Sequence[int] = (500, 500, 500),
    losses: Sequence[str] = PROJECT_LOSSES["imbalanced"],
    seeds: Sequence[str] = PROJECT_SEEDS,
    k: int = 3,
    confidence_z: float = 1.96,
    show: bool = False,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    metric_columns = _metric_columns(k)
    plot_metric_columns = _aggregate_plot_metric_columns(k)
    output_dir = Path(output_dir)
    seed_rows = compute_imbalanced_minor_seed_metrics(
        data_root=data_root,
        class_sample_counts=minor_class_counts,
        losses=losses,
        seeds=seeds,
        k=k,
    )
    aggregate_rows = aggregate_seed_metrics(
        seed_rows,
        metric_columns=metric_columns,
        confidence_z=confidence_z,
    )

    write_rows_csv(
        seed_rows,
        output_dir / "imbalanced_minor_seed_metrics.csv",
        fieldnames=["split", "criterion", "seed", "epoch", *metric_columns],
    )
    write_rows_csv(
        aggregate_rows,
        output_dir / "imbalanced_minor_aggregate_metrics.csv",
        fieldnames=aggregate_fieldnames(metric_columns),
    )

    for loss_name in losses:
        loss_rows = [row for row in aggregate_rows if str(row["criterion"]) == loss_name]
        plot_aggregate_metrics(
            loss_rows,
            output_path=output_dir / f"{loss_name}_minor_mean_ci.png",
            show=show,
            title=f"Imbalanced Minor Classes {loss_name}: Mean with 95% CI",
            metric_columns=plot_metric_columns,
            include_confidence_intervals=True,
            zoom_to_data=True,
        )

    plot_aggregate_metrics(
        aggregate_rows,
        output_path=output_dir / "imbalanced_minor_all_losses_mean.png",
        show=show,
        title="Imbalanced Minor Classes Loss Comparison: Mean with 95% CI",
        metric_columns=plot_metric_columns,
        include_confidence_intervals=True,
        zoom_to_data=True,
    )
    plot_terminal_loss_comparison(
        aggregate_rows,
        output_path=output_dir / "imbalanced_minor_terminal_loss_comparison.png",
        show=show,
        title="Imbalanced Minor Classes Terminal Loss Comparison",
        metric_columns=plot_metric_columns,
    )

    return seed_rows, aggregate_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregate metrics for imbalanced results using only minor-class "
            "examples whose true labels are 0, 1, or 2."
        )
    )
    parser.add_argument("--data-root", default="training_data")
    parser.add_argument(
        "--output-dir",
        default="plots/aggregate_metrics/imbalanced_minor",
    )
    parser.add_argument(
        "--minor-class-counts",
        default=None,
        help="Comma-separated training counts for classes 0,1,2. Defaults to 500,500,500.",
    )
    parser.add_argument("--seeds", nargs="+", default=PROJECT_SEEDS)
    parser.add_argument("--losses", nargs="+", default=PROJECT_LOSSES["imbalanced"])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--confidence-z", type=float, default=1.96)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    minor_counts = _parse_counts(args.minor_class_counts, MINOR_CLASSES)
    plot_imbalanced_minor_aggregate_metrics(
        data_root=args.data_root,
        output_dir=args.output_dir,
        minor_class_counts=minor_counts,
        losses=args.losses,
        seeds=args.seeds,
        k=args.k,
        confidence_z=args.confidence_z,
        show=args.show,
    )


if __name__ == "__main__":
    main()
