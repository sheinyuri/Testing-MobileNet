from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt


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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _loss_label(loss_name: str) -> str:
    return loss_name.replace("_", " ").title().replace("Ce", "CE")


def read_normalized_loss_curve(loss_path: str | Path) -> list[tuple[int, float]]:
    """Read one training_losses.csv and normalize first loss to 1, final loss to 0."""
    loss_path = Path(loss_path)
    rows = []

    with loss_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for record in reader:
            rows.append((int(record["epoch"]), float(record["training_loss"])))

    if not rows:
        raise ValueError(f"{loss_path} has no loss rows.")

    rows.sort(key=lambda row: row[0])
    start_loss = rows[0][1]
    end_loss = rows[-1][1]
    loss_range = start_loss - end_loss
    if loss_range == 0:
        raise ValueError(
            f"{loss_path} has the same start and end loss, so it cannot be normalized."
        )

    return [(epoch, (loss - end_loss) / loss_range) for epoch, loss in rows]


def collect_split_curves(
    data_root: str | Path,
    split: str,
    losses: Sequence[str],
) -> dict[str, dict[int, float]]:
    """Return mean normalized loss by loss function and epoch for one split."""
    data_root = Path(data_root)
    split_curves = {}

    for loss_name in losses:
        loss_dir = data_root / split / loss_name
        if not loss_dir.exists():
            continue

        epoch_values: dict[int, list[float]] = defaultdict(list)
        for loss_path in sorted(loss_dir.glob("seed_*/training_losses.csv")):
            for epoch, normalized_loss in read_normalized_loss_curve(loss_path):
                epoch_values[epoch].append(normalized_loss)

        if epoch_values:
            split_curves[loss_name] = {
                epoch: _mean(values) for epoch, values in sorted(epoch_values.items())
            }

    return split_curves


def plot_normalized_loss_curves(
    split_curves: Mapping[str, Mapping[int, float]],
    output_path: str | Path,
    title: str,
    show: bool = False,
) -> None:
    if not split_curves:
        raise ValueError(f"No normalized loss curves available for {title}.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for loss_name, epoch_to_loss in split_curves.items():
        epochs = sorted(epoch_to_loss)
        ax.plot(
            epochs,
            [epoch_to_loss[epoch] for epoch in epochs],
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            label=_loss_label(loss_name),
        )

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Training Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot normalized training loss curves. Each seed curve is scaled as "
            "(loss - end_loss) / (start_loss - end_loss), so every curve starts "
            "at 1 and ends at 0."
        )
    )
    parser.add_argument(
        "--data-root",
        default="training_data",
        help="Root folder containing balanced/ and imbalanced/ results.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/normalized_loss",
        help="Directory where balanced and imbalanced plots are saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot windows after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    for split, losses in PROJECT_LOSSES.items():
        split_curves = collect_split_curves(args.data_root, split, losses)
        if not split_curves:
            print(f"Skipping {split}: no training_losses.csv files found.")
            continue

        plot_normalized_loss_curves(
            split_curves,
            output_path=output_dir / f"{split}_normalized_loss.png",
            title=f"{split.title()} Normalized Training Loss",
            show=args.show,
        )


if __name__ == "__main__":
    main()
