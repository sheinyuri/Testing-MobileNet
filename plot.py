from metric_plotting import compute_metrics, plot_metrics

class_counts = [5000] * 10

folders = [
    "training_data/balanced/ce/seed_2",
    "training_data/balanced/focal_loss/seed_2",
    "training_data/balanced/label_smoothing_ce/seed_2",
]

all_rows = []
for folder in folders:
    rows = compute_metrics(folder, class_counts)

    # Optional: rename criterion using the folder name if the CSV criterion labels overlap
    loss_name = folder.split("/")[-2]
    for row in rows:
        row["criterion"] = loss_name

    all_rows.extend(rows)

plot_metrics(
    all_rows,
    output_path="plots/seed_2_loss_function_comparison.png",
    title="Loss Function Comparison: Seed 2",
)