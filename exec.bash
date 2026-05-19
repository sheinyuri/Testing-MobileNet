# For plotting the metrics for both balanced and imbalanced
python metric_plotting.py --project-plots --project-split both   --imbalanced-class-counts 500,500,500,5000,5000,5000,5000,5000,5000,5000



python metric_plotting.py 80epochs_training_data/imbalanced/focal_loss/seed_45 \
  --epoch-confidence-accuracy \
  --output plots/epoch_confidence_accuracy/80epochs_focal_loss_seed_45.png \
  --metrics-csv plots/epoch_confidence_accuracy/80epochs_focal_loss_seed_45.csv

python metric_plotting.py 80epochs_training_data/imbalanced/weighted_ce/seed_45 \
  --epoch-confidence-accuracy \
  --output plots/epoch_confidence_accuracy/80epochs_weighted_ce_seed_45.png \
  --metrics-csv plots/epoch_confidence_accuracy/80epochs_weighted_ce_seed_45.csv

python metric_plotting.py 80epochs_training_data/imbalanced/focal_loss/seed_45 \
  --class-counts 500,500,500,5000,5000,5000,5000,5000,5000,5000 \
  --output 80epochs_training_data/focal_loss_seed_45_metrics.png \
  --metrics-csv 80epochs_training_data/focal_loss_seed_45_metrics.csv

python metric_plotting.py 80epochs_training_data/imbalanced/weighted_ce/seed_45 \
  --class-counts 500,500,500,5000,5000,5000,5000,5000,5000,5000 \
  --output 80epochs_training_data/weighted_ce_seed_45_metrics.png \
  --metrics-csv 80epochs_training_data/weighted_ce_seed_45_metrics.csv