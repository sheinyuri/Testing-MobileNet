import os
import pandas as pd
import numpy as np

def calculate_metrics(logits_df, subset_classes):
    # Filter for the subset of samples based on true label
    subset_df = logits_df[logits_df['true_label'].isin(subset_classes)]
    
    # Logit columns are from index 5 to 14
    logit_cols = [f'logit_class_{i}' for i in range(10)]
    
    # Top-1 predictions
    preds = subset_df[logit_cols].values.argmax(axis=1)
    true_labels = subset_df['true_label'].values
    
    # Accuracy (Top-1)
    accuracy = (preds == true_labels).mean()
    
    # Top-3 Accuracy
    top3_preds = subset_df[logit_cols].values.argsort(axis=1)[:, -3:]
    top3_accuracy = np.array([true_labels[i] in top3_preds[i] for i in range(len(true_labels))]).mean()
    
    # For Macro Precision and F1, we need per-class metrics on the FULL dataset
    # but we only average them over the subset_classes.
    full_preds = logits_df[logit_cols].values.argmax(axis=1)
    full_true = logits_df['true_label'].values
    
    # Confusion matrix for all 10 classes
    cm = np.zeros((10, 10))
    for t, p in zip(full_true, full_preds):
        cm[t, p] += 1
        
    precisions = []
    f1_scores = []
    
    for i in subset_classes:
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        f1_scores.append(f1)
        
    macro_precision = np.mean(precisions)
    macro_f1 = np.mean(f1_scores)
    
    return accuracy, macro_precision, macro_f1, top3_accuracy

def main():
    loss_functions = ['ce', 'class_balanced_loss', 'focal_loss', 'label_smoothing_ce', 'weighted_ce']
    seeds = [18, 2, 255, 33, 45]
    
    subsets = {
        'All Classes': list(range(10)),
        'Minority Classes (0,1,2)': [0, 1, 2],
        'Majority Classes (3-9)': list(range(3, 10))
    }
    
    all_results = []
    
    for subset_name, subset_classes in subsets.items():
        subset_results = []
        for loss in loss_functions:
            seed_metrics = []
            for seed in seeds:
                path = f'training_data/imbalanced/{loss}/seed_{seed}/test_logits.csv'
                if not os.path.exists(path):
                    print(f"Warning: {path} not found.")
                    continue
                
                print(f"Processing {loss} seed {seed}...")
                df = pd.read_csv(path)
                # Filter for the last epoch
                last_epoch = df['epoch'].max()
                df_last = df[df['epoch'] == last_epoch]
                
                metrics = calculate_metrics(df_last, subset_classes)
                seed_metrics.append(metrics)
            
            if seed_metrics:
                mean_metrics = np.mean(seed_metrics, axis=0)
                subset_results.append({
                    'Section': subset_name,
                    'Loss Function': loss,
                    'Accuracy': mean_metrics[0],
                    'Precision': mean_metrics[1],
                    'Macro F1': mean_metrics[2],
                    'Top-3 Accuracy': mean_metrics[3]
                })
        all_results.extend(subset_results)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('imbalanced_summary_metrics.csv', index=False)
    print("Summary CSV generated: imbalanced_summary_metrics.csv")

if __name__ == "__main__":
    main()
