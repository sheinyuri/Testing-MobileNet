import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

def export_interactive_3d_ribbons(path: str, out_path: str, target_class: int = 0):
    """
    Reads CNN logit data and exports a fully interactive 3D HTML plot.
    The ribbons represent the 25th-75th percentile (IQR) of logit values.
    Axes: X=Epoch, Y=Predicted Class, Z=Logit Value.
    """
    # 1. Load data
    df = pd.read_csv(path)
    class_df = df[df['true_label'] == target_class]
    
    epochs = sorted(class_df['epoch'].unique())
    num_classes = 10 
    
    # 2. Initialize the Plotly figure
    fig = go.Figure()
    
    # 3. Create ribbons and lines for each predicted class
    for pred_cls in range(num_classes):
        q1_vals, q3_vals, median_vals, ep_vals = [], [], [], []
        
        # Calculate percentiles per epoch
        for epoch in epochs:
            logits = class_df[class_df['epoch'] == epoch][f'logit_class_{pred_cls}'].values
            if len(logits) > 0:
                ep_vals.append(epoch)
                q1_vals.append(np.percentile(logits, 25))
                median_vals.append(np.median(logits))
                q3_vals.append(np.percentile(logits, 75))
        
        if not ep_vals:
            continue
            
        # Create a 2D grid for the surface plot (X=Epochs, Y=Constant Class, Z=Q1 to Q3)
        X = np.array([ep_vals, ep_vals])
        Y = np.array([np.full(len(ep_vals), pred_cls), np.full(len(ep_vals), pred_cls)])
        Z = np.array([q1_vals, q3_vals])
        
        # Styling based on whether this is the target class
        if pred_cls == target_class:
            fill_color = 'rgba(214, 39, 40, 0.7)' # Red, highly visible
            line_color = 'darkred'
            legend_name = f"Class {pred_cls} (TARGET)"
        else:
            fill_color = 'rgba(31, 119, 180, 0.2)' # Blue, highly translucent
            line_color = 'rgba(11, 83, 148, 0.6)'
            legend_name = f"Class {pred_cls}"
            
        # A uniform colorscale to make the ribbon a solid color
        custom_colorscale = [[0, fill_color], [1, fill_color]]
        legend_group = f"group_{pred_cls}"
        
        # Add Median Line (This acts as our interactive legend toggle)
        fig.add_trace(go.Scatter3d(
            x=ep_vals,
            y=np.full(len(ep_vals), pred_cls),
            z=median_vals,
            mode='lines',
            line=dict(color=line_color, width=5),
            name=legend_name,
            legendgroup=legend_group,
            showlegend=True
        ))
        
        # Add the IQR Ribbon (Surface bounded by Q1 and Q3)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=custom_colorscale,
            showscale=False,
            name=f"{legend_name} Ribbon",
            legendgroup=legend_group, # Links the surface to the line toggle
            showlegend=False,
            hoverinfo='skip' # Keeps hover UI clean by only triggering on the median line
        ))

    # 4. Final layout configurations
    fig.update_layout(
        title=f'Interactive 3D Logit Distribution Over Time (True Class {target_class})<br><sup>Shaded Ribbon = 25th-75th Percentile | Solid Line = Median</sup>',
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='Predicted Class',
            zaxis_title='Logit Value',
            yaxis=dict(tickvals=list(range(num_classes))) # Lock Y ticks to integers
        ),
        width=1100,
        height=800,
        scene_camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=0.5) # Initial viewing angle looking slightly up the epoch tunnel
        )
    )
    
    # 5. Export
    output_filename = f"{out_path}/interactive_ribbons_class_{target_class}.html"
    fig.write_html(output_filename)
    
    print(f"Success! Saved interactive ribbon plot to: {output_filename}")

from typing import List

def export_averaged_interactive_3d_ribbons(paths: List[str], out_path: str, target_class: int = 0):
    """
    Reads multiple CNN logit CSVs (different seeds), averages the logit predictions 
    for equivalent samples across all runs, and exports a fully interactive 3D HTML plot.
    Axes: X=Epoch, Y=Predicted Class, Z=Logit Value.
    """
    # 1. Load and combine all datasets
    print(f"Loading {len(paths)} files...")
    dfs = [pd.read_csv(path) for path in paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 2. Average the logits across different seeds for the EXACT same sample
    # We identify a unique sample at a specific point in time using these 4 columns
    groupby_cols = ['epoch', 'batch_idx', 'sample_idx', 'true_label']
    logit_cols = [col for col in combined_df.columns if col.startswith('logit_class_')]
    num_classes = len(logit_cols)
    
    print("Averaging predictions across seeds...")
    # This averages the logit columns while keeping the grouping columns intact
    averaged_df = combined_df.groupby(groupby_cols)[logit_cols].mean().reset_index()
    
    # 3. Filter for the specific target class we want to visualize
    class_df = averaged_df[averaged_df['true_label'] == target_class]
    epochs = sorted(class_df['epoch'].unique())
    
    # 4. Initialize the Plotly figure
    fig = go.Figure()
    
    print(f"Generating interactive 3D ribbons for True Class {target_class}...")
    # 5. Create ribbons and lines for each predicted class
    for pred_cls in range(num_classes):
        q1_vals, q3_vals, median_vals, ep_vals = [], [], [], []
        
        # Calculate percentiles per epoch on the averaged data
        for epoch in epochs:
            logits = class_df[class_df['epoch'] == epoch][f'logit_class_{pred_cls}'].values
            if len(logits) > 0:
                ep_vals.append(epoch)
                q1_vals.append(np.percentile(logits, 25))
                median_vals.append(np.median(logits))
                q3_vals.append(np.percentile(logits, 75))
        
        if not ep_vals:
            continue
            
        # Create a 2D grid for the surface plot (X=Epochs, Y=Constant Class, Z=Q1 to Q3)
        X = np.array([ep_vals, ep_vals])
        Y = np.array([np.full(len(ep_vals), pred_cls), np.full(len(ep_vals), pred_cls)])
        Z = np.array([q1_vals, q3_vals])
        
        # Styling based on whether this is the target class
        if pred_cls == target_class:
            fill_color = 'rgba(214, 39, 40, 0.7)' # Red, highly visible
            line_color = 'darkred'
            legend_name = f"Class {pred_cls} (TARGET)"
        else:
            fill_color = 'rgba(31, 119, 180, 0.2)' # Blue, highly translucent
            line_color = 'rgba(11, 83, 148, 0.6)'
            legend_name = f"Class {pred_cls}"
            
        custom_colorscale = [[0, fill_color], [1, fill_color]]
        legend_group = f"group_{pred_cls}"
        
        # Add Median Line (Acts as the interactive legend toggle)
        fig.add_trace(go.Scatter3d(
            x=ep_vals,
            y=np.full(len(ep_vals), pred_cls),
            z=median_vals,
            mode='lines',
            line=dict(color=line_color, width=5),
            name=legend_name,
            legendgroup=legend_group,
            showlegend=True
        ))
        
        # Add the IQR Ribbon (Surface bounded by Q1 and Q3)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=custom_colorscale,
            showscale=False,
            name=f"{legend_name} Ribbon",
            legendgroup=legend_group,
            showlegend=False,
            hoverinfo='skip'
        ))

    # 6. Final layout configurations
    fig.update_layout(
        title=(f'Averaged Interactive 3D Logit Distribution ({len(paths)} Seeds)<br>'
               f'<sup>True Class {target_class} | Shaded Ribbon = 25th-75th Percentile | Solid Line = Median</sup>'),
        scene=dict(
            xaxis_title='Epoch',
            yaxis_title='Predicted Class',
            zaxis_title='Mean Logit Value',
            yaxis=dict(tickvals=list(range(num_classes))) 
        ),
        width=1100,
        height=800,
        scene_camera=dict(
            eye=dict(x=-1.5, y=-1.5, z=0.5)
        )
    )
    
    # 7. Export
    output_filename = f"{out_path}/interactive_ribbons_class_{target_class}.html"
    fig.write_html(output_filename)
    
    print(f"Success! Saved averaged interactive ribbon plot to: {output_filename}")

# Example usage:
# file_paths = ['cnn_logits_seed_42.csv', 'cnn_logits_seed_99.csv', 'cnn_logits_seed_123.csv']
# export_averaged_interactive_3d_ribbons(file_paths, target_class=0)

# Example usage:
# export_interactive_3d_ribbons('cnn_logits.csv', target_class=0)
mode = 0o755
parent_path = "/students/u7947738/Documents/comp3242/project_alt/Testing-MobileNet"

datasets = ["balanced","imbalanced"]
loss_functions = ["ce","focal_loss","label_smoothing_ce","class_balanced_loss","weighted_ce"]
seeds = [2,18,33,45,255]

def make_plots(parent):
    for dataset in datasets:
        dataset_folder = dataset
        os.mkdir(dataset_folder,mode)
        for loss_function in loss_functions[:len(loss_functions)- (2 if dataset == "balanced" else 0)]:
            loss_folder = dataset_folder + "/" + loss_function
            os.mkdir(loss_folder,mode)
            seed_data = []
            for seed in seeds:
                data_path = parent_path + f'/training_data/{dataset}/{loss_function}/seed_{seed}/test_logits.csv'
                seed_data.append(data_path)
            for i in range(10):
                export_averaged_interactive_3d_ribbons(seed_data, loss_folder, i)

make_plots(parent=parent_path)
