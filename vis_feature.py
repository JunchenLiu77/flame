"""
Visualize high-dimensional tensor distributions using PCA.
Creates summary figures for each tensor group across all blocks.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse


def load_tensor(base_path, residual, block_idx, tensor_name):
    """Load a tensor from the saved .pt file."""
    path = os.path.join(base_path, residual, str(block_idx), f"{tensor_name}.pt")
    tensor = torch.load(path, map_location='cpu')
    # Shape: [B, L, D] where B=1, squeeze to [L, D]
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    # Convert bfloat16 to float32 (numpy doesn't support bfloat16)
    return tensor.detach().float().numpy()


def create_summary_figure(base_path, ttt_loss_type, save_dir='output/vis_feature/plots'):
    """Create a summary figure with all blocks in a grid."""
    os.makedirs(save_dir, exist_ok=True)
    
    layer_path = os.path.join(base_path, ttt_loss_type)
    block_indices = sorted([int(d) for d in os.listdir(layer_path) if os.path.isdir(os.path.join(layer_path, d))])
    
    n_blocks = len(block_indices)
    n_cols = 4
    n_rows = (n_blocks + n_cols - 1) // n_cols
    
    # Colors
    colors = {
        'q': '#E63946', 'k': '#457B9D', 'v': '#2A9D8F', 'o': '#E9C46A', 'fast_q': '#9B5DE5', 'fast_k': '#F4A261', 'fast_v': '#C1121F', 'fw_x': '#003566', 'fast_qk': '#577590',
    }
    markers = {'q': 'o', 'k': 'o', 'v': 'o', 'o': 'o', 'fast_q': 'o', 'fast_k': 'o', 'fast_v': 'o', 'fw_x': 'o', 'fast_qk': 'o'}
    
    # for group_name, tensor_names in [('ttt', ['fast_q', 'fast_k', 'fast_v', 'fw_x'])]:
    # for group_name, tensor_names in [('sdpa', ['q', 'k', 'v', 'o']), ('ttt', ['fast_q', 'fast_k', 'fast_v', 'fw_x'])]:
    # for group_name, tensor_names in [('ttt_qk', ['fast_q', 'fast_k']), ('ttt_ov', ['fast_v', 'fw_x'])]:
    for group_name, tensor_names in [('ttt_qk', ['fast_q', 'fast_k', 'fast_qk'])]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_blocks > 1 else [axes]
        
        for ax_idx, block_idx in enumerate(block_indices):
            ax = axes[ax_idx]
            
            # Collect data
            all_data = []
            data_info = []
            for tensor_name in tensor_names:
                data = load_tensor(base_path, ttt_loss_type, block_idx, tensor_name)
                all_data.append(data)
                data_info.append((tensor_name, data.shape[0]))
            
            all_data = np.vstack(all_data)
            
            # PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings = reducer.fit_transform(all_data)
            
            # Plot
            idx = 0
            for tensor_name, n_points in data_info:
                emb = embeddings[idx:idx + n_points]
                idx += n_points
                ax.scatter(
                    emb[:, 0], emb[:, 1],
                    c=colors[tensor_name],
                    marker=markers[tensor_name],
                    alpha=0.6,
                    s=15,
                    edgecolors='white',
                    linewidths=0.3
                )
            
            ax.set_title(f'Block {block_idx}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for ax_idx in range(len(block_indices), len(axes)):
            axes[ax_idx].axis('off')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = []
        for tensor_name in tensor_names:
            legend_elements.append(
                Line2D([0], [0], marker=markers[tensor_name], color='w',
                        markerfacecolor=colors[tensor_name], markersize=8,
                        label=f'{tensor_name}',
                        markeredgecolor='black')
            )
        
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(tensor_names),
                   bbox_to_anchor=(0.5, 1.02), fontsize=10)
        
        plt.suptitle(f'{ttt_loss_type.upper()} {group_name.upper()} Features Across Blocks (PCA)', fontsize=14, y=1.06)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{ttt_loss_type}_{group_name}_pca.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == '__main__':
    # uv run vis_feature.py --ttt_loss_type dot_product
    parser = argparse.ArgumentParser(description='Visualize tensor feature distributions with PCA')
    parser.add_argument('--base_path', type=str, default='exp/vis',
                        help='Base path to vis_feature directory')
    parser.add_argument('--save_dir', type=str, default='exp/vis/plots',
                        help='Directory to save plots')
    parser.add_argument('--ttt_loss_type', type=str, default='dot_product',
                        help='TTT loss type')
    args = parser.parse_args()
    
    create_summary_figure(args.base_path, args.ttt_loss_type, args.save_dir)
    
    print("\n" + "=" * 50)
    print("Visualization complete!")
    print(f"Plots saved to: {args.save_dir}")
