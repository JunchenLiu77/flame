# -*- coding: utf-8 -*-
"""
Visualize per-position perplexity results from multiple models.

Reads CSV files from exp/{model_name}/eval/ppl_by_position_len{len}_n{n}.csv
and generates comparison plots.

Usage:
    python plot_ppl_by_position.py --exp_names model1 model2 model3 --seq_length 32768 --num_samples 10
    python plot_ppl_by_position.py --exp_dir exp --seq_length 32768 --num_samples 10  # auto-discover models
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-position perplexity comparison across models"
    )
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs="+",
        default=None,
        help="List of experiment names (model directories) to compare",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="Base experiment directory (default: exp)",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=32768,
        help="Sequence length used in evaluation (default: 32768)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples used in evaluation. If not specified, auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the plot (default: exp_dir/ppl_by_position_comparison.pdf)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["perplexity", "loss"],
        default="perplexity",
        help="Metric to plot: perplexity or loss (default: perplexity)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Figure size (width, height) in inches (default: 10 6)",
    )
    parser.add_argument(
        "--legend_loc",
        type=str,
        default="best",
        help="Legend location (default: best)",
    )
    parser.add_argument(
        "--no_grid",
        action="store_true",
        help="Disable grid lines",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Custom labels for each model (must match number of exp_names)",
    )
    return parser.parse_args()


def find_csv_file(exp_dir: str, exp_name: str, seq_length: int, num_samples: Optional[int] = None) -> Optional[str]:
    """Find the CSV file for a given experiment."""
    eval_dir = Path(exp_dir) / exp_name / "eval"
    
    if not eval_dir.exists():
        return None
    
    # If num_samples specified, look for exact match
    if num_samples is not None:
        csv_path = eval_dir / f"ppl_by_position_len{seq_length}_n{num_samples}.csv"
        if csv_path.exists():
            return str(csv_path)
        return None
    
    # Otherwise, find any matching file
    pattern = f"ppl_by_position_len{seq_length}_n*.csv"
    matches = list(eval_dir.glob(pattern))
    
    if matches:
        # Return the one with most samples (largest n)
        def extract_n(path):
            name = path.stem
            try:
                n_part = name.split("_n")[-1]
                return int(n_part)
            except:
                return 0
        matches.sort(key=extract_n, reverse=True)
        return str(matches[0])
    
    return None


def read_csv(csv_path: str) -> Tuple[List[int], List[float], List[float]]:
    """Read positions, losses, and perplexities from CSV."""
    positions = []
    losses = []
    perplexities = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append(int(row['position']))
            losses.append(float(row['avg_loss']))
            perplexities.append(float(row['perplexity']))
    
    return positions, losses, perplexities


def discover_experiments(exp_dir: str, seq_length: int, num_samples: Optional[int] = None) -> List[str]:
    """Discover all experiments that have per-position perplexity results."""
    exp_path = Path(exp_dir)
    experiments = []
    
    for model_dir in exp_path.iterdir():
        if model_dir.is_dir():
            csv_file = find_csv_file(exp_dir, model_dir.name, seq_length, num_samples)
            if csv_file:
                experiments.append(model_dir.name)
    
    return sorted(experiments)


def plot_comparison(
    data: Dict[str, Tuple[List[int], List[float]]],
    metric: str,
    output_path: str,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    legend_loc: str = "best",
    show_grid: bool = True,
    labels: Optional[Dict[str, str]] = None,
):
    """Generate comparison plot."""
    # Set up the figure with a nice style
    plt.style.use('seaborn-v0_8-whitegrid' if show_grid else 'seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    
    # Plot each model
    for idx, (model_name, (positions, values)) in enumerate(data.items()):
        label = labels.get(model_name, model_name) if labels else model_name
        ax.plot(
            positions, 
            values, 
            marker='o',
            markersize=4,
            linewidth=2,
            label=label,
            color=colors[idx],
            alpha=0.8,
        )
    
    # Labels and title
    ax.set_xlabel("Position in Sequence", fontsize=12)
    ylabel = "Perplexity" if metric == "perplexity" else "Cross-Entropy Loss"
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Per-Position {ylabel} Comparison", fontsize=14, fontweight='bold')
    
    # Format x-axis with K notation
    def format_position(x, p):
        if x >= 1000:
            return f"{int(x/1000)}K"
        return str(int(x))
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_position))
    
    # Legend
    ax.legend(loc=legend_loc, fontsize=10, framealpha=0.9)
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PNG if output is PDF
    if output_path.endswith('.pdf'):
        png_path = output_path.replace('.pdf', '.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Plot also saved to: {png_path}")
    
    plt.close()


def main():
    args = parse_args()
    
    # Discover or use specified experiments
    if args.exp_names:
        exp_names = args.exp_names
    else:
        print(f"Auto-discovering experiments in {args.exp_dir}...")
        exp_names = discover_experiments(args.exp_dir, args.seq_length, args.num_samples)
        if not exp_names:
            print(f"No experiments found with ppl_by_position results for seq_length={args.seq_length}")
            return
        print(f"Found {len(exp_names)} experiments: {exp_names}")
    
    # Read data from each experiment
    data = {}
    for exp_name in exp_names:
        csv_path = find_csv_file(args.exp_dir, exp_name, args.seq_length, args.num_samples)
        if csv_path is None:
            print(f"Warning: No CSV found for {exp_name}, skipping...")
            continue
        
        print(f"Reading: {csv_path}")
        positions, losses, perplexities = read_csv(csv_path)
        
        if args.metric == "perplexity":
            data[exp_name] = (positions, perplexities)
        else:
            data[exp_name] = (positions, losses)
    
    if not data:
        print("No data to plot!")
        return
    
    # Prepare labels
    labels = None
    if args.labels:
        if len(args.labels) != len(exp_names):
            print(f"Warning: Number of labels ({len(args.labels)}) doesn't match experiments ({len(exp_names)})")
        else:
            labels = dict(zip(exp_names, args.labels))
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        n_str = f"_n{args.num_samples}" if args.num_samples else ""
        output_path = os.path.join(
            args.exp_dir, 
            f"ppl_by_position_comparison_len{args.seq_length}{n_str}.pdf"
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Plot
    plot_comparison(
        data=data,
        metric=args.metric,
        output_path=output_path,
        title=args.title,
        figsize=tuple(args.figsize),
        legend_loc=args.legend_loc,
        show_grid=not args.no_grid,
        labels=labels,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
