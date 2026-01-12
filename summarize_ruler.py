# -*- coding: utf-8 -*-
"""
Summarize RULER benchmark evaluation results from multiple experiments.
Reads .out files and generates CSV/LaTeX tables.

Usage:
    python summarize_ruler.py --exp_dir exp --output_format csv
    python summarize_ruler.py --exp_dir exp --output_format latex
    python summarize_ruler.py --exp_dir exp --output_format both
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from pathlib import Path


VALID_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multiquery",
    "niah_multivalue",
]

# Mapping for LaTeX table headers (like the example image)
TASK_DISPLAY_NAMES = {
    "niah_single_1": "S-NIAH-1 (pass-key retrieval)",
    "niah_single_2": "S-NIAH-2 (number in haystack)",
    "niah_single_3": "S-NIAH-3 (uuid in haystack)",
    "niah_multikey_1": "M-NIAH-1 (multi-key)",
    "niah_multiquery": "M-NIAH (multi-query)",
    "niah_multivalue": "M-NIAH (multi-value)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize RULER benchmark results from multiple experiments"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp",
        help="Base experiment directory (default: exp)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["csv", "latex", "both"],
        default="both",
        help="Output format: csv, latex, or both (default: both)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to exp_dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Filter by specific limit value (e.g., 500). If not set, groups by limit.",
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Fixed sequence lengths to show (e.g., 1024 2048 4096 8192). If not set, uses all found lengths.",
    )
    return parser.parse_args()


def parse_out_file(filepath: str) -> dict:
    """
    Parse a RULER .out file and extract results from the table output.
    
    The .out file contains a table like:
    |    Tasks    |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
    |-------------|------:|------|-----:|-----:|---|----:|---|------|
    |niah_single_2|      1|none  |     0|  4096|↑  |  0.5|±  |   N/A|
    
    Returns dict with:
        - task: task name
        - seq_length: sequence length
        - limit: number of examples
        - score: accuracy score
        - model_name: experiment/model name
    """
    filename = os.path.basename(filepath)
    
    # Parse filename: ruler_{task}_len{seq_length}_limit{limit}.out
    match = re.match(r"ruler_(.+)_len(\d+)_limit(\d+)\.out", filename)
    if not match:
        return None
    
    task = match.group(1)
    seq_length = int(match.group(2))
    limit = int(match.group(3))
    
    # Extract model name from path
    # filepath: exp/{model_name}/eval/ruler_...
    parts = Path(filepath).parts
    try:
        exp_idx = parts.index("exp")
        model_name = parts[exp_idx + 1]
    except (ValueError, IndexError):
        model_name = "unknown"
    
    # Parse the output file to find the score from the table
    score = None
    try:
        with open(filepath, "r") as f:
            content = f.read()
        
        # Primary method: parse the table output at the end of the file
        # Format: |task_name|version|filter|n-shot|metric|arrow|value|±|stderr|
        # Example: |niah_single_2|      1|none  |     0|  4096|↑  |  0.5|±  |   N/A|
        
        # Look for table rows with the task name
        # The metric column contains the sequence length, value column contains the score
        table_pattern = rf"\|{re.escape(task)}\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|[↑↓]?\s*\|\s*([\d.]+)\s*\|"
        table_matches = re.findall(table_pattern, content)
        
        for metric_len, value in table_matches:
            if int(metric_len) == seq_length:
                score = float(value)
                break
        
        # Fallback: try a more flexible pattern
        if score is None:
            # Try matching with potential whitespace variations
            flexible_pattern = rf"\|\s*{re.escape(task)}\s*\|.*?\|\s*{seq_length}\s*\|.*?\|\s*([\d.]+)\s*\|"
            flex_match = re.search(flexible_pattern, content)
            if flex_match:
                score = float(flex_match.group(1))
    
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
        return None
    
    if score is None:
        return None
    
    return {
        "task": task,
        "seq_length": seq_length,
        "limit": limit,
        "score": score,
        "model_name": model_name,
    }


def find_all_out_files(exp_dir: str) -> list:
    """Find all RULER .out files in the experiment directory."""
    pattern = os.path.join(exp_dir, "**/eval/ruler_*.out")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def organize_results(results: list) -> dict:
    """
    Organize results by task and limit.
    
    Returns:
        {
            (task, limit): {
                model_name: {seq_length: score, ...},
                ...
            },
            ...
        }
    """
    organized = defaultdict(lambda: defaultdict(dict))
    
    for r in results:
        if r is None:
            continue
        key = (r["task"], r["limit"])
        organized[key][r["model_name"]][r["seq_length"]] = r["score"]
    
    return organized


def format_seq_length(seq_length: int) -> str:
    """Format sequence length for display (e.g., 4096 -> '4K')."""
    if seq_length >= 1024:
        if seq_length % 1024 == 0:
            return f"{seq_length // 1024}K"
        else:
            return f"{seq_length / 1024:.1f}K"
    return str(seq_length)


def generate_csv(organized: dict, output_dir: str, fixed_seq_lengths: list = None):
    """Generate CSV files for each (task, limit) combination."""
    for (task, limit), model_results in organized.items():
        # Collect all sequence lengths or use fixed ones
        if fixed_seq_lengths:
            seq_lengths = sorted(fixed_seq_lengths)
        else:
            all_seq_lengths = set()
            for scores in model_results.values():
                all_seq_lengths.update(scores.keys())
            seq_lengths = sorted(all_seq_lengths)
        
        # Output file
        output_file = os.path.join(output_dir, f"ruler_{task}_limit{limit}.csv")
        
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Model"] + [format_seq_length(s) for s in seq_lengths]
            writer.writerow(header)
            
            # Data rows
            for model_name in sorted(model_results.keys()):
                scores = model_results[model_name]
                row = [model_name]
                for seq_len in seq_lengths:
                    score = scores.get(seq_len, "")
                    if isinstance(score, float):
                        row.append(f"{score * 100:.1f}")  # Convert to percentage
                    else:
                        row.append(score)
                writer.writerow(row)
        
        print(f"  Saved: {output_file}")


def generate_latex(organized: dict, output_dir: str, fixed_seq_lengths: list = None):
    """Generate LaTeX tables for each (task, limit) combination."""
    latex_output = []
    
    for (task, limit), model_results in sorted(organized.items()):
        # Collect all sequence lengths or use fixed ones
        if fixed_seq_lengths:
            seq_lengths = sorted(fixed_seq_lengths)
        else:
            all_seq_lengths = set()
            for scores in model_results.values():
                all_seq_lengths.update(scores.keys())
            seq_lengths = sorted(all_seq_lengths)
        
        # Get display name for task
        task_display = TASK_DISPLAY_NAMES.get(task, task)
        
        # Build LaTeX table
        num_cols = len(seq_lengths) + 1  # Model + seq lengths
        col_spec = "l" + "c" * len(seq_lengths)
        
        latex = []
        latex.append(f"% Table for {task} (limit={limit})")
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{task_display} (limit={limit})}}")
        latex.append(f"\\label{{tab:{task}_limit{limit}}}")
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"\toprule")
        
        # Header row
        header_cells = ["Model"] + [format_seq_length(s) for s in seq_lengths]
        latex.append(" & ".join(header_cells) + r" \\")
        latex.append(r"\midrule")
        
        # Find best scores for each column (for bolding)
        best_scores = {}
        for seq_len in seq_lengths:
            scores_for_len = [
                model_results[m].get(seq_len, -1)
                for m in model_results
                if isinstance(model_results[m].get(seq_len), (int, float))
            ]
            if scores_for_len:
                best_scores[seq_len] = max(scores_for_len)
        
        # Data rows
        for model_name in sorted(model_results.keys()):
            scores = model_results[model_name]
            cells = [model_name.replace("_", r"\_")]
            
            for seq_len in seq_lengths:
                score = scores.get(seq_len, None)
                if score is not None and isinstance(score, (int, float)):
                    score_pct = score * 100
                    # Bold if best
                    if abs(score - best_scores.get(seq_len, -1)) < 0.001:
                        cells.append(f"\\textbf{{{score_pct:.1f}}}")
                    else:
                        cells.append(f"{score_pct:.1f}")
                else:
                    cells.append("-")
            
            latex.append(" & ".join(cells) + r" \\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")
        
        latex_output.extend(latex)
    
    # Save to file
    output_file = os.path.join(output_dir, "ruler_summary.tex")
    with open(output_file, "w") as f:
        f.write("\n".join(latex_output))
    
    print(f"  Saved: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 60)
    print("LaTeX Output:")
    print("=" * 60)
    print("\n".join(latex_output))


def generate_combined_latex(organized: dict, output_dir: str, fixed_seq_lengths: list = None):
    """
    Generate a single combined LaTeX table with multiple tasks as column groups.
    Similar to the example image format.
    """
    # Group by limit
    by_limit = defaultdict(dict)
    for (task, limit), model_results in organized.items():
        by_limit[limit][(task, limit)] = model_results
    
    latex_output = []
    
    for limit, task_results in sorted(by_limit.items()):
        # Collect all models and tasks
        all_models = set()
        all_tasks = set()
        all_seq_lengths_by_task = {}
        
        for (task, _), model_results in task_results.items():
            all_tasks.add(task)
            all_models.update(model_results.keys())
            if fixed_seq_lengths:
                all_seq_lengths_by_task[task] = sorted(fixed_seq_lengths)
            else:
                seq_lengths = set()
                for scores in model_results.values():
                    seq_lengths.update(scores.keys())
                all_seq_lengths_by_task[task] = sorted(seq_lengths)
        
        all_tasks = sorted(all_tasks)
        all_models = sorted(all_models)
        
        # Build table
        latex = []
        latex.append(f"% Combined table for limit={limit}")
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\small")
        latex.append(f"\\caption{{RULER Benchmark Results (limit={limit})}}")
        latex.append(f"\\label{{tab:ruler_combined_limit{limit}}}")
        
        # Calculate column spec
        # Model | task1 cols | task2 cols | ...
        total_cols = 1  # Model column
        col_spec_parts = ["l"]
        
        for task in all_tasks:
            num_seq = len(all_seq_lengths_by_task.get(task, []))
            total_cols += num_seq
            col_spec_parts.append("c" * num_seq)
        
        col_spec = "|".join(col_spec_parts)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"\toprule")
        
        # Multi-row header
        # First row: task names spanning columns
        header1_cells = [""]
        for task in all_tasks:
            seq_lengths = all_seq_lengths_by_task.get(task, [])
            task_short = task.replace("niah_", "").replace("_", "-")
            task_display = TASK_DISPLAY_NAMES.get(task, task_short)
            # Extract just the short name for header
            short_name = task_short.upper()
            if len(seq_lengths) > 1:
                header1_cells.append(f"\\multicolumn{{{len(seq_lengths)}}}{{c}}{{{short_name}}}")
            elif len(seq_lengths) == 1:
                header1_cells.append(short_name)
        
        latex.append(" & ".join(header1_cells) + r" \\")
        
        # Second row: sequence lengths
        header2_cells = ["Model"]
        for task in all_tasks:
            seq_lengths = all_seq_lengths_by_task.get(task, [])
            for seq_len in seq_lengths:
                header2_cells.append(format_seq_length(seq_len))
        
        latex.append(" & ".join(header2_cells) + r" \\")
        latex.append(r"\midrule")
        
        # Find best scores for bolding
        best_scores = {}
        for task in all_tasks:
            for seq_len in all_seq_lengths_by_task.get(task, []):
                key = (task, seq_len)
                task_model_results = task_results.get((task, limit), {})
                scores = [
                    task_model_results[m].get(seq_len, -1)
                    for m in task_model_results
                    if isinstance(task_model_results[m].get(seq_len), (int, float))
                ]
                if scores:
                    best_scores[key] = max(scores)
        
        # Data rows
        for model_name in all_models:
            cells = [model_name.replace("_", r"\_")]
            
            for task in all_tasks:
                seq_lengths = all_seq_lengths_by_task.get(task, [])
                task_model_results = task_results.get((task, limit), {})
                model_scores = task_model_results.get(model_name, {})
                
                for seq_len in seq_lengths:
                    score = model_scores.get(seq_len, None)
                    if score is not None and isinstance(score, (int, float)):
                        score_pct = score * 100
                        key = (task, seq_len)
                        if abs(score - best_scores.get(key, -1)) < 0.001:
                            cells.append(f"\\textbf{{{score_pct:.1f}}}")
                        else:
                            cells.append(f"{score_pct:.1f}")
                    else:
                        cells.append("-")
            
            latex.append(" & ".join(cells) + r" \\")
        
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        latex.append("")
        
        latex_output.extend(latex)
    
    # Save to file
    output_file = os.path.join(output_dir, "ruler_summary_combined.tex")
    with open(output_file, "w") as f:
        f.write("\n".join(latex_output))
    
    print(f"  Saved: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 60)
    print("Combined LaTeX Output:")
    print("=" * 60)
    print("\n".join(latex_output))


def print_summary_table(organized: dict, fixed_seq_lengths: list = None):
    """Print a summary table to console."""
    print("\n" + "=" * 80)
    print("RULER Benchmark Summary")
    print("=" * 80)
    
    for (task, limit), model_results in sorted(organized.items()):
        print(f"\n{task} (limit={limit})")
        print("-" * 60)
        
        # Collect all sequence lengths or use fixed ones
        if fixed_seq_lengths:
            seq_lengths = sorted(fixed_seq_lengths)
        else:
            all_seq_lengths = set()
            for scores in model_results.values():
                all_seq_lengths.update(scores.keys())
            seq_lengths = sorted(all_seq_lengths)
        
        # Header
        header = f"{'Model':<40}"
        for seq_len in seq_lengths:
            header += f" {format_seq_length(seq_len):>8}"
        print(header)
        print("-" * len(header))
        
        # Data rows
        for model_name in sorted(model_results.keys()):
            scores = model_results[model_name]
            row = f"{model_name:<40}"
            for seq_len in seq_lengths:
                score = scores.get(seq_len, None)
                if score is not None:
                    row += f" {score * 100:>7.1f}%"
                else:
                    row += f" {'-':>8}"
            print(row)


def main():
    args = parse_args()
    
    exp_dir = args.exp_dir
    output_dir = args.output_dir or exp_dir
    
    print(f"\n{'='*60}")
    print(f"RULER Benchmark Results Summarizer")
    print(f"{'='*60}")
    print(f"  Experiment directory: {exp_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output format: {args.output_format}")
    
    # Find all .out files
    out_files = find_all_out_files(exp_dir)
    print(f"\n  Found {len(out_files)} .out files")
    
    if not out_files:
        print("  No RULER output files found!")
        return
    
    # Parse all files
    results = []
    for filepath in out_files:
        result = parse_out_file(filepath)
        if result:
            results.append(result)
            print(f"    Parsed: {os.path.basename(filepath)} -> {result['task']}, len={result['seq_length']}, score={result['score']:.2f}")
        else:
            print(f"    Warning: Could not parse {filepath}")
    
    if not results:
        print("\n  No valid results found!")
        return
    
    # Filter by limit if specified
    if args.limit:
        results = [r for r in results if r["limit"] == args.limit]
        print(f"\n  Filtered to {len(results)} results with limit={args.limit}")
    
    # Organize results
    organized = organize_results(results)
    
    # Get fixed sequence lengths if specified
    fixed_seq_lengths = args.seq_lengths
    if fixed_seq_lengths:
        print(f"\n  Using fixed sequence lengths: {[format_seq_length(s) for s in fixed_seq_lengths]}")
    
    # Print summary to console
    print_summary_table(organized, fixed_seq_lengths)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate outputs
    print(f"\n{'='*60}")
    print("Generating output files...")
    print(f"{'='*60}")
    
    if args.output_format in ["csv", "both"]:
        generate_csv(organized, output_dir, fixed_seq_lengths)
    
    if args.output_format in ["latex", "both"]:
        generate_latex(organized, output_dir, fixed_seq_lengths)
        generate_combined_latex(organized, output_dir, fixed_seq_lengths)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Always show 1K, 2K, 4K, 8K columns even if some models don't have all results
    # python summarize_ruler.py --exp_dir exp --seq_lengths 1024 2048 4096 8192
    main()
