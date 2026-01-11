# -*- coding: utf-8 -*-
# Evaluate a checkpoint on RULER benchmark (needle-in-haystack tasks)
# This script combines:
#   1. Converting DCP checkpoint to HuggingFace format
#   2. Running lm-eval on RULER tasks

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import custom_models  # registers LaCTSWIGLU with HF

import torch
torch._dynamo.disable()

# Pre-download NLTK resources to avoid race conditions in multi-process
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


VALID_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multiquery",
    "niah_multivalue",
]

VALID_SEQ_LENGTHS = [4096, 8192, 16384, 32768]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DCP checkpoint to HF format and evaluate on RULER benchmark"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint folder (e.g., exp/lact_baseline_bs16_20k_ga_dot_product/checkpoint/step-16000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model config JSON file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="fla-hub/transformer-1.3B-100B",
        help="Tokenizer name or path (default: fla-hub/transformer-1.3B-100B)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="niah_single_2",
        choices=VALID_TASKS,
        help=f"RULER task to evaluate (default: niah_single_2). Valid: {VALID_TASKS}",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        required=True,
        choices=VALID_SEQ_LENGTHS,
        help=f"Sequence length for evaluation. Valid: {VALID_SEQ_LENGTHS}",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for evaluation (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of examples to evaluate (default: 500)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output files. Defaults to checkpoint parent directory.",
    )
    parser.add_argument(
        "--skip_convert",
        action="store_true",
        help="Skip DCP to HF conversion (use if already converted)",
    )
    return parser.parse_args()


def extract_step_and_exp_path(checkpoint_path: str):
    """
    Extract step number, experiment path, and model name from checkpoint folder.
    
    Example:
        Input: exp/lact_baseline_bs16_20k_ga_dot_product/checkpoint/step-16000
        Output: (16000, exp/lact_baseline_bs16_20k_ga_dot_product, lact_baseline_bs16_20k_ga_dot_product)
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    
    # Get step from folder name (step-XXXXX)
    step_folder = checkpoint_path.name
    if not step_folder.startswith("step-"):
        raise ValueError(f"Checkpoint folder must be named 'step-XXXXX', got: {step_folder}")
    
    step = int(step_folder.replace("step-", ""))
    
    # Get experiment path (parent of 'checkpoint' folder)
    if checkpoint_path.parent.name != "checkpoint":
        raise ValueError(
            f"Expected checkpoint folder structure: .../checkpoint/step-XXXXX, "
            f"got: {checkpoint_path}"
        )
    
    exp_path = checkpoint_path.parent.parent
    
    # Extract model name (the folder name under exp/)
    model_name = exp_path.name
    
    return step, exp_path, model_name


def check_hf_checkpoint_exists(exp_path: Path):
    """Check if HuggingFace checkpoint files exist."""
    required_files = [
        "config.json",
        "model.safetensors",
    ]
    missing_files = []
    for f in required_files:
        if not (exp_path / f).exists():
            missing_files.append(f)
    
    if missing_files:
        raise FileNotFoundError(
            f"HF checkpoint missing required files in {exp_path}: {missing_files}"
        )
    
    print(f"  ✓ HF checkpoint verified: config.json and model.safetensors exist")


def convert_dcp_to_hf(exp_path: Path, step: int, config: str, tokenizer: str):
    """Convert DCP checkpoint to HuggingFace format."""
    print(f"\n{'='*60}")
    print(f"Step 1: Converting DCP checkpoint to HuggingFace format")
    print(f"{'='*60}")
    print(f"  Experiment path: {exp_path}")
    print(f"  Step: {step}")
    print(f"  Config: {config}")
    print(f"  Tokenizer: {tokenizer}")
    
    cmd = [
        "uv", "run", "python", "-m", "flame.utils.convert_dcp_to_hf",
        "--path", str(exp_path),
        "--step", str(step),
        "--config", config,
        "--tokenizer", tokenizer,
    ]
    
    print(f"\n  Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"DCP to HF conversion failed with return code {result.returncode}")
    
    # Sanity check: verify HF checkpoint files exist
    check_hf_checkpoint_exists(exp_path)
    
    print(f"\n  Conversion completed successfully!")


def run_ruler_eval(
    exp_path: Path,
    task: str,
    seq_length: int,
    num_gpus: int,
    batch_size: int,
    limit: int,
    output_dir: Path,
):
    """Run RULER benchmark evaluation using lm-eval with accelerate."""
    print(f"\n{'='*60}")
    print(f"Step 2: Running RULER evaluation")
    print(f"{'='*60}")
    print(f"  Model path: {exp_path}")
    print(f"  Task: {task}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Number of GPUs: {num_gpus}")
    print(f"  Batch size: {batch_size}")
    print(f"  Limit: {limit}")
    print(f"  Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file name
    output_file = output_dir / f"ruler_{task}_len{seq_length}_limit{limit}.out"
    
    # Build the command - use this script itself as the entry point
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    
    # Build lm_eval CLI args
    lm_eval_args = [
        "--model", "hf",
        "--model_args", f"pretrained={exp_path},dtype=bfloat16,max_length={seq_length},trust_remote_code=True",
        "--tasks", task,
        "--metadata", json.dumps({"max_seq_lengths": [seq_length]}),
        "--batch_size", str(batch_size),
        "--limit", str(limit),
        "--device", "cuda",
        "--show_config",
    ]
    
    cmd = [
        "uv", "run", "accelerate", "launch",
        "--num_processes", str(num_gpus),
        sys.argv[0],  # This script
        "--run_lm_eval",  # Special flag to run lm_eval directly
    ] + lm_eval_args
    
    print(f"\n  Running: {' '.join(cmd)}")
    print(f"  Output will be saved to: {output_file}\n")
    
    # Run and capture output
    with open(output_file, "w") as f:
        # Write command info to output file
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*60}\n\n")
        f.flush()
        
        # Run the process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Stream output to both console and file
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        
        process.wait()
        
        if process.returncode != 0:
            error_msg = f"\nEvaluation failed with return code {process.returncode}"
            print(error_msg)
            f.write(error_msg)
            raise RuntimeError(error_msg)
    
    print(f"\n  Evaluation completed!")
    print(f"  Results saved to: {output_file}")
    
    return output_file


def main():
    args = parse_args()
    
    # Parse checkpoint path
    step, exp_path, model_name = extract_step_and_exp_path(args.checkpoint)
    
    print(f"\n{'='*60}")
    print(f"RULER Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model name: {model_name}")
    print(f"  Extracted step: {step}")
    print(f"  Experiment path: {exp_path}")
    print(f"  Task: {args.task}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Limit: {args.limit}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = exp_path / "eval"
    
    # Step 1: Convert DCP to HF (unless skipped or already exists)
    if args.skip_convert:
        print(f"\n  Skipping DCP to HF conversion (--skip_convert flag set)")
    elif (exp_path / "config.json").exists() and (exp_path / "model.safetensors").exists():
        print(f"\n  Skipping DCP to HF conversion (config.json and model.safetensors already exist)")
    else:
        convert_dcp_to_hf(exp_path, step, args.config, args.tokenizer)
    
    # Step 2: Run RULER evaluation
    output_file = run_ruler_eval(
        exp_path=exp_path,
        task=args.task,
        seq_length=args.seq_length,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=output_dir,
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"  Results saved to: {output_file}")


if __name__ == "__main__":
    # Check if we're being called to run lm_eval directly (via accelerate launch)
    if "--run_lm_eval" in sys.argv:
        # Remove our custom flag and pass the rest to lm_eval
        sys.argv.remove("--run_lm_eval")
        sys.argv[0] = "lm_eval"  # Set proper program name
        from lm_eval.__main__ import cli_evaluate
        cli_evaluate()
    else:
        main()
