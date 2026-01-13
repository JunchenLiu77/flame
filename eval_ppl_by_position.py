# -*- coding: utf-8 -*-
"""
Evaluate per-position perplexity on Book3 dataset.
This computes loss at different positions within long sequences to show
how the model performs as context length increases.

Output: CSV with columns [position, avg_loss, perplexity, num_samples]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import custom_models  # registers LaCTSWIGLU with HF

import pyarrow.parquet as pq
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

torch._dynamo.disable()


BOOK3_PATH = "/datasets/DL3DV-DSO/book3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate per-position perplexity on Book3 dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer path. If not specified, uses model_path.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32768,
        help="Maximum sequence length to evaluate (default: 32768)",
    )
    parser.add_argument(
        "--position_interval",
        type=int,
        default=1024,
        help="Interval for computing per-position loss (default: 1024)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of long sequences to evaluate. If not set, uses --target_tokens to determine.",
    )
    parser.add_argument(
        "--target_tokens",
        type=float,
        default=2.5e9,
        help="Target total tokens to evaluate (default: 2.5B). Used if --num_samples not set.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output CSV. Defaults to model_path/eval/",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=BOOK3_PATH,
        help=f"Path to Book3 dataset (default: {BOOK3_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    return parser.parse_args()


def load_book3_texts(dataset_path: str):
    """Load all texts from Book3 parquet files."""
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        
        text_col = None
        for col in ['text', 'content', 'book_text', 'body']:
            if col in table.column_names:
                text_col = col
                break
        
        if text_col is None:
            continue
        
        text_column = table.column(text_col)
        for i in range(len(table)):
            text = text_column[i].as_py()
            if isinstance(text, str) and len(text) > 0:
                yield text


def get_long_sequences(texts, tokenizer, max_seq_length: int, num_samples: int):
    """
    Get long sequences of at least max_seq_length tokens.
    Concatenates multiple documents if needed.
    Yields sequences one at a time (streaming) to show progress.
    """
    buffer = []
    samples_collected = 0
    
    for text in texts:
        tokens = tokenizer(text, return_attention_mask=False)['input_ids']
        buffer.extend(tokens)
        
        while len(buffer) >= max_seq_length and samples_collected < num_samples:
            # Extract exactly max_seq_length tokens
            chunk = buffer[:max_seq_length]
            buffer = buffer[max_seq_length:]
            
            yield torch.tensor(chunk, dtype=torch.long)
            samples_collected += 1
            
            if samples_collected >= num_samples:
                return


@torch.no_grad()
def evaluate_per_position(
    model,
    tokenizer,
    texts,
    max_seq_length: int,
    position_interval: int,
    num_samples: int,
    device: str,
    output_csv: str,
):
    """
    Evaluate per-position perplexity.
    
    For each position p, we compute the average loss of predicting token at position p
    given all previous tokens [0, p-1].
    """
    model.eval()
    
    # Dictionary to accumulate loss at each position
    # position -> (total_loss, count)
    position_losses = defaultdict(lambda: [0.0, 0])
    
    # Loss function (no reduction - we want per-token loss)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print(f"\nStarting evaluation (tokenization + inference)...")
    print(f"Processing {num_samples} sequences of length {max_seq_length}")
    
    # Process sequences one by one with progress bar
    seq_count = 0
    for input_ids in tqdm(get_long_sequences(texts, tokenizer, max_seq_length, num_samples), 
                          total=num_samples, desc="Tokenizing & evaluating"):
        seq_count += 1
        input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Compute per-token loss
        # Shift: predict token[i+1] from logits[i]
        shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab]
        shift_labels = input_ids[:, 1:].contiguous()    # [1, seq_len-1]
        
        # Per-token loss: [seq_len-1]
        per_token_loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Accumulate loss at each position
        # Note: per_token_loss[i] is the loss of predicting token[i+1] from context [0:i+1]
        # So position i+1 has loss per_token_loss[i]
        for i, loss in enumerate(per_token_loss.cpu().tolist()):
            position = i + 1  # Position of the predicted token
            position_losses[position][0] += loss
            position_losses[position][1] += 1
    
    # Compute average loss at each position interval
    results = []
    positions = sorted(position_losses.keys())
    
    # Group by intervals
    interval_losses = defaultdict(lambda: [0.0, 0])
    for pos in positions:
        # Round to nearest interval
        interval_pos = ((pos - 1) // position_interval + 1) * position_interval
        interval_losses[interval_pos][0] += position_losses[pos][0]
        interval_losses[interval_pos][1] += position_losses[pos][1]
    
    # Compute final results
    print(f"\nPer-position perplexity results:")
    print(f"{'Position':<12} {'Avg Loss':<12} {'Perplexity':<12} {'Samples':<12}")
    print("-" * 48)
    
    for pos in sorted(interval_losses.keys()):
        if pos > max_seq_length:
            continue
        total_loss, count = interval_losses[pos]
        if count > 0:
            avg_loss = total_loss / count
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            
            result = {
                "position": pos,
                "avg_loss": avg_loss,
                "perplexity": ppl,
                "num_samples": count,
            }
            results.append(result)
            
            print(f"{pos:<12} {avg_loss:<12.4f} {ppl:<12.2f} {count:<12}")
    
    # Save to CSV
    save_results_to_csv(results, output_csv)
    print(f"\nResults saved to: {output_csv}")
    
    return results


def save_results_to_csv(results: list, output_csv: str):
    """Save results to CSV file."""
    if not results:
        return
    
    fieldnames = ["position", "avg_loss", "perplexity", "num_samples"]
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main():
    args = parse_args()
    
    # Determine number of samples
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        # Calculate from target_tokens
        num_samples = int(args.target_tokens / args.max_seq_length)
    
    print(f"\n{'='*60}")
    print(f"Per-Position Perplexity Evaluation")
    print(f"{'='*60}")
    print(f"  Model path: {args.model_path}")
    print(f"  Max sequence length: {args.max_seq_length}")
    print(f"  Position interval: {args.position_interval}")
    print(f"  Target tokens: {args.target_tokens:,.0f}")
    print(f"  Number of samples: {num_samples:,} (= {num_samples * args.max_seq_length:,.0f} tokens)")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model_path) / "eval"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load texts
    texts = load_book3_texts(args.dataset_path)
    
    # Output file
    output_csv = output_dir / f"ppl_by_position_len{args.max_seq_length}_n{num_samples}.csv"
    
    # Evaluate
    results = evaluate_per_position(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_seq_length=args.max_seq_length,
        position_interval=args.position_interval,
        num_samples=num_samples,
        device=args.device,
        output_csv=str(output_csv),
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
