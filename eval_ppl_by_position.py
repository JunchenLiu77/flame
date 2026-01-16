# -*- coding: utf-8 -*-
"""
Evaluate per-position perplexity on Book3 dataset.
This computes loss at different positions within long sequences to show
how the model performs as context length increases.

Output: CSV with columns [position, avg_loss, perplexity, num_samples]

Usage:
    # Single GPU
    python eval_ppl_by_position.py --model_path exp/model

    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 eval_ppl_by_position.py --model_path exp/model
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import custom_models  # registers LaCTSWIGLU with HF

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
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
    return parser.parse_args()


# ============================================================================
# Distributed utilities
# ============================================================================

def setup_distributed():
    """Initialize distributed training if available."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed(world_size: int):
    """Cleanup distributed process group."""
    if world_size > 1:
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_rank0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main_process(rank):
        print(msg)


# ============================================================================
# Data loading
# ============================================================================

def load_book3_texts(dataset_path: str):
    """Load all texts from Book3 parquet files (generator)."""
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "*.parquet")))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_path}")
    
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


def get_sequences_for_rank(
    texts,
    tokenizer,
    max_seq_length: int,
    num_samples: int,
    rank: int,
    world_size: int,
):
    """
    Tokenize texts and yield sequences assigned to this rank.
    Each rank processes sequences where global_idx % world_size == rank.
    All ranks tokenize the same data but only keep their assigned sequences.
    """
    buffer = []
    global_idx = 0
    
    for text in texts:
        tokens = tokenizer(text, return_attention_mask=False)['input_ids']
        buffer.extend(tokens)
        
        while len(buffer) >= max_seq_length and global_idx < num_samples:
            chunk = buffer[:max_seq_length]
            buffer = buffer[max_seq_length:]
            
            # Only yield if this sequence is assigned to this rank
            if global_idx % world_size == rank:
                yield torch.tensor(chunk, dtype=torch.long)
            
            global_idx += 1
            
            if global_idx >= num_samples:
                return
        
        if global_idx >= num_samples:
            return


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_per_position(
    model,
    tokenizer,
    texts,
    max_seq_length: int,
    num_samples: int,
    device: str,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Evaluate per-position perplexity.
    
    For each position p, we compute the average loss of predicting token at position p
    given all previous tokens [0, p-1].
    
    In multi-GPU mode, each GPU processes a subset of sequences (round-robin).
    """
    model.eval()
    
    # Dictionary to accumulate loss at each position
    # position -> [total_loss, count]
    position_losses = defaultdict(lambda: [0.0, 0])
    
    # Loss function (no reduction - we want per-token loss)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Calculate how many sequences this rank will process
    my_num_samples = len([i for i in range(num_samples) if i % world_size == rank])
    
    if world_size > 1:
        print(f"[Rank {rank}] Processing {my_num_samples} sequences (tokenizing on-the-fly)")
    else:
        print(f"\nStarting evaluation...")
        print(f"Processing {num_samples} sequences of length {max_seq_length}")
    
    # Process sequences assigned to this rank
    pbar = tqdm(
        get_sequences_for_rank(texts, tokenizer, max_seq_length, num_samples, rank, world_size),
        total=my_num_samples,
        desc=f"GPU {rank}" if world_size > 1 else "Evaluating",
        position=rank if world_size > 1 else 0,
    )
    
    for input_ids in pbar:
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
    
    return position_losses


def reduce_position_losses(local_losses: dict, world_size: int, local_rank: int) -> dict:
    """Reduce position losses across all ranks using all_reduce."""
    if world_size == 1:
        return local_losses
    
    device = f"cuda:{local_rank}"
    
    # Find global max position
    local_max_pos = max(local_losses.keys()) if local_losses else 0
    max_pos_tensor = torch.tensor([local_max_pos], dtype=torch.long, device=device)
    dist.all_reduce(max_pos_tensor, op=dist.ReduceOp.MAX)
    global_max_pos = max_pos_tensor.item()
    
    if global_max_pos == 0:
        return local_losses
    
    # Create tensors for total_loss and count for each position
    total_loss_tensor = torch.zeros(global_max_pos, dtype=torch.float64, device=device)
    count_tensor = torch.zeros(global_max_pos, dtype=torch.long, device=device)
    
    for pos, (total_loss, count) in local_losses.items():
        if 0 < pos <= global_max_pos:
            total_loss_tensor[pos - 1] = total_loss
            count_tensor[pos - 1] = count
    
    # All-reduce to sum across ranks
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    
    # Convert back to dict
    merged_losses = defaultdict(lambda: [0.0, 0])
    for pos in range(1, global_max_pos + 1):
        total_loss = total_loss_tensor[pos - 1].item()
        count = count_tensor[pos - 1].item()
        if count > 0:
            merged_losses[pos] = [total_loss, count]
    
    return merged_losses


def compute_and_save_results(
    position_losses: dict,
    max_seq_length: int,
    position_interval: int,
    output_csv: str,
):
    """Compute final results and save to CSV."""
    # Group by intervals
    interval_losses = defaultdict(lambda: [0.0, 0])
    for pos, (total_loss, count) in position_losses.items():
        interval_pos = ((pos - 1) // position_interval + 1) * position_interval
        interval_losses[interval_pos][0] += total_loss
        interval_losses[interval_pos][1] += count
    
    # Compute final results
    results = []
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
    if results:
        fieldnames = ["position", "avg_loss", "perplexity", "num_samples"]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nResults saved to: {output_csv}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    
    # Determine number of samples
    if args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = int(args.target_tokens / args.max_seq_length)
    
    print_rank0(f"\n{'='*60}", rank)
    print_rank0(f"Per-Position Perplexity Evaluation", rank)
    print_rank0(f"{'='*60}", rank)
    print_rank0(f"  Model path: {args.model_path}", rank)
    print_rank0(f"  Max sequence length: {args.max_seq_length}", rank)
    print_rank0(f"  Position interval: {args.position_interval}", rank)
    print_rank0(f"  Target tokens: {args.target_tokens:,.0f}", rank)
    print_rank0(f"  Number of samples: {num_samples:,} (= {num_samples * args.max_seq_length:,.0f} tokens)", rank)
    if world_size > 1:
        print_rank0(f"  World size: {world_size} GPUs", rank)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.model_path) / "eval"
    
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    # Load tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model_path
    print_rank0(f"\nLoading tokenizer from {tokenizer_path}...", rank)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    
    # Load model
    print_rank0(f"Loading model from {args.model_path}...", rank)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    
    print_rank0(f"Model loaded: {type(model).__name__}", rank)
    print_rank0(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", rank)
    
    # Load texts (generator - each rank reads independently)
    print_rank0(f"\nLoading dataset from {args.dataset_path}...", rank)
    texts = load_book3_texts(args.dataset_path)
    
    # Evaluate (each rank tokenizes and processes its own subset)
    position_losses = evaluate_per_position(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_seq_length=args.max_seq_length,
        num_samples=num_samples,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    
    # Synchronize and reduce results
    if world_size > 1:
        dist.barrier()
        print_rank0("\nReducing results across GPUs...", rank)
        position_losses = reduce_position_losses(position_losses, world_size, local_rank)
    
    # Save results (only on rank 0)
    if is_main_process(rank):
        output_csv = output_dir / f"ppl_by_position_len{args.max_seq_length}_n{num_samples}.csv"
        compute_and_save_results(
            position_losses=position_losses,
            max_seq_length=args.max_seq_length,
            position_interval=args.position_interval,
            output_csv=str(output_csv),
        )
        
        print(f"\n{'='*60}")
        print(f"Evaluation Complete!")
        print(f"{'='*60}")
    
    # Cleanup
    cleanup_distributed(world_size)


if __name__ == "__main__":
    main()
