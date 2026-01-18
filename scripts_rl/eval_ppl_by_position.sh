#!/bin/bash
# Evaluate per-position perplexity on Book3 dataset (Multi-GPU)
# This shows how model perplexity changes at different positions within a sequence
#
# Usage:
#   # Single GPU
#   ./scripts_rl/eval_ppl_by_position.sh --model_path exp/model
#
#   # Multi-GPU (8 GPUs)
#   ./scripts_rl/eval_ppl_by_position.sh --model_path exp/model --num_gpus 8

# Environment variables
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# For SLURM: use the first node as master
MASTER_ADDR="localhost"
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT=$(python3 -c "import socket as s; x=s.socket(s.AF_INET,s.SOCK_STREAM); x.bind(('',0)); print(x.getsockname()[1]); x.close()")
fi

# Default values
MODEL_PATH=""
TOKENIZER=""
MAX_SEQ_LENGTH=32768
POSITION_INTERVAL=1024
NUM_SAMPLES=""
TARGET_TOKENS="2.5e9"
DATASET_PATH="/root/.cache/huggingface/hub/datasets--Geralt-Targaryen--books3/snapshots/669aefeaf7e17e3f2039004d603c7a4cc4163af9"
OUTPUT_DIR=""
NUM_GPUS=1
BATCH_SIZE=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        --max_seq_length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --position_interval)
            POSITION_INTERVAL="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --target_tokens)
            TARGET_TOKENS="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --model_path         Path to HuggingFace model directory"
    echo ""
    echo "Optional:"
    echo "  --tokenizer          Tokenizer path (default: model_path)"
    echo "  --max_seq_length     Max sequence length (default: 32768)"
    echo "  --position_interval  Position interval for aggregation (default: 1024)"
    echo "  --num_samples        Number of sequences (overrides --target_tokens if set)"
    echo "  --target_tokens      Target total tokens to evaluate (default: 2.5e9)"
    echo "  --dataset_path       Path to Book3 dataset"
    echo "  --output_dir         Output directory"
    echo "  --num_gpus           Number of GPUs to use (default: 1)"
    echo "  --batch_size         Batch size for evaluation (default: 1)"
    exit 1
fi

echo
echo "=============================================="
echo "Per-Position Perplexity Evaluation"
echo "=============================================="
echo "  Model Path: $MODEL_PATH"
echo "  Max Seq Length: $MAX_SEQ_LENGTH"
echo "  Position Interval: $POSITION_INTERVAL"
echo "  Target Tokens: $TARGET_TOKENS"
echo "  Num Samples: ${NUM_SAMPLES:-'(auto from target_tokens)'}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Num GPUs: $NUM_GPUS"
if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "  Master Addr: $MASTER_ADDR"
    echo "  Master Port: $MASTER_PORT"
fi
echo "=============================================="
echo

# Build Python arguments
PYTHON_ARGS="eval_ppl_by_position.py \
    --model_path \"$MODEL_PATH\" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --position_interval $POSITION_INTERVAL \
    --target_tokens $TARGET_TOKENS \
    --batch_size $BATCH_SIZE \
    --dataset_path \"$DATASET_PATH\""

if [[ -n "$NUM_SAMPLES" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --num_samples $NUM_SAMPLES"
fi

if [[ -n "$TOKENIZER" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --tokenizer \"$TOKENIZER\""
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --output_dir \"$OUTPUT_DIR\""
fi

# Run the evaluation
if [[ "$NUM_GPUS" -gt 1 ]]; then
    echo "Running with torchrun (${NUM_GPUS} GPUs)..."
    CMD="uv run torchrun --nproc_per_node=$NUM_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $PYTHON_ARGS"
else
    echo "Running with single GPU..."
    CMD="uv run python $PYTHON_ARGS"
fi

echo "Command: $CMD"
echo
eval $CMD

echo
echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "=============================================="

exit 0
