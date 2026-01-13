#!/bin/bash
# Evaluate per-position perplexity on Book3 dataset
# This shows how model perplexity changes at different positions within a sequence

# Environment variables
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Default values
MODEL_PATH=""
TOKENIZER=""
MAX_SEQ_LENGTH=32768
POSITION_INTERVAL=1024
NUM_SAMPLES=""
TARGET_TOKENS="2.5e9"
DATASET_PATH="/datasets/DL3DV-DSO/book3"
OUTPUT_DIR=""

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
echo "=============================================="
echo

# Build command
CMD="uv run python eval_ppl_by_position.py \
    --model_path \"$MODEL_PATH\" \
    --max_seq_length $MAX_SEQ_LENGTH \
    --position_interval $POSITION_INTERVAL \
    --target_tokens $TARGET_TOKENS \
    --dataset_path \"$DATASET_PATH\""

if [[ -n "$NUM_SAMPLES" ]]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [[ -n "$TOKENIZER" ]]; then
    CMD="$CMD --tokenizer \"$TOKENIZER\""
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output_dir \"$OUTPUT_DIR\""
fi

# Run the evaluation
eval $CMD

echo
echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "=============================================="

exit 0
