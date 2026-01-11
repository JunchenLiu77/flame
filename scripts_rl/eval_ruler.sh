#!/bin/bash
# Evaluate a checkpoint on RULER benchmark (needle-in-haystack tasks)

# Environment variables
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# uv experimental features
export UV_PREVIEW_FEATURES=extra-build-dependencies

# Default values
CHECKPOINT=""
CONFIG=""
TASK="niah_single_2"
SEQ_LENGTH=""
NUM_GPUS=1
BATCH_SIZE=1
LIMIT=500
TOKENIZER="fla-hub/transformer-1.3B-100B"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
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
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --tokenizer)
            TOKENIZER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required"
    echo "Usage: $0 --checkpoint <path> --config <path> --seq_length <len> [options]"
    echo ""
    echo "Required:"
    echo "  --checkpoint    Path to checkpoint folder (e.g., exp/.../checkpoint/step-16000)"
    echo "  --config        Path to model config JSON file (e.g., configs/760M_lact_swiglu_nh4_fwlow_rank_momentum_muon.json)"
    echo "  --seq_length    Sequence length (4096, 8192, 16384, or 32768)"
    echo ""
    echo "Optional:"
    echo "  --task          RULER task (default: niah_single_2)"
    echo "  --num_gpus      Number of GPUs (default: 1)"
    echo "  --batch_size    Batch size (default: 1)"
    echo "  --limit         Number of examples (default: 500)"
    echo "  --tokenizer     Tokenizer name/path (default: fla-hub/transformer-1.3B-100B)"
    exit 1
fi

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi

if [[ -z "$SEQ_LENGTH" ]]; then
    echo "Error: --seq_length is required"
    exit 1
fi

echo
echo "=============================================="
echo "RULER Benchmark Evaluation"
echo "=============================================="
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Task: $TASK"
echo "  Seq Length: $SEQ_LENGTH"
echo "  Num GPUs: $NUM_GPUS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Limit: $LIMIT"
echo "=============================================="
echo

# Run the evaluation
uv run python eval_ruler.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --task "$TASK" \
    --seq_length "$SEQ_LENGTH" \
    --num_gpus "$NUM_GPUS" \
    --batch_size "$BATCH_SIZE" \
    --limit "$LIMIT" \
    --tokenizer "$TOKENIZER"

echo
echo "=============================================="
echo "Evaluation completed at: $(date)"
echo "=============================================="

exit 0
