#!/bin/bash
#
# Plot per-position perplexity comparison across multiple models
#
# Usage:
#   ./scripts_rl/plot_ppl_by_position.sh --exp_names model1 model2 --seq_length 32768 --num_samples 10
#   ./scripts_rl/plot_ppl_by_position.sh --seq_length 32768 --num_samples 10  # auto-discover all models
#
# Examples:
#   # Compare specific models
#   ./scripts_rl/plot_ppl_by_position.sh \
#       --exp_names lact_baseline_bs16_20k_ga_dot_product lact_baseline_bs16_32k_ga_dot_product \
#       --seq_length 32768 --num_samples 76293
#
#   # Auto-discover all models with results
#   ./scripts_rl/plot_ppl_by_position.sh --seq_length 32768 --num_samples 76293
#
#   # Custom output and labels
#   ./scripts_rl/plot_ppl_by_position.sh \
#       --exp_names model1 model2 \
#       --labels "Model 1 (20K)" "Model 2 (32K)" \
#       --output results/comparison.pdf
#

set -e

# Default values
EXP_DIR="exp"
SEQ_LENGTH=32768
NUM_SAMPLES=""
OUTPUT=""
METRIC="perplexity"
TITLE=""
EXP_NAMES=""
LABELS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp_names)
            shift
            EXP_NAMES=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXP_NAMES="$EXP_NAMES $1"
                shift
            done
            ;;
        --exp_dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --metric)
            METRIC="$2"
            shift 2
            ;;
        --title)
            TITLE="$2"
            shift 2
            ;;
        --labels)
            shift
            LABELS=""
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LABELS="$LABELS $1"
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "=============================================="
echo "Plot Per-Position Perplexity Comparison"
echo "=============================================="
echo "  Exp Dir: $EXP_DIR"
echo "  Seq Length: $SEQ_LENGTH"
echo "  Num Samples: ${NUM_SAMPLES:-'(auto-detect)'}"
echo "  Metric: $METRIC"
if [ -n "$EXP_NAMES" ]; then
    echo "  Experiments:$EXP_NAMES"
else
    echo "  Experiments: (auto-discover)"
fi
if [ -n "$OUTPUT" ]; then
    echo "  Output: $OUTPUT"
fi
echo "=============================================="
echo ""

# Build command
CMD="uv run python plot_ppl_by_position.py"
CMD="$CMD --exp_dir $EXP_DIR"
CMD="$CMD --seq_length $SEQ_LENGTH"
CMD="$CMD --metric $METRIC"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi

if [ -n "$TITLE" ]; then
    CMD="$CMD --title \"$TITLE\""
fi

if [ -n "$EXP_NAMES" ]; then
    CMD="$CMD --exp_names$EXP_NAMES"
fi

if [ -n "$LABELS" ]; then
    CMD="$CMD --labels$LABELS"
fi

# Run
echo "Running: $CMD"
echo ""
eval $CMD

echo ""
echo "=============================================="
echo "Done!"
echo "=============================================="
