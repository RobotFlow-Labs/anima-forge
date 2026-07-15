#!/bin/bash
# run_vla_eval.sh — Run VLA evaluation benchmarks against a FORGE student
#
# Usage:
#   bash scripts/run_vla_eval.sh <checkpoint_path> [variant] [benchmark]
#
# Examples:
#   bash scripts/run_vla_eval.sh ./outputs/checkpoints/best.pt
#   bash scripts/run_vla_eval.sh ./outputs/checkpoints/best.pt nano libero
#   bash scripts/run_vla_eval.sh ./outputs/checkpoints/best.pt small all
set -e

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [variant] [benchmark]}"
VARIANT="${2:-nano}"
BENCHMARK="${3:-libero}"
DEVICE="${FORGE_DEVICE:-cuda}"
EPISODES="${VLA_EVAL_EPISODES:-20}"
MAX_TASKS="${VLA_EVAL_MAX_TASKS:-10}"
OUTPUT_DIR="./outputs/eval"

echo "╔══════════════════════════════════════════════════╗"
echo "║       FORGE VLA Evaluation Harness               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Checkpoint: $CHECKPOINT"
echo "  Variant:    $VARIANT"
echo "  Benchmark:  $BENCHMARK"
echo "  Device:     $DEVICE"
echo "  Episodes:   $EPISODES"
echo "  Max Tasks:  $MAX_TASKS"
echo ""

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker first."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Ensure eval deps installed
uv sync --quiet

# Run evaluation
if [ "$BENCHMARK" = "all" ]; then
    echo ">>> Running ALL benchmarks (libero, simpler, vlabench)..."
    uv run forge eval run-all \
        --checkpoint "$CHECKPOINT" \
        --variant "$VARIANT" \
        --device "$DEVICE" \
        --output-dir "$OUTPUT_DIR"
else
    echo ">>> Running $BENCHMARK benchmark..."
    uv run forge eval run "$BENCHMARK" \
        --checkpoint "$CHECKPOINT" \
        --variant "$VARIANT" \
        --device "$DEVICE" \
        --episodes "$EPISODES" \
        --max-tasks "$MAX_TASKS" \
        --output-dir "$OUTPUT_DIR"
fi

echo ""
echo ">>> Results:"
uv run forge eval results

echo ""
echo "Done. Results saved to $OUTPUT_DIR"
