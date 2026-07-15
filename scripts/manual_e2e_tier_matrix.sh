#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${FORGE_MATRIX_OUTPUT_DIR:-./outputs/validation/gpu_full/manual_e2e_${TS}}"
mkdir -p "$OUT"

MANUAL_RESULTS="$OUT/manual_e2e_results.csv"
RUN_STEPS=0
PASS_STEPS=0
FAILED_STEPS=0
DEVICE="${FORGE_MATRIX_DEVICE:-auto}"

echo "step,status,duration_s,log" > "$MANUAL_RESULTS"

run_step() {
  local step="$1"
  shift
  local log="$OUT/${step}.log"
  local start end status duration

  RUN_STEPS=$((RUN_STEPS + 1))
  start=$(date +%s)
  set +e
  "$@" >"$log" 2>&1
  status=$?
  set -e
  end=$(date +%s)
  duration=$((end - start))

  echo "${step},${status},${duration}s,${log}" >> "$MANUAL_RESULTS"
  if [ "$status" -eq 0 ]; then
    PASS_STEPS=$((PASS_STEPS + 1))
  else
    FAILED_STEPS=$((FAILED_STEPS + 1))
  fi
}

run_step "manual_tier0_uv_sync" uv sync --group dev
run_step "manual_tier0_pytest_smoke" uv run pytest -q tests/ --maxfail=1
run_step "manual_tier0_pytest_gpu" uv run pytest -q tests/ -m gpu

run_step "manual_tier1_root_help" uv run forge --help
for cmd in top info pipeline report serve web status autosense export teacher students benchmark embodiment embodyments demo quantize universal-distill curriculum profile train metrics models hyperparam finetune telemetry transfer eval; do
  run_step "manual_tier1_${cmd//-/_}_help" uv run forge "$cmd" --help
done

run_step "manual_tier1_json_contracts" uv run forge top --json

run_step "manual_tier2_info" uv run forge info
run_step "manual_tier2_pipeline_contract" uv run forge pipeline --help
run_step "manual_tier2_real_smoke" uv run forge eval smoke --checkpoint outputs/real_training/checkpoints/best.pt --json --device "${DEVICE}"
run_step "manual_tier2_benchmark" uv run forge benchmark run --device "${DEVICE}" --json
run_step "manual_tier2_eval_libero" uv run forge eval run libero --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${DEVICE}" --json
run_step "manual_tier2_eval_simpler" uv run forge eval run simpler --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${DEVICE}" --json
run_step "manual_tier2_eval_vlabench" uv run forge eval run vlabench --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${DEVICE}" --json

run_step "manual_tier3_pipeline_short" uv run forge pipeline --config configs/forge_nano.yaml --device "${DEVICE}" --max-steps 5
run_step "manual_tier3_serve_smoke" uv run forge serve --device "${DEVICE}" --checkpoint outputs/real_training/checkpoints/best.pt --port 8011

{
  echo
  echo "Manual e2e run complete."
  echo "Output dir: $OUT"
  echo "Total: $RUN_STEPS"
  echo "Pass: $PASS_STEPS"
  echo "Fail: $FAILED_STEPS"
  echo
  echo "Post-conditions to verify:"
  echo "- outputs/real_training/checkpoints/best.pt exists before serving/eval steps."
  echo "- outputs/validation/gpu_full/final_orchestrated/* for historical package/pytest checkpoints."
  echo "- outputs/validation/gpu_full/cli_contract_checks_uv_20260327/* for legacy CLI contract baselines."
  echo "- outputs/validation/gpu_full/tier3_final_20260327_182709/* for prior pipeline/eval service references."
  echo "- Any new logs under: $OUT"
} > "$OUT/manual_e2e_summary.md"
