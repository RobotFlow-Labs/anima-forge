#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${FORGE_MATRIX_OUTPUT_DIR:-./outputs/validation/gpu_full/matrix_full_${TS}}"
mkdir -p "$OUT"

RESULTS="$OUT/matrix_results.csv"
TIER4_MATRIX="$OUT/tier4_failure_matrix.csv"
SUMMARY="$OUT/matrix_status_summary.md"
MATRIX_SUMMARY_SCRIPT="${OUT}/_matrix_summary.py"
cat > "$MATRIX_SUMMARY_SCRIPT" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
for row in csv.DictReader(path.open(newline="", encoding="utf-8")):
    print(
        f"| {row['tier']} | {row['step']} | {row['status']} | {row['duration_s']} | {row['command']} | {row['log']} | {row['category']} |"
    )
PY

FORGE_MATRIX_STEP3_ONLY="${FORGE_MATRIX_STEP3_ONLY:-0}"
FORGE_MATRIX_RESUME_FROM="${FORGE_MATRIX_RESUME_FROM:-}"
FORGE_MATRIX_PROFILE_INTERVAL="${FORGE_MATRIX_PROFILE_INTERVAL:-10}"
TIER0_ENV_SCRIPT="${OUT}/tier0_env.py"
FAILED_STEPS=0
PASS_STEPS=0
BLOCKING_FAILURES=0
NONBLOCKING_FAILURES=0
CURRENT_TIER=0

cat > "$TIER0_ENV_SCRIPT" <<'PY'
import json
import os
import platform
import torch

print("python", platform.python_version())
print("platform", platform.platform())
print("cwd", os.getcwd())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count", torch.cuda.device_count())
    print("cuda_device", torch.cuda.get_device_name(0))
PY

declare -A TIER_PASS
declare -A TIER_FAIL
TIER_PASS[0]=0
TIER_PASS[1]=0
TIER_PASS[2]=0
TIER_PASS[3]=0
TIER_FAIL[0]=0
TIER_FAIL[1]=0
TIER_FAIL[2]=0
TIER_FAIL[3]=0

FORGE_REQUIRE_GPU="${FORGE_REQUIRE_GPU:-1}"
FORGE_QUANTIZE_GPUS="${FORGE_QUANTIZE_GPUS:-6,7}"
FORGE_MATRIX_DEVICE="${FORGE_MATRIX_DEVICE:-$(
  if [ "${FORGE_REQUIRE_GPU}" = "1" ]; then
    echo "cuda"
  else
    echo "auto"
  fi
)}"
FORGE_PIPELINE_TIMEOUT="${FORGE_PIPELINE_TIMEOUT:-900}"
FORGE_ALLOW_NONBLOCKING="${FORGE_ALLOW_NONBLOCKING:-0}"

declare -A PROFILED_STEPS=(
  [tier1_json_benchmark_run]=1
  [tier1_json_quant_bench]=1
  [tier1_json_quant_run]=1
  [tier1_json_train_start]=1
  [tier1_json_ud_start]=1
  [tier1_json_eval_smoke]=1
  [tier2_benchmark]=1
  [tier2_eval_smoke]=1
  [tier2_eval_libero]=1
  [tier2_eval_simpler]=1
  [tier2_eval_vlabench]=1
  [tier2_train_start]=1
  [tier2_profile_benchmark]=1
  [tier3_pipeline_short]=1
  [tier3_serve]=1
  [tier3_eval_serve]=1
  [tier3_eval_smoke_short]=1
  [tier3_eval_run_all]=1
)

if [ "${FORGE_MATRIX_STEP3_ONLY}" = "1" ] && [ ! -f "$RESULTS" ]; then
  echo "[FATAL] FORGE_MATRIX_STEP3_ONLY=1 requires existing results file at ${RESULTS}"
  exit 1
fi

if [ "${FORGE_MATRIX_STEP3_ONLY}" = "1" ]; then
  while IFS="=" read -r key value; do
    case "$key" in
      PASS_STEPS) PASS_STEPS="$value" ;;
      FAILED_STEPS) FAILED_STEPS="$value" ;;
    BLOCKING_FAILURES) BLOCKING_FAILURES="$value" ;;
    NONBLOCKING_FAILURES) NONBLOCKING_FAILURES="$value" ;;
    TIER_PASS_0) TIER_PASS[0]="${value}" ;;
    TIER_PASS_1) TIER_PASS[1]="${value}" ;;
    TIER_PASS_2) TIER_PASS[2]="${value}" ;;
    TIER_PASS_3) TIER_PASS[3]="${value}" ;;
      TIER_FAIL_0) TIER_FAIL[0]="${value}" ;;
      TIER_FAIL_1) TIER_FAIL[1]="${value}" ;;
      TIER_FAIL_2) TIER_FAIL[2]="${value}" ;;
      TIER_FAIL_3) TIER_FAIL[3]="${value}" ;;
    esac
  done < <(python3 - "$RESULTS" "$FORGE_MATRIX_RESUME_FROM" "$OUT" <<'PY'
import csv
import sys

matrix_path = sys.argv[1]
resume_from = sys.argv[2].strip()
out_dir = sys.argv[3]

tier3_steps = [
    "tier3_pipeline_short",
    "tier3_serve",
    "tier3_eval_serve",
    "tier3_eval_smoke_short",
    "tier3_eval_run_all",
]

resume_index = {name: idx for idx, name in enumerate(tier3_steps)}.get(resume_from, None)
if resume_from and resume_index is None:
    raise SystemExit(f"[FATAL] FORGE_MATRIX_RESUME_FROM='{resume_from}' is not a recognized tier3 step.")

keep_rows = []
with open(matrix_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tier = row.get("tier", "")
        step = row.get("step", "")
        if tier == "3":
            if not row.get("step"):
                continue
            step_idx = {"tier3_pipeline_short": 0, "tier3_serve": 1, "tier3_eval_serve": 2, "tier3_eval_smoke_short": 3, "tier3_eval_run_all": 4}.get(step)
            if step_idx is None:
                keep_rows.append(row)
                continue
            if resume_index is not None and step_idx < resume_index:
                keep_rows.append(row)
            elif resume_index is not None and step_idx >= resume_index:
                continue
            elif resume_index is None:
                continue
        else:
            keep_rows.append(row)

with open(matrix_path, "w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "tier",
        "step",
        "status",
        "duration_s",
        "command",
        "log",
        "category",
        "reason",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in keep_rows:
        writer.writerow(row)

pass_steps = 0
fail_steps = 0
block_fail = 0
nonblock_fail = 0
tier_pass = [0, 0, 0, 0]
tier_fail = [0, 0, 0, 0]

for row in keep_rows:
    status = int(row.get("status", "0") or 0)
    category = row.get("category", "code defect")
    tier = int(row.get("tier", 0) or 0)
    if status == 0:
        pass_steps += 1
        tier_pass[tier] += 1
    else:
        fail_steps += 1
        tier_fail[tier] += 1
        if category == "missing test coverage":
            nonblock_fail += 1
        else:
            block_fail += 1

artifact_root = out_dir
print(f"PASS_STEPS={pass_steps}")
print(f"FAILED_STEPS={fail_steps}")
print(f"BLOCKING_FAILURES={block_fail}")
print(f"NONBLOCKING_FAILURES={nonblock_fail}")
print(f"FORGE_MATRIX_RESUME_FROM={resume_from}")
print(f"ARTIFACT_ROOT={artifact_root}")
for i, val in enumerate(tier_pass):
    print(f"TIER_PASS_{i}={val}")
for i, val in enumerate(tier_fail):
    print(f"TIER_FAIL_{i}={val}")
PY
  )
else
  echo "tier,step,status,duration_s,command,log,category,reason" > "$RESULTS"
fi

if [ "${FORGE_MATRIX_STEP3_ONLY}" != "1" ]; then
  echo "tier,step,category,status,reason,log" > "$TIER4_MATRIX"
elif [ ! -f "$TIER4_MATRIX" ]; then
  echo "tier,step,category,status,reason,log" > "$TIER4_MATRIX"
fi

echo "# Matrix summary ${TS}" > "$SUMMARY"
echo "Output dir: $OUT" >> "$SUMMARY"
echo "Execution timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
echo "Required checkpoints referenced: outputs/real_training/checkpoints/best.pt" >> "$SUMMARY"
echo >> "$SUMMARY"

escape_csv() {
  local value="$1"
  printf '"%s"' "$(printf '%s' "${value}" | sed 's/"/""/g' | sed ':a;N;$!ba;s/\n/\\n/g')" # forge-public-audit: allow[private-unc-path]
}

a_record() {
  local tier="$1"
  local step="$2"
  local status="$3"
  local duration="$4"
  local cmd="$5"
  local log="$6"
  local category="$7"
  local reason="${8:-}"
  echo "$(escape_csv "$tier"),$(escape_csv "$step"),$(escape_csv "$status"),$(escape_csv "$duration"),$(escape_csv "$cmd"),$(escape_csv "$log"),$(escape_csv "$category"),$(escape_csv "$reason")" >> "$RESULTS"

  if [ "$status" -eq 0 ]; then
    PASS_STEPS=$((PASS_STEPS + 1))
    TIER_PASS[$tier]=$((TIER_PASS[$tier] + 1))
    return
  fi

  FAILED_STEPS=$((FAILED_STEPS + 1))
  TIER_FAIL[$tier]=$((TIER_FAIL[$tier] + 1))
  if [ "$category" = "missing test coverage" ]; then
    NONBLOCKING_FAILURES=$((NONBLOCKING_FAILURES + 1))
  else
    BLOCKING_FAILURES=$((BLOCKING_FAILURES + 1))
  fi
  echo "$(escape_csv "$tier"),$(escape_csv "$step"),$(escape_csv "$category"),$(escape_csv "$status"),$(escape_csv "$reason"),$(escape_csv "$log")" >> "$TIER4_MATRIX"
}

classify_failure() {
  local step="$1"
  local log="$2"
  local category="code defect"

  if [ -f "$log" ]; then
    if grep -Eq "FileNotFoundError|file not found|No such file|missing checkpoint|asset gap|FORGE_MODEL_DIR|weights not found|weights directory" "$log"; then
      category="asset/data gap"
    elif grep -Eq "ModuleNotFoundError|No module named|ImportError|cannot import|ResolutionImpossible|Could not find a version|uv sync" "$log"; then
      category="packaging/import defect"
    elif grep -Eq "docker|Docker|container|compose|runtime|socket|connection refused|cannot bind|eval serve|serve --checkpoint|heartbeat|api status" "$log"; then
      category="Docker/eval infra gap"
    elif grep -Eq "timed out|timeout after|out of memory|OOM|MemoryError" "$log"; then
      category="performance regression"
    elif grep -Eq "Traceback|ValueError|RuntimeError|AssertionError|TypeError|KeyError|AttributeError" "$log"; then
      category="code defect"
    fi
  fi

  if [ "$CURRENT_TIER" = "0" ] && [ "$category" = "code defect" ]; then
    category="packaging/import defect"
  fi

  if [ "$CURRENT_TIER" = "1" ] && [ "$category" = "code defect" ]; then
    category="CLI contract defect"
  fi

  echo "$category"
}

check_gpu_available() {
  if [ "${FORGE_REQUIRE_GPU}" != "1" ]; then
    return 0
  fi
  if ! uv run python -c "import torch, sys; import sys as _sys; _sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[FATAL] CUDA requested (FORGE_REQUIRE_GPU=1) but torch.cuda.is_available() is false."
    exit 1
  fi
}

run_step() {
  local step="$1"
  shift
  local log="$OUT/${step}.log"
  local cmd=( "$@" )
  local start end status duration
  local category="pass"
  local reason=""

  echo "[START] ${step} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  start=$(date +%s)
  set +e
  "${cmd[@]}" >> "$log" 2>&1
  status=$?
  set -e
  end=$(date +%s)
  duration=$((end - start))
  echo "[END] ${step} rc=${status} duration=${duration}s" | tee -a "$log"

  if [ "$status" -ne 0 ]; then
    category="$(classify_failure "$step" "$log")"
    if [ "$category" = "missing test coverage" ]; then
      reason="non-blocking known gap"
    elif [ "$category" = "pass" ]; then
      reason="classify-failed: command failed without classifier hit"
    else
      reason="classified-${category}"
    fi
  fi
  a_record "$CURRENT_TIER" "$step" "$status" "$duration" "${cmd[*]}" "$log" "$category" "$reason"
}

run_step_profiled() {
  local step="$1"
  shift
  local log="$OUT/${step}.log"
  local profiler_csv="$OUT/${step}_prof.csv"
  local cmd=( "$@" )
  local start end status duration
  local category="pass"
  local reason=""
  local cmd_str=""

  for arg in "${cmd[@]}"; do
    if [ -z "$cmd_str" ]; then
      printf -v cmd_str "%q" "$arg"
    else
      printf -v cmd_str " %s %q" "$cmd_str" "$arg"
    fi
  done

  echo "[START] ${step} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  start=$(date +%s)
  set +e
  uv run python scripts/gpu_fit_profiler.py --command "$cmd_str" --out "$profiler_csv" --interval "${FORGE_MATRIX_PROFILE_INTERVAL}" --json >> "$log" 2>&1
  status=$?
  set -e
  end=$(date +%s)
  duration=$((end - start))
  echo "[END] ${step} rc=${status} duration=${duration}s" | tee -a "$log"

  if [ "$status" -ne 0 ]; then
    category="$(classify_failure "$step" "$log")"
    if [ "$category" = "missing test coverage" ]; then
      reason="non-blocking known gap"
    elif [ "$category" = "pass" ]; then
      reason="classify-failed: command failed without classifier hit"
    else
      reason="classified-${category}"
    fi
  elif [ ! -f "${profiler_csv}.json" ]; then
    status=1
    category="performance regression"
    reason="missing-profile-json"
  fi
  a_record "$CURRENT_TIER" "$step" "$status" "$duration" "${cmd[*]}" "$log" "$category" "$reason"
}

run_step_timeout() {
  local timeout_s="$1"
  shift
  local step="$1"
  shift
  run_step "$step" timeout "$timeout_s" "$@"
}

run_step_auto() {
  local step="$1"
  shift
  if [ "${PROFILED_STEPS[$step]:-0}" = "1" ]; then
    run_step_profiled "$step" "$@"
  else
    run_step "$step" "$@"
  fi
}

run_step_timeout_profiled() {
  local timeout_s="$1"
  local step="$2"
  shift 2
  run_step_profiled "$step" timeout "$timeout_s" "$@"
}

run_python_step() {
  local step="$1"
  local script="$2"
  local log="$OUT/${step}.log"
  local tmp status duration start end
  if [ -f "$script" ]; then
    tmp="$script"
  else
    tmp="$(mktemp)"
    printf '%s\n' "$script" > "$tmp"
  fi

  echo "[START] ${step} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  start=$(date +%s)
  set +e
  uv run python "$tmp" >> "$log" 2>&1
  status=$?
  set -e
  end=$(date +%s)
  duration=$((end - start))
  if [ ! -f "$script" ]; then
    rm -f "$tmp"
  fi
  echo "[END] ${step} rc=${status} duration=${duration}s" | tee -a "$log"
  a_record "$CURRENT_TIER" "$step" "$status" "$duration" "uv run python" "$log" "pass" ""
}

run_service_smoke() {
  local step="$1"
  local port="$2"
  shift 2
  local log="$OUT/${step}.log"
  local pid_file="$OUT/${step}.pid"
  local start end duration
  local status=0
  local ready=0
  local pid

  echo "[START] ${step} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  start=$(date +%s)

  set +e
  "$@" > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$pid_file"

  for _ in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=1
      break
    fi
    if uv run python -c "import socket; sock = socket.socket(); sock.settimeout(1.0); sock.connect((\"127.0.0.1\", ${port})); sock.close()"; then
      ready=1
      break
    fi
    sleep 1
  done

  if [ "$ready" -eq 0 ]; then
    status=1
  fi

  if kill -0 "$pid" 2>/dev/null; then
    if [ "$status" -eq 0 ]; then
      kill "$pid" >/dev/null 2>&1 || status=1
      wait "$pid" >/dev/null 2>&1 || true
    else
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  fi
  set -e

  end=$(date +%s)
  duration=$((end - start))
  echo "[END] ${step} rc=${status} duration=${duration}s" | tee -a "$log"

  local category="pass"
  local reason=""
  if [ "$status" -ne 0 ]; then
    category="$(classify_failure "$step" "$log")"
    reason="classified-${category}"
  fi
  a_record "$CURRENT_TIER" "$step" "$status" "$duration" "service:${*}" "$log" "$category" "$reason"
}

run_service_smoke_profiled() {
  local step="$1"
  local port="$2"
  shift 2
  local log="$OUT/${step}.log"
  local pid_file="$OUT/${step}.pid"
  local profiler_csv="$OUT/${step}_prof.csv"
  local start end duration
  local status=0
  local ready=0
  local pid
  local profiler_status=0

  echo "[START] ${step} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log"
  start=$(date +%s)

  set +e
  "$@" > "$log" 2>&1 &
  pid=$!
  echo "$pid" > "$pid_file"
  uv run python scripts/gpu_fit_profiler.py --pid "$pid" --out "$profiler_csv" --interval "${FORGE_MATRIX_PROFILE_INTERVAL}" --json >> "$log" 2>&1 &
  profiler_status=$!

  for _ in $(seq 1 90); do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=1
      break
    fi
    if uv run python -c "import socket; sock = socket.socket(); sock.settimeout(1.0); sock.connect((\"127.0.0.1\", ${port})); sock.close()"; then
      ready=1
      break
    fi
    sleep 1
  done

  if [ "$ready" -eq 0 ]; then
    status=1
  fi

  if kill -0 "$pid" 2>/dev/null; then
    if [ "$status" -eq 0 ]; then
      kill "$pid" >/dev/null 2>&1 || status=1
      wait "$pid" >/dev/null 2>&1 || true
    else
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  fi
  wait "$profiler_status" 2>/dev/null || true
  set -e

  end=$(date +%s)
  duration=$((end - start))
  echo "[END] ${step} rc=${status} duration=${duration}s" | tee -a "$log"

  local category="pass"
  local reason=""
  if [ "$status" -ne 0 ]; then
    category="$(classify_failure "$step" "$log")"
    reason="classified-${category}"
  elif [ ! -f "${profiler_csv}.json" ]; then
    status=1
    category="performance regression"
    reason="missing-profile-json"
  fi
  a_record "$CURRENT_TIER" "$step" "$status" "$duration" "service:${*}" "$log" "$category" "$reason"
}

step3_index() {
  case "$1" in
    tier3_pipeline_short) echo 0 ;;
    tier3_serve) echo 1 ;;
    tier3_eval_serve) echo 2 ;;
    tier3_eval_smoke_short) echo 3 ;;
    tier3_eval_run_all) echo 4 ;;
    *) echo "" ;;
  esac
}

should_run_tier3_step() {
  local step="$1"
  local step_idx
  local resume_idx
  step_idx="$(step3_index "$step")"
  if [ -z "${FORGE_MATRIX_RESUME_FROM}" ]; then
    [ -n "$step_idx" ] && return 0
  fi
  resume_idx="$(step3_index "${FORGE_MATRIX_RESUME_FROM}")"
  [ -n "$step_idx" ] && [ -n "$resume_idx" ] && [ "$step_idx" -ge "$resume_idx" ]
}

check_gpu_available

run_tier0_steps() {
  CURRENT_TIER=0
  run_step "tier0_uv_sync" uv sync --group dev
  run_step "tier0_uv_build" uv build
  run_python_step "tier0_env" "$TIER0_ENV_SCRIPT"
  run_step "tier0_pytest_all" uv run pytest -q -o timeout=0 tests/ --maxfail=1
  run_step "tier0_pytest_gpu" uv run pytest -q tests/ -m gpu
}

run_tier1_steps() {
  CURRENT_TIER=1
  for cmd in "top" "agent" "agent-top" "top-agent" "info" "pipeline" "report" "serve" "web" "status" "autosense" "export" "teacher" "students" "benchmark" "embodiment" "embodyments" "demo" "quantize" "universal-distill" "curriculum" "profile" "train" "metrics" "models" "hyperparam" "finetune" "telemetry" "transfer" "eval"; do
    run_step "tier1_help_${cmd//-/_}" uv run forge "$cmd" --help
  done

  run_step_auto "tier1_json_top" uv run forge top --json
  run_step_auto "tier1_json_status" uv run forge status --json
  run_step_auto "tier1_json_autosense" uv run forge autosense --json
  run_step_auto "tier1_json_benchmark_run" uv run forge benchmark run --device "${FORGE_MATRIX_DEVICE}" --json
  run_step_auto "tier1_json_quant_bench" CUDA_VISIBLE_DEVICES="${FORGE_QUANTIZE_GPUS}" uv run forge quantize bench --device "${FORGE_MATRIX_DEVICE}" --method turboquant-mse --bits 3 --json
  run_step_auto "tier1_json_quant_run" CUDA_VISIBLE_DEVICES="${FORGE_QUANTIZE_GPUS}" uv run forge quantize run --method turboquant-mse --bits 3 --json --device "${FORGE_MATRIX_DEVICE}"
  run_step_auto "tier1_json_train_start" uv run forge train start --json --device "${FORGE_MATRIX_DEVICE}" --max-steps 5
  run_step "tier1_json_train_status" uv run forge train status --json
  run_step_auto "tier1_json_ud_start" uv run forge universal-distill start --mode staged --staged --json --device "${FORGE_MATRIX_DEVICE}"
  run_step "tier1_json_ud_status" uv run forge universal-distill status --json
  run_step_auto "tier1_json_eval_smoke" uv run forge eval smoke --checkpoint outputs/real_training/checkpoints/best.pt --json --device "${FORGE_MATRIX_DEVICE}"
}

run_tier2_steps() {
  CURRENT_TIER=2
  run_step "tier2_info" uv run forge info
  run_step "tier2_autosense" uv run forge autosense --json
  run_step "tier2_eval_setup" uv run forge eval setup --skip-pull
  run_step_auto "tier2_benchmark" uv run forge benchmark run --device "${FORGE_MATRIX_DEVICE}" --json
  run_step_auto "tier2_eval_smoke" uv run forge eval smoke --checkpoint outputs/real_training/checkpoints/best.pt --json --device "${FORGE_MATRIX_DEVICE}"
  run_step_auto "tier2_eval_libero" uv run forge eval run libero --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${FORGE_MATRIX_DEVICE}" --json
  run_step_auto "tier2_eval_simpler" uv run forge eval run simpler --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${FORGE_MATRIX_DEVICE}" --json
  run_step_auto "tier2_eval_vlabench" uv run forge eval run vlabench --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${FORGE_MATRIX_DEVICE}" --json
  run_step "tier2_demo" uv run forge demo --device "${FORGE_MATRIX_DEVICE}" --steps 20
  run_step_auto "tier2_train_start" uv run forge train start --json --device "${FORGE_MATRIX_DEVICE}" --max-steps 10
  run_step_auto "tier2_profile_benchmark" uv run forge profile benchmark --device "${FORGE_MATRIX_DEVICE}" --variant nano --samples 10 --json
}

run_tier3_steps() {
  CURRENT_TIER=3
  if should_run_tier3_step "tier3_pipeline_short"; then
    run_step_timeout_profiled "${FORGE_PIPELINE_TIMEOUT}" tier3_pipeline_short uv run forge pipeline --config configs/forge_nano.yaml --device "${FORGE_MATRIX_DEVICE}" --max-steps 5
  fi
  if should_run_tier3_step "tier3_serve"; then
    run_service_smoke_profiled "tier3_serve" 8013 uv run forge serve --device "${FORGE_MATRIX_DEVICE}" --checkpoint outputs/real_training/checkpoints/best.pt --port 8013
  fi
  if should_run_tier3_step "tier3_eval_serve"; then
    run_service_smoke_profiled "tier3_eval_serve" 8014 uv run forge eval serve --checkpoint outputs/real_training/checkpoints/best.pt --port 8014 --device "${FORGE_MATRIX_DEVICE}"
  fi
  if should_run_tier3_step "tier3_eval_smoke_short"; then
    run_step_auto "tier3_eval_smoke_short" uv run forge eval smoke --checkpoint outputs/real_training/checkpoints/best.pt --json --device "${FORGE_MATRIX_DEVICE}"
  fi
  if should_run_tier3_step "tier3_eval_run_all"; then
    run_step_timeout_profiled 1800 tier3_eval_run_all uv run forge eval run-all --checkpoint outputs/real_training/checkpoints/best.pt --episodes 1 --max-tasks 1 --device "${FORGE_MATRIX_DEVICE}" --json
  fi
}

if [ "${FORGE_MATRIX_STEP3_ONLY}" = "1" ]; then
  run_tier3_steps
else
  run_tier0_steps
  run_tier1_steps
  run_tier2_steps
  run_tier3_steps
fi

{
  echo ""
  echo "## Command results"
  echo ""
  echo "Executed on: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Total checks: $((PASS_STEPS + FAILED_STEPS))"
  echo "Pass: $PASS_STEPS"
  echo "Fail: $FAILED_STEPS"
  echo "Blocking failures: $BLOCKING_FAILURES"
  echo "Non-blocking failures: $NONBLOCKING_FAILURES"
  echo
  echo "| Tier | Step | Status | Duration (s) | Command | Log | Category |"
  echo "| --- | --- | --- | --- | --- | --- | --- |"
  uv run python "$MATRIX_SUMMARY_SCRIPT" "$RESULTS"
} >> "$SUMMARY"

{
  echo ""
  echo "## Tier 4 summary"
  echo "| Tier | Pass | Fail |"
  echo "| --- | --- | --- |"
  echo "| 0 | ${TIER_PASS[0]} | ${TIER_FAIL[0]} |"
  echo "| 1 | ${TIER_PASS[1]} | ${TIER_FAIL[1]} |"
  echo "| 2 | ${TIER_PASS[2]} | ${TIER_FAIL[2]} |"
  echo "| 3 | ${TIER_PASS[3]} | ${TIER_FAIL[3]} |"
} >> "$SUMMARY"

{
  echo ""
  echo "## Gate-closure criteria"
  echo "- Tier 0/1/2/3 all pass: PASS"
  echo "- No category in Tier4 blocker list except \`missing test coverage\`"
  echo "- Blocking categories: packaging/import defect, code defect, CLI contract defect, asset/data gap, Docker/eval infra gap, performance regression"
  echo "- Non-blocking category: missing test coverage"
  echo "- Required checkpoints/artifacts: \`outputs/real_training/checkpoints/best.pt\` and \`outputs/checkpoints/final.pt\` when produced"
  echo ""
  echo "Existing artifact roots for review:"
  echo "- outputs/validation/gpu_full/final_orchestrated/*"
  echo "- outputs/validation/gpu_full/cli_contract_checks_uv_20260327/*"
  echo "- outputs/validation/gpu_full/manual_test_matrix.md"
  echo "- outputs/validation/gpu_full/tier3_final_20260327_182709/*"
  echo "- /tmp/eval_smoke_run.log and /tmp/eval_vlabench_run.log"
  echo "- ${OUT}"
} >> "$SUMMARY"

if [ "${TIER_FAIL[0]}" -ne 0 ] || [ "${TIER_FAIL[1]}" -ne 0 ] || [ "${TIER_FAIL[2]}" -ne 0 ] || [ "${TIER_FAIL[3]}" -ne 0 ]; then
  if [ "$BLOCKING_FAILURES" -ne 0 ]; then
    echo "GATE: BLOCKED - blocking failures detected." >> "$SUMMARY"
    exit 1
  fi
  if [ "$FORGE_ALLOW_NONBLOCKING" = "1" ]; then
    echo "GATE: CONDITIONAL-OPEN - only non-blocking failures." >> "$SUMMARY"
    exit 0
  fi
  echo "GATE: BLOCKED - failures found and no explicit non-blocking override." >> "$SUMMARY"
  exit 1
fi

echo "GATE: OPEN - all tiers and categories closed." >> "$SUMMARY"
exit 0
