#!/usr/bin/env bash
set -xeuo pipefail

LOG_DIR= ... # set your log directory
PATH_TO_GPT_MODEL= ... # set your path to gpt-oss-120b model

# ---------------------- CLI args ---------------------- #
RUN_ID="run_64k_prompt1"
SP_ID=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --sp-id)  SP_ID="$2";  shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--run-id my_run_tag] [--sp-id 1|2|3]" >&2
      exit 2
      ;;
  esac
done

# 0) per-job unique ID
if [ -n "${SLURM_JOB_ID-}" ]; then
  JOB_ID=$SLURM_JOB_ID
else
  JOB_ID=$(date +%Y%m%d_%H%M%S)_$$
fi
echo "JOB_ID = $JOB_ID"

# 1) cluster / node settings
export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-0}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-8}
export TP_SIZE=${TP_SIZE:-2}
export DP_SIZE=${DP_SIZE:-4}
echo "Node $NODE_RANK/$NNODES  •  GPUs/node=$GPUS_PER_NODE  •  tp=$TP_SIZE  •  dp=$DP_SIZE"
echo "Run: $RUN_ID  •  System Prompt ID: $SP_ID"

mkdir -p "$LOG_DIR"
LOG_SGLANG=${LOG_DIR}/${JOB_ID}_sglang_node${NODE_RANK}.log

# 4) launch SGLang in background, capture wrapper PID
python3 -u -m sglang.launch_server \
  --model-path $PATH_TO_GPT_MODEL \
  --tp "$TP_SIZE" \
  --dp "$DP_SIZE" \
  --port 30000 \
  >"$LOG_SGLANG" 2>&1 &
WRAPPER_PID=$!
echo "Launched sglang wrapper (PID=$WRAPPER_PID), logging to $LOG_SGLANG"

# 4a) give wrapper a moment to fork real server
sleep 1

# 4b) find the real server PID(s) listening on port 30000
SERVER_PIDS=( $(lsof -t -iTCP:30000 -sTCP:LISTEN || true) )
if [ ${#SERVER_PIDS[@]} -eq 0 ]; then
  echo "⚠️  couldn't find server listening on :30000; continuing with wrapper PID"
  SERVER_PIDS=($WRAPPER_PID)
fi
echo "  → server PID(s): ${SERVER_PIDS[*]}"

# 5) health-check loop
MODELS_URL="http://localhost:30000/v1/models"
MAX_ATTEMPTS=300
SLEEP_SECS=2

echo "Waiting for SGLang to report healthy via $MODELS_URL…"
for ((i=1; i<=MAX_ATTEMPTS; i++)); do
  http_code=$(curl -s -o /dev/null -w '%{http_code}' "$MODELS_URL" || echo 000)
  if [ "$http_code" = "200" ]; then
    echo "✓ SGLang healthy after $i attempts (~$((i*SLEEP_SECS))s)"
    break
  fi
  printf "  • attempt %3d/%d – status=%s\n" "$i" "$MAX_ATTEMPTS" "$http_code"
  sleep $SLEEP_SECS
  if [ $i -eq $MAX_ATTEMPTS ]; then
    echo "✗ SGLang did not become healthy after $((MAX_ATTEMPTS*SLEEP_SECS))s" >&2
    exit 1
  fi
done


LOG_INF=${LOG_DIR}/${JOB_ID}_infer_node${NODE_RANK}.log
python infer_llm.py \
  --run-identifier "$RUN_ID" \
  --system-prompt-id "$SP_ID" \
  >"$LOG_INF" 2>&1

# 7) shutdown: kill real server(s), then wrapper
for pid in "${SERVER_PIDS[@]}"; do
  kill "$pid" || true
done
kill "$WRAPPER_PID" || true
wait "$WRAPPER_PID" 2>/dev/null || true

echo "Node $NODE_RANK finished (JOB_ID=$JOB_ID)"