#!/usr/bin/env bash
set -xeuo pipefail

LOG_DIR=logs

# ---------------------- Parse CLI args ---------------------- #
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --model-path PATH     Model path (required)"
    echo "  -o, --output-path PATH    Output path (required)"
    echo "  -p, --port PORT           Port number (default: 30000)"
    echo "  -h, --help               Show this help message"
    exit 1
}

# Default values
MODEL_PATH=""
OUTPUT_PATH=""
TEMPERATURE=0.6
TOP_P=0.95
PRESENCE_PENALTY=0
FREQUENCY_PENALTY=0
PORT="${PORT:-30000}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -o|--output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top_p)
            TOP_P="$2"
            shift 2
            ;;
        --presence_penalty)
            PRESENCE_PENALTY="$2"
            shift 2
            ;;
        --frequency_penalty)
            FREQUENCY_PENALTY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: Model path is required (-m/--model-path)" >&2
    usage
fi

if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: Output path is required (-o/--output-path)" >&2
    usage
fi

echo "MODEL_PATH = $MODEL_PATH"
echo "OUTPUT_PATH = $OUTPUT_PATH"
echo "PORT = $PORT"

# 0) per-job unique ID
if [ -n "${SLURM_JOB_ID-}" ]; then
    JOB_ID=$SLURM_JOB_ID
else
    JOB_ID=$(date +%Y%m%d_%H%M%S)_$$
fi
echo "JOB_ID = $JOB_ID"

export NNODES=${MLP_WORKER_NUM:-1}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-0}
export GPUS_PER_NODE=${MLP_WORKER_GPU:-8}
export TP_SIZE=${TP_SIZE:-2}
export DP_SIZE=${DP_SIZE:-4}
echo "Node $NODE_RANK/$NNODES • GPUs/node=$GPUS_PER_NODE • tp=$TP_SIZE • dp=$DP_SIZE"

mkdir -p "$LOG_DIR"
LOG_SGLANG=${LOG_DIR}/${JOB_ID}_sglang_node${NODE_RANK}.log

MODEL_PATH_LOWER=$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
if [[ "$MODEL_PATH_LOWER" == *"7b"* ]] && [[ "$MODEL_PATH_LOWER" == *"miromind"* ]]; then
    python3 -u -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp "$TP_SIZE" \
        --dp "$DP_SIZE" \
        --port "$PORT" \
        --trust-remote-code \
        >"$LOG_SGLANG" 2>&1 &
else
    python3 -u -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp "$TP_SIZE" \
        --dp "$DP_SIZE" \
        --port "$PORT" \
        --trust-remote-code \
        --json-model-override-args '{"rope_scaling":{"type":"yarn","factor":2.0,"original_max_position_embeddings":32768}, "max_position_embeddings":65536}' \
        >"$LOG_SGLANG" 2>&1 &
fi
WRAPPER_PID=$!
echo "Launched sglang wrapper (PID=$WRAPPER_PID), logging to $LOG_SGLANG"

sleep 1

SERVER_PIDS=( $(lsof -t -iTCP:$PORT -sTCP:LISTEN || true) )
if [ ${#SERVER_PIDS[@]} -eq 0 ]; then
    echo "⚠️ couldn't find server listening on :$PORT; continuing with wrapper PID"
    SERVER_PIDS=($WRAPPER_PID)
fi
echo " → server PID(s): ${SERVER_PIDS[*]}"

MODELS_URL="http://localhost:${PORT}/v1/models"
MAX_ATTEMPTS=300
SLEEP_SECS=2
echo "Waiting for SGLang to report healthy via $MODELS_URL…"

for ((i=1; i<=MAX_ATTEMPTS; i++)); do
    http_code=$(curl -s -o /dev/null -w '%{http_code}' "$MODELS_URL" || echo 000)
    if [ "$http_code" = "200" ]; then
        echo "✓ SGLang healthy after $i attempts (~$((i*SLEEP_SECS))s)"
        break
    fi
    printf " • attempt %3d/%d – status=%s\n" "$i" "$MAX_ATTEMPTS" "$http_code"
    sleep $SLEEP_SECS
    if [ $i -eq $MAX_ATTEMPTS ]; then
        echo "✗ SGLang did not become healthy after $((MAX_ATTEMPTS*SLEEP_SECS))s" >&2
        exit 1
    fi
done

LOG_INF=${LOG_DIR}/${JOB_ID}_infer_node${NODE_RANK}.log
python collect_rollout_using_sglang.py \
    --model-str "$MODEL_PATH" \
    --out-dir "$OUTPUT_PATH" \
    --max-len 32768 \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --presence_penalty $PRESENCE_PENALTY \
    --frequency_penalty $FREQUENCY_PENALTY

# 7) shutdown: kill real server(s), then wrapper
for pid in "${SERVER_PIDS[@]}"; do
    kill "$pid" || true
done
kill "$WRAPPER_PID" || true
wait "$WRAPPER_PID" 2>/dev/null || true

echo "Node $NODE_RANK finished (JOB_ID=$JOB_ID)"