#!/usr/bin/env bash
# Inputs for the restore-versus-recompute estimate: prefill time against prompt length for
# each model in $MODELS, and host-device copy bandwidth.
#
# Usage: prefill_and_copy.sh
#
# Writes $W/prefill/<model>.json (prefill_curve.py) and $W/copy_bandwidth.json
# (copy_bandwidth.py). Nemotron-H models do not start on ROCm without the fixes in
# nemotron_h_repro.sh, so the default models are Qwen3.5 hybrids.
set -u
W=${W:-/work}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
MODELS=${MODELS:-"Qwen/Qwen3.5-9B Qwen/Qwen3.5-35B-A3B"}
LENGTHS=${LENGTHS:-512,1024,2048,4096,8192,16384,32768,65536,131072}
REPS=${REPS:-5}
PORT=${PORT:-39400}
export HF_HOME=${HF_HOME:-$W/hf} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
mkdir -p "$W/prefill"
echo "=== start $(date -Is) models [$MODELS]"
bash "$HERE/env_check.sh" --idle
trap 'stop_server' EXIT

for M in $MODELS; do
  prefetch "$M" || continue
  NAME=$(basename "$M")
  start_server "prefill-$NAME" "$PORT" --model-path "$M" --context-length 140000 \
    --chunked-prefill-size 16384 || continue
  python3 "$HERE/prefill_curve.py" "http://127.0.0.1:$PORT" "$W/prefill/$NAME.json" \
    --lengths "$LENGTHS" --reps "$REPS" | tail -n 12
  cp "$W/logs/prefill-$NAME.server.log" "$W/prefill/$NAME.server.log"
  stop_server
done

echo "=== copy bandwidth $(date -Is)"
python3 "$HERE/copy_bandwidth.py" "$W/copy_bandwidth.json"
bash "$HERE/env_check.sh" --idle
echo "=== done $(date -Is)"
