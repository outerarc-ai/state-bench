#!/usr/bin/env bash
# Paired measurement of SGLang's per-path recurrent-state cap (--mamba-max-states-per-path)
# on one GPU, at several state-pool sizes.
#
# Usage: paired_path_cap.sh
#
# For each pool size in $POOLS (a --max-mamba-cache-size value, or "default" for SGLang's
# own sizing) this runs `state-bench paired run` with $PAIRS alternating pairs:
#   baseline   the server with the arguments below
#   candidate  the same server plus --mamba-max-states-per-path $CAP
# and writes $W/paired/pool-<size>/ with the per-request records, server logs, the run
# manifest, and report.json. The load is `state-bench paired load`: $SESSIONS sessions that
# share a $SHARED-token prefix and grow by $TURN_MIN-$TURN_MAX tokens per turn for $TURNS
# turns, sent one request at a time in round-robin order (an assumption, not a recorded
# workload). Run hybrid_cache_tests.sh gsm8k first: the candidate must pass the accuracy
# check before its timing means anything.
set -u
W=${W:-/work}
HERE=$(cd "$(dirname "$0")" && pwd)
STATE_BENCH=${STATE_BENCH:-$(cd "$HERE/../.." && pwd)}
. "$HERE/lib.sh"
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
POOLS=${POOLS:-"64 256 default"}
CAP=${CAP:-2}
PAIRS=${PAIRS:-5}
SESSIONS=${SESSIONS:-32}; TURNS=${TURNS:-8}; SHARED=${SHARED:-2048}
TURN_MIN=${TURN_MIN:-512}; TURN_MAX=${TURN_MAX:-2048}; MAX_NEW=${MAX_NEW:-32}
WARMUP=${WARMUP:-$SESSIONS}
PORT=${PORT:-39300}
export HF_HOME=${HF_HOME:-$W/hf} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export PYTHONPATH=$STATE_BENCH/src${PYTHONPATH:+:$PYTHONPATH}
echo "=== start $(date -Is) model $MODEL pools [$POOLS] cap $CAP pairs $PAIRS"
python3 -c "import jsonschema" 2>/dev/null || pip install -q "jsonschema>=4.18,<5"
python3 -m state_bench.cli --help >/dev/null || { echo "STOP: state-bench not importable"; exit 1; }
require_flag --mamba-max-states-per-path
pgrep -af "python[0-9.]* -m sglang[.]launch_server" && { echo "STOP: an SGLang server is already running"; exit 1; }
prefetch "$MODEL" || { echo "STOP: model download failed"; exit 1; }
mkdir -p "$W/paired"

for POOL in $POOLS; do
  RUN=$W/paired/pool-$POOL
  [ -e "$RUN" ] && { echo "skip pool $POOL: $RUN exists"; continue; }
  python3 - "$RUN.json" "$POOL" <<PY
import json, sys
out, pool = sys.argv[1], sys.argv[2]
server = ["python3", "-m", "sglang.launch_server", "--model-path", "$MODEL",
          "--host", "127.0.0.1", "--port", "$PORT", "--context-length", "32768",
          "--mamba-radix-cache-strategy", "extra_buffer"]
if pool != "default":
    server += ["--max-mamba-cache-size", pool]
cfg = {
    "name": f"path-cap-{pool}",
    "pairs": $PAIRS,
    "order": "abba",
    "warmup_requests": $WARMUP,
    "slo": {"ttft_s": None, "e2e_s": None},
    "device_check": ["bash", "$HERE/env_check.sh", "--idle"],
    "device_idle_pattern": "device idle",
    "ready": {"command": ["curl", "-sf", "-m", "2", "http://127.0.0.1:$PORT/health"],
              "timeout_s": 1800, "interval_s": 5},
    "load": ["python3", "-m", "state_bench.cli", "paired", "load", "http://127.0.0.1:$PORT",
             "-o", "{records}", "--sessions", "$SESSIONS", "--turns", "$TURNS",
             "--shared-prefix", "$SHARED", "--turn-min", "$TURN_MIN",
             "--turn-max", "$TURN_MAX", "--max-new-tokens", "$MAX_NEW"],
    "arms": {
        "baseline": {"start": server},
        "candidate": {"start": server + ["--mamba-max-states-per-path", "$CAP"]},
    },
}
json.dump(cfg, open(out, "w"), indent=2)
PY
  echo "=== pool $POOL run $(date -Is)"
  python3 -m state_bench.cli paired run "$RUN.json" -o "$RUN" || { echo "paired run failed"; continue; }
  rm -f "$RUN.json"  # the run directory keeps it as config.json
  python3 -m state_bench.cli paired report "$RUN" > "$RUN/report.json"
  python3 - "$RUN/report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("complete pairs", d["complete_pairs"], "warnings", d["warnings"])
for r in d["runs"]:
    m = r.get("metrics") or {}
    print(r["run"], r["pair"], r["arm"], r["status"], r.get("denominators"),
          {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
for c in d["comparisons"]:
    print(f"{c['metric']}: {c['verdict']}; median diff {c['diff_median']:.4f} "
          f"CI [{c['ci_low']:.4f}, {c['ci_high']:.4f}]")
PY
  echo "--- state pool size and cap seen by the servers"
  grep -h -o -E "max_mamba_cache_size=[0-9None]+|mamba_max_states_per_path=-?[0-9]+|Mamba Cache is allocated[^,]*" \
    "$RUN"/raw/*.server.log | sort | uniq -c
  grep -h -n -m2 -E "Traceback|HIP error|OutOfResources" "$RUN"/raw/*.server.log | cut -c1-200
done
echo "=== done $(date -Is)"
