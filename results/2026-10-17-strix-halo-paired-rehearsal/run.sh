#!/usr/bin/env bash
# Rehearsal of `state-bench paired` on Strix Halo (gfx1151), shared host: Qwen3.5-4B,
# baseline versus --mamba-max-states-per-path 2, five alternating pairs (abba). Checks the
# tool end to end; not a performance measurement. Run from a clone of this repository at
# commit f50d140 (the source copied into the container):
#   bash results/2026-10-17-strix-halo-paired-rehearsal/run.sh <ssh-host> 2>&1 | tee run.log
# Touches only container "claude-statebench" and /tmp/claude-statebench inside it; port 39117.
set -u
HOST=${1:?usage: run.sh <ssh-host>}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
NAME=claude-statebench
echo "=== copy state-bench f50d140 src into the container"
git -C "$REPO" archive f50d140 src | ssh "$HOST" "docker exec -i $NAME sh -c 'mkdir -p /tmp/claude-statebench/paired17 && tar -x -C /tmp/claude-statebench/paired17'"
echo "=== part B"
ssh "$HOST" "docker exec -i $NAME bash -s" <<'IN2'
set -u
W=/tmp/claude-statebench/paired17
MD=/models/Qwen/Qwen3.5-4B
[ -f "$MD/config.json" ] || { echo "STOP: $MD missing"; exit 0; }
stop_servers() {  # stop every SGLang server process in this container (ours only)
  local pat="python[0-9.]* -m sglang[.]launch_server|^sglang::"
  pgrep -f "$pat" >/dev/null || return 0
  pkill -f "$pat" 2>/dev/null
  for _ in $(seq 60); do pgrep -f "$pat" >/dev/null || return 0; sleep 1; done
  pkill -9 -f "$pat" 2>/dev/null; sleep 3
  ! pgrep -af "$pat"
}
if pgrep -af "python[0-9.]* -m sglang[.]launch_server|^sglang::"; then
  echo "--- server processes still running from an earlier run; stopping them"
  stop_servers || { echo "STOP: could not stop them"; exit 0; }
fi
cd "$W" && rm -rf run cfg.json load.py
trap 'stop_servers; echo "=== part B cleanup done"' EXIT
export PYTHONPATH="$W/src"
cat > load.py <<'PY'
"""Deterministic multi-turn load for the paired rehearsal (writes state-bench records).

Usage: python load.py URL OUT [--sessions 8 --turns 6 --seed 17]

Sessions share a 1024-token prefix (a stand-in system prompt). Each turn appends 256-1024
new tokens to the previous prompt (an append chain). Requests are sent one at a time in
round-robin order across sessions, so sessions interleave and compete for the state
pool. Token ids are placeholders drawn from a seeded generator.
"""

import argparse
import json
import random
import time
import urllib.request


def stream_generate(url, ids, max_new):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}, "stream": True}
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    meta = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            if ttft is None:
                ttft = time.perf_counter() - t0
            obj = json.loads(payload)
            meta = obj.get("meta_info", meta)
    return ttft, time.perf_counter() - t0, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-new", type=int, default=16)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = lambda n: [rng.randrange(1000, 100000) for _ in range(n)]  # noqa: E731
    shared = tok(1024)
    prompts = [list(shared) for _ in range(a.sessions)]
    adds = [[rng.randrange(256, 1025) for _ in range(a.turns)] for _ in range(a.sessions)]
    adds_tok = [[tok(n) for n in row] for row in adds]
    start = time.perf_counter()
    with open(a.out, "w") as f:
        for t in range(a.turns):
            for s in range(a.sessions):
                prompts[s] = prompts[s] + adds_tok[s][t]
                rec = {"req_id": f"s{s}-t{t}", "t_send_s": round(time.perf_counter() - start, 6),
                       "prompt_tokens": len(prompts[s]), "valid": None}
                try:
                    ttft, e2e, meta = stream_generate(a.url, prompts[s], a.max_new)
                    rec.update(admitted=True, completed=True, ttft_s=ttft, e2e_s=e2e,
                               output_tokens=meta.get("completion_tokens"),
                               cached_tokens=meta.get("cached_tokens"))
                except Exception as e:  # recorded, not raised
                    rec.update(admitted=False, completed=False, error=f"{type(e).__name__}: {e}"[:300])
                f.write(json.dumps(rec) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
PY
echo "=== part B sizing"
MEM=$(python3 - <<'PY'
import psutil, sys
avail = psutil.virtual_memory().available / 2**30
need = 9.0 + 1.0 + 2.0 + 3.0   # weights, 16 state slots, KV for 64k tokens, headroom (GiB)
f = need / avail
print(f"{f:.3f}" if avail >= need / 0.9 else "0")
print(f"# available {avail:.1f} GiB, footprint {need:.1f} GiB, fraction {f:.3f}", file=sys.stderr)
PY
)
[ "$MEM" != "0" ] || { echo "STOP: not enough free memory now; retry later"; exit 0; }
python3 - "$MD" "$MEM" <<'PY'
import json, sys
md, mem = sys.argv[1], sys.argv[2]
server = ["python3", "-m", "sglang.launch_server", "--model-path", md, "--host", "127.0.0.1",
          "--port", "39117", "--mem-fraction-static", mem, "--max-mamba-cache-size", "16",
          "--max-running-requests", "2", "--mamba-radix-cache-strategy", "extra_buffer",
          "--mamba-track-interval", "128", "--attention-backend", "triton",
          "--linear-attn-backend", "triton", "--mamba-backend", "triton",
          "--chunked-prefill-size", "2048", "--context-length", "16384"]
env = {"SGLANG_DISABLE_LEAN_ATTENTION": "1"}
cfg = {
    "name": "strix-halo-paired-rehearsal",
    "pairs": 5, "order": "abba", "warmup_requests": 2, "slo": {"ttft_s": 10.0},
    "device_check": ["sh", "-c", "rocm-smi --showpids 2>/dev/null | grep -E '^[0-9]' | head -20; echo checked"],
    "ready": {"command": ["curl", "-sf", "-m", "2", "http://127.0.0.1:39117/health"], "timeout_s": 900, "interval_s": 5},
    "load": ["python3", "load.py", "http://127.0.0.1:39117", "{records}", "--sessions", "8", "--turns", "6"],
    "arms": {"baseline": {"start": server, "env": env},
             "candidate": {"start": server + ["--mamba-max-states-per-path", "2"], "env": env}},
}
json.dump(cfg, open("cfg.json", "w"), indent=1)
PY
echo "=== part B run $(date -Is)"
python3 -m state_bench.cli paired run cfg.json -o run
python3 -m state_bench.cli paired report run > run/report.json; echo "report exit $?"
python3 - <<'PY'
import json
d = json.load(open("run/report.json"))
print("complete pairs", d["complete_pairs"], "warnings", d["warnings"])
for r in d["runs"]:
    m = r.get("metrics") or {}
    print(r["run"], r["pair"], r["arm"], r["status"], r.get("denominators"), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
for c in d["comparisons"]:
    print(c["metric"], c["verdict"], "median diff", round(c["diff_median"], 4), "CI", round(c["ci_low"], 4), round(c["ci_high"], 4))
PY
echo "--- server log errors (if any)"; grep -h -n -E "Traceback|Error|OutOfResources" run/raw/*.server.log | head -8 | cut -c1-200
echo "--- per-path cap evidence"; grep -h -o "max_states_per_path=[-0-9]*" run/raw/*.server.log | sort | uniq -c
echo "=== part B artifacts (base64 tar.gz)"
tar -czf - run/config.json run/report.json run/raw/runs.json run/raw/*.jsonl | base64 -w0; echo
echo "=== end artifacts"
IN2
echo "=== done $(date -Is)"
