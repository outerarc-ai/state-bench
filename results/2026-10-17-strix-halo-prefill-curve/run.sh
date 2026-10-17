#!/usr/bin/env bash
# Prefill time against prompt length for NVIDIA-Nemotron-Nano-9B-v2 on Strix Halo
# (gfx1151), shared host. Functional input for the restore-versus-recompute estimate; not a
# performance measurement. Run from a workstation:
#   ssh <host> 'bash -s' < run.sh 2>&1 | tee run.log
# Touches only container "claude-statebench" and /tmp/claude-statebench inside it; uses
# port 39116; restores the two temporary source patches on exit.
set -u
NAME=claude-statebench
echo "=== start $(date -Is)"
echo "=== preflight"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "STOP: container $NAME does not exist"; exit 0; }
docker inspect -f 'state {{.State.Status}} exit {{.State.ExitCode}} oom {{.State.OOMKilled}} started {{.State.StartedAt}} finished {{.State.FinishedAt}} restarts {{.RestartCount}}' "$NAME"
if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
  echo "--- last container log lines before it stopped"
  docker logs --tail 15 "$NAME" 2>&1 | cut -c1-200
  echo "--- host uptime and memory"; uptime; free -g | head -2
  echo "=== starting $NAME (our own container; nothing else is touched)"
  docker start "$NAME" >/dev/null || { echo "STOP: docker start failed"; exit 0; }
  for _ in $(seq 30); do [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] && break; sleep 2; done
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] || { echo "STOP: $NAME did not stay up"; docker logs --tail 15 "$NAME" 2>&1 | cut -c1-200; exit 0; }
  sleep 5
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = "true" ] || { echo "STOP: $NAME exited right after start"; docker logs --tail 15 "$NAME" 2>&1 | cut -c1-200; exit 0; }
fi
docker exec "$NAME" true || { echo "STOP: cannot exec in $NAME"; exit 0; }
docker exec "$NAME" sh -c 'ls -d /tmp/claude-statebench /tmp/claude-statebench/models/* 2>&1 | head; df -h /tmp | tail -1' 
echo "--- GPU users (any process here belongs to another job; we stop rather than share the GPU)"
GPU_PIDS=$(docker exec "$NAME" sh -c "rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[[:space:]]/{print \$1\"(\"\$2\")\"}'" | tr '\n' ' ')
echo "kfd processes: ${GPU_PIDS:-none}"
if [ -n "$GPU_PIDS" ] && [ "${ALLOW_SHARED_GPU:-0}" != "1" ]; then
  echo "STOP: the GPU is in use by another job. Run this round later, or, if those processes"
  echo "      are idle and memory allows, rerun with: ssh <host> 'ALLOW_SHARED_GPU=1 bash -s' < run.sh"
  exit 0
fi
echo "=== part A"
docker exec -i "$NAME" bash -s <<'IN'
set -u
cd /tmp/claude-statebench
MD=/tmp/claude-statebench/models/NVIDIA-Nemotron-Nano-9B-v2
[ -f "$MD/config.json" ] || { echo "STOP: model missing"; exit 0; }
stop_servers() {  # stop every SGLang server process in this container (ours only)
  local pat="python[0-9.]* -m sglang[.]launch_server|^sglang::"
  pgrep -f "$pat" >/dev/null || return 0
  pkill -f "$pat" 2>/dev/null
  for _ in $(seq 60); do pgrep -f "$pat" >/dev/null || return 0; sleep 1; done
  pkill -9 -f "$pat" 2>/dev/null; sleep 3
  ! pgrep -af "$pat"
}
if pgrep -af "python[0-9.]* -m sglang[.]launch_server|^sglang::"; then
  echo "--- leftover server processes from an earlier run; stopping them"
  stop_servers || { echo "STOP: could not stop them"; exit 0; }
fi
echo "=== config"
python3 - "$MD" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
pat = c.get("hybrid_override_pattern", "")
keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "attention_head_dim", "intermediate_size", "vocab_size"]
print({k: c.get(k) for k in keys})
print("pattern", pat, "M", pat.count("M"), "attn", pat.count("*"), "mlp", pat.count("-"))
PY
echo "=== sizing"
MEM=$(python3 - <<'PY'
import psutil
avail = psutil.virtual_memory().available / 2**30
# weights 16.5 + 16 state slots 2.2 + KV for 70k tokens 1.1 + headroom 3 GiB; our cap 40 GiB
need = 16.5 + 2.2 + 1.1 + 3.0
f = need / avail
print(f"{f:.3f}" if avail >= need / 0.9 and need <= 40 else "0")
import sys; print(f"# available {avail:.1f} GiB, footprint {need:.1f} GiB, fraction {f:.3f}", file=sys.stderr)
PY
)
[ "$MEM" != "0" ] || { echo "STOP: not enough free memory now; retry later"; exit 0; }

MF=/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/mamba.py
AF=/sgl-workspace/sglang/python/sglang/srt/layers/activation.py
cp "$MF" mamba.py.orig; cp "$AF" activation.py.orig
restore() { cp mamba.py.orig "$MF"; cp activation.py.orig "$AF"; stop_servers && echo "=== restored sources, server stopped" || echo "WARNING: server processes still running"; }
trap restore EXIT
python3 - "$MF" "$AF" <<'PY'
import sys
m, a = sys.argv[1], sys.argv[2]
s = open(m).read()
assert "elif is_xpu():" in s and "is_hip()" not in s
s = s.replace("elif is_xpu():", "elif is_hip():\n    from sglang.kernels.ops.mamba.causal_conv1d_triton import (\n        causal_conv1d_fn as causal_conv1d_fn,\n        causal_conv1d_fn as causal_conv1d_fn_triton,\n        causal_conv1d_update as causal_conv1d_update,\n        causal_conv1d_update as causal_conv1d_update_triton,\n    )\nelif is_xpu():", 1)
s = s.replace("    is_cuda,\n", "    is_cuda,\n    is_hip,\n", 1)
open(m, "w").write(s)
s = open(a).read()
anchor = "    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:\n        return relu2(x)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + "\n    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:\n        return self.forward_native(x)\n")
open(a, "w").write(s)
PY

echo "=== launch $(date -Is)"
SGLANG_DISABLE_LEAN_ATTENTION=1 nohup python3 -m sglang.launch_server --model-path "$MD" \
  --host 127.0.0.1 --port 39116 --max-mamba-cache-size 16 --mem-fraction-static "$MEM" \
  --attention-backend triton --mamba-backend triton --moe-runner-backend triton \
  --chunked-prefill-size 8192 --max-running-requests 1 --context-length 70000 \
  > server16.log 2>&1 &
for i in $(seq 1 120); do
  curl -s -m 2 http://127.0.0.1:39116/health >/dev/null && break
  pgrep -f "sglang.launch_server.*39116" >/dev/null || { echo "STOP: server exited"; tail -30 server16.log | cut -c1-200; exit 0; }
  sleep 5
done
grep -E "KV Cache is allocated|max_total_num_tokens|chunked_prefill_size=" server16.log | head -3 | cut -c1-200
echo "=== measure $(date -Is)"
python3 - <<'PY'
import json, random, statistics, time, urllib.request
URL = "http://127.0.0.1:39116"
def post(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    raw = urllib.request.urlopen(req, timeout=600).read()
    try:
        return json.loads(raw or b"{}")
    except ValueError:  # /flush_cache answers in plain text
        if path == "/flush_cache" and not getattr(post, "shown", False):
            post.shown = True
            print("flush_cache:", raw[:120])
        return {}
rng = random.Random(16)
def run(n):
    post("/flush_cache")
    ids = [rng.randrange(1000, 100000) for _ in range(n)]
    t = time.perf_counter()
    out = post("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
    dt = time.perf_counter() - t
    return dt, out.get("meta_info", {}).get("cached_tokens")
for n in (256, 1024):  # warmup
    run(n)
lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
res = {n: [] for n in lengths}
for rep in range(3):
    order = lengths if rep % 2 == 0 else lengths[::-1]
    for n in order:
        dt, cached = run(n)
        res[n].append(dt)
        print(f"rep {rep} L {n} time {dt*1e3:.1f} ms cached {cached}", flush=True)
print("=== summary (measured, shared gfx1151, end-to-end request time with 1 output token)")
for n in lengths:
    med = statistics.median(res[n])
    print(f"L {n:6d}  median {med*1e3:9.1f} ms  per token {med/n*1e6:7.2f} us  runs {[round(x*1e3,1) for x in res[n]]}")
PY
echo "--- errors (if any)"; grep -n -E "Traceback|Error|OutOfResources" server16.log | head -8 | cut -c1-200
echo "--- server log tracebacks (context)"; grep -n -A8 "Traceback" server16.log | head -45 | cut -c1-200
IN

echo "=== done $(date -Is)"
