#!/usr/bin/env bash
# Record the software and device environment, and check that the GPU is idle.
#
# Usage: env_check.sh            full record, written to $W/env/env-<time>.txt and stdout
#        env_check.sh --idle     one line: "device idle" or "device busy: ..." (for the
#                                device_check of a paired run; match "device idle")
#
# The idle check lists processes with an open KFD handle (rocm-smi --showpids, falling back
# to amd-smi), which covers processes in other containers on the same host.
set -u
GPU=${HIP_VISIBLE_DEVICES:-0}

idle_line() {
  local out
  out=$(rocm-smi --showpids 2>/dev/null) || out=""
  if [ -n "$out" ]; then
    if echo "$out" | grep -q "No KFD PIDs currently running"; then echo "device idle"; return; fi
    local pids
    pids=$(echo "$out" | awk '/^[0-9]+[[:space:]]/{print $1"("$2")"}' | tr '\n' ' ')
    if [ -z "$pids" ]; then echo "device idle"; else echo "device busy: $pids"; fi
    return
  fi
  out=$(amd-smi process --json 2>/dev/null) || { echo "device unknown: no rocm-smi or amd-smi"; return; }
  python3 - "$out" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
except ValueError:
    print("device unknown: unreadable amd-smi output"); sys.exit()
procs = []
def walk(x):
    if isinstance(x, dict):
        if "pid" in x and isinstance(x.get("pid"), int):
            procs.append(f"{x['pid']}({x.get('name', '?')})")
        for v in x.values():
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(d)
print("device busy: " + " ".join(procs) if procs else "device idle")
PY
}

if [ "${1:-}" = "--idle" ]; then idle_line; exit 0; fi

W=${W:-/work}; mkdir -p "$W/env"
OUT=$W/env/env-$(date -u +%Y%m%dT%H%M%SZ).txt
{
  echo "=== time $(date -Is)"
  echo "=== host $(hostname) kernel $(uname -r)"
  echo "image ${IMAGE_TAG:-unknown}"
  echo "=== GPU"
  amd-smi static --asic --board --vbios --driver -g "$GPU" 2>/dev/null \
    || rocm-smi --showproductname --showvbios --showdriverversion 2>/dev/null
  echo "=== clocks, power, and performance level"
  amd-smi metric --clock --power -g "$GPU" 2>/dev/null | head -60 \
    || rocm-smi --showclocks --showpower --showperflevel 2>/dev/null
  rocm-smi --showperflevel --showmaxpower 2>/dev/null | grep -E "Performance Level|Max Graphics Package Power" || true
  echo "=== memory"
  rocm-smi --showmeminfo vram 2>/dev/null | grep -E "Total|Used" || true
  echo "=== software"
  cat /opt/rocm/.info/version 2>/dev/null | sed 's/^/rocm /'
  python3 - <<'PY'
import importlib.metadata as m
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    print(f"device {p.name} arch {getattr(p, 'gcnArchName', '?')} "
          f"mem {p.total_memory / 2**30:.0f} GiB count {torch.cuda.device_count()}")
    print("torch", torch.__version__, "hip", torch.version.hip)
except Exception as e:  # recorded, not fatal
    print("torch unavailable:", e)
for pkg in ("sglang", "triton", "aiter", "transformers", "flashinfer-python"):
    try:
        print(pkg, m.version(pkg))
    except Exception:
        print(pkg, "not installed")
PY
  for d in /sgl-workspace/sglang /sgl-workspace/aiter; do
    [ -d "$d/.git" ] && echo "$(basename "$d") commit $(git -C "$d" rev-parse HEAD 2>/dev/null)"
  done
  echo "=== idle check"
  idle_line
} 2>&1 | tee "$OUT"
echo "written $OUT"
