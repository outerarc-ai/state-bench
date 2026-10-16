#!/usr/bin/env bash
# Reproduce on MI300X (gfx942) the three issues that stop NVIDIA Nemotron-H (Mamba2 hybrid)
# models from starting under SGLang on ROCm, then run SGLang's Mamba2 extra_buffer KL test
# with fixes for all three applied. First seen on Strix Halo (gfx1151); see state-bench
# results/2026-10-14-strix-halo-mamba2-extra-buffer-kl.
#
# Issues (SGLang upstream main c4870d98, 2026-10-15; the three sites are unchanged on
# a448fdf3, 2026-10-16):
#   1. arg_groups/model_overrides/nemotron_h.py: moe_runner_backend "auto" resolves to
#      flashinfer_cutlass on every non-sm100 platform, ROCm included; the FlashInfer warmup
#      autotune then imports flashinfer, which ROCm images do not ship.
#   2. layers/attention/mamba/mamba.py: causal_conv1d kernels are bound only for
#      CUDA/NPU/XPU, so the names are undefined on ROCm.
#   3. layers/activation.py: relu2 is imported only on CUDA and ReLU2 has no forward_hip.
#
# Stages (run in order; each prints what it found):
#   env       record GPU, image, SGLang commit; check the GPU is idle; check the three sites
#             on the image's source and on upstream main
#   baseline  launch the server with the upstream test's arguments, adding the fixes one by
#             one: none -> fix 1 -> fixes 1+2 -> fixes 1+2+3. Expected: the first three
#             launches fail with issues 1, 2, 3 in turn; the last one starts
#   kl        run the upstream test file unmodified with fixes 1+2+3 applied
#   all       env + baseline + kl (default)
#
# Fixes are applied to the container's source for the run and reverted on exit. They are
# minimal test patches, not the proposed upstream change:
#   fix 1: leave moe_runner_backend "auto" on non-CUDA platforms (nemotron_h.py)
#   fix 2: bind the portable Triton causal_conv1d kernels on ROCm, as for XPU (mamba.py)
#   fix 3: ReLU2.forward_hip falls back to forward_native (activation.py)
#
# Where it runs:
#   - On a GPU host with Docker (e.g. an AMD Developer Cloud droplet): starts container
#     "sb-mi300x" from $IMAGE and runs inside it. Work dir: $HOST_WORK (default ~/sb-mi300x).
#   - Inside an SGLang ROCm container already (e.g. a RunPod pod on that image): runs in place.
#
# Usage, from a clone of the repository on the GPU host (survives a dropped SSH connection):
#   nohup envs/mi300x/nemotron_h_repro.sh all > nemotron_h.log 2>&1 &
#   tail -n 40 nemotron_h.log                      # "=== done" marks the end
# It can also be copied to the host and run on its own; it needs nothing else from the
# repository.
# Expected time on one MI300X (estimate): about 1-2 h, most of it model download and the
# four server launches (the first launch compiles kernels).
set -u
STAGE=${1:-all}
IMAGE=${IMAGE:-rocm/sgl-dev:v0.5.21-rocm724-mi30x-20261014}
NAME=${NAME:-sb-mi300x-nemotron}
HOST_WORK=${HOST_WORK:-$HOME/sb-mi300x}
case "$STAGE" in env|patchcheck|baseline|kl|all) ;; *) echo "usage: $0 [env|patchcheck|baseline|kl|all]"; exit 2 ;; esac
echo "=== start $(date -Is) stage=$STAGE"

inner() {
cat <<'INNER'
set -u
STAGE=$1
W=${W:-/work}; mkdir -p "$W"; cd "$W"
export HF_HOME=$W/hf XDG_CACHE_HOME=$W/cache HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
SRC=${SRC:-/sgl-workspace/sglang}
PY=$SRC/python/sglang/srt
F1=$PY/arg_groups/model_overrides/nemotron_h.py
F2=$PY/layers/attention/mamba/mamba.py
F3=$PY/layers/activation.py
TEST=$SRC/test/registered/radix_cache/test_mamba2_extra_buffer_kl.py
ARGS=(--max-mamba-cache-size 256 --mem-fraction-static 0.8 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 128)

for f in "$F1" "$F2" "$F3"; do [ -f "$f" ] || { echo "STOP: $f not found (image layout differs)"; exit 0; }; done
mkdir -p "$W/orig"
for f in "$F1" "$F2" "$F3"; do [ -f "$W/orig/$(basename "$f")" ] || cp "$f" "$W/orig/$(basename "$f")"; done
restore() { for f in "$F1" "$F2" "$F3"; do cp "$W/orig/$(basename "$f")" "$f"; done; }
trap 'restore; echo "=== restored original sources"' EXIT
restore

fix1() { python3 - "$F1" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '        else:\n            overrides["moe_runner_backend"] = "flashinfer_cutlass"\n'
assert s.count(a) == 1, "fix 1: anchor not found (code changed or already fixed?)"
s = s.replace(a, '        elif get_platform().is_cuda:  # test patch: not on ROCm\n            overrides["moe_runner_backend"] = "flashinfer_cutlass"\n')
open(p, "w").write(s)
PY
}
fix2() { python3 - "$F2" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "elif is_xpu():"
assert a in s and "is_hip()" not in s, "fix 2: anchor not found (code changed or already fixed?)"
b = ("elif is_hip():\n"
     "    # test patch: bind the portable Triton kernels on ROCm, as for XPU\n"
     "    from sglang.kernels.ops.mamba.causal_conv1d_triton import (\n"
     "        causal_conv1d_fn as causal_conv1d_fn,\n"
     "        causal_conv1d_fn as causal_conv1d_fn_triton,\n"
     "        causal_conv1d_update as causal_conv1d_update,\n"
     "        causal_conv1d_update as causal_conv1d_update_triton,\n"
     "    )\n")
s = s.replace(a, b + a, 1)
assert s.count("    is_cuda,\n") >= 1, "fix 2: import anchor not found"
s = s.replace("    is_cuda,\n", "    is_cuda,\n    is_hip,\n", 1)
open(p, "w").write(s)
PY
}
fix3() { python3 - "$F3" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:\n        return relu2(x)\n"
assert s.count(a) == 1, "fix 3: anchor not found (code changed or already fixed?)"
s = s.replace(a, a + "\n    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:\n        # test patch: relu2 is imported only on CUDA\n        return self.forward_native(x)\n")
open(p, "w").write(s)
PY
}

stage_env() {
  echo "=== env"
  (amd-smi static --asic --board 2>/dev/null || rocm-smi --showproductname 2>/dev/null) | grep -i -E "market|product|name|gfx|target|sku|vbios" | head -12
  python3 - <<'PY'
import torch, importlib.metadata as m
p = torch.cuda.get_device_properties(0)
print(f"device {p.name} arch {getattr(p, 'gcnArchName', '?')} mem {p.total_memory/2**30:.0f} GiB count {torch.cuda.device_count()}")
print("torch", torch.__version__, "hip", torch.version.hip)
for pkg in ("sglang", "triton", "aiter", "flashinfer-python", "flashinfer"):
    try: print(pkg, m.version(pkg))
    except Exception: print(pkg, "not installed")
PY
  echo "image ${IMAGE_TAG:-unknown}"
  echo "sglang commit $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "--- GPU processes (should be none besides ours)"
  (amd-smi process 2>/dev/null || rocm-smi --showpids 2>/dev/null) | head -20
  echo "--- the three sites in the image's source"
  grep -n -B1 'overrides\["moe_runner_backend"\] = "flashinfer_cutlass"' "$F1"
  grep -n -E '^(if|elif) is_' "$F2"
  grep -n -A2 "def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:" "$F3" | grep -A1 -B1 "relu2(x)"
  grep -c "forward_hip" "$F3" | sed 's/^/forward_hip count in activation.py: /'
  echo "--- the three sites on upstream main"
  MAIN=$(git ls-remote https://github.com/sgl-project/sglang refs/heads/main 2>/dev/null | cut -c1-40)
  echo "upstream main $MAIN"
  if [ -n "$MAIN" ]; then
    R=https://raw.githubusercontent.com/sgl-project/sglang/$MAIN/python/sglang/srt
    curl -fsS "$R/arg_groups/model_overrides/nemotron_h.py" | grep -n -B1 'overrides\["moe_runner_backend"\] = "flashinfer_cutlass"'
    curl -fsS "$R/layers/attention/mamba/mamba.py" | grep -n -E '^(if|elif) is_'
    curl -fsS "$R/layers/activation.py" | awk '/^class ReLU2/{p=1} /^class QuickGELU/{p=0} p' | grep -n "def forward_"
  fi
  patchcheck
}

patchcheck() {
  echo "--- fixes apply cleanly to the image's source?"
  restore
  fix1 && fix2 && fix3 && echo "all three fixes apply"
  for f in "$F1" "$F2" "$F3"; do diff -u "$W/orig/$(basename "$f")" "$f"; done | grep -c "^+ " | sed 's/^/added lines: /'
  restore
}

launch() {  # $1 = label; returns after the server starts or fails
  local label=$1 log=$W/launch_$1.log
  echo "=== launch [$label] $(date -Is)"
  python3 - "$MODEL" "${ARGS[@]}" > "$log" 2>&1 <<'PY'
import sys, time
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server, terminate_and_kill_process_tree
model, args = sys.argv[1], sys.argv[2:]
t = time.time()
try:
    p = popen_launch_server(model, DEFAULT_URL_FOR_TEST, timeout=1800, other_args=args,
                            env={"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1"})
except Exception as e:
    print(f"RESULT: launch failed after {time.time()-t:.0f}s: {type(e).__name__}: {e}"); sys.exit(0)
print(f"RESULT: server ready after {time.time()-t:.0f}s")
terminate_and_kill_process_tree(p, wait_timeout=60)
PY
  grep -m1 "RESULT:" "$log"
  grep -o -E "moe_runner_backend='[a-z_]+'|attention_backend='[a-z_]+'|mamba_backend='[a-z_]+'" "$log" | sort -u | tr '\n' ' '; echo
  echo "--- first error"
  grep -n -m3 -E "ModuleNotFoundError|NameError|ImportError|OutOfResources|RuntimeError|AssertionError|HIP error|Error:" "$log" | cut -c1-260
  echo "--- traceback tail"
  awk '/Traceback \(most recent call last\)/{p=1; n=0} p{buf[n++]=$0} END{s=(n>14?n-14:0); for(i=s;i<n;i++) print buf[i]}' "$log" | cut -c1-220
  pkill -f "sglang.launch_server" 2>/dev/null; sleep 5
}

prefetch() {
  echo "=== prefetch model $(date -Is)"
  python3 - "$MODEL" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
t = time.time(); d = snapshot_download(sys.argv[1]); print(f"model at {d} in {time.time()-t:.0f}s")
PY
}

stage_baseline() {
  prefetch
  restore;             launch 0-nofix
  restore; fix1;       launch 1-fix1
  restore; fix1; fix2; launch 2-fix12
  restore; fix1; fix2; fix3; launch 3-fix123
  restore
}

stage_kl() {
  [ -f "$TEST" ] || { echo "STOP: upstream test file not in image: $TEST"; return; }
  echo "=== kl test (upstream file, unmodified) sha256 $(sha256sum "$TEST" | cut -c1-16)"
  restore; fix1; fix2; fix3
  echo "--- applied fixes"; for f in "$F1" "$F2" "$F3"; do diff -u "$W/orig/$(basename "$f")" "$f"; done | grep -E "^[-+]" | grep -v -E "^(\+\+\+|---)"
  cp "$TEST" "$W/test_mamba2_extra_buffer_kl.py"
  echo "=== run $(date -Is)"
  SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=1 timeout 5400 python3 -m pytest -v -s -p no:cacheprovider "$W/test_mamba2_extra_buffer_kl.py" > "$W/kl.log" 2>&1
  echo "=== pytest exit $? $(date -Is)"
  grep -n -E "max_mamba_cache_size|max_running_requests|Mamba Cache|KV Cache is allocated" "$W/kl.log" | head -8 | cut -c1-240
  grep -n -E "avg_kl_div|kl_divs|PASSED|FAILED|ERROR" "$W/kl.log" | cut -c1-240 | tail -30
  echo "--- max mamba usage"; grep -o "mamba usage: [0-9.]*" "$W/kl.log" | sort -t: -k2 -n | tail -1
  echo "--- scheduler exception (if any)"; awk '/Scheduler hit an exception/{p=1} p{print; n++} n>60{exit}' "$W/kl.log" | cut -c1-220 | tail -25
  restore
}

case "$STAGE" in
  env) stage_env ;;
  patchcheck) patchcheck ;;
  baseline) stage_baseline ;;
  kl) stage_kl ;;
  all) stage_env; stage_baseline; stage_kl ;;
esac
pgrep -af "sglang.launch_server" || echo "no leftover server"
INNER
}

if [ -d /sgl-workspace/sglang ]; then
  echo "=== running inside the container"
  mkdir -p /work
  IMAGE_TAG=${IMAGE_TAG:-unknown} bash -s "$STAGE" <<<"$(inner)"
else
  command -v docker >/dev/null || { echo "STOP: neither inside an SGLang container nor Docker available"; exit 1; }
  mkdir -p "$HOST_WORK"
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "=== pull $IMAGE $(date -Is)"; docker pull -q "$IMAGE" || { echo "STOP: pull failed"; exit 1; }
    docker rm -f "$NAME" >/dev/null 2>&1
    docker run -d --name "$NAME" --device=/dev/kfd --device=/dev/dri --group-add video \
      --ipc=host --shm-size 32g --security-opt seccomp=unconfined \
      -v "$HOST_WORK":/work "$IMAGE" sleep infinity >/dev/null || { echo "STOP: docker run failed"; exit 1; }
  fi
  echo "image digest $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null)"
  inner | docker exec -i -e IMAGE_TAG="$IMAGE" "$NAME" bash -s "$STAGE"
  echo "container $NAME left running; remove with: docker rm -f $NAME"
fi
echo "=== done $(date -Is)"
