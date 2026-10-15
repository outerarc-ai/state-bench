#!/usr/bin/env bash
# Run SGLang's test_mamba2_extra_buffer_kl.py (adapted) on Strix Halo (gfx1151).
# Prerequisite: prepare.sh. Touches only container "claude-statebench" and
# /tmp/claude-statebench inside it.
#
# Deviations from the upstream test, each needed to start the server on gfx1151:
# - Triton attention and Mamba backends (the default aiter attention aborts: MFMA).
# - State pool sized to fit a shared GPU (upstream: 256 slots, mem fraction 0.8).
# - --moe-runner-backend triton: Nemotron-H otherwise defaults to flashinfer_cutlass,
#   whose warmup autotune imports flashinfer, absent from ROCm images.
# - SGLANG_DISABLE_LEAN_ATTENTION=1: the ROCm Lean decode kernel needs more shared
#   memory than gfx1151 has.
# - Two temporary source patches (local.patch), reverted after the run: mamba.py binds
#   causal_conv1d kernels only for CUDA/NPU/XPU, and ReLU2 has no ROCm path.
set -u
NAME=claude-statebench
echo "=== start $(date -Is)"
docker exec -i "$NAME" bash -s <<'IN'
set -u
cd /tmp/claude-statebench
MD=/tmp/claude-statebench/models/NVIDIA-Nemotron-Nano-9B-v2
[ -f "$MD/config.json" ] || { echo "STOP: model missing"; exit 0; }
pgrep -af "sglang.launch_server" && { echo "STOP: a server from an earlier round is still running"; exit 0; }
export XDG_CACHE_HOME=/tmp/claude-statebench/cache
ls "$XDG_CACHE_HOME"/sglang/longbench/ | head -3

echo "=== sizing"
PLAN=$(python3 - "$MD" <<'PY'
import json, sys, torch
c = json.load(open(sys.argv[1] + "/config.json"))
pat = c.get("hybrid_override_pattern", "")
layers = pat.count("M")
h, d, n = c["mamba_num_heads"], c["mamba_head_dim"], c["ssm_state_size"]
g, k = c.get("n_groups", 1), c.get("conv_kernel", 4)
ssm_b = 4 if str(c.get("mamba_ssm_cache_dtype", "float32")) in ("float32", "fp32") else 2
per_req = layers * (h * d * n * ssm_b + (h * d + 2 * g * n) * (k - 1) * 2)
# Same measure SGLang uses (sglang/srt/utils/common.py:452-459): on an integrated GPU,
# system available memory, which other users' containers share.
import psutil
props = torch.cuda.get_device_properties(0)
if props.is_integrated:
    free = psutil.virtual_memory().available
else:
    free, _ = torch.cuda.mem_get_info()
free_gb = free / 2**30
weights_gb, kv_gb, footprint_cap_gb = 16.5, 4.0, 40.0
print(f"# mamba layers {layers}, heads {h}x{d}, state {n}, groups {g}, conv {k}, ssm bytes {ssm_b}", file=sys.stderr)
print(f"# per slot {per_req/2**20:.1f} MiB; integrated {props.is_integrated}; available {free_gb:.1f} GiB; our cap {footprint_cap_gb} GiB", file=sys.stderr)
for slots in (128, 96, 64):
    pool = slots * per_req / 2**30
    f = 1 - (free_gb - weights_gb - pool - kv_gb) / free_gb + 0.03
    print(f"# slots {slots}: pool {pool:.1f} GiB, mem fraction {f:.3f}, footprint {f*free_gb:.1f} GiB", file=sys.stderr)
    if f * free_gb <= footprint_cap_gb:
        print(slots, f"{f:.2f}")
        break
else:
    print("0 0")
PY
)
echo "plan: slots mem_fraction = $PLAN"
set -- $PLAN; SLOTS=$1; MEM=$2
[ "$SLOTS" -gt 0 ] || { echo "STOP: not enough free GPU memory now; retry later"; exit 0; }

cat > test_mamba2_extra_buffer_kl_local.py <<PY
# Adapted from SGLang 9f5d8712514c2d9dd29c76c6080774db3fd50aa2
# test/registered/radix_cache/test_mamba2_extra_buffer_kl.py (Apache-2.0).
import unittest
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

class TestMamba2ExtraBufferKL(KLDivergenceMixin, DefaultServerBase):
    model = "$MD"
    server_env = {"SGLANG_ENABLE_RANK_CONSENSUS_CHECKER": "1", "SGLANG_DISABLE_LEAN_ATTENTION": "1"}
    kl_div_thres = 0.005
    kl_div_thres_prefill = 0.012
    kl_div_max_samples = 16
    other_args = [
        "--max-mamba-cache-size", "$SLOTS",        # upstream 256; sized to fit a shared GPU
        "--mem-fraction-static", "$MEM",           # upstream 0.8; shared GPU
        "--mamba-radix-cache-strategy", "extra_buffer",
        "--mamba-track-interval", "128",
        # gfx1151: default aiter attention aborts at init (MFMA); use Triton.
        "--attention-backend", "triton",
        "--mamba-backend", "triton",
        # Nemotron-H defaults moe_runner_backend to flashinfer_cutlass on non-sm100
        # platforms, including ROCm; warmup autotune then imports flashinfer, which the
        # ROCm image does not ship. The model has no MoE layers, so triton is inert.
        "--moe-runner-backend", "triton",
    ]

if __name__ == "__main__":
    unittest.main()
PY

MF=/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/mamba.py
cp "$MF" /tmp/claude-statebench/mamba.py.orig
restore() { cp /tmp/claude-statebench/mamba.py.orig "$MF"; echo "=== restored original mamba.py"; }
trap restore EXIT
python3 - "$MF" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "elif is_xpu():"
assert anchor in s and "is_hip()" not in s, "unexpected mamba.py layout"
branch = (
    "elif is_hip():\n"
    "    # Local test patch: bind the portable Triton kernels on ROCm, as for XPU.\n"
    "    from sglang.kernels.ops.mamba.causal_conv1d_triton import (\n"
    "        causal_conv1d_fn as causal_conv1d_fn,\n"
    "        causal_conv1d_fn as causal_conv1d_fn_triton,\n"
    "        causal_conv1d_update as causal_conv1d_update,\n"
    "        causal_conv1d_update as causal_conv1d_update_triton,\n"
    "    )\n"
)
s = s.replace(anchor, branch + anchor, 1)
s = s.replace("    is_cuda,\n", "    is_cuda,\n    is_hip,\n", 1)
open(p, "w").write(s)
PY
AF=/sgl-workspace/sglang/python/sglang/srt/layers/activation.py
cp "$AF" /tmp/claude-statebench/activation.py.orig
restore() {
  cp /tmp/claude-statebench/mamba.py.orig "$MF"
  cp /tmp/claude-statebench/activation.py.orig "$AF"
  echo "=== restored original mamba.py and activation.py"
}
python3 - "$AF" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:\n        return relu2(x)\n"
assert s.count(anchor) == 1, "unexpected activation.py layout"
s = s.replace(anchor, anchor + "\n    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:\n        # Local test patch: relu2 is imported only on CUDA.\n        return self.forward_native(x)\n")
open(p, "w").write(s)
PY
echo "=== local patch"; diff -u /tmp/claude-statebench/mamba.py.orig "$MF" | head -40; diff -u /tmp/claude-statebench/activation.py.orig "$AF"

echo "=== run $(date -Is)"
SGLANG_DISABLE_LEAN_ATTENTION=1 timeout 3000 python3 -m pytest -v -s -p no:cacheprovider test_mamba2_extra_buffer_kl_local.py > kl_round15.log 2>&1
echo "=== pytest exit $? $(date -Is)"
grep -n -E "max_mamba_cache_size|max_running_requests|avail mem=|Mamba Cache|mamba_cache|KV Cache is allocated|Memory pool end" kl_round15.log | head -12 | cut -c1-260
grep -n -E "kl_divs|avg_kl_div|PASSED|FAILED|ERROR|Error|expected cached_tokens|cached_tokens" kl_round15.log | grep -v -i "custom_all_reduce\|ninja\|CalledProcess\|nodiscard" | cut -c1-260 | tail -40
echo "--- scheduler exception (if any)"; awk '/Scheduler hit an exception/{p=1} p{print; n++} n>70{exit}' kl_round15.log | grep -v "^\s*\^" | cut -c1-220 | tail -25
echo "--- max mamba usage"; grep -o "mamba usage: [0-9.]*" kl_round15.log | sort -t: -k2 -n | tail -1
echo "--- tail"; tail -8 kl_round15.log | cut -c1-260
pgrep -af "sglang.launch_server" || echo "no leftover server"
IN
echo "=== done $(date -Is)"
