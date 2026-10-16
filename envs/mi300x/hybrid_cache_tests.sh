#!/usr/bin/env bash
# Single-GPU correctness checks of SGLang's hybrid (recurrent + attention) prefix cache on
# ROCm. Run before any performance measurement of a cache change (docs/MEASUREMENT.md,
# "Correctness first").
#
# Usage: hybrid_cache_tests.sh [mamba_cache_kl|unified_mamba_kl|gsm8k|all]
#
#   mamba_cache_kl    upstream test/registered/radix_cache/test_mamba_cache_kl.py, unmodified
#                     (Qwen3.5-0.8B; registered upstream for one GPU)
#   unified_mamba_kl  upstream test/registered/radix_cache/unified_radix_tree/
#                     test_unified_radix_cache_kl_mamba.py with --tp-size 4 changed to 1
#                     (Qwen3-Next-80B-A3B-Instruct-FP8; registered upstream for four GPUs).
#                     Covers the per-path cap (--mamba-max-states-per-path 3) and HiCache.
#   gsm8k             few-shot GSM8K accuracy, baseline server versus the same server with
#                     --mamba-max-states-per-path $CAP (model $GSM8K_MODEL)
#
# The Mamba2 extra_buffer KL test for Nemotron-H needs three ROCm fixes and is run by
# nemotron_h_repro.sh. Outputs go to $W/hybrid. Tests run from the image's SGLang source.
set -u
STAGE=${1:-all}
W=${W:-/work}; SRC=${SRC:-/sgl-workspace/sglang}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
GSM8K_MODEL=${GSM8K_MODEL:-Qwen/Qwen3.5-35B-A3B}
GSM8K_QUESTIONS=${GSM8K_QUESTIONS:-400}
CAP=${CAP:-2}
PORT=${PORT:-39200}
export HF_HOME=${HF_HOME:-$W/hf} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
OUT=$W/hybrid; mkdir -p "$OUT"
case "$STAGE" in mamba_cache_kl|unified_mamba_kl|gsm8k|all) ;;
  *) echo "usage: $0 [mamba_cache_kl|unified_mamba_kl|gsm8k|all]"; exit 2 ;; esac
echo "=== start $(date -Is) stage=$STAGE sglang $(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
bash "$HERE/env_check.sh" --idle

run_pytest() {  # run_pytest LABEL FILE
  local label=$1 file=$2 log=$OUT/$1.log
  echo "=== pytest [$label] $(date -Is) sha256 $(sha256sum "$file" | cut -c1-16)"
  SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=1 timeout "${TEST_TIMEOUT_S:-7200}" \
    python3 -m pytest -v -s -p no:cacheprovider "$file" > "$log" 2>&1
  echo "pytest [$label] exit $? $(date -Is)"
  grep -E "(PASSED|FAILED|ERROR|SKIPPED)( |$)" "$log" | grep "::" | cut -c1-200
  grep -o -E "(kl[_a-z]*|KL)[^,;]{0,40}[0-9]\.[0-9]+(e-?[0-9]+)?" "$log" | sort | uniq -c | sort -rn | head -12
  echo "--- first error (if any)"
  grep -n -m3 -E "Traceback|AssertionError|HIP error|OutOfResources|RuntimeError" "$log" | cut -c1-220
}

stage_mamba_cache_kl() {
  local f=$SRC/test/registered/radix_cache/test_mamba_cache_kl.py
  [ -f "$f" ] || { echo "STOP: $f not in image"; return; }
  cp "$f" "$OUT/test_mamba_cache_kl.py"
  run_pytest mamba_cache_kl "$OUT/test_mamba_cache_kl.py"
}

stage_unified_mamba_kl() {
  local f=$SRC/test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_mamba.py
  [ -f "$f" ] || { echo "STOP: $f not in image"; return; }
  python3 - "$f" "$OUT/test_unified_radix_cache_kl_mamba_1gpu.py" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
pat = re.compile(r'("--tp-size",\s*)"4"')
n = len(pat.findall(s))
assert n >= 1, "no --tp-size 4 in the test (file changed?)"
s = pat.sub(r'\1"1"', s)
open(sys.argv[2], "w").write(s)
print(f"single-GPU copy: {n} --tp-size values changed from 4 to 1")
PY
  prefetch Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
  run_pytest unified_mamba_kl "$OUT/test_unified_radix_cache_kl_mamba_1gpu.py"
}

gsm8k_arm() {  # gsm8k_arm LABEL EXTRA_ARGS...
  local label=$1; shift
  start_server "gsm8k-$label" "$PORT" --model-path "$GSM8K_MODEL" \
    --mamba-radix-cache-strategy extra_buffer "$@" || return
  python3 -m sglang.test.few_shot_gsm8k --port "$PORT" --num-questions "$GSM8K_QUESTIONS" \
    --parallel 64 > "$OUT/gsm8k-$label.log" 2>&1
  echo "gsm8k [$label] exit $? $(grep -E "^(Accuracy|Invalid|Latency|Output throughput)" "$OUT/gsm8k-$label.log" | tr '\n' ' ')"
  stop_server
}

stage_gsm8k() {
  require_flag --mamba-max-states-per-path
  prefetch "$GSM8K_MODEL"
  gsm8k_arm baseline
  gsm8k_arm "cap$CAP" --mamba-max-states-per-path "$CAP"
  echo "note: $GSM8K_QUESTIONS questions; a difference of a few points is within sampling noise"
}

trap 'stop_server' EXIT
case "$STAGE" in
  mamba_cache_kl) stage_mamba_cache_kl ;;
  unified_mamba_kl) stage_unified_mamba_kl ;;
  gsm8k) stage_gsm8k ;;
  all) stage_mamba_cache_kl; stage_gsm8k; stage_unified_mamba_kl ;;
esac
bash "$HERE/env_check.sh" --idle
echo "=== done $(date -Is)"
