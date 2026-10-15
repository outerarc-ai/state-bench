#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root.
# Simulated: recorded kv-cache-tester traces replayed through the snapshot simulator.
# Session start times (--session-interval) and reply round-trip are assumptions.
# Downloads the traces (about 630 MB) at a pinned commit into a temporary directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
COMMIT=94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf
git clone -q --filter=blob:none --no-checkout https://github.com/callanjfox/kv-cache-tester "$tmp/kvct"
git -C "$tmp/kvct" sparse-checkout set traces
git -C "$tmp/kvct" checkout -q "$COMMIT"
mapfile -t files < <(ls "$tmp/kvct/traces"/trace_*.json | sort)
POLICIES="sglang_current prompt_end_anchor marconi_admission pin_recent boundary_value boundary_value:demand=true offline_oracle"
SLOTS="none 256 64 16"
for s in 0 1 2; do
  subset=("${files[@]:$((s * 40)):40}")
  uv run state-bench import weka "${subset[@]}" -o "$tmp/subset-$s.jsonl" \
    --source "kv-cache-tester@${COMMIT:0:7}" --license Apache-2.0 --session-interval 60
  uv run state-bench stats "$tmp/subset-$s.jsonl" > "$here/raw/subset-$s-stats.json"
done
one() {
  s=$1; variant=$2
  flag=""; [ "$variant" = no-roundtrip ] && flag="--no-output-roundtrip"
  args=()
  for p in $POLICIES; do args+=(--policy "$p"); done
  for n in $SLOTS; do args+=(--state-slots "$n"); done
  uv run state-bench sim "$tmp/subset-$s.jsonl" --chunked-prefill-size 8192 \
    --first-chunk-cut hashed $flag "${args[@]}" > "$here/raw/subset-$s-$variant.json"
}
export -f one
export tmp here POLICIES SLOTS
printf '%s\n' "0 roundtrip" "1 roundtrip" "2 roundtrip" "0 no-roundtrip" "1 no-roundtrip" "2 no-roundtrip" \
  | xargs -P "$(nproc)" -L1 bash -c 'one "$@"' _
