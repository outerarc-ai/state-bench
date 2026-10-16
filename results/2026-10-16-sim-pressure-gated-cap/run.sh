#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root. About 40 min on two cores.
# Simulated: recorded kv-cache-tester traces replayed through the snapshot simulator with a
# per-path snapshot cap that applies only while the state pool is nearly full. Session
# start times and reply round-trip are assumptions. Downloads the traces (about 630 MB) at
# a pinned commit into a temporary directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
COMMIT=94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf
git clone -q --filter=blob:none --no-checkout https://github.com/callanjfox/kv-cache-tester "$tmp/kvct"
git -C "$tmp/kvct" sparse-checkout set traces
git -C "$tmp/kvct" checkout -q "$COMMIT"
mapfile -t files < <(ls "$tmp/kvct/traces"/trace_*.json | sort)
SLOTS="none 256 128 64 32 16"
for s in 0 1 2; do
  subset=("${files[@]:$((s * 40)):40}")
  uv run state-bench import weka "${subset[@]}" -o "$tmp/subset-$s.jsonl" \
    --source "kv-cache-tester@${COMMIT:0:7}" --license Apache-2.0 --session-interval 60
done
one() {
  s=$1; variant=$2; cap=$3; usage=$4
  flag=""; [ "$variant" = no-roundtrip ] && flag="--no-output-roundtrip"
  args=(--policy sglang_current --max-states-per-path "$cap" --path-cap-min-usage "$usage")
  for n in $SLOTS; do args+=(--state-slots "$n"); done
  uv run state-bench sim "$tmp/subset-$s.jsonl" --chunked-prefill-size 8192 \
    --first-chunk-cut hashed $flag "${args[@]}" > "$here/raw/subset-$s-$variant-cap-$cap-usage-$usage.json"
}
export -f one
export tmp here SLOTS
for s in 0 1 2; do for v in roundtrip no-roundtrip; do for c in 1 2; do for u in 0.8 0.95; do
  echo "$s $v $c $u"; done; done; done; done | xargs -P "$(nproc)" -L1 bash -c 'one "$@"' _
uv run python "$here/analyze.py" "$here" > "$here/raw/summary.json"
