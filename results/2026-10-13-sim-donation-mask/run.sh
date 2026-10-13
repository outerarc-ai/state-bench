#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root.
# Simulated output on synthetic traces; every trace parameter is an assumption.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for preset in append mixed; do
  uv run state-bench synth --preset "$preset" --sessions 64 --seed 1 -o "$tmp/$preset.jsonl" >/dev/null
  uv run state-bench stats "$tmp/$preset.jsonl" > "$here/raw/trace-$preset-stats.json"
  for chunk in 2048 8192 16384; do
    for cut in none hashed; do
      for rt in roundtrip no-roundtrip; do
        flag=""
        [ "$rt" = no-roundtrip ] && flag="--no-output-roundtrip"
        uv run state-bench sim "$tmp/$preset.jsonl" --chunked-prefill-size "$chunk" \
          --first-chunk-cut "$cut" $flag \
          --policy sglang_current --policy sglang_absolute_mask \
          > "$here/raw/$preset-chunk$chunk-$cut-$rt.json"
      done
    done
  done
done
