#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root.
# Simulated output on synthetic traces; every trace parameter is an assumption.
# Seeds 1-3 were used while designing boundary_value; see summary.md for what was
# changed after seeing seeds 11-13. Seeds 21-22 were run only after the last change.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
POLICIES="sglang_current prompt_end_anchor marconi_admission pin_recent boundary_value offline_oracle"
SLOTS="none 256 64 16"

cases=()
for preset in mixed fork; do
  for seed in 11 12 13; do cases+=("$preset-$seed||"); done
done
cases+=("mixed-11|--no-shared-system|no-shared-system")
cases+=("mixed-11||no-roundtrip")
cases+=("mixed-21||no-roundtrip")
cases+=("fork-22||no-roundtrip")

one() {
  IFS='|' read -r base synth_flag variant <<< "$1"
  preset="${base%-*}"; seed="${base#*-}"
  name="$base${variant:+-$variant}"
  trace="$tmp/$name.jsonl"
  uv run state-bench synth --preset "$preset" --sessions 64 --seed "$seed" $synth_flag -o "$trace" >/dev/null
  uv run state-bench stats "$trace" > "$here/raw/$name-stats.json"
  sim_flag=""
  [ "$variant" = no-roundtrip ] && sim_flag="--no-output-roundtrip"
  args=()
  for p in $POLICIES; do args+=(--policy "$p"); done
  for s in $SLOTS; do args+=(--state-slots "$s"); done
  uv run state-bench sim "$trace" --chunked-prefill-size 8192 --first-chunk-cut hashed \
    $sim_flag "${args[@]}" > "$here/raw/$name.json"
}
export -f one
export tmp here POLICIES SLOTS
printf '%s\n' "${cases[@]}" | xargs -P "$(nproc)" -I{} bash -c 'one "$@"' _ {}
