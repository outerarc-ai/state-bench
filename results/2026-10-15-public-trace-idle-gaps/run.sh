#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root.
# Downloads the kv-cache-tester traces (about 630 MB) at a pinned commit into a temporary
# directory; nothing from the corpus is kept in this repository.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
COMMIT=94f6046dfd9c0c8bc0cdb24a5aa6579bac8669bf
git clone -q --filter=blob:none --no-checkout https://github.com/callanjfox/kv-cache-tester "$tmp/kvct"
git -C "$tmp/kvct" sparse-checkout set traces
git -C "$tmp/kvct" checkout -q "$COMMIT"
uv run python "$here/analyze.py" "$tmp/kvct/traces" > "$here/raw/idle-gaps.json"
