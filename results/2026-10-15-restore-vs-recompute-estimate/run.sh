#!/usr/bin/env bash
# Regenerates raw/ for this result. Run from the repository root. Estimate only.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
uv run python "$here/estimate.py" "$here/inputs.json" > "$here/raw/estimate.json"
