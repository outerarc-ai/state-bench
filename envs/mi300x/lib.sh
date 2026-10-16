# shellcheck shell=bash
# Shared helpers for the scripts in this directory. Source it; do not run it.
# Expects W (work directory) and SRC (SGLang source tree) to be set by the caller.

SERVER_PID=""

# start_server LABEL PORT ARGS... : launch an SGLang server in the background and wait until
# /health answers. Returns 1 if the server exits or is not ready within READY_TIMEOUT_S.
start_server() {
  local label=$1 port=$2; shift 2
  local log=$W/logs/$label.server.log
  mkdir -p "$W/logs"
  echo "=== server [$label] start $(date -Is)"
  python3 -m sglang.launch_server --host 127.0.0.1 --port "$port" "$@" > "$log" 2>&1 &
  SERVER_PID=$!
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "${READY_TIMEOUT_S:-1800}" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server [$label] exited during startup; last lines:"; tail -n 25 "$log" | cut -c1-220
      SERVER_PID=""; return 1
    fi
    if curl -sf -m 2 "http://127.0.0.1:$port/health" >/dev/null; then
      echo "server [$label] ready after $((SECONDS - t0)) s"; return 0
    fi
    sleep 5
  done
  echo "server [$label] not ready after ${READY_TIMEOUT_S:-1800} s"; stop_server; return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null
  pkill -f "python[0-9.]* -m sglang[.]launch_server" 2>/dev/null
  SERVER_PID=""; sleep 5
}

# prefetch REPO_ID... : download model snapshots before timing anything.
prefetch() {
  python3 - "$@" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    t = time.time()
    path = snapshot_download(repo)
    print(f"model {repo} at {path} in {time.time() - t:.0f} s")
PY
}

require_flag() {  # require_flag --some-flag : stop if the installed SGLang lacks a server flag
  python3 -m sglang.launch_server --help 2>/dev/null | grep -q -- "$1" || {
    echo "STOP: this SGLang build has no $1"; exit 1; }
}
