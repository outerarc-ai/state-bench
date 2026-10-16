#!/usr/bin/env bash
# Run one of the scripts in this directory inside an SGLang ROCm container on a GPU host.
#
# Usage: envs/mi300x/container.sh <script> [args...]
#   e.g. envs/mi300x/container.sh env_check.sh
#
# Starts (or reuses) container $NAME from $IMAGE with the repository mounted read-only at
# /state-bench and $HOST_WORK mounted at /work, then runs the script there. Outputs go to
# $HOST_WORK on the host. When already inside an SGLang container, runs the script in place.
set -euo pipefail
IMAGE=${IMAGE:-rocm/sgl-dev:v0.5.21-rocm724-mi30x-20261014}
NAME=${NAME:-sb-mi300x}
HOST_WORK=${HOST_WORK:-$HOME/sb-mi300x}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
[ $# -ge 1 ] || { echo "usage: $0 <script> [args...]"; exit 2; }
SCRIPT=$1; shift
[ -f "$HERE/$SCRIPT" ] || { echo "no such script: $HERE/$SCRIPT"; exit 2; }

if [ -d /sgl-workspace/sglang ]; then
  export STATE_BENCH=$REPO W=${W:-/work}
  exec bash "$HERE/$SCRIPT" "$@"
fi

command -v docker >/dev/null || { echo "STOP: Docker not found and not inside an SGLang container"; exit 1; }
mkdir -p "$HOST_WORK"
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "=== pull $IMAGE $(date -Is)"
  docker pull -q "$IMAGE"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size 32g --security-opt seccomp=unconfined \
    -v "$REPO":/state-bench:ro -v "$HOST_WORK":/work \
    -e HF_HOME=/work/hf -e IMAGE_TAG="$IMAGE" \
    "$IMAGE" sleep infinity >/dev/null
fi
docker exec "$NAME" test -d /state-bench/envs/mi300x || {
  echo "STOP: container $NAME exists without the repository mounted; remove it (docker rm -f $NAME)"
  exit 1
}
echo "image $IMAGE digest $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE" 2>/dev/null || echo unknown)"
exec docker exec -i -e STATE_BENCH=/state-bench -e W=/work "$NAME" \
  bash "/state-bench/envs/mi300x/$SCRIPT" "$@"
