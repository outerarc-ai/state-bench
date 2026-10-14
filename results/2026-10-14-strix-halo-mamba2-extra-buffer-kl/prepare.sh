#!/usr/bin/env bash
# Prepare the model and offline inputs for run.sh (Strix Halo, container claude-statebench).
# Downloads NVIDIA-Nemotron-Nano-9B-v2 (Hugging Face, then hf-mirror, then ModelScope) and
# writes 48 prompts cut from the container's Python stdlib as an offline stand-in for the
# test's default input set.
# Touches only: container "claude-statebench", /tmp/claude-statebench inside it.
set -u
NAME=claude-statebench
echo "=== start $(date -Is)"
docker exec -i "$NAME" bash -s <<'IN'
set -u
cd /tmp/claude-statebench
MD=/tmp/claude-statebench/models/NVIDIA-Nemotron-Nano-9B-v2
echo "=== disk"; df -h /tmp | tail -1
FREE=$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9); [ "$FREE" -ge 30 ] || { echo "STOP: <30 GB free"; exit 0; }
echo "=== reachability"
for u in https://huggingface.co https://hf-mirror.com https://www.modelscope.cn; do
  printf "%-28s " "$u"; curl -s -o /dev/null -m 10 -w "%{http_code}\n" "$u" || echo fail; done
if [ ! -f "$MD/config.json" ]; then
  echo "=== download $(date -Is)"
  python3 - "$MD" <<'PY'
import os, sys, time
dst = sys.argv[1]; os.makedirs(dst, exist_ok=True)
pats = ["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "tokenizer*"]
def hf(endpoint):
    if endpoint: os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download
    return snapshot_download("nvidia/NVIDIA-Nemotron-Nano-9B-v2", local_dir=dst, allow_patterns=pats)
def ms():
    from modelscope import snapshot_download
    return snapshot_download("nv-community/NVIDIA-Nemotron-Nano-9B-v2", local_dir=dst)
for name, fn in [("huggingface", lambda: hf(None)), ("hf-mirror", lambda: hf("https://hf-mirror.com")), ("modelscope", ms)]:
    t = time.time()
    try:
        fn(); print(f"downloaded via {name} in {time.time()-t:.0f}s"); break
    except Exception as e:
        print(f"{name} failed: {type(e).__name__}: {str(e)[:200]}")
PY
fi
ls -la "$MD" | head -30; du -sh "$MD"
[ -f "$MD/config.json" ] || { echo "STOP: model not downloaded"; exit 0; }

export XDG_CACHE_HOME=/tmp/claude-statebench/cache
python3 - "$MD" <<'PY'
# Offline inputs: 48 prompts, 0.5x-1.5x of 3000 tokens, from the container's Python stdlib.
import glob, json, os, random, sys, hashlib
from transformers import AutoTokenizer
M = sys.argv[1]
tok = AutoTokenizer.from_pretrained(M)
text = "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3.12/*.py")))
ids = tok.encode(text); rng = random.Random(0); out, pos = [], 0
for _ in range(48):
    n = int(3000 * rng.uniform(0.5, 1.5)); out.append(ids[pos:pos + n]); pos += n
d = os.path.join(os.environ["XDG_CACHE_HOME"], "sglang", "longbench"); os.makedirs(d, exist_ok=True)
p = os.path.join(d, f"input_ids_{M.replace('/', '_')}_3000_48.json"); json.dump(out, open(p, "w"))
print("prompts", len(out), "corpus tokens", len(ids), "sha256", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])
PY

IN
echo "=== done $(date -Is)"
