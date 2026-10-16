"""Host-device copy bandwidth on one GPU, the transfer rate a host-tier state cache sees.

Usage: python3 copy_bandwidth.py OUT.json [--sizes-mib 1,4,16,64,256,1024] [--reps 20]

For each size, times host-to-device and device-to-host copies of a contiguous byte buffer
with GPU events, from pinned and from pageable host memory, after warm-up copies. Reports
the median over repetitions in GB/s (1e9 bytes per second). Measured on the visible device;
a recurrent-state snapshot is one contiguous block per layer, so sizes of 1-100 MiB are the
relevant range for a single state.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def time_copy(dst: torch.Tensor, src: torch.Tensor, reps: int) -> list[float]:
    for _ in range(3):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src, non_blocking=True)
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--sizes-mib", default="1,4,16,64,256,1024")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    rows = []
    for mib in (int(x) for x in a.sizes_mib.split(",")):
        n = mib << 20
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        for host_kind in ("pinned", "pageable"):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=host_kind == "pinned")
            for direction, dst, src in (("h2d", dev, host), ("d2h", host, dev)):
                ts = time_copy(dst, src, a.reps)
                med = statistics.median(ts)
                rows.append(
                    {
                        "size_mib": mib,
                        "host": host_kind,
                        "direction": direction,
                        "median_s": med,
                        "gbps": n / med / 1e9,
                        "times_s": ts,
                    }
                )
                print(
                    f"{mib:6d} MiB {host_kind:8s} {direction}  {n / med / 1e9:7.1f} GB/s",
                    flush=True,
                )
            del host
        del dev
        torch.cuda.empty_cache()
    out = {
        "provenance": "measured",
        "device": props.name,
        "arch": getattr(props, "gcnArchName", None),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "reps": a.reps,
        "rows": rows,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
