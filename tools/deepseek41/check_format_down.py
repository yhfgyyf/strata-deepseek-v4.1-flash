#!/usr/bin/env python3
"""Validate numerical records as well as native exit status for the fused-down probe."""
import csv
import io
import math
import subprocess
import sys


def main(binary):
    process = subprocess.Popen([binary], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=150)
    except subprocess.TimeoutExpired:
        process.kill()
        print("format-down probe timed out", file=sys.stderr)
        return 1
    print(stdout, end="")
    print(stderr, end="", file=sys.stderr)
    rows = list(csv.DictReader(io.StringIO(stdout)))
    cases = {(int(r["weight_type"]), int(r["seed"]), int(r["tokens"])) for r in rows}
    expected = {(t, 41+t+offset, n) for t in (10, 11) for offset in (0, 1000) for n in (1, 3, 8)}
    if process.returncode != 0 or len(rows) != 12 or cases != expected:
        print("incomplete or failed format-down execution", file=sys.stderr)
        return 1
    for row in rows:
        for name, limit in (("fused_vs_unfused_l2", 1e-3), ("gpu_vs_bf16_l2", 1e-2),
                            ("cpu_vs_gpu_l2", 1e-2), ("cpu_vs_bf16_l2", 1e-2)):
            value = float(row[name])
            if not math.isfinite(value) or value > limit:
                print(f"precision gate failed: type={row['weight_type']} seed={row['seed']} "
                      f"tokens={row['tokens']} {name}={value} > {limit}", file=sys.stderr)
                return 1
    if "all format-down correctness and A/B/A samples completed" not in stderr:
        print("native completion marker is missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
