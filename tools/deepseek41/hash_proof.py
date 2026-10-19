#!/usr/bin/env python3
"""Create native-loader proofs after full SHA verification of the approved GGUFs."""
import argparse
import hashlib
from pathlib import Path
import time

EXPECTED = {
    "main": "11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53",
    "draft": "d5de12076cac2378cf8cdbf0853d8e0afb0a0cd06e6ef2829d0aceee69ff94a9",
    "vision": "bd8a7593fc3721fad8ef84689beeb65437cd7a75326f005d6f741521a872d09e",
}


def identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def verify(paths, output):
    records = []
    for kind, path in paths.items():
        if not path:
            continue
        before = identity(path)
        digest = hashlib.sha256()
        started = time.monotonic()
        with path.open("rb") as stream:
            while block := stream.read(16 << 20):
                digest.update(block)
        if identity(path) != before or digest.hexdigest() != EXPECTED[kind]:
            raise ValueError(kind + " full SHA-256 failed; no new proof was written")
        records.append(" ".join([kind, digest.hexdigest(), *(str(x) for x in before)]))
        print(kind, "passed", "seconds", round(time.monotonic() - started, 2), flush=True)
    if not records:
        raise ValueError("at least one GGUF is required")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x") as stream:
        stream.write("STRATA_DSV41_SHA256_V1\n" + "\n".join(records) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in EXPECTED:
        parser.add_argument("--" + kind, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--quantized-draft", action="store_true",
                        help="verify the tested Q2_K/Q4 DSpark derivative instead of the original draft")
    args = parser.parse_args()
    if args.quantized_draft:
        if not args.draft:
            parser.error("--quantized-draft requires --draft")
        EXPECTED["draft"] = "9269e73cbd04c493d2510d3513da85429441949f6f10ed8130392ae0de2a949b"
    verify({kind: getattr(args, kind) for kind in EXPECTED}, args.output)
