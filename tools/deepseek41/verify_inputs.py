#!/usr/bin/env python3
"""Verify the pinned model inputs without changing any model file."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def read_checksums(root):
    entries = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        relative = Path(name)
        if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or relative.is_absolute() or ".." in relative.parts or name in entries):
            raise ValueError("Invalid or duplicate checksum entry: " + name)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Checksum entry escapes model directory: " + name)
        entries[name] = digest
    required = {
        "DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf": "11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53",
        "DeepSeek-V4.1-Flash-DSpark-draft.gguf": "d5de12076cac2378cf8cdbf0853d8e0afb0a0cd06e6ef2829d0aceee69ff94a9",
        "DeepSeek-V4.1-Flash-mmproj-BF16.gguf": "bd8a7593fc3721fad8ef84689beeb65437cd7a75326f005d6f741521a872d09e",
    }
    if any(entries.get(name) != digest for name, digest in required.items()):
        raise ValueError("Checksums do not match the approved model revision")
    return entries


def verify(root, output):
    entries = read_checksums(root)
    report = {"status": "running", "pid": os.getpid(), "root": str(root),
              "revision": "8270aceb26eb5590ea40f347256a1736999f551d",
              "started_at": time.time(), "files": [], "current": None}

    def save():
        report["updated_at"] = time.time()
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    save()
    # Check small inputs first, then stream each large file with bounded memory.
    names = sorted(entries, key=lambda name: (root / name).stat().st_size
                   if (root / name).is_file() else -1)
    for name in names:
        path = root / name
        record = {"path": name, "expected_sha256": entries[name]}
        try:
            before = path.stat()
            digest = hashlib.sha256()
            count = 0
            updated = time.monotonic()
            report["current"] = {"path": name, "bytes": 0, "size": before.st_size}
            save()
            with path.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    digest.update(block)
                    count += len(block)
                    if time.monotonic() - updated >= 10:
                        report["current"]["bytes"] = count
                        save()
                        updated = time.monotonic()
            after = path.stat()
            stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns)
            record.update(bytes=count, sha256=digest.hexdigest(), stable=stable,
                          passed=stable and digest.hexdigest() == entries[name])
        except OSError as error:
            record.update(passed=False, error=str(error))
        report["files"].append(record)
        report["current"] = None
        save()
    report["status"] = "passed" if all(f["passed"] for f in report["files"]) else "failed"
    report["finished_at"] = time.time()
    save()
    print(json.dumps({"status": report["status"], "files": len(report["files"]),
                      "failures": [f for f in report["files"] if not f["passed"]]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(verify(args.root, args.output))
