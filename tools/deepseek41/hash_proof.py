#!/usr/bin/env python3
"""Create native-loader proofs after full SHA verification of the approved GGUFs."""
import argparse
import hashlib
from pathlib import Path
import re
import time

EXPECTED = {
    "main": "11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53",
    "draft": "d5de12076cac2378cf8cdbf0853d8e0afb0a0cd06e6ef2829d0aceee69ff94a9",
    "vision": "bd8a7593fc3721fad8ef84689beeb65437cd7a75326f005d6f741521a872d09e",
}
SIDECARS_1017 = {
    "draft": "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a",
    "vision": "9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9",
}
IQ3XXS_SHA256 = [
    "0c2a2031bd93ebd60aa6a4f3912775ba3990b5f39b24c6edf9ea029d2f387a28",
    "aa7827cc3c88958f4bc4134fab2c2d63dff1e53f947f76cbb34ce429205d551c",
    "7e5bc7c88e2bbcdadafcf672a1959581aeb2f2b23732ffd77823a756151a4e76",
    "a6b33c1cc73bbd2a54617f0714c5144ee8ee6d6403ceea49628633a7a2be21de",
    "5e392a945c534eb439279bbb638e7ea47e5a1cfca620b96c3fd47da87d5923a8",
    "e1b4a74844e637d8b15c94c30757e2359f81f287824c421e8b5b746b010e030e",
    "9d21807ac1a40cd1635634c14135b5acca1c8b855dae228170361602c7de560b",
]


Q2K_XL_SHA256 = [
    "0c2a2031bd93ebd60aa6a4f3912775ba3990b5f39b24c6edf9ea029d2f387a28",
    "605972fece1b43277b9ff86a2cd79c9002ff424caefc507f20b6ccd47ae4ab39",
    "1cb58786245bd8c38478174eecc648b12691bb79c2bb2f446c135ab00263580e",
    "7b753295dd32534dfea52b2e9b5c4741b8007a955e38702ee37034271df7526e",
    "336be674937fe29683e7aeee7c6af32f896b864931a6c916907b78e0f3692de4",
    "192694fb09612e1200cce9ae35f3645ad39cf8aa43e94417070af9c4f4ecb43b",
    "a653c5309b7c74a6c41580aaa8c2d2bbdae0342253bca2cd3fc9e69d05f3d0eb",
]


def approved_files(paths, main_q2_k_xl=False, sidecars_1017=False):
    result = []
    expected = {**EXPECTED, **(SIDECARS_1017 if sidecars_1017 else {})}
    for kind, path in paths.items():
        if not path:
            continue
        match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path.name)
        if kind == "main" and match:
            if int(match[3]) != 7 or not 1 <= int(match[2]) <= 7:
                raise ValueError("unapproved DeepSeek main shard count")
            result.extend(
                ("main." + str(i), path.with_name(f"{match[1]}-{i:05d}-of-00007.gguf"), digest)
                for i, digest in enumerate(Q2K_XL_SHA256 if main_q2_k_xl else IQ3XXS_SHA256, 1))
        else:
            result.append((kind, path, expected[kind]))
    # Refuse an incomplete download before spending time hashing any other shard.
    for _, path, _ in result:
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def identity(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def verify(paths, output, main_q2_k_xl=False, sidecars_1017=False):
    records = []
    for kind, path, expected in approved_files(paths, main_q2_k_xl, sidecars_1017):
        before = identity(path)
        digest = hashlib.sha256()
        started = time.monotonic()
        with path.open("rb") as stream:
            while block := stream.read(16 << 20):
                digest.update(block)
        if identity(path) != before or digest.hexdigest() != expected:
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
    parser.add_argument("--sidecars-1017", action="store_true",
                        help="verify the 1017 Q8_0 DSpark and BF16 vision checkpoint")
    parser.add_argument("--main-q2-k-xl", action="store_true", help="verify the approved 1017 UD-Q2_K_XL seven-shard checkpoint")
    args = parser.parse_args()
    if args.quantized_draft:
        if not args.draft:
            parser.error("--quantized-draft requires --draft")
        EXPECTED["draft"] = "9269e73cbd04c493d2510d3513da85429441949f6f10ed8130392ae0de2a949b"
    if args.sidecars_1017:
        if args.quantized_draft:
            parser.error("--sidecars-1017 conflicts with --quantized-draft")
        EXPECTED["draft"] = "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"
        EXPECTED["vision"] = "9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9"
    verify({kind: getattr(args, kind) for kind in EXPECTED}, args.output, args.main_q2_k_xl)
