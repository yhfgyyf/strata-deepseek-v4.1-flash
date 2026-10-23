"""Prepare launch metadata without rewriting, merging or requantizing GGUFs."""
from pathlib import Path
import hashlib
import struct

from tools.deepseek41 import hash_proof

VARIANTS = {"UD-IQ3_XXS": 67636039680, "UD-Q2_K_XL": 68664950784}
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


def starter_profile():
    # Complete, workload-independent ranking. Calibrated rankings remain optional.
    pairs = [(layer, expert) for expert in range(384) for layer in range(40)]
    inverse = [0] * len(pairs)
    for slot, (layer, expert) in enumerate(pairs):
        inverse[layer * 384 + expert] = slot
    return (b"STRP" + struct.pack("<5I", 1, 40, 384, len(pairs), len(pairs)) +
            b"".join(struct.pack("<HH", *pair) for pair in pairs) +
            struct.pack("<15360i", *inverse))


def proof_matches(path, approved):
    try:
        lines = path.read_text().splitlines()
        expected = ["STRATA_DSV41_SHA256_V1"] + [
            " ".join([kind, digest, *(str(x) for x in hash_proof.identity(asset))])
            for kind, asset, digest in approved]
        return lines == expected
    except (OSError, ValueError):
        return False


def write_metadata(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("generated startup metadata differs; use an explicit profile/heat or a new runtime_dir: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def prepare(cfg):
    if cfg.get("variant") and hashlib.sha256(Path(cfg["tokenizer"]).read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer SHA-256 mismatch; use the matching official tokenizer.json")
    if cfg.get("_auto_profile"):
        write_metadata(cfg["profile"], starter_profile())
        write_metadata(cfg["heat"], b"0\n" * (2 * 40 * 384))
        print("Using the uncalibrated starter ranking and zero initial heat; benchmark rankings are separate.", flush=True)
    if cfg.get("_auto_integrity"):
        paths = {"main": Path(cfg["model"]), "draft": Path(cfg["draft"]),
                 "vision": Path(cfg["mmproj"]) if cfg.get("mmproj") else None}
        q2 = cfg["variant"] == "UD-Q2_K_XL"
        approved = hash_proof.approved_files(paths, q2, sidecars_1017=True)
        output = Path(cfg["integrity"])
        if not proof_matches(output, approved):
            output.parent.mkdir(parents=True, exist_ok=True)
            print("Verifying all GGUF files with full SHA-256. The first launch can take several minutes.", flush=True)
            hash_proof.verify(paths, output, q2, sidecars_1017=True)
