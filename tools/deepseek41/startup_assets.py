"""Prepare launch metadata without rewriting, merging or requantizing GGUFs."""
from pathlib import Path
import hashlib
import os
import struct
import tempfile

from tools.deepseek41 import hash_proof

GSQ_VARIANT = "GSQ-RCO-IQ2_XS"
GSQ_UNCENSORED_VARIANT = "GSQ-RCO-IQ2_XS-Uncensored"
GSQ_VARIANTS = (GSQ_VARIANT, GSQ_UNCENSORED_VARIANT)
GSQ_VISION_BIAS = "exp_probs_b_vl.f32"
GSQ_VISION_BIAS_SHA256 = "710b11267b4db392d1fdb558f36ba8e0e91a91f4e8180c953c912e18eacf843d"
VARIANTS = {"UD-IQ3_XXS": 67636039680, "UD-Q2_K_XL": 68664950784,
            GSQ_VARIANT: "auto", GSQ_UNCENSORED_VARIANT: "auto"}
TOKENIZER_SHA256 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


def gsq_ple_model(cfg):
    return (cfg.get("variant") == GSQ_VARIANT and
            Path(cfg.get("model", "")).name == "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf")


def vision_enabled(cfg):
    # A legacy GSQ config could reserve mmproj memory while serving only text.
    return cfg.get("vision", cfg.get("variant") not in GSQ_VARIANTS and bool(cfg.get("mmproj")))


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


def gsq_launch_proof(cfg):
    lines = Path(cfg["integrity"]).read_text().splitlines()
    fourth = [row[1:] for row in map(str.split, lines) if row[:1] == ["main.4"]]
    expected = (hash_proof.GSQ_PLE_SHA256[3] if gsq_ple_model(cfg) else
                hash_proof.GSQ_UNCENSORED_SHARD4_SHA256 if cfg["variant"] == GSQ_UNCENSORED_VARIANT
                else hash_proof.GSQ_RCO_SHA256[3])
    model = Path(cfg["model"])
    shard = model.with_name(model.name.replace("-00001-of-", "-00004-of-"))
    if fourth != [[expected, *map(str, hash_proof.identity(shard))]]:
        raise ValueError("GSQ fourth-shard proof does not match the selected model")
    if not cfg.get("_gsq_vision_bias"):
        return cfg["integrity"]
    bias = Path(cfg["_gsq_vision_bias"])
    before = hash_proof.identity(bias)
    if (before[2] != 40 * 384 * 4 or hashlib.sha256(bias.read_bytes()).hexdigest() != GSQ_VISION_BIAS_SHA256
            or hash_proof.identity(bias) != before):
        raise ValueError("GSQ image router bias SHA-256 or size mismatch: " + str(bias))
    # Preserve explicit and shared proofs. The native loader checks every model row.
    rows = [line for line in lines if line.split(maxsplit=1)[:1] != ["vision_bias"]]
    rows.append(" ".join(["vision_bias", GSQ_VISION_BIAS_SHA256, *map(str, before)]))
    content = ("\n".join(rows) + "\n").encode()
    runtime = Path(cfg["runtime_dir"])
    runtime.mkdir(parents=True, exist_ok=True)
    proof = runtime / ("gsq-vision-" + hashlib.sha256(content).hexdigest() + ".proof")
    with tempfile.NamedTemporaryFile(dir=runtime, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.replace(temp, proof)
        finally:
            temp.unlink(missing_ok=True)
    return str(proof)


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
        gsq = cfg["variant"] in GSQ_VARIANTS
        uncensored = cfg["variant"] == GSQ_UNCENSORED_VARIANT
        approved = hash_proof.approved_files(paths, q2, sidecars_1017=True, main_gsq_rco=gsq,
                                            main_gsq_rco_uncensored=uncensored)
        output = Path(cfg["integrity"])
        if not proof_matches(output, approved):
            output.parent.mkdir(parents=True, exist_ok=True)
            print("Verifying all GGUF files with full SHA-256. The first launch can take several minutes.", flush=True)
            hash_proof.verify(paths, output, q2, sidecars_1017=True, main_gsq_rco=gsq,
                              main_gsq_rco_uncensored=uncensored)
    if cfg.get("variant") in GSQ_VARIANTS:
        return gsq_launch_proof(cfg)
    return cfg.get("integrity")
