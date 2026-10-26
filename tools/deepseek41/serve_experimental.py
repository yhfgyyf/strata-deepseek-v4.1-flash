"""Serve the fixed RTX PRO 6000 experimental profile from local asset paths."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.deepseek41.startup_assets import GSQ_VARIANT, VARIANTS, prepare
from tools.deepseek41.hash_proof import GSQ_ENGRAM_FILE


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    host, port = cfg.get("host", "127.0.0.1"), cfg.get("port", 30011)
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.get("api_key"):
        raise ValueError("a non-loopback DeepSeek API requires api_key")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in 1..65535")
    variant = cfg.get("variant")
    if variant is not None and variant not in VARIANTS:
        raise ValueError("variant must be " + ", ".join(VARIANTS))
    if variant == GSQ_VARIANT:
        if cfg.get("vision") is True:
            raise ValueError("GSQ-RCO is text-only; vision must be false")
        cfg.setdefault("vision", False)
    # Explicit calibrated assets retain their existing behavior. The supplied
    # variant configs can instead prepare only small metadata beside the config.
    if variant:
        cfg.setdefault("hot_bytes", VARIANTS[variant])
        runtime = Path(cfg.get("runtime_dir", "runtime/" + path.stem)).expanduser()
        if not runtime.is_absolute():
            runtime = path.parent / runtime
        for name, filename in (("costs", "dspark-costs-rtxpro6000.txt"),
                               ("draft_input_vocab", "draft-input-vocab-114801.bin"),
                               ("draft_output_vocab", "draft-vocab-90000.bin")):
            cfg.setdefault(name, str(ROOT / "data" / filename))
        if bool(cfg.get("profile")) != bool(cfg.get("heat")):
            raise ValueError("supply both profile and heat, or omit both for starter metadata")
        cfg["_auto_profile"] = not cfg.get("profile")
        cfg["_auto_integrity"] = not cfg.get("integrity")
        if cfg["_auto_profile"]:
            cfg.update(profile=str(runtime / "starter.strp"), heat=str(runtime / "starter-heat.txt"))
        if cfg["_auto_integrity"]:
            cfg["integrity"] = str(runtime / "integrity.txt")
    if "hot_bytes" in cfg and cfg["hot_bytes"] != "auto" and (type(cfg["hot_bytes"]) is not int or not 0 < cfg["hot_bytes"] < 1 << 63):
        raise ValueError("hot_bytes must be auto or a positive integer below 2^63")
    if "vision" in cfg and type(cfg["vision"]) is not bool:
        raise ValueError("vision must be a boolean")
    if cfg.get("vision") and not cfg.get("mmproj"):
        raise ValueError("vision requires mmproj")
    if type(cfg.get("prefix_cache_mib", 0)) is not int or not 0 <= cfg.get("prefix_cache_mib", 0) <= 4096:
        raise ValueError("prefix_cache_mib must be an integer from 0 to 4096")
    for name in ("engine", "model", "draft", "tokenizer", "integrity", "profile", "heat",
                 "costs", "draft_input_vocab", "draft_output_vocab", "mmproj", "log"):
        value = cfg.get(name)
        if name in ("mmproj", "log") and not value:
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(name + " must be a file path")
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        # Split GGUF peers are discovered beside the supplied first-shard path.
        # Preserve an intentional symlink directory that assembles those peers.
        resolved = Path(os.path.abspath(resolved)) if name == "model" else resolved.resolve()
        generated = (name == "integrity" and cfg.get("_auto_integrity") or
                     name in ("profile", "heat") and cfg.get("_auto_profile"))
        if name != "log" and not generated and not resolved.is_file():
            raise ValueError(name + " file is missing: " + str(resolved))
        cfg[name] = str(resolved)
    if not os.access(cfg["engine"], os.X_OK):
        raise ValueError("engine must be executable")
    if variant:
        prefix = "DeepSeek-V4.1-Flash-1017-" + variant
        shards = 4 if variant == GSQ_VARIANT else 7
        if Path(cfg["model"]).name != prefix + f"-00001-of-{shards:05d}.gguf":
            count = "four" if shards == 4 else "seven"
            raise ValueError("model must name the first of the " + count + " " + variant + " shards")
        for i in range(1, shards + 1):
            peer = Path(cfg["model"]).with_name(prefix + f"-{i:05d}-of-{shards:05d}.gguf")
            if not peer.is_file():
                raise ValueError("model shard is missing: " + str(peer))
        if variant == GSQ_VARIANT and not Path(cfg["model"]).with_name(GSQ_ENGRAM_FILE).is_file():
            raise ValueError("GSQ-RCO external Engram is missing: " + GSQ_ENGRAM_FILE)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-config", action="store_true", help="check local paths without starting inference")
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check_config:
        print("Configuration paths are valid. Generated metadata and full hashes are checked at launch. No model or GPU was loaded.")
        return
    prepare(cfg)
    from serve.deepseek41_engine import DeepSeekEngine
    from serve.deepseek41_frontend import DeepSeekService
    from serve.deepseek41_tokenizer import DeepSeekTokenizer
    from serve.server import serve

    command = [cfg["engine"], cfg["model"], cfg["draft"], cfg.get("mmproj") or "-",
               cfg["integrity"], cfg["profile"], cfg["heat"], cfg["costs"],
               cfg["draft_input_vocab"], cfg["draft_output_vocab"]]
    if "hot_bytes" in cfg:
        command.append(str(cfg["hot_bytes"]))
    if cfg.get("log"):
        Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("STRATA_DS41_ADAPTIVE_POLICY", "maya")
    os.environ["STRATA_DS41_BLOCK_SEGMENTS"] = "40"
    os.environ["STRATA_DS41_BLOCK_HUGE_MIN_PERCENT"] = "100"
    os.environ["STRATA_DS41_DECODE_SEGMENTS"] = "2"
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    engine = DeepSeekEngine(command, 1048576, 1, True, True, log_path=cfg.get("log"),
                           prefix_cache_mib=cfg.get("prefix_cache_mib", 0))
    server = None
    try:
        tokenizer = DeepSeekTokenizer(cfg["tokenizer"])
        service = DeepSeekService(engine, tokenizer, cfg.get("alias", "deepseek-v4.1"),
                                 vision=cfg.get("vision", bool(cfg.get("mmproj"))),
                                 sampling_defaults={"temperature": 0.0, "top_k": 1,
                                                    "top_p": 1.0, "seed": 1234})
        service.set_shared({"max_tokens": 32768})
        service.api_key = cfg.get("api_key", "")
        server = serve(service, cfg.get("host", "127.0.0.1"), cfg.get("port", 30011))
        if args.ready_file:
            from tools.deepseek41.download_assets import private_json
            private_json(args.ready_file, {"pid": os.getpid(), "host": server.server_address[0],
                                           "port": server.server_address[1]})
        print("DeepSeek ready on {}:{}".format(*server.server_address[:2]), flush=True)
        while not stop.wait(1):
            if not engine.alive():
                raise RuntimeError("native engine exited")
    finally:
        if server:
            server.shutdown()
            server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
