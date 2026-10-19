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


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    host, port = cfg.get("host", "127.0.0.1"), cfg.get("port", 30011)
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.get("api_key"):
        raise ValueError("a non-loopback DeepSeek API requires api_key")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in 1..65535")
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
        resolved = resolved.resolve()
        if name != "log" and not resolved.is_file():
            raise ValueError(name + " file is missing: " + str(resolved))
        cfg[name] = str(resolved)
    if not os.access(cfg["engine"], os.X_OK):
        raise ValueError("engine must be executable")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-config", action="store_true", help="check local paths without starting inference")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.check_config:
        print("Configuration paths are valid. No model or GPU was loaded.")
        return
    from serve.deepseek41_engine import DeepSeekEngine
    from serve.deepseek41_frontend import DeepSeekService
    from serve.deepseek41_tokenizer import DeepSeekTokenizer
    from serve.server import serve

    command = [cfg["engine"], cfg["model"], cfg["draft"], cfg.get("mmproj") or "-",
               cfg["integrity"], cfg["profile"], cfg["heat"], cfg["costs"],
               cfg["draft_input_vocab"], cfg["draft_output_vocab"]]
    if cfg.get("log"):
        Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX", "0")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    engine = DeepSeekEngine(command, 32768, 1, True, True, log_path=cfg.get("log"))
    server = None
    try:
        tokenizer = DeepSeekTokenizer(cfg["tokenizer"])
        service = DeepSeekService(engine, tokenizer, cfg.get("alias", "deepseek-v4.1"),
                                 vision=bool(cfg.get("mmproj")),
                                 sampling_defaults={"temperature": 0.0, "top_k": 1,
                                                    "top_p": 1.0, "seed": 1234})
        service.api_key = cfg.get("api_key", "")
        server = serve(service, cfg.get("host", "127.0.0.1"), cfg.get("port", 30011))
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
