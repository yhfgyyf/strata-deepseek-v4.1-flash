"""Experimental DeepSeek service: python -m serve.deepseek41 --config FILE."""
import argparse
import json
import re
from pathlib import Path
import signal
import time

from serve.deepseek41_engine import DeepSeekEngine
from serve.deepseek41_frontend import DeepSeekService
from serve.deepseek41_tokenizer import DeepSeekTokenizer
from serve.server import serve


def chunk_option(cfg):
    def parse(value):
        if value == "auto":
            return "auto"
        if type(value) not in (int, str):
            raise ValueError("chunk/prefill must be an integer, auto, or auto:N")
        match = re.fullmatch(r"(auto:)?([0-9]+)", str(value))
        if match is None:
            raise ValueError("chunk/prefill must be an integer, auto, or auto:N")
        automatic, number = match.groups()
        tokens = int(number)
        if not 1 <= tokens <= 32768 or automatic and (tokens < 256 or tokens % 256):
            raise ValueError("chunk/prefill must be 1..32768; auto:N needs a multiple of 256")
        return ("auto" if tokens == 32768 else f"auto:{tokens}") if automatic else str(tokens)

    value = parse(cfg.get("chunk", cfg.get("prefill", 1024)))
    if "chunk" in cfg and "prefill" in cfg and value != parse(cfg["prefill"]):
        raise ValueError("chunk and prefill specify different values")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    host, port = cfg.get("host", "127.0.0.1"), cfg.get("port", 30011)
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.get("api_key"):
        raise ValueError("a non-loopback DeepSeek API requires api_key")
    context, slots = cfg.get("context", 32768), cfg.get("slots", 1)
    command = [cfg["engine"], "--serve", "--model", cfg["model"], "--integrity", cfg["integrity"],
               "--context", str(context), "--chunk", chunk_option(cfg), "--slots", str(slots)]
    for key, flag in [("draft", "--draft"), ("mmproj", "--mmproj"), ("profile", "--profile"),
                      ("hot_front_gib", "--hot-front-gib"), ("hot_back_gib", "--hot-back-gib"),
                      ("cpu_workers", "--cpu-workers"), ("engram_cache_mib", "--engram-cache-mib"),
                      ("work_mib", "--work-mib"), ("state_mib", "--state-mib"),
                      ("gpu_headroom_mib", "--gpu-headroom-mib")]:
        if cfg.get(key) is not None:
            if key in ("work_mib", "state_mib", "gpu_headroom_mib") and (
                    type(cfg[key]) is not int or cfg[key] < (0 if key == "gpu_headroom_mib" else 1) or
                    cfg[key] > (((1 << 64) - 1) >> 20)):
                raise ValueError(key + " must be an in-range integer MiB budget")
            command.extend([flag, str(cfg[key])])
    if type(cfg.get("hot_auto", False)) is not bool:
        raise ValueError("hot_auto must be a boolean")
    if cfg.get("hot_auto", False):
        if not cfg.get("profile"):
            raise ValueError("hot_auto requires a profile")
        command.append("--hot-auto")
    if cfg.get("spec", False):
        command.append("--spec")
    if cfg.get("trace", False):
        command.append("--trace")
    if cfg.get("expert_down_fusion", False):
        command.append("--expert-down-fusion")
    if type(cfg.get("adaptive_experts", False)) is not bool:
        raise ValueError("adaptive_experts must be a boolean")
    if cfg.get("adaptive_experts", False):
        if slots != 1 or cfg.get("phase_residency", False) or cfg.get("spec", False):
            raise ValueError("adaptive_experts requires one slot without phase residency or speculation")
        command.append("--adaptive-experts")
    if cfg.get("adaptive_experts", False) or cfg.get("phase_residency", False):
        if cfg.get("migration_mib") is not None:
            value = cfg["migration_mib"]
            minimum = 1 if cfg.get("adaptive_experts", False) else 0
            if type(value) is not int or value < minimum or value > (((1 << 64) - 1) >> 20):
                raise ValueError("migration_mib must be an in-range integer budget")
            command.extend(["--migration-mib", str(value)])
    if cfg.get("phase_residency", False):
        command.append("--phase-residency")
        for key, flag in [("borrow_gib", "--borrow-gib"), ("prefill_profile", "--prefill-profile")]:
            if cfg.get(key) is not None:
                command.extend([flag, str(cfg[key])])
    engine = DeepSeekEngine(command, context, slots, cfg.get("mode", "ced") == "ced", cfg.get("spec", False),
                            log_path=cfg.get("log"))
    try:
        tokenizer = DeepSeekTokenizer(cfg["tokenizer"])
        svc = DeepSeekService(engine, tokenizer, cfg.get("alias", "deepseek-v4.1"), bool(cfg.get("mmproj")),
                              sampling_defaults=cfg.get("sampling"))
        svc.api_key = cfg.get("api_key", "")
        server = serve(svc, host, port)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
            server.server_close()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
