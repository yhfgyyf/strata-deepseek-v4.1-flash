"""Install and manage the fixed DeepSeek profile. No GPU is loaded by plan or gui."""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.deepseek41.download_assets import (SOURCES, assets_for_config, download_file, private_json, read_json)
from tools.deepseek41.startup_assets import GSQ_VARIANT, VARIANTS

ENTRY = ROOT / "tools/deepseek41/serve_experimental.py"
PATH_KEYS = ("engine", "model", "draft", "mmproj", "tokenizer", "runtime_dir", "profile", "heat", "integrity",
             "costs", "draft_input_vocab", "draft_output_vocab", "log")
EDIT_KEYS = set(PATH_KEYS) | {"variant", "host", "port", "alias", "api_key", "vision", "hot_bytes", "prefix_cache_mib"}
SETUP_DEFAULTS = dict(source="modelscope", data_dir=str(ROOT / "models"), build_dir=str(ROOT / "build"),
                      cuda="/usr/local/cuda-13.2/bin/nvcc", ggml_dir="", jobs=2, no_proxy=False)


def process_identity(pid):
    """PID reuse and zombie processes must not be mistaken for a running owned service."""
    try:
        if sys.platform.startswith("linux"):
            proc = Path("/proc") / str(int(pid))
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            return dict(start=fields[19], boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                        command=hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest())
        # Used by the management UI's CPU-only tests on macOS; native serving still requires Linux.
        result = subprocess.run(["ps", "-p", str(int(pid)), "-o", "lstart=,stat=,command="],
                                capture_output=True, text=True, timeout=3)
        line = result.stdout.strip()
        if result.returncode or not line or "Z" in line.split()[5]:
            return None
        # The process state (R/S/...) changes normally; it is not part of its identity.
        fields = line.split()
        return dict(ps=hashlib.sha256(" ".join(fields[:5] + fields[6:]).encode()).hexdigest())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def owned(record):
    return bool(record and record.get("pid") and record.get("identity")
                and process_identity(record["pid"]) == record["identity"])


def signal_owned(record, sig, group=False):
    fd = None
    try:
        pid = record["pid"]
        if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
            fd = os.pidfd_open(pid)
        if not owned(record):
            return False
        if group:
            if os.getpgid(pid) != pid:
                return False
            os.killpg(pid, sig)
        elif fd is not None:
            signal.pidfd_send_signal(fd, sig)
        else:
            os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


@contextlib.contextmanager
def exclusive(path, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError("another launcher operation is already running") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run_output(command, timeout=10):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return 1, "", str(error)


def endpoint(cfg):
    host = cfg.get("host", "127.0.0.1")
    host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else "::1" if host == "::" else host
    return "http://" + ("[" + host + "]" if ":" in host else host) + ":" + str(cfg.get("port", 30011))


def health(cfg):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(endpoint(cfg) + "/health", timeout=2) as response:
            value = json.load(response)
        return value if value.get("service") == "strata" and value.get("loaded") is True else None
    except (OSError, ValueError):
        return None


def default_config(variant, setup):
    example = {GSQ_VARIANT: "gsq-rco-iq2-xs", "UD-Q2_K_XL": "ud-q2-k-xl", "UD-IQ3_XXS": "ud-iq3-xxs"}[variant]
    cfg = read_json(ROOT / ("examples/rtxpro6000." + example + ".json"))
    data = Path(setup["data_dir"]).expanduser().absolute()
    cfg.update(engine=str(Path(setup["build_dir"]).expanduser().absolute() / "strata-deepseek41-experimental"),
               model=str(data / variant / Path(cfg["model"]).name),
               draft=str(data / "shared" / Path(cfg["draft"]).name),
               mmproj=str(data / "shared" / "mmproj-BF16.gguf"), tokenizer=str(data / "shared/tokenizer.json"),
               runtime_dir=str(ROOT / "runtime" / example), log=str(ROOT / "runtime" / example / "engine.log"))
    return cfg


def validate_config(cfg):
    if not isinstance(cfg, dict) or cfg.get("variant") not in VARIANTS:
        raise ValueError("choose GSQ-RCO-IQ2_XS, UD-Q2_K_XL or UD-IQ3_XXS")
    for key in ("engine", "model", "draft", "tokenizer"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(key + " must be a file path")
    for key in PATH_KEYS:
        if cfg.get(key) is not None and (not isinstance(cfg[key], str) or any(c in cfg[key] for c in "\r\n\0")):
            raise ValueError("invalid path: " + key)
    port = cfg.get("port", 30011)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in 1..65535")
    host = cfg.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        raise ValueError("invalid API host")
    if not isinstance(cfg.get("api_key", ""), str):
        raise ValueError("api_key must be a string")
    if any(c in cfg.get("api_key", "") for c in "\r\n\0"):
        raise ValueError("invalid API key")
    if not isinstance(cfg.get("alias", "deepseek-v4.1"), str) or not cfg.get("alias", "deepseek-v4.1"):
        raise ValueError("alias must be a nonempty string")
    if host not in ("127.0.0.1", "localhost", "::1") and not cfg.get("api_key"):
        raise ValueError("a non-loopback API requires api_key")
    if "vision" in cfg and type(cfg["vision"]) is not bool:
        raise ValueError("vision must be a boolean")
    if cfg.get("vision") and (cfg["variant"] == GSQ_VARIANT or not cfg.get("mmproj")):
        raise ValueError("vision requires a UD model and mmproj")
    cache = cfg.get("prefix_cache_mib", 0)
    if type(cache) is not int or not 0 <= cache <= 4096:
        raise ValueError("prefix_cache_mib must be an integer in 0..4096")
    hot = cfg.get("hot_bytes", "auto")
    if hot != "auto" and (type(hot) is not int or not 0 < hot < 1 << 63):
        raise ValueError("hot_bytes must be auto or a positive integer below 2^63")
    if bool(cfg.get("profile")) != bool(cfg.get("heat")):
        raise ValueError("supply both profile and heat, or neither")


class Manager:
    def __init__(self, config=None, overrides=None):
        self.config = Path(config or ROOT / "config.local.json").expanduser().resolve()
        key = hashlib.sha256(str(self.config).encode()).hexdigest()[:16]
        self.state_dir = ROOT / "runtime/launcher" / key
        self.overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
        self.process = None

    @property
    def python(self):
        return ROOT / ".venv/bin/python"

    def setup(self):
        return {**SETUP_DEFAULTS, **read_json(self.state_dir / "setup.json", {}), **self.overrides}

    def config_value(self, resolve=False):
        cfg = read_json(self.config)
        if cfg is None:
            cfg = default_config(GSQ_VARIANT, self.setup())
        validate_config(cfg)
        if resolve:
            cfg = dict(cfg)
            for key in PATH_KEYS:
                if cfg.get(key):
                    path = Path(cfg[key]).expanduser()
                    cfg[key] = os.path.abspath(path if path.is_absolute() else self.config.parent / path)
        return cfg

    def public_config(self):
        cfg = self.config_value()
        has_key = bool(cfg.pop("api_key", ""))
        return dict(path=str(self.config), exists=self.config.is_file(), values=cfg, has_api_key=has_key,
                    setup=self.setup(), variants=[GSQ_VARIANT, "UD-Q2_K_XL", "UD-IQ3_XXS"])

    def job(self):
        job = read_json(self.state_dir / "job.json", {})
        queued = read_json(self.state_dir / "queued.json", {})
        if queued.get("started", 0) > job.get("started", 0):
            job = queued
        elif queued.get("pid") == job.get("pid") and queued.get("state") == "failed":
            job = {**job, **queued}
        cancelled = read_json(self.state_dir / "cancelled.json", {})
        if job.get("pid") and all(cancelled.get(k) == job.get(k) for k in ("pid", "identity")):
            return {**job, "state": "cancelled", "phase": "cancelled"}
        if job.get("state") == "queued" and not owned(job):
            return {**job, "state": "failed", "message": "preparation exited; inspect the preparation log"}
        if job.get("state") == "running" and not owned(job):
            return {**job, "state": "interrupted", "message": "preparation exited; logs and partial downloads retained"}
        return job

    def status(self):
        state = read_json(self.state_dir / "service.json", {})
        running = owned(state.get("supervisor"))
        if (state.get("state") in ("starting", "ready", "stopping") and not running
                and (state.get("supervisor") or time.time() - state.get("started", 0) > 5)):
            state = {**state, "state": "orphaned" if owned(state.get("worker")) else "exited",
                     "message": "supervisor is no longer running; inspect the recorded processes and logs"}
        return {"state": state.get("state", "stopped"), **state, "running": running,
                "ready": running and state.get("state") == "ready", "config": str(self.config),
                "url": state.get("url", endpoint(self.config_value())), "log": str(self.state_dir / "service.log")}

    def ensure_idle(self):
        status = self.status()
        if status["running"] or status["state"] == "orphaned":
            raise RuntimeError("stop this managed service before changing configuration or installing")
        job = self.job()
        if job.get("state") in ("running", "queued") and job.get("pid") != os.getpid():
            raise RuntimeError("preparation is already running")

    def save_config(self, changes=None, setup=None, reset_paths=False, clear_api_key=False):
        changes, setup = changes or {}, setup or {}
        if not isinstance(changes, dict) or not isinstance(setup, dict):
            raise ValueError("config and setup changes must be JSON objects")
        if type(reset_paths) is not bool or type(clear_api_key) is not bool:
            raise ValueError("reset_paths and clear_api_key must be booleans")
        if set(changes) - EDIT_KEYS or set(setup) - SETUP_DEFAULTS.keys():
            raise ValueError("unknown configuration field")
        settings = {**self.setup(), **setup}
        if settings["source"] not in SOURCES or type(settings["no_proxy"]) is not bool:
            raise ValueError("invalid download source or no_proxy")
        if type(settings["jobs"]) is not int or not 1 <= settings["jobs"] <= 16:
            raise ValueError("build jobs must be an integer in 1..16")
        for key in ("data_dir", "build_dir", "cuda", "ggml_dir"):
            if not isinstance(settings[key], str) or any(c in settings[key] for c in "\r\n\0"):
                raise ValueError("invalid setup path: " + key)
            if settings[key]:
                settings[key] = str(Path(settings[key]).expanduser().absolute())
        if not all(settings[k] for k in ("data_dir", "build_dir", "cuda")):
            raise ValueError("data_dir, build_dir and cuda cannot be empty")
        with exclusive(self.state_dir / "operation.lock"):
            self.ensure_idle()
            current = self.config_value()
            variant = changes.get("variant", current["variant"])
            if variant not in VARIANTS:
                raise ValueError("unsupported variant")
            cfg = (default_config(variant, settings) if reset_paths or not self.config.exists() else current)
            # A blank password box preserves the existing key. Clearing is always explicit.
            if not changes.get("api_key") and not clear_api_key:
                changes = {k: v for k, v in changes.items() if k != "api_key"}
                if current.get("api_key"):
                    cfg["api_key"] = current["api_key"]
            cfg.update(changes)
            if "hot_bytes" in changes and changes["hot_bytes"] is None:
                cfg.pop("hot_bytes", None)
            if clear_api_key:
                cfg.pop("api_key", None)
            validate_config(cfg)
            backup = None
            if self.config.exists():
                backup = self.config.with_name(self.config.name + f".bak.{time.time_ns()}")
                shutil.copy2(self.config, backup)
                backup.chmod(0o600)
            private_json(self.config, cfg)
            private_json(self.state_dir / "setup.json", settings)
        return dict(ok=True, config=str(self.config), backup=str(backup) if backup else None)

    def plan(self):
        cfg, settings = self.config_value(resolve=True), self.setup()
        assets = assets_for_config(cfg)
        groups = {}
        for asset in assets:
            path = Path(asset["destination"])
            part = path.with_name(path.name + ".part")
            have = path.stat().st_size if path.exists() else part.stat().st_size if part.exists() else 0
            asset.update(exists=path.is_file(), partial_bytes=part.stat().st_size if part.exists() else 0,
                         download_bytes=0 if path.exists() else max(0, asset["bytes"] - have),
                         size_matches=path.is_file() and have == asset["bytes"], verified=False)
            parent = path.parent
            while not parent.exists():
                parent = parent.parent
            device = str(parent.stat().st_dev)
            group = groups.setdefault(device, dict(path=str(parent), free_bytes=shutil.disk_usage(parent).free, needed_bytes=0))
            group["needed_bytes"] += asset["download_bytes"]
        for group in groups.values():
            group["enough"] = group["free_bytes"] >= group["needed_bytes"] + (5 << 30 if group["needed_bytes"] else 0)
        expected_engine = Path(settings["build_dir"]) / "strata-deepseek41-experimental"
        return dict(ok=True, config=str(self.config), config_exists=self.config.is_file(), variant=cfg["variant"],
                    source=settings["source"], no_proxy=settings["no_proxy"], assets=assets,
                    download_bytes=sum(x["download_bytes"] for x in assets), disks=list(groups.values()),
                    build=str(expected_engine) == cfg["engine"] or not Path(cfg["engine"]).is_file(),
                    engine=cfg["engine"], url=endpoint(cfg),
                    profile="Linux x86-64 / CUDA 13.2 / RTX PRO 6000 96GB; 8192 chunk, 40 blocks, 100% huge pages",
                    note="Existing sizes are not SHA256 proof. Install verifies all assets; start verifies native-loader metadata.")

    def doctor(self, stage="run"):
        if stage not in ("build", "run"):
            raise ValueError("stage must be build or run")
        checks = []
        def check(name, passed, detail):
            checks.append(dict(name=name, ok=bool(passed), detail=detail))
        check("platform", sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"),
              platform.system() + " " + platform.machine())
        check("python", sys.version_info >= (3, 10), platform.python_version() + " (requires 3.10+)")
        if stage == "build":
            for name, command in (("CMake", ["cmake", "--version"]), ("C++", ["c++", "--version"]),
                                  ("Git", ["git", "--version"]), ("CUDA 13.2", [self.setup()["cuda"], "--version"])):
                code, out, error = run_output(command)
                supported = code == 0 and (name != "CUDA 13.2" or "release 13.2" in out)
                if name == "CMake" and supported:
                    import re
                    version = re.search(r"version (\d+)\.(\d+)", out)
                    supported = bool(version and tuple(map(int, version.groups())) >= (3, 24))
                check(name, supported, (out or error)[:500])
        else:
            code, out, error = run_output([str(self.python), "-c", "import numpy,jinja2,regex,PIL,requests,psutil"])
            check("runtime dependencies", code == 0, "ready" if code == 0 else (error or out)[-600:])
            code, out, error = run_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,compute_cap",
                                           "--format=csv,noheader,nounits"])
            devices = list(csv.reader(out.splitlines())) if code == 0 else []
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            selected = [r for r in devices if len(r) == 7 and (visible is None and r[0].strip() == "0"
                        or visible is not None and visible in (r[0].strip(), r[1].strip()))]
            check("GPU selection", len(selected) == 1, out or error or "no GPU")
            if len(selected) == 1:
                gpu = [v.strip() for v in selected[0]]
                check("GPU profile", "RTX PRO 6000" in gpu[2] and gpu[6] == "12.0" and float(gpu[3]) >= 90000, gpu[2])
                check("GPU available", float(gpu[4]) >= 90000 and float(gpu[5]) <= 10,
                      f"{gpu[4]} MiB free, {gpu[5]}% busy")
                code, processes, error = run_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
                check("no competing GPU workload", code == 0 and gpu[1] not in processes, processes or error or "none")
            try:
                mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
                check("available RAM", mem["MemAvailable"] >= 160 * 1024 * 1024,
                      f"{mem['MemAvailable'] / 1048576:.1f} GiB available; starter requires at least 160 GiB")
                cpus = os.sched_getaffinity(0)
                check("CPU 8", 8 in cpus, "CPU 8 must be in the process affinity set")
            except (OSError, KeyError, AttributeError, ValueError) as error:
                check("host resources", False, str(error))
            cfg = self.config_value(resolve=True)
            try:
                from tools.deepseek41.serve_experimental import load_config
                load_config(self.config)
                check("config paths", True, str(self.config))
            except (OSError, ValueError) as error:
                check("config paths", False, str(error))
            check("huge-page policy", True, "Native startup enforces 40 registered blocks and 100% coverage; no system settings changed")
        return dict(ok=all(x["ok"] for x in checks), stage=stage, checks=checks)

    def install(self, yes=False, wait_lock=False):
        if not yes:
            raise ValueError("inspect plan, then pass --yes to install dependencies/build/download the selected files")
        if not self.config.exists():
            self.save_config()
        with exclusive(self.state_dir / "operation.lock", wait=wait_lock):
            self.ensure_idle()
            plan, cfg, settings = self.plan(), self.config_value(resolve=True), self.setup()
            if not all(d["enough"] for d in plan["disks"]):
                raise RuntimeError("insufficient disk space; inspect plan for each destination filesystem")
            expected = str(Path(settings["build_dir"]) / "strata-deepseek41-experimental")
            if plan["build"] and cfg["engine"] != expected:
                raise ValueError("missing custom engine; build into build_dir or point engine at an existing executable")
            check = self.doctor("build") if plan["build"] else dict(ok=sys.platform.startswith("linux") and sys.version_info >= (3, 10))
            if not check["ok"]:
                raise RuntimeError("build requirements are not met: " + json.dumps(check))
            job = dict(state="running", pid=os.getpid(), identity=process_identity(os.getpid()), started=time.time())
            def progress(**value):
                job.update(value, updated=time.time())
                private_json(self.state_dir / "job.json", job)
                print(json.dumps(value), file=sys.stderr, flush=True)
            env = dict(os.environ)
            if settings["no_proxy"]:
                env = {k: v for k, v in env.items() if not k.lower().endswith("_proxy")}
            env["CUDA_VISIBLE_DEVICES"] = ""
            def command(args, phase):
                progress(phase=phase, command=args)
                subprocess.run(args, cwd=ROOT, env=env, check=True, stdout=sys.stderr, stderr=sys.stderr)
            try:
                progress(phase="dependencies")
                if not self.python.exists():
                    command([sys.executable, "-m", "venv", str(self.python.parent.parent)], "virtualenv")
                command([str(self.python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], "dependencies")
                if plan["build"]:
                    args = ["cmake", "-S", str(ROOT), "-B", settings["build_dir"], "-DCMAKE_BUILD_TYPE=Release",
                            "-DSTRATA_ENABLE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=120",
                            "-DCMAKE_CUDA_COMPILER=" + settings["cuda"], "-DSTRATA_BUILD_TESTS=OFF",
                            "-DSTRATA_BUILD_DEEPSEEK41_TESTS=ON"]
                    if settings["ggml_dir"]:
                        args.append("-DSTRATA_GGML_DIR=" + settings["ggml_dir"])
                    command(args, "configuring")
                    command(["cmake", "--build", settings["build_dir"], "--parallel", str(settings["jobs"]), "--target",
                             "strata-deepseek41", "strata-deepseek41-experimental", "strata-deepseek41-inspect"], "building")
                receipts = read_json(self.state_dir / "downloads.json", {})
                for asset in plan["assets"]:
                    download_file(asset, settings["source"], receipts, settings["no_proxy"], progress)
                    private_json(self.state_dir / "downloads.json", receipts)
                command([str(self.python), str(ENTRY), "--config", str(self.config), "--check-config"], "validating")
                progress(state="complete", phase="complete", completed=time.time())
            except BaseException as error:
                progress(state="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", phase="failed",
                         error=str(error), completed=time.time())
                raise
        return dict(ok=True, config=str(self.config), message="Installed and verified; the model has not been started")

    def start(self, wait=True, timeout=1800, check=True):
        if not math.isfinite(timeout) or not 0 < timeout <= 86400:
            raise ValueError("timeout must be a finite number in 0..86400 seconds")
        with exclusive(self.state_dir / "operation.lock"):
            self.ensure_idle()
            if check:
                report = self.doctor()
                if not report["ok"]:
                    raise RuntimeError("preflight failed: " + json.dumps(report))
            cfg = self.config_value(resolve=True)
            instance = secrets.token_hex(16)
            state = dict(state="starting", instance=instance, started=time.time(), url=endpoint(cfg))
            private_json(self.state_dir / "service.json", state)
            with (self.state_dir / "service.log").open("ab", buffering=0) as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_supervise", "--config", str(self.config),
                                         "--instance", instance, "--timeout", str(timeout)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log, start_new_session=True)
            # The supervisor fills in its own identity before launching the model. Do not overwrite its receipt.
            self.process = proc
            threading.Thread(target=proc.wait, daemon=True).start()
        deadline = time.monotonic() + (timeout + 10 if wait else 5)
        while time.monotonic() < deadline:
            state = self.status()
            if state["ready"] or (not wait and state["running"]):
                return dict(ok=True, **state)
            if state["state"] in ("failed", "stopped") or proc.poll() is not None:
                raise RuntimeError("service failed to start; inspect " + str(self.state_dir / "service.log"))
            time.sleep(.2)
        raise RuntimeError("startup is still pending; inspect status and logs, or stop this managed service")

    def stop(self, timeout=90):
        with exclusive(self.state_dir / "operation.lock"):
            state = self.status()
            if state["state"] == "orphaned":
                raise RuntimeError("supervisor is missing; refusing an unverified process-group stop")
            if not state["running"]:
                return dict(ok=True, state=state["state"], message="no owned service is running")
            record = state["supervisor"]
            if not owned(record):
                raise RuntimeError("process identity changed; no signal sent")
            signal_owned(record, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.status()
            if not result["running"] and not owned(result.get("worker")):
                return dict(ok=True, **result)
            time.sleep(.2)
        raise RuntimeError("stop is still pending; inspect status and logs; no unrelated process was signalled")

    def logs(self, kind="service", limit=24000):
        if kind not in ("service", "install"):
            raise ValueError("unknown log")
        path = self.state_dir / (kind + ".log")
        paths = [(kind, path)]
        native = self.config_value(resolve=True).get("log") if kind == "service" else None
        if native and Path(native).resolve() != path.resolve():
            paths.append(("native", Path(native)))
        parts = []
        budget = limit // len(paths)
        for label, target in paths:
            try:
                with target.open("rb") as stream:
                    stream.seek(max(0, os.fstat(stream.fileno()).st_size - budget))
                    value = stream.read(budget).decode("utf-8", errors="replace")
                    if value:
                        parts.append(("[" + label + "]\n" if len(paths) > 1 else "") + value)
            except FileNotFoundError:
                pass
        return "\n".join(parts)

    def prepare_background(self, launch=False):
        with exclusive(self.state_dir / "operation.lock"):
            self.ensure_idle()
            if not self.config.exists():
                raise ValueError("save the configuration first")
            with (self.state_dir / "install.log").open("ab", buffering=0) as log:
                proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "launch" if launch else "install",
                                         "--config", str(self.config), "--yes", "--json", "--wait-lock"], cwd=ROOT,
                                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            # The child acquires operation.lock after this method returns.
            record = dict(state="queued", pid=proc.pid, identity=process_identity(proc.pid), started=time.time())
            with exclusive(self.state_dir / "queue.lock"):
                private_json(self.state_dir / "queued.json", record)
            def finish():
                code = proc.wait()
                with exclusive(self.state_dir / "queue.lock", wait=True):
                    if read_json(self.state_dir / "queued.json", {}).get("pid") == proc.pid:
                        private_json(self.state_dir / "queued.json", {**record, "state": "complete" if code == 0 else "failed",
                                     "exit_code": code, "completed": time.time()})
            threading.Thread(target=finish, daemon=True).start()
        return dict(ok=True, **record)

    def cancel_prepare(self):
        if self.status()["running"]:
            raise RuntimeError("model startup has already begun; use Stop to end the managed service")
        job = self.job()
        if job.get("state") not in ("running", "queued") or not owned(job):
            return dict(ok=True, message="no owned preparation job is running")
        if os.getpgid(job["pid"]) != job["pid"]:
            raise RuntimeError("preparation was started in another terminal; use Ctrl+C there")
        if not owned(job):
            raise RuntimeError("preparation process identity changed; no signal sent")
        private_json(self.state_dir / "cancelled.json", {k: job[k] for k in ("pid", "identity")})
        signal_owned(job, signal.SIGTERM, group=True)
        return dict(ok=True, message="preparation cancelled; partial files and logs retained")


def supervise(manager, instance, timeout):
    worker, stop = None, threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    record = dict(state="starting", instance=instance, started=time.time(),
                  supervisor=dict(pid=os.getpid(), identity=process_identity(os.getpid())))
    state_path = manager.state_dir / "service.json"
    ready_path = manager.state_dir / ("ready-" + instance + ".json")
    locks = contextlib.ExitStack()
    def save(**values):
        record.update(values, updated=time.time())
        private_json(state_path, record)
    try:
        if sys.platform.startswith("linux"):
            import ctypes
            # Reap the native child even if the Python frontend crashes before it can close it.
            if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                raise OSError("could not establish native child cleanup")
        cfg = manager.config_value(resolve=True)
        record["url"] = endpoint(cfg)
        locks.enter_context(exclusive(manager.state_dir / "running.lock"))
        locks.enter_context(exclusive(ROOT / "runtime/launcher" / f"port-{cfg.get('port', 30011)}.lock"))
        # This checks for another service before loading hundreds of GB. Never adopt or stop it.
        family = socket.AF_INET6 if ":" in cfg.get("host", "") else socket.AF_INET
        with socket.socket(family) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((cfg.get("host", "127.0.0.1"), cfg.get("port", 30011)))
        save()
        worker = subprocess.Popen([str(manager.python), "-u", str(ENTRY), "--config", str(manager.config),
                                   "--ready-file", str(ready_path)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                  start_new_session=True)
        worker_record = dict(pid=worker.pid, identity=process_identity(worker.pid))
        save(worker=worker_record)
        deadline = time.monotonic() + timeout
        while not stop.wait(.5):
            code = worker.poll()
            if code is not None:
                raise RuntimeError(f"service process exited with code {code}")
            if record["state"] == "starting":
                ready = read_json(ready_path, {})
                if ready.get("pid") == worker.pid and health(cfg):
                    save(state="ready", ready_at=time.time())
                elif time.monotonic() > deadline:
                    raise RuntimeError("model startup timed out")
        save(state="stopping")
    except BaseException as error:
        save(state="failed", error=str(error))
        print("Launcher: " + str(error), file=sys.stderr, flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            # First let the Python frontend close the native engine and its cache normally.
            worker.terminate()
            try:
                worker.wait(timeout=45)
            except subprocess.TimeoutExpired:
                signal_owned(record["worker"], signal.SIGTERM, group=True)
                try:
                    worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    signal_owned(record["worker"], signal.SIGKILL, group=True)
                    worker.wait(timeout=10)
        if sys.platform.startswith("linux"):
            children_path = Path(f"/proc/self/task/{os.getpid()}/children")
            # These are only our adopted descendants; never enumerate or signal by process name.
            for child in children_path.read_text().split():
                pid = int(child)
                child_record = dict(pid=pid, identity=process_identity(pid))
                if owned(child_record):
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 5
                    while owned(child_record) and time.monotonic() < deadline:
                        time.sleep(.1)
                    if owned(child_record):
                        os.kill(pid, signal.SIGKILL)
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(pid, 0)
        if record["state"] != "failed":
            save(state="stopped", stopped_at=time.time(), exit_code=worker.returncode if worker else None)
        if ready_path.exists():
            ready_path.unlink()
        locks.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "configure", "install", "launch", "start", "stop", "cancel", "status", "doctor", "logs", "gui", "_supervise"):
        p = commands.add_parser(name, help=argparse.SUPPRESS if name == "_supervise" else name)
        p.add_argument("--config", type=Path, default=ROOT / "config.local.json")
        p.add_argument("--json", action="store_true", help="machine-readable result; progress goes to stderr")
        if name in ("plan", "configure", "install", "launch", "gui"):
            p.add_argument("--source", choices=SOURCES)
            p.add_argument("--data-dir")
            p.add_argument("--build-dir")
            p.add_argument("--cuda")
            p.add_argument("--ggml-dir")
            p.add_argument("--jobs", type=int)
            p.add_argument("--no-proxy", action="store_true", default=None)
        if name == "configure":
            p.add_argument("--model", choices=VARIANTS)
            p.add_argument("--reset-paths", action="store_true", help="replace asset paths with the selected data directory")
            p.add_argument("--set", type=Path, help="JSON object of config fields; API keys stay in a private file")
        if name in ("install", "launch"):
            p.add_argument("--yes", action="store_true")
            p.add_argument("--wait-lock", action="store_true", help=argparse.SUPPRESS)
        if name in ("start", "launch", "_supervise"):
            p.add_argument("--timeout", type=float, default=1800)
            p.add_argument("--no-wait", action="store_true")
        if name == "doctor":
            p.add_argument("--stage", choices=("build", "run"), default="run")
        if name == "logs":
            p.add_argument("--kind", choices=("service", "install"), default="service")
        if name == "gui":
            p.add_argument("--port", type=int, default=30010)
            p.add_argument("--no-browser", action="store_true")
        if name == "_supervise":
            p.add_argument("--instance", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv if argv is not None else sys.argv[1:] or ["gui"])
    manager = Manager(args.config, {k: getattr(args, k, None) for k in SETUP_DEFAULTS})
    try:
        if args.command == "configure":
            changes = read_json(args.set, {}) if args.set else {}
            if not isinstance(changes, dict):
                raise ValueError("--set must contain a JSON object")
            changes.setdefault("variant", args.model or manager.config_value()["variant"])
            value = manager.save_config(changes, reset_paths=args.reset_paths)
        elif args.command == "plan":
            value = manager.plan()
        elif args.command in ("install", "launch"):
            if manager.overrides:
                manager.save_config(setup=manager.overrides)
            value = manager.install(args.yes, args.wait_lock)
            if args.command == "launch":
                value = manager.start(not args.no_wait, args.timeout)
        elif args.command == "start":
            value = manager.start(not args.no_wait, args.timeout)
        elif args.command == "stop":
            value = manager.stop()
        elif args.command == "cancel":
            value = manager.cancel_prepare()
        elif args.command == "status":
            value = dict(ok=True, **manager.status(), job=manager.job())
        elif args.command == "doctor":
            value = manager.doctor(args.stage)
        elif args.command == "logs":
            value = dict(ok=True, text=manager.logs(args.kind))
            if not args.json:
                print(value["text"], end="")
                return 0
        elif args.command == "gui":
            from tools.deepseek41.launcher_web import run
            run(manager, args.port, not args.no_browser)
            return 0
        elif args.command == "_supervise":
            supervise(manager, args.instance, args.timeout)
            return 0
        print(json.dumps(value, indent=2))
        return 0 if value.get("ok", True) else 1
    except KeyboardInterrupt:
        print(json.dumps(dict(ok=False, error="interrupted; partial files and logs retained")), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(ok=False, error=str(error))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
