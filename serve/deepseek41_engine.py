"""Bounded, request-keyed IPC to one native DeepSeek runtime with up to two slots."""
from __future__ import annotations

import itertools
import contextlib
import math
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading
import time

from serve.server import EngineDied, EngineSilent


TIMING_FIELDS = (
    ("prompt_ms", float), ("decode_ms", float), ("drafts_offered", int), ("drafts_accepted", int),
    ("engram_wait_ms", float), ("expert_gpu", int), ("expert_cpu", int), ("expert_h2d_bytes", int),
    ("encoder_token_layers", int), ("decoder_token_layers", int), ("kv_only_tokens", int),
    ("replay_tokens", int), ("draft_ms", float), ("verify_ms", float),
    ("phase_borrow_bytes", int), ("phase_backup_bytes", int), ("phase_encoder_h2d_bytes", int),
    ("phase_restore_h2d_bytes", int), ("phase_replay_reuse_bytes", int),
    ("phase_dynamic_h2d_bytes", int), ("phase_dynamic_d2h_bytes", int), ("phase_migrations", int),
    ("phase_ready_waits", int), ("phase_prepare_ms", float), ("phase_host_wait_ms", float),
    ("phase_restore_submit_ms", float), ("phase_dynamic_ms", float),
)


def sampling_params(sampling, prompt_tokens):
    """Validate and encode the native sampler's types/ranges before writing any GEN payload."""
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise ValueError("DeepSeek sampling must be an object")
    if any(sampling.get(k) for k in ("gumbel", "logit_bias", "grammar")):
        raise ValueError("DeepSeek gumbel/logit_bias/grammar sampling is not implemented")
    defaults = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "seed": 0,
                "repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0,
                "penalty_last_n": prompt_tokens}
    values = dict(defaults)
    # Keep the original backend spellings while accepting the serving API's canonical keys.
    aliases = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}
    for name, value in sampling.items():
        key = aliases.get(name, name)
        if key not in defaults:
            continue
        if key in ("top_k", "penalty_last_n", "seed"):
            low, high = (0, 2**64 - 1) if key == "seed" else (-(2**31), 2**31 - 1)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"DeepSeek sampling.{name} must be a native integer")
        else:
            if type(value) not in (int, float):
                raise ValueError(f"DeepSeek sampling.{name} must be a finite number")
            try:
                value = struct.unpack("f", struct.pack("f", value))[0]
            except (OverflowError, struct.error):
                raise ValueError(f"DeepSeek sampling.{name} exceeds native float range") from None
            if not math.isfinite(value):
                raise ValueError(f"DeepSeek sampling.{name} must be a finite number")
            if ((key == "temperature" and value < 0) or
                (key == "top_p" and not 0 < value <= 1) or
                (key == "min_p" and not 0 <= value <= 1) or
                (key == "repetition_penalty" and value <= 0)):
                raise ValueError(f"DeepSeek sampling.{name} is outside its native range")
        if name not in aliases or key not in sampling:
            values[key] = value
    return values


def quote(text):
    if not isinstance(text, str) or any(c in text for c in "\r\n\0"):
        raise ValueError("invalid native IPC path")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class DeepSeekEngine:
    def __init__(self, command, context=32768, slots=1, ced=True, speculative=False,
                 log_path=None, directory=None, ready_timeout=900):
        if slots not in (1, 2) or context < 1:
            raise ValueError("invalid DeepSeek context or slots")
        self.max_context = self.known_ctx = context
        self.batch, self.ced, self.speculative = slots, ced, speculative
        self.info = {"architecture": "deepseek-v4.1", "version": "experimental", "validated_context": 0}
        self.waiting, self.last, self.progress = 0, None, None
        self._slots = threading.BoundedSemaphore(slots)
        self._ids = itertools.count(1)
        self._lock, self._write_lock = threading.Lock(), threading.Lock()
        self._channels, self._active = {}, {}
        self._local = threading.local()
        self._ready = queue.Queue()
        self._temporary = tempfile.TemporaryDirectory(prefix="strata-ds41-", dir=directory)
        self.dir = Path(self._temporary.name)
        self.log_path = str(log_path or self.dir / "engine.log")
        self.log_start = Path(self.log_path).stat().st_size if Path(self.log_path).exists() else 0
        try:
            self._log = open(self.log_path, "ab", buffering=0)
            self.proc = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=self._log, text=True, bufsize=1)
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            ready = self._ready.get(timeout=ready_timeout)
            if isinstance(ready, Exception):
                raise ready
            if ready != (context, slots):
                raise EngineDied("native DeepSeek READY context/slot mismatch")
        except Exception:
            self.close()
            raise

    def _read(self):
        failure = EngineDied("native DeepSeek exited; inspect its private engine log")
        try:
            for line in self.proc.stdout:
                fields = line.rstrip("\n").split()
                if fields[:1] == ["READY"] and len(fields) == 3:
                    self._ready.put((int(fields[1]), int(fields[2])))
                    continue
                if not fields or fields[0] not in ("T", "D", "E") or len(fields) < 3:
                    raise EngineDied("invalid native DeepSeek IPC output")
                rid = int(fields[1])
                if rid == 0:
                    raise EngineDied("native DeepSeek IPC control error")
                with self._lock:
                    channel = self._channels.get(rid)
                if channel is None:
                    raise EngineDied("native DeepSeek output references an inactive request")
                channel.put((fields[0], fields[2:]))
        except Exception as error:
            failure = error
        finally:
            self._ready.put(failure)
            with self._lock:
                channels = list(self._channels.values())
            for channel in channels:
                channel.put(("F", failure))

    def _send(self, text):
        with self._write_lock:
            if self.proc.poll() is not None:
                raise EngineDied("native DeepSeek is no longer running")
            try:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise EngineDied("native DeepSeek input pipe ended") from error

    def alive(self):
        return self.proc.poll() is None

    def slots_view(self):
        with self._lock:
            active = list(self._active.items())
        return [{"slot": i, "state": "processing", "request_id": rid, "prompt_tokens": n} for i, (rid, n) in enumerate(active)] + \
               [{"slot": i, "state": "idle"} for i in range(len(active), self.batch)]

    def last_for_thread(self):
        return getattr(self._local, "last", None)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, original_prompt_length=None,
                 prefix_load="", prefix_save=""):
        if type(max_new) is not int or not ids or max_new < 0 or len(ids) + max_new > self.max_context:
            raise ValueError("DeepSeek request exceeds its context")
        if any(type(t) is not int or t < 0 or t >= 129280 for t in ids):
            raise ValueError("DeepSeek token IDs must be vocabulary integers")
        original_prompt_length = len(ids) if original_prompt_length is None else original_prompt_length
        if type(original_prompt_length) is not int or not 1 <= original_prompt_length <= len(ids):
            raise ValueError("DeepSeek original prompt length must be within its history")
        sampling = sampling_params(sampling, len(ids))
        prefix_paths = quote(prefix_load), quote(prefix_save)
        embeddings = list(embeddings or [])
        if len(embeddings) > 64:
            raise ValueError("DeepSeek request has too many images")
        for begin, image in embeddings:
            if (type(begin) is not int or begin < 0 or begin + image.tokens > original_prompt_length or
                type(image.height) is not int or type(image.width) is not int or
                image.height < 1 or image.width < 1 or image.height * image.width > 9198):
                raise ValueError("DeepSeek image span must fit the original prompt")
        with self._lock:
            self.waiting += 1
        acquired = False
        try:
            while not cancel.is_set():
                if self._slots.acquire(timeout=0.1):
                    acquired = True
                    break
            if not acquired:
                return
        finally:
            with self._lock:
                self.waiting -= 1
        rid, channel = next(self._ids), queue.Queue()
        terminal, sent, cancelled = False, False, False
        self._local.last = None
        with self._lock:
            self._channels[rid] = channel
            self._active[rid] = len(ids)
        temporary = None
        try:
            temporary = tempfile.TemporaryDirectory(prefix="request-", dir=self.dir)
            with contextlib.nullcontext(temporary.name) as directory:
                image_lines = []
                for i, (begin, image) in enumerate(embeddings):
                    path = Path(directory) / f"image-{i}.f32"
                    image.values.tofile(path)
                    image_lines.append(f"{begin} {image.height} {image.width} {quote(str(path))}")
                # Paths for prefix control are backend operations, never forwarded from HTTP sampling.
                header = ["GEN", rid, len(ids), max_new, int(self.ced), int(self.speculative),
                          sampling["temperature"], sampling["top_k"], sampling["top_p"], sampling["min_p"],
                          sampling["seed"], sampling["repetition_penalty"], sampling["frequency_penalty"],
                          sampling["presence_penalty"], sampling["penalty_last_n"], len(image_lines),
                          *prefix_paths, original_prompt_length]
                self._send(" ".join(map(str, header)) + "\n" + " ".join(map(str, ids)) +
                           ("\n" + "\n".join(image_lines) if image_lines else ""))
                sent = True
                while True:
                    if cancel.is_set() and not cancelled:
                        self._send(f"CANCEL {rid}")
                        cancelled = True
                    try:
                        kind, data = channel.get(timeout=0.2)
                    except queue.Empty:
                        if not self.alive():
                            raise EngineDied("native DeepSeek died during a request")
                        yield None
                        continue
                    if kind == "T":
                        if not cancelled:
                            yield int(data[0])
                    elif kind == "D":
                        terminal = True
                        if len(data) != len(TIMING_FIELDS):
                            raise EngineDied("invalid native DeepSeek timing record")
                        try:
                            last = {name: cast(value) for (name, cast), value in zip(TIMING_FIELDS, data)}
                            if any(value < 0 or (cast is int and value >= 2**64) or
                                   (cast is float and not math.isfinite(value))
                                   for name, cast in TIMING_FIELDS for value in (last[name],)):
                                raise ValueError("invalid timing value")
                        except (ValueError, OverflowError):
                            raise EngineDied("invalid native DeepSeek timing record") from None
                        last.update(prompt_tokens=len(ids), original_prompt_tokens=original_prompt_length, reused=0)
                        self._local.last = last
                        self.last = last
                        break
                    elif kind == "E":
                        terminal = True
                        if not cancelled:
                            raise RuntimeError("native DeepSeek: " + " ".join(data))
                        break
                    elif kind == "F":
                        terminal = True
                        raise data
        finally:
            try:
                if sent and not terminal and self.alive():
                    self._send(f"CANCEL {rid}")
                    deadline = time.monotonic() + 90
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            self.proc.terminate()
                            raise EngineSilent("native DeepSeek did not acknowledge cancellation")
                        try:
                            kind, _ = channel.get(timeout=min(0.2, remaining))
                        except queue.Empty:
                            continue
                        if kind in ("D", "E", "F"):
                            break
            finally:
                with self._lock:
                    self._channels.pop(rid, None)
                    self._active.pop(rid, None)
                if temporary is not None:
                    temporary.cleanup()
                self._slots.release()

    def close(self):
        if hasattr(self, "proc") and self.alive():
            try:
                self._send("QUIT")
                self.proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, OSError, EngineDied):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if hasattr(self, "_reader"):
            self._reader.join(timeout=1)
        if hasattr(self, "proc"):
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream is not None:
                    with contextlib.suppress(OSError):
                        stream.close()
        if hasattr(self, "_log"):
            self._log.close()
        if hasattr(self, "_temporary"):
            self._temporary.cleanup()
