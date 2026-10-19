"""DeepSeek grammar adapter for the existing Chat/Responses HTTP handlers."""
from __future__ import annotations

import contextlib
import copy
import json
import time

from serve import deepseek41_encoding as codec
from serve.deepseek41_images import expand_images, preprocess
from serve.frontend import Event, ToolCall, _object_list, _parts_of, _tool_list, _text_of, tool_choice_of
from serve.responses import ResponsesError
from serve.server import Detokenizer, EngineStarting, Service, StopMatcher, Vision, stop_strings


def effort_kwargs(req, shared=None):
    reasoning = req.get("reasoning") or {}
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object")
    custom = req.get("chat_template_kwargs") or {}
    if not isinstance(custom, dict):
        raise ValueError("chat_template_kwargs must be an object")
    effort = custom.get("reasoning_effort", req.get("reasoning_effort", reasoning.get("effort")))
    if effort is None:
        effort = (shared or {}).get("reasoning_effort", "high")
    thinking = custom.get("enable_thinking", effort not in ("none", "minimal"))
    if effort in ("none", "minimal"):
        effort = "high"
    if type(thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    if not (type(effort) is int and 1 <= effort <= 100) and effort not in ("low", "high", "max"):
        raise ValueError("DeepSeek reasoning effort must be low/high/max or an integer from 1 to 100")
    return {"enable_thinking": thinking, "reasoning_effort": effort}


def official_messages(messages, tools):
    messages = copy.deepcopy(messages)
    for message in messages:
        message["role"] = "system" if message.get("role") == "developer" else message.get("role")
        if isinstance(message.get("content"), list):
            message["content_blocks"] = message.pop("content")
            for part in message["content_blocks"]:
                if part.get("type") == "image" and isinstance(part.get("source"), str):
                    part["url"] = part.pop("source")
    if tools:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        messages[0]["tools"] = [{"type": "function", "function": tool} for tool in tools]
    return messages


class DeepSeekTemplate:
    caps = {"tools": True, "tool_calls": True, "parallel_tool_calls": True, "thinking": True}
    source = "DeepSeek-V4.1 official encoding.py"

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        if not add_generation_prompt:
            raise ValueError("DeepSeek message-boundary pinning is not implemented; use native prefix files")
        return codec.encode_messages(official_messages(messages, tools),
            "thinking" if kwargs.get("enable_thinking", True) else "chat",
            reasoning_effort=kwargs.get("reasoning_effort", "high"))


class DeepSeekOutputParser:
    CALL = "<｜DSML｜ calls>"
    END_CALL = "</｜DSML｜ calls>"

    def __init__(self, thinking=True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""

    def feed(self, text):
        if self.state == "done":
            if text.strip():
                raise ValueError("unexpected text after DeepSeek EOS")
            return []
        self.buf += text
        output = []
        while self.buf:
            if self.state == "calls":
                end = self.buf.find(self.END_CALL)
                if end < 0:
                    if len(self.buf) > 8 << 20:
                        raise ValueError("DeepSeek tool block exceeds 8 MiB")
                    break
                block = self.buf[:end + len(self.END_CALL)]
                index, _, calls = codec.parse_tool_calls(len(self.CALL) - 1, block)
                if index != len(block):
                    raise ValueError("invalid DeepSeek tool block")
                for raw in calls:
                    name = raw["name"]
                    if raw.get("namespace"):
                        name = raw["namespace"] + "::" + name
                    args = json.loads(raw["arguments"])
                    if not isinstance(args, dict):
                        raise ValueError("DeepSeek tool arguments must be an object")
                    call = ToolCall(name=name, arguments=args)
                    output.extend([Event("tool_start", call=call), Event("tool_args", json.dumps(args, ensure_ascii=False), call),
                                   Event("tool_call", call=call)])
                self.buf = self.buf[len(block):]
                self.state = "content"
                continue
            stops = ["</think>"] if self.state == "reasoning" else [self.CALL, codec.eos_token]
            found = [(self.buf.find(stop), stop) for stop in stops if stop in self.buf]
            if found:
                index, stop = min(found)
                if index:
                    output.append(Event(self.state, self.buf[:index]))
                self.buf = self.buf[index + len(stop):]
                if stop == "</think>":
                    self.state = "content"
                elif stop == self.CALL:
                    self.buf = stop + self.buf
                    self.state = "calls"
                else:
                    self.state = "done"
                    if self.buf.strip():
                        raise ValueError("unexpected content after DeepSeek EOS")
                    self.buf = ""
                continue
            keep = max([0, *(k for stop in stops for k in range(1, len(stop)) if self.buf.endswith(stop[:k]))])
            take = len(self.buf) - keep
            if take:
                output.append(Event(self.state, self.buf[:take]))
                self.buf = self.buf[take:]
            break
        return output

    def finish(self, reason=None):
        if self.state == "calls":
            if reason == "length":
                self.buf = ""
                return []
            raise ValueError("incomplete DeepSeek tool block")
        result = [Event(self.state, self.buf)] if self.buf else []
        self.buf = ""
        return result


class DeepSeekService(Service):
    def __init__(self, engine, tokenizer, model_name="deepseek-v4.1", vision=False, **kwargs):
        super().__init__(engine, tokenizer, DeepSeekTemplate(), model_name, **kwargs)
        self.stop_ids = {tokenizer.eos_id}
        self.vision = engine if vision else None
        self.repeat_stop_tokens = 0

    def openai_messages(self, req):
        messages = []
        for raw in _object_list(req.get("messages"), "messages"):
            message = {"role": raw.get("role"), "content": _parts_of(raw.get("content"))}
            if message["role"] not in ("system", "developer", "user", "assistant", "tool"):
                raise ValueError("invalid message role")
            for field in ("reasoning_content", "tool_calls", "tool_call_id"):
                if raw.get(field) is not None:
                    message[field] = raw[field]
            if message["role"] in ("system", "developer"):
                message["content"] = _text_of(raw.get("content"))
            messages.append(message)
        tools = [t.get("function", t) if t.get("type") == "function" else t
                 for t in _tool_list(req.get("tools"), "function")] or None
        return messages, tools, effort_kwargs(req, self.shared)

    def response_kwargs(self, req):
        try:
            self.force_call(req.get("tool_choice"), None)
        except ValueError as error:
            raise ResponsesError(str(error), "tool_choice", "unsupported_parameter") from None
        return effort_kwargs(req, self.shared)

    def force_call(self, choice, tools):
        kind, name = tool_choice_of(choice)
        if kind in ("none", "auto"):
            return None
        # A forced opening cannot be inserted mid-reasoning without a grammar
        # continuation transaction. Refuse explicitly until that path is verified.
        raise ValueError("DeepSeek forced tool_choice is not implemented; use auto or none")

    def encode_prompt(self, messages, tools, kwargs):
        return self.tok.encode(self.template.render(messages, tools, **kwargs), parse_special=True)

    def prepare(self, messages, tools, kwargs, max_new=None, force=None, req=None):
        self.drop_embeddings()
        kwargs = {**effort_kwargs(req, self.shared), **kwargs} if req is not None else kwargs
        prompt, media = codec.encode_messages(official_messages(messages, tools),
            "thinking" if kwargs.get("enable_thinking", True) else "chat",
            reasoning_effort=kwargs.get("reasoning_effort", "high"), return_multi_modal_data=True)
        patches = []
        for record in media["images"]:
            if not self.vision:
                raise ValueError("DeepSeek mmproj is not configured")
            source = record.get("url") or record.get("data") or (record.get("source") or {}).get("data")
            if isinstance(source, bytes):
                data = source
            elif record.get("data") is not None:
                import base64
                data = base64.b64decode(record["data"], validate=True)
            else:
                data = Vision.load(source)
            patches.append(preprocess(data))
        ids, spans = expand_images(self.tok.encode(prompt, parse_special=True), patches)
        ctx = self.reported_ctx()
        if ctx < 1:
            raise EngineStarting("DeepSeek is not ready")
        room = ctx - len(ids)
        if room < 1:
            raise ValueError("DeepSeek prompt leaves no room in the configured context")
        maximum = room if max_new is None or max_new <= 0 else max_new
        if maximum > room:
            if not self.fit_max_tokens:
                raise ValueError("DeepSeek prompt plus max tokens exceeds the configured context")
            maximum = room
        if req and req.get("strata_prefix") is not None:
            raise ValueError("DeepSeek message-boundary prefix pinning is not implemented")
        # Reject invalid native parameters while the HTTP handler can still return a client error.
        from serve.deepseek41_engine import sampling_params
        sampling_params({**self.sampling_defaults, **self.shared,
                         **{k: v for k, v in (req or {}).items() if v is not None}}, len(ids))
        self.embeddings.path = spans or None
        return ids, kwargs.get("enable_thinking", True), maximum

    def drop_embeddings(self):
        self.embeddings.path = None

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None):
        sampling = {**self.sampling_defaults, **self.shared, **{k: v for k, v in (sampling or {}).items() if v is not None}}
        parser, detok = DeepSeekOutputParser(thinking), Detokenizer(self.tok)
        stops = StopMatcher(stop_strings(sampling))
        images, self.embeddings.path = getattr(self.embeddings, "path", None), None
        state = {"busy": True, "phase": "reading the prompt", "prompt_tokens": len(ids), "generated": 0,
                 "max_tokens": max_new, "started": time.time(), "first_token": None}
        waiting = time.perf_counter()
        trace = getattr(self.request_trace, "record", None)
        n, reasoning_n, finish = 0, 0, "length"
        with self.status_lock:
            self.status["queued"] += 1
        try:
            with (contextlib.nullcontext() if getattr(self.engine, "batch", 0) else self.fifo):
                with self.status_lock:
                    self.status["queued"] -= 1
                    self.live_reqs[id(state)] = (state, self.rate.copy())
                    self.status.update(state)
                if trace is not None:
                    trace["queue_s"] += time.perf_counter() - waiting
                    trace["state"] = "generating"
                gen = self.engine.generate(ids, max_new, sampling, cancel, embeddings=images)
                try:
                    for token in gen:
                        if token is None:
                            yield "ping", None
                            continue
                        n += 1
                        if token in self.stop_ids:
                            finish = "stop"
                            break
                        reasoning_n += parser.state == "reasoning"
                        for event in parser.feed(detok.push(token)):
                            if event.kind == "content":
                                event.text = stops.push(event.text)
                                if not event.text:
                                    continue
                            self._note(n, [event], state)
                            yield "event", event
                        if stops.hit is not None:
                            finish = "stop"
                            break
                finally:
                    gen.close()
                if cancel.is_set():
                    finish = "cancelled"
                for event in parser.finish(finish):
                    yield "event", event
                held = stops.flush()
                if held:
                    yield "event", Event("content", held)
                last = getattr(self.engine, "last_for_thread", lambda: getattr(self.engine, "last", None))() or {}
                with self.status_lock:
                    self.totals["requests"] += 1
                    self.totals["prompt_tokens"] += len(ids)
                    self.totals["output_tokens"] += n
                    self.totals["prompt_ms"] += last.get("prompt_ms", 0)
                    self.totals["decode_ms"] += last.get("decode_ms", 0)
                    self.last_request_at = time.time()
                    self.latencies.observe((state["first_token"] - state["started"]) if state["first_token"] else None,
                                           n, last.get("decode_ms", 0) / 1000, time.time() - state["started"])
                yield "done", {"finish": finish, "completion_tokens": n, "reasoning_tokens": reasoning_n}
        finally:
            with self.status_lock:
                self.live_reqs.pop(id(state), None)
                self.status["busy"] = bool(self.live_reqs)
                self.last_request_at = time.time()
