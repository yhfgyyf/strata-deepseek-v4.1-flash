import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve import deepseek41_encoding as codec
from serve import deepseek41 as launcher
from serve.deepseek41_engine import DeepSeekEngine, TIMING_FIELDS, sampling_params
from serve.deepseek41_frontend import DeepSeekOutputParser, DeepSeekService, DeepSeekTemplate, effort_kwargs
from serve.server import ByteTokenizer, EngineDied, serve


class Tokens(ByteTokenizer):
    SPECIALS = [codec.bos_token, codec.eos_token, "<｜User｜>", "<｜Assistant｜>", "<｜System｜>", "<think>", "</think>"]
    eos_id = 257


class Launcher(unittest.TestCase):
    def command(self, **options):
        cfg = {"engine": "native", "model": "main", "integrity": "proof", **options}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json";path.write_text(json.dumps(cfg))
            with mock.patch.object(sys, "argv", ["deepseek41", "--config", str(path)]), \
                    mock.patch.object(launcher, "DeepSeekEngine", side_effect=RuntimeError("captured")) as engine:
                with self.assertRaisesRegex(RuntimeError, "captured"):
                    launcher.main()
        return engine.call_args.args[0]

    def test_prefill_auto_and_numeric_chunks(self):
        for options, expected in (({}, "1024"), ({"chunk": 8192}, "8192"),
                                  ({"chunk": "auto"}, "auto"), ({"prefill": "auto"}, "auto"),
                                  ({"chunk": "auto:4096"}, "auto:4096"),
                                  ({"prefill": 32768}, "32768"),
                                  ({"chunk": "auto", "prefill": "auto:32768"}, "auto")):
            with self.subTest(options=options):
                command = self.command(**options)
                self.assertEqual(command[command.index("--chunk") + 1], expected)

    def test_invalid_prefill_chunks(self):
        for value in (0, -1, 32769, True, 1.5, None, "", "Auto", "auto:", "auto:257",
                      "auto:32769", "auto:128", "8192oops", " 8192", "1.5", []):
            for key in ("chunk", "prefill"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.command(**{key: value})
        with self.assertRaisesRegex(ValueError, "different values"):
            self.command(chunk=8192, prefill="auto")

    def test_hot_auto_and_capacity_flags(self):
        command = self.command(profile="decode.strp", hot_auto=True, work_mib=3072, state_mib=256,
                               gpu_headroom_mib=1024, spec=True)
        self.assertIn("--hot-auto", command);self.assertIn("--spec", command)
        for flag, value in (("--profile", "decode.strp"), ("--work-mib", "3072"),
                            ("--state-mib", "256"), ("--gpu-headroom-mib", "1024")):
            self.assertEqual(command[command.index(flag) + 1], value)
        default = self.command()
        self.assertTrue(all(flag not in default for flag in (
            "--hot-auto", "--work-mib", "--state-mib", "--gpu-headroom-mib")))

    def test_invalid_new_capacity_configuration(self):
        for options in ({"hot_auto": True}, {"hot_auto": "false"}, {"work_mib": 0},
                        {"state_mib": -1}, {"work_mib": True}, {"state_mib": 1.5},
                        {"gpu_headroom_mib": -1}, {"gpu_headroom_mib": 1 << 44}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.command(**options)

    def test_adaptive_experts_command(self):
        command = self.command(adaptive_experts=True, migration_mib=2048)
        self.assertIn("--adaptive-experts", command)
        self.assertEqual(command[command.index("--migration-mib") + 1], "2048")
        self.assertNotIn("--adaptive-experts", self.command())
        phase = self.command(phase_residency=True, migration_mib=0)
        self.assertEqual(phase[phase.index("--migration-mib") + 1], "0")

    def test_invalid_adaptive_experts_configuration(self):
        for options in ({"adaptive_experts": "false"}, {"adaptive_experts": True, "slots": 2},
                        {"adaptive_experts": True, "spec": True}, {"adaptive_experts": True, "phase_residency": True},
                        {"adaptive_experts": True, "migration_mib": 0}, {"adaptive_experts": True, "migration_mib": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.command(**options)


class ScriptEngine:
    max_context = 32768

    def __init__(self, tokenizer, script):
        self.ids = tokenizer.encode(script, parse_special=True) + [tokenizer.eos_id]

    def generate(self, ids, maximum, sampling, cancel, embeddings=None):
        for token in self.ids[:maximum]:
            if cancel.is_set():
                return
            yield token


PIPE = '''
import sys,threading,time,shlex,pathlib
lock=threading.Lock(); jobs={}
def say(s):
 with lock: print(s,flush=True)
say("READY 4096 2")
def run(rid,maximum,images):
 event=jobs[rid]
 for i in range(maximum):
  if event.wait(.005):
   # Request image storage must outlive cancellation and its acknowledgement.
   if not all(pathlib.Path(x).exists() for x in images):say("E 0 image storage released before cancellation")
   else:say(f"E {rid} cancelled")
   return
  say(f"T {rid} {100+rid}")
 say(f"D {rid} " + " ".join(["0"]*27))
for line in sys.stdin:
 a=shlex.split(line)
 if not a:continue
 if a[0]=="QUIT":
  for e in jobs.values():e.set()
  break
 if a[0]=="CANCEL":
  jobs[int(a[1])].set();continue
 rid=int(a[1]);n=int(a[2]);maximum=int(a[3]);nimages=int(a[15])
 assert len(a)==19 and 1<=int(a[18])<=n
 ids=sys.stdin.readline().split();assert len(ids)==n
 images=[shlex.split(sys.stdin.readline())[3] for _ in range(nimages)]
 jobs[rid]=threading.Event()
 threading.Thread(target=run,args=(rid,maximum,images)).start()
'''


class Frontend(unittest.TestCase):
    def test_standard_chat_tools_are_flattened_for_official_encoder(self):
        tok = Tokens()
        svc = DeepSeekService(ScriptEngine(tok, "OK"), tok)
        function = {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
        for tool in ({"type": "function", "function": function}, function):
            with self.subTest(tool=tool):
                req = {"messages": [{"role": "user", "content": "Look it up."}],
                       "tools": [tool], "reasoning_effort": "none"}
                messages, tools, kwargs = svc.openai_messages(req)
                self.assertEqual(tools, [function])
                ids, thinking, maximum = svc.prepare(messages, tools, kwargs, 100, req=req)
                self.assertTrue(ids)
                self.assertFalse(thinking)
                self.assertEqual(maximum, 100)

    def test_anthropic_disabled_thinking_survives_prepare(self):
        from serve.frontend import anthropic_to_messages
        tok = Tokens()
        svc = DeepSeekService(ScriptEngine(tok, "OK"), tok)
        for stream in (False, True):
            req = {"messages": [{"role": "user", "content": "Hello"}],
                   "thinking": {"type": "disabled"}, "stream": stream, "max_tokens": 100}
            messages, tools, kwargs = anthropic_to_messages(req)
            ids, thinking, maximum = svc.prepare(messages, tools, kwargs, 100, req=req)
            self.assertFalse(thinking)
            self.assertTrue(tok.decode(ids).endswith("<｜Assistant｜></think>"))
            self.assertEqual(maximum, 100)

    def test_responses_forced_choice_is_rejected_before_generation(self):
        tok = Tokens()
        engine = ScriptEngine(tok, "OK")
        svc = DeepSeekService(engine, tok)
        server = serve(svc, "127.0.0.1", 0)
        try:
            with mock.patch.object(engine, "generate", wraps=engine.generate) as generate:
                for stream in (False, True):
                    for choice in ("required", {"type": "function", "name": "lookup"},
                                   {"type": "unknown"}):
                        body = {"input": "Hello", "tools": [{"type": "function", "name": "lookup",
                                "parameters": {"type": "object"}}], "tool_choice": choice,
                                "stream": stream, "max_output_tokens": 100, "reasoning": {"effort": "none"}}
                        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/responses",
                            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                        with self.subTest(stream=stream, choice=choice):
                            with self.assertRaises(urllib.error.HTTPError) as error:
                                urllib.request.urlopen(request, timeout=10)
                            self.assertEqual(error.exception.code, 400)
                            result = json.load(error.exception)
                            self.assertEqual(result["error"]["param"], "tool_choice")
                            self.assertEqual(result["error"]["code"], "unsupported_parameter")
                            error.exception.close()
                self.assertEqual(generate.call_count, 0)
            for choice in (None, "auto", "none"):
                self.assertFalse(svc.response_kwargs({"tool_choice": choice,
                                                      "reasoning": {"effort": "none"}})["enable_thinking"])
        finally:
            server.shutdown(); server.server_close()

    def test_effort_and_mid_system(self):
        self.assertEqual(effort_kwargs({"reasoning_effort": 63})["reasoning_effort"], 63)
        self.assertFalse(effort_kwargs({"reasoning_effort": "none"})["enable_thinking"])
        for bad in (0, 101, True, "xhigh", "medium"):
            with self.assertRaises(ValueError):
                effort_kwargs({"reasoning_effort": bad})
        template = DeepSeekTemplate()
        prompt = template.render([{"role": "user", "content": "A"}, {"role": "system", "content": "B"}],
                                 reasoning_effort=63)
        self.assertIn("Reasoning Effort: 63", prompt)
        self.assertIn("<｜System｜>B", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))

    def test_split_reasoning_and_dsml_against_official_parser(self):
        text = 'Checking.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="read">\n<｜DSML｜ parameter name="path" string="true">a\\b.txt</｜DSML｜ parameter>\n<｜DSML｜ parameter name="limit" string="false">3</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>'
        expected = codec.parse_message_from_completion_text(text + codec.eos_token, "thinking")
        parser = DeepSeekOutputParser(True)
        events = []
        for char in text:
            events.extend(parser.feed(char))
        events.extend(parser.finish("stop"))
        self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), expected["reasoning_content"])
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "read")
        self.assertEqual(calls[0].arguments, json.loads(expected["tool_calls"][0]["function"]["arguments"]))

    def test_http_chat_and_responses(self):
        tok = Tokens()
        svc = DeepSeekService(ScriptEngine(tok, "Checking.</think>Answer."), tok)
        server = serve(svc, "127.0.0.1", 0)
        try:
            for path, body in [("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}],
                                                          "reasoning_effort": 63, "max_tokens": 100}),
                               ("/v1/responses", {"input": "Hello", "reasoning": {"effort": "high"},
                                                  "max_output_tokens": 100})]:
                request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=10) as response:
                    result = json.load(response)
                if path.endswith("completions"):
                    message = result["choices"][0]["message"]
                    self.assertEqual(message["content"], "Answer.")
                    self.assertEqual(message["reasoning_content"], "Checking.")
                else:
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["output"][-1]["content"][0]["text"], "Answer.")
        finally:
            server.shutdown();server.server_close()

    def test_http_sampling_error_leaves_native_pipe_usable(self):
        engine = DeepSeekEngine([sys.executable, "-u", "-c", PIPE], context=4096, slots=2, ready_timeout=5)
        svc = DeepSeekService(engine, Tokens())
        server = serve(svc, "127.0.0.1", 0)
        try:
            with mock.patch.object(engine, "_send", wraps=engine._send) as send:
                for path, request_fields in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}),
                                             ("/v1/responses", {"input": "Hi", "max_output_tokens": 1})):
                    for invalid in ("0\nQUIT", float("nan"), True):
                        body = {**request_fields, "temperature": invalid}
                        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}",
                            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                        with self.subTest(path=path, invalid=invalid), self.assertRaises(urllib.error.HTTPError) as error:
                            urllib.request.urlopen(request, timeout=10)
                        self.assertEqual(error.exception.code, 400)
                        error.exception.close()
                self.assertEqual(send.call_count, 0)
            body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1,
                    "temperature": 0, "reasoning_effort": "none"}
            request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "e")
            self.assertTrue(engine.alive())
        finally:
            server.shutdown();server.server_close();engine.close()


class Pipe(unittest.TestCase):
    def test_http_slots_uses_native_slot_contract(self):
        engine = DeepSeekEngine([sys.executable, "-u", "-c", PIPE], context=4096, slots=2, ready_timeout=5)
        service = DeepSeekService(engine, Tokens())
        server = serve(service, "127.0.0.1", 0)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/slots", timeout=5) as response:
                slots = json.load(response)
            self.assertEqual([slot["id"] for slot in slots], [0, 1])
            self.assertTrue(all(not slot["is_processing"] for slot in slots))
            self.assertEqual([slot["n_ctx"] for slot in slots], [4096, 4096])
        finally:
            server.shutdown(); server.server_close(); engine.close()

    def test_failed_spawn_cleans_private_files(self):
        opened = []
        def track_open(*args, **kwargs):
            file = open(*args, **kwargs)
            opened.append(file)
            return file
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("serve.deepseek41_engine.open", side_effect=track_open, create=True):
                with self.assertRaises(FileNotFoundError):
                    DeepSeekEngine([str(Path(directory) / "missing-engine")], directory=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(len(opened), 1)
            self.assertTrue(opened[0].closed)

    def test_single_slot_default(self):
        pipe = PIPE.replace("READY 4096 2", "READY 4096 1")
        engine = DeepSeekEngine([sys.executable, "-u", "-c", pipe], context=4096, ready_timeout=5)
        try:
            self.assertEqual(engine.batch, 1)
            self.assertEqual(list(engine.generate([1], 1, {}, threading.Event())), [101])
        finally:
            engine.close()
        self.assertTrue(engine.proc.stdin.closed)
        self.assertTrue(engine.proc.stdout.closed)

    def test_invalid_sampling_never_writes_gen_and_next_request_succeeds(self):
        engine = DeepSeekEngine([sys.executable, "-u", "-c", PIPE], context=4096, slots=2, ready_timeout=5)
        cases = [{"temperature": x} for x in (-1, float("nan"), float("inf"), 1e100, True, "0\nQUIT")]
        cases += [{"top_p": x} for x in (0, 1.5, float("nan"), "0.9", False)]
        cases += [{"min_p": x} for x in (-1, 2, float("inf"), None)]
        cases += [{"repetition_penalty": x} for x in (0, -1, 1e-100, float("nan"), "1")]
        cases += [{key: value} for key in ("frequency_penalty", "presence_penalty", "repeat_penalty")
                  for value in (float("inf"), True, "0")]
        cases += [{key: value} for key in ("top_k", "penalty_last_n", "repeat_last_n")
                  for value in (1.5, True, 2**31, -(2**31)-1, "1")]
        cases += [{"seed": x} for x in (-1, 2**64, True, 1.5, "0")]
        cases += [[], {"gumbel": True}, {"grammar": "x"}, {"logit_bias": {"1": 1}}]
        try:
            with mock.patch.object(engine, "_send", wraps=engine._send) as send:
                for sampling in cases:
                    with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                        list(engine.generate([1], 1, sampling, threading.Event()))
                for maximum in (True, 1.5, "1"):
                    with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                        list(engine.generate([1], maximum, {}, threading.Event()))
                self.assertEqual(send.call_count, 0)
                self.assertEqual(engine.waiting, 0)
                self.assertTrue(all(s["state"] == "idle" for s in engine.slots_view()))
            self.assertEqual(list(engine.generate([1], 1, {}, threading.Event())), [101])
            self.assertTrue(engine.alive())
        finally:
            engine.close()

    def test_native_ranges_and_penalty_keys(self):
        values = sampling_params({"temperature": 1e-45, "top_k": -1, "top_p": 1, "min_p": 1,
                                  "repetition_penalty": 2, "repeat_penalty": 3,
                                  "frequency_penalty": -2, "presence_penalty": -3,
                                  "penalty_last_n": -1, "seed": 2**64 - 1}, 3)
        self.assertGreater(values["temperature"], 0)
        self.assertEqual(values["top_k"], -1)
        self.assertEqual(values["repetition_penalty"], 2)
        self.assertEqual(values["frequency_penalty"], -2)
        self.assertEqual(values["presence_penalty"], -3)
        self.assertEqual(values["penalty_last_n"], -1)
        self.assertEqual(values["seed"], 2**64 - 1)
        self.assertEqual(sampling_params({"top_k": 65, "repeat_penalty": 2, "repeat_last_n": 0}, 3)["top_k"], 65)
        self.assertEqual(sampling_params({"repeat_penalty": 2, "repeat_last_n": 0}, 3)["repetition_penalty"], 2)

    def test_history_header_and_phase_timing_order(self):
        values = [str(i) if cast is int else str(i + 0.5) for i, (_, cast) in enumerate(TIMING_FIELDS)]
        values[7] = str(2**63 + 1)  # Keep byte counters exact beyond FP64's integer precision.
        done = ' say(f"D {rid} " + " ".join(["0"]*27))'
        pipe = PIPE.replace(done, f' say(f"D {{rid}} " + {" ".join(values)!r})')
        engine = DeepSeekEngine([sys.executable, "-u", "-c", pipe], context=4096, slots=2, ready_timeout=5)
        try:
            with mock.patch.object(engine, "_send", wraps=engine._send) as send:
                result = list(engine.generate([1, 2, 3], 1, {"repetition_penalty": 2, "penalty_last_n": 4},
                    threading.Event(), original_prompt_length=2, prefix_load='a "quoted" path', prefix_save="b path"))
                import shlex
                header = shlex.split(send.call_args_list[0].args[0].splitlines()[0])
                self.assertEqual(header[11], "2.0")
                self.assertEqual(header[14], "4")
                self.assertEqual(header[15:], ["0", 'a "quoted" path', "b path", "2"])
            self.assertEqual(result, [101])
            self.assertEqual(engine.last["prompt_tokens"], 3)
            self.assertEqual(engine.last["original_prompt_tokens"], 2)
            for (name, cast), value in zip(TIMING_FIELDS, values):
                self.assertEqual(engine.last[name], cast(value), name)
                self.assertIs(type(engine.last[name]), cast, name)
            self.assertEqual(engine.last_for_thread(), engine.last)
        finally:
            engine.close()

    def test_invalid_history_or_prefix_is_rejected_before_gen(self):
        engine = DeepSeekEngine([sys.executable, "-u", "-c", PIPE], context=4096, slots=2, ready_timeout=5)
        try:
            with mock.patch.object(engine, "_send", wraps=engine._send) as send:
                for original in (0, 3, True, 1.5, "1"):
                    with self.subTest(original=original), self.assertRaises(ValueError):
                        list(engine.generate([1, 2], 1, {}, threading.Event(), original_prompt_length=original))
                for key in ("prefix_load", "prefix_save"):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        list(engine.generate([1], 1, {}, threading.Event(), **{key: "x\nQUIT"}))
                image = mock.Mock(tokens=2, height=3, width=3)
                with self.assertRaisesRegex(ValueError, "original prompt"):
                    list(engine.generate([1, 2, 3], 1, {}, threading.Event(),
                                         embeddings=[(1, image)], original_prompt_length=2))
                self.assertEqual(send.call_count, 0)
        finally:
            engine.close()

    def test_malformed_timing_record_is_rejected(self):
        done = ' say(f"D {rid} " + " ".join(["0"]*27))'
        for index, value in ((None, None), (0, "nan"), (2, "1.5"), (7, str(2**64)), (23, "-1")):
            fields = ["0"] * 27
            if index is None:
                fields = fields[:14]
            else:
                fields[index] = value
            pipe = PIPE.replace(done, f' say(f"D {{rid}} " + {" ".join(fields)!r})')
            with self.subTest(index=index, value=value):
                engine = DeepSeekEngine([sys.executable, "-u", "-c", pipe], context=4096, slots=2, ready_timeout=5)
                try:
                    with self.assertRaisesRegex(EngineDied, "timing record"):
                        list(engine.generate([1], 1, {}, threading.Event()))
                finally:
                    engine.close()

    def test_interleaved_requests_cancel_and_slot_reuse(self):
        engine = DeepSeekEngine([sys.executable, "-u", "-c", PIPE], context=4096, slots=2, ready_timeout=5)
        outputs, errors = {}, []
        try:
            def request(key):
                try:
                    outputs[key] = [t for t in engine.generate([1, 2], 8, {"temperature": 0}, threading.Event()) if t is not None]
                except Exception as error:
                    errors.append(error)
            a, b = threading.Thread(target=request, args=("a",)), threading.Thread(target=request, args=("b",))
            a.start();b.start();a.join(10);b.join(10)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertFalse(errors)
            self.assertEqual(sorted(outputs.values()), [[101] * 8, [102] * 8])
            cancel = threading.Event()
            image = mock.Mock(tokens=4, height=3, width=3)
            image.values.tofile.side_effect = lambda path: path.write_bytes(b"\0" * (3 * 3 * 588 * 4))
            gen = engine.generate([1, 2, 3, 4], 100, {}, cancel, embeddings=[(0, image)])
            self.assertEqual(next(gen), 103)
            cancel.set()
            self.assertEqual(list(gen), [])
            result = [t for t in engine.generate([2], 1, {}, threading.Event()) if t is not None]
            self.assertEqual(result, [104])
            self.assertTrue(engine.alive())
            self.assertTrue(all(s["state"] == "idle" for s in engine.slots_view()))
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
