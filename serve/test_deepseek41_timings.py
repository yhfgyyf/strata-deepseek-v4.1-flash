"""Regression coverage for EOS-drained timings, new wall fields and request isolation."""
import json
import sys
import threading
import unittest
import urllib.request

from serve.deepseek41_engine import DeepSeekEngine
from serve.deepseek41_frontend import DeepSeekService
from serve.server import EngineDied, serve
from serve.test_deepseek41 import Tokens


PIPE = '''
import sys, shlex
print("READY 4096 1", flush=True)
for line in sys.stdin:
    a = shlex.split(line)
    if a[0] == "QUIT": break
    if a[0] == "CANCEL": continue
    rid = int(a[1]); ids = sys.stdin.readline().split()
    values = [str(10*rid), str(20*rid)] + ["0"]*25
    if MODE != "legacy": values += [str(100*rid), str(200*rid), str(50*rid)]
    if MODE == "invalid": values[-1] = "nan"
    if MODE == "partial": values.pop()
    print(f"T {rid} {EOS}", flush=True)
    print(f"D {rid} " + " ".join(values), flush=True)
'''


def engine(mode, eos=1):
    script = f"MODE={mode!r}\nEOS={eos}\n" + PIPE
    return DeepSeekEngine([sys.executable, "-u", "-c", script], context=4096, slots=1)


class Timings(unittest.TestCase):
    def test_eos_close_preserves_legacy_stats_per_request(self):
        eng = engine("legacy")
        try:
            for rid in (1, 2):
                gen = eng.generate([100, 101], 4, {}, threading.Event())
                self.assertEqual(next(gen), 1)
                gen.close()
                last = eng.last_for_thread()
                self.assertIsNotNone(last)
                self.assertEqual(last["prompt_ms"], 10*rid)
                self.assertEqual(last["decode_ms"], 20*rid)
                self.assertEqual(last["prompt_tokens"], 2)
                self.assertEqual(eng.slots_view()[0]["state"], "idle")
        finally:
            eng.close()

    def test_new_wall_fields_normal_completion(self):
        eng = engine("wall")
        try:
            self.assertEqual(list(eng.generate([100], 4, {}, threading.Event())), [1])
            last = eng.last_for_thread()
            self.assertEqual(last["prefill_wall_ms"], 100)
            self.assertEqual(last["generation_wall_ms"], 200)
            self.assertEqual(last["cleanup_wall_ms"], 50)
            self.assertEqual(last["decode_ms"], 20)
        finally:
            eng.close()

    def test_malformed_wall_fields_rejected_on_eos_close(self):
        for mode in ("invalid", "partial"):
            with self.subTest(mode=mode):
                eng = engine(mode)
                try:
                    gen = eng.generate([100], 4, {}, threading.Event())
                    self.assertEqual(next(gen), 1)
                    with self.assertRaises(EngineDied):
                        gen.close()
                    self.assertIsNone(eng.last_for_thread())
                    self.assertEqual(eng.slots_view()[0]["state"], "idle")
                finally:
                    eng.close()

    def test_http_eos_returns_full_wall_timings_stream_and_nonstream(self):
        tok = Tokens(); eng = engine("wall", tok.eos_id)
        svc = DeepSeekService(eng, tok)
        server = serve(svc, "127.0.0.1", 0)
        try:
            for rid, stream in enumerate((False, True), 1):
                body = {"messages": [{"role": "user", "content": "Hi"}],
                        "reasoning_effort": "none", "max_tokens": 4, "stream": stream,
                        "stream_options": {"include_usage": True}}
                req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    if stream:
                        items = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                                 if line.startswith("data: ") and line[6:] != "[DONE]"]
                        result = next(item for item in items if item.get("timings"))
                    else:
                        result = json.load(response)
                t = result["timings"]
                self.assertEqual(t["prompt_ms"], 100*rid)
                self.assertEqual(t["predicted_ms"], 250*rid)
                self.assertEqual(t["predicted_n"], 1)
                self.assertEqual(t["generation_ms"], 200*rid)
                self.assertEqual(t["cleanup_ms"], 50*rid)
                self.assertEqual(t["target_forward_ms"], 20*rid)
            self.assertEqual(svc.totals["prompt_ms"], 300)
            self.assertEqual(svc.totals["decode_ms"], 750)
        finally:
            server.shutdown(); server.server_close(); eng.close()


if __name__ == "__main__":
    unittest.main()
