"""Observable cache/IPC ownership checks; real model state is tested separately."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request

from serve.deepseek41_engine import DeepSeekEngine, PrefixCache, sampling_params
from serve.deepseek41_frontend import DeepSeekService
from serve.server import ByteTokenizer, EngineDied, serve


class Tokens(ByteTokenizer):
    SPECIALS = ['<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>', '<｜User｜>',
                '<｜Assistant｜>', '<｜System｜>', '<think>', '</think>']
    eos_id = 257


PIPE = r'''
import json,pathlib,shlex,sys,threading,time
lock=threading.Lock();jobs={}
def say(text):
 with lock: print(text,flush=True)
def run(a,ids,images):
 rid,n,maximum=int(a[1]),int(a[2]),int(a[3]);event=jobs[rid]
 load,save=a[16:18];original=int(a[18]);reused=0
 load_n,save_n=map(int,a[19:21]) if len(a)==21 else (original if load else 0,original if save else 0)
 if load:
  try: restored=json.loads(pathlib.Path(load).read_text())
  except Exception: say(f'E {rid} checkpoint rejected');return
  if restored!=ids[:load_n]:say(f'E {rid} token mismatch');return
  reused=len(restored)
 if save:pathlib.Path(save).write_text(json.dumps(ids[:save_n]))
 if ids[-1]!=9001:say(f'R {rid} {reused+1 if ids[-1]==9002 else reused}')
 values=['0']*30;values[10]=str(n-reused);values[27:]=['10','2','1']
 if maximum==1000:
  say(f'T {rid} 65')
  event.wait(5)
  if not all(pathlib.Path(x).is_file() for x in images):say('E 0 image lifetime');return
  say(f'E {rid} cancelled');return
 for token in [65,257][:maximum]:say(f'T {rid} {token}')
 say(f'D {rid} '+' '.join(values))
print('READY 4096 1',flush=True)
for line in sys.stdin:
 a=shlex.split(line)
 if not a:continue
 if a[0]=='QUIT':
  for e in jobs.values():e.set()
  break
 if a[0]=='CANCEL':jobs[int(a[1])].set();continue
 assert a[0]=='GEN' and len(a) in (19,21)
 ids=list(map(int,sys.stdin.readline().split()));assert len(ids)==int(a[2])
 images=[shlex.split(sys.stdin.readline())[3] for _ in range(int(a[15]))]
 jobs[int(a[1])]=threading.Event()
 threading.Thread(target=run,args=(a,ids,images)).start()
'''


class Checkpoints(unittest.TestCase):
    def test_longest_prefix_root_retention_and_lru(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = PrefixCache(Path(directory) / 'cache', 1024)
            paths = []
            for i in range(1, 7):
                path = cache.directory / str(i)
                path.write_bytes(b'x' * 10)
                cache.publish(path, list(range(i)))
                paths.append(path)
            self.assertEqual(len(cache.entries), 4)
            self.assertTrue(paths[0].is_file(), 'retain the root of an extended conversation')
            self.assertFalse(paths[1].exists())
            self.assertEqual(cache.match(list(range(7)))[0], paths[-1])
            self.assertEqual(cache.match([0, 99])[0], paths[0])
            self.assertIsNone(cache.match([99]))

    def test_byte_budget_overrides_root_and_oversize_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = PrefixCache(Path(directory) / 'cache', 12)
            for name, ids, size in [('root', [1], 8), ('next', [1, 2], 8), ('large', [7], 13)]:
                path = cache.directory / name
                path.write_bytes(b'x' * size)
                cache.publish(path, ids)
            self.assertEqual([x[0].name for x in cache.entries], ['next'])
            self.assertEqual([p.name for p in cache.directory.iterdir()], ['next'])

    def test_missing_and_resized_files_become_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = PrefixCache(Path(directory) / 'cache', 1024)
            for mode in ('missing', 'resized'):
                path = cache.directory / mode
                path.write_bytes(b'abc'); cache.publish(path, [1, 2])
                if mode == 'missing': path.unlink()
                else: path.write_bytes(b'x')
                self.assertIsNone(cache.match([1, 2, 3]))
                self.assertFalse(cache.entries)


class AutomaticCache(unittest.TestCase):
    def setUp(self):
        self.engine = DeepSeekEngine([sys.executable, '-u', '-c', PIPE], context=4096,
                                     ready_timeout=5, prefix_cache_mib=1)
        self.addCleanup(self.engine.close)

    def request(self, ids, **sampling):
        result = list(self.engine.generate(ids, 2, sampling, threading.Event()))
        self.assertEqual([x for x in result if x is not None], [65, 257])
        last = self.engine.last_for_thread()
        self.assertEqual(last['kv_only_tokens'], len(ids) - last['reused'])
        return last['reused']

    def test_extension_repeat_interleaved_conversations_and_bypass(self):
        self.assertEqual(self.request([1, 2, 3]), 0)
        self.assertEqual(self.request([9, 8, 7]), 0)
        self.assertEqual(self.request([1, 2, 3, 4, 5]), 2)
        self.assertEqual(self.request([9, 8, 7, 6]), 2)
        self.assertEqual(self.request([1, 2, 3, 4, 5]), 4)
        entries = list(self.engine.prefix_cache.entries)
        self.assertEqual(self.request([1, 2, 3, 4, 5], cache_prompt=False), 0)
        self.assertEqual(self.engine.prefix_cache.entries, entries)
        self.assertEqual(self.request([1, 88, 3]), 0)

    def test_cancel_and_generator_close_preserve_previous_cache(self):
        self.request([1, 2, 3])
        for mode in ('event', 'close'):
            before = list(self.engine.prefix_cache.entries)
            cancel = threading.Event()
            gen = self.engine.generate([1, 2, 3, 4], 1000, {}, cancel)
            for token in gen:
                if token is not None: break
            if mode == 'event':
                cancel.set(); list(gen)
            else:
                gen.close()
            self.assertEqual(self.engine.prefix_cache.entries, before)
            self.assertEqual(set(self.engine.prefix_cache.directory.iterdir()), {e[0] for e in before})
            self.assertEqual(self.request([1, 2, 3]), 2)
            self.assertTrue(self.engine.alive())

    def test_bad_checkpoint_invalidated_then_next_request_cold(self):
        self.request([1, 2, 3])
        path, _, size = self.engine.prefix_cache.entries[0]
        path.write_bytes(b'x' * size)
        with self.assertRaisesRegex(RuntimeError, 'checkpoint rejected'):
            list(self.engine.generate([1, 2, 3, 4], 2, {}, threading.Event()))
        self.assertFalse(path.exists())
        self.assertEqual(self.request([1, 2, 3, 4]), 0)

    def test_native_reuse_ack_required_and_checked(self):
        for last in (9001, 9002):
            with self.assertRaises(EngineDied):
                list(self.engine.generate([1, 2, last], 2, {}, threading.Event()))
            self.assertFalse(self.engine.prefix_cache.entries)
            self.assertFalse(list(self.engine.prefix_cache.directory.iterdir()))

    def test_images_bypass_cache(self):
        class Values:
            def tofile(self, path):
                Path(path).write_bytes(b'\0' * (588 * 4))
        class Image:
            height = width = tokens = 1
            values = Values()
        self.request([1, 2, 3])
        before = list(self.engine.prefix_cache.entries)
        list(self.engine.generate([1, 2, 3], 2, {}, threading.Event(), embeddings=[(0, Image())]))
        self.assertEqual(self.engine.last_for_thread()['reused'], 0)
        self.assertEqual(self.engine.prefix_cache.entries, before)

    def test_single_token_and_cleanup(self):
        self.assertEqual(self.request([1]), 0)
        self.assertFalse(self.engine.prefix_cache.entries)
        self.request([1, 2, 3])
        directory = self.engine.dir
        self.engine.close()
        self.assertFalse(directory.exists())

    def test_http_usage_timing_and_metrics_report_native_hit(self):
        service = DeepSeekService(self.engine, Tokens())
        server = serve(service, '127.0.0.1', 0)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 2,
                'temperature': 0, 'reasoning_effort': 'none'}
        values = []
        for _ in range(2):
            req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                values.append(json.load(response))
        expected = values[1]['usage']['prompt_tokens'] - 1
        self.assertEqual(values[0]['usage']['prompt_tokens_details']['cached_tokens'], 0)
        self.assertEqual(values[1]['usage']['prompt_tokens_details']['cached_tokens'], expected)
        self.assertEqual(values[1]['timings']['cache_n'], expected)
        self.assertEqual(values[1]['timings']['prompt_n'], 1)
        self.assertEqual(service.totals['reused'], expected)


class Configuration(unittest.TestCase):
    def test_bounds_before_process_launch(self):
        for value, slots in ((True, 1), (-1, 1), (4097, 1), (1, 2)):
            with self.assertRaises(ValueError):
                DeepSeekEngine(['never-started'], prefix_cache_mib=value, slots=slots)
        for value in ('false', 0, None):
            with self.assertRaises(ValueError): sampling_params({'cache_prompt': value}, 10)


if __name__ == '__main__':
    unittest.main()
