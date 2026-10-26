"""Small HTTP/filesystem/process fixtures; no real model download or GPU needed."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from tools.deepseek41 import download_assets as downloads
from tools.deepseek41 import launcher
from tools.deepseek41.launcher_web import make_server

SOURCE_ROOT = launcher.ROOT


class Downloads(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = bytes(range(256)) * 64
        self.mode = "normal"
        self.ranges = []
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                requested = self.headers.get("Range")
                owner.ranges.append(requested)
                offset = int(requested.split("=")[1].split("-")[0]) if requested else 0
                if owner.mode == "ignore-range":
                    offset = 0
                code = 206 if requested and owner.mode != "ignore-range" else 200
                self.send_response(code)
                if code == 206:
                    start = offset + (1 if owner.mode == "wrong-range" else 0)
                    self.send_header("Content-Range", f"bytes {start}-{len(owner.data)-1}/{len(owner.data)}")
                self.send_header("Content-Length", str(len(owner.data) - offset))
                self.end_headers()
                self.wfile.write(owner.data[offset:])
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = "http://127.0.0.1:" + str(self.server.server_port) + "/fixture"
        self.path = self.root / "model.gguf"
        self.asset = dict(destination=str(self.path), bytes=len(self.data), sha256=hashlib.sha256(self.data).hexdigest())
        self.receipts = {}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def download(self):
        downloads.download_file(self.asset, "modelscope", self.receipts, no_proxy=True, url=self.url)

    def test_new_download_and_cached_verification(self):
        self.download()
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.path.with_suffix(".gguf.part").exists())
        with patch.object(downloads.hashlib, "sha256", side_effect=AssertionError("should reuse unchanged identity")):
            self.download()
        self.assertEqual(len(self.ranges), 1)

    def test_resume_retains_exact_prefix(self):
        partial = self.path.with_suffix(".gguf.part")
        partial.write_bytes(self.data[:511])
        self.download()
        self.assertEqual(self.ranges, ["bytes=511-"])
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_empty_partial_can_restart(self):
        self.path.with_suffix(".gguf.part").touch()
        self.download()
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_ignored_or_wrong_resume_range_preserves_partial(self):
        partial = self.path.with_suffix(".gguf.part")
        partial.write_bytes(self.data[:511])
        for mode in ("ignore-range", "wrong-range"):
            self.mode = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "resume range"):
                self.download()
            self.assertEqual(partial.read_bytes(), self.data[:511])
            self.assertFalse(self.path.exists())

    def test_hash_mismatch_keeps_partial_and_existing_files(self):
        self.asset["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.download()
        self.assertFalse(self.path.exists())
        self.assertEqual(self.path.with_suffix(".gguf.part").read_bytes(), self.data)
        self.path.write_bytes(self.data)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.download()
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_modified_file_invalidates_verification_receipt(self):
        self.download()
        self.path.write_bytes(b"x" + self.data[1:])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.download()

    def test_wrong_size_existing_file_is_never_overwritten(self):
        self.path.write_bytes(b"keep me")
        with self.assertRaisesRegex(ValueError, "file size"):
            self.download()
        self.assertEqual(self.path.read_bytes(), b"keep me")
        self.assertEqual(self.ranges, [])

    def test_all_sources_use_same_expected_hashes(self):
        for variant in launcher.VARIANTS:
            assets = downloads.catalog(variant)
            self.assertEqual(len(assets), 8 if variant == launcher.GSQ_VARIANT else 10)
            for source in downloads.SOURCES:
                urls = [downloads.asset_url(a, source) for a in assets]
                self.assertTrue(all(u.startswith("https://") for u in urls))
                self.assertTrue(all("/resolve/" in u for u in urls))


class ManagerFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "examples").mkdir()
        for p in (SOURCE_ROOT / "examples").glob("rtxpro6000.*.json"):
            (self.root / "examples" / p.name).write_bytes(p.read_bytes())
        self.patcher = patch.object(launcher, "ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.manager = launcher.Manager(self.root / "config.local.json", dict(data_dir=str(self.root / "models"),
                                                                              build_dir=str(self.root / "build")))


class Configuration(ManagerFixture):
    def test_plan_is_read_only_and_accounts_for_all_files(self):
        plan = self.manager.plan()
        self.assertFalse(self.manager.config.exists())
        self.assertFalse(self.manager.state_dir.exists())
        self.assertEqual(len(plan["assets"]), 8)
        self.assertEqual(plan["download_bytes"], sum(a["bytes"] for a in plan["assets"]))
        self.assertTrue(any("Engram" in a["path"] for a in plan["assets"]))

    def test_config_backup_key_redaction_and_preservation(self):
        self.manager.save_config(dict(api_key="private-test-key", host="0.0.0.0"))
        original = self.manager.config.read_bytes()
        self.assertNotIn("private-test-key", json.dumps(self.manager.public_config()))
        result = self.manager.save_config(dict(api_key="", prefix_cache_mib=512))
        self.assertEqual(Path(result["backup"]).read_bytes(), original)
        self.assertEqual(self.manager.config_value()["api_key"], "private-test-key")
        self.assertEqual(self.manager.config.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "requires api_key"):
            self.manager.save_config(clear_api_key=True)
        self.assertEqual(self.manager.config_value()["api_key"], "private-test-key")

    def test_invalid_config_does_not_replace_previous_file(self):
        self.manager.save_config()
        before = self.manager.config.read_bytes()
        for change in (dict(port=True), dict(vision=True), dict(hot_bytes=0), dict(prefix_cache_mib=-1),
                       dict(profile="one-without-heat"), dict(command="not an allowed field")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.manager.save_config(change)
            self.assertEqual(self.manager.config.read_bytes(), before)

    def test_relative_paths_are_relative_to_config(self):
        self.manager.save_config(dict(draft="assets/draft.gguf"))
        self.assertEqual(self.manager.config_value(resolve=True)["draft"], str(self.root / "assets/draft.gguf"))

    def test_config_symlink_uses_same_paths_and_process_ownership(self):
        self.manager.save_config(dict(draft="assets/draft.gguf"))
        alias=self.root/"alias/config.json";alias.parent.mkdir();alias.symlink_to(self.manager.config)
        other=launcher.Manager(alias)
        self.assertEqual(other.state_dir,self.manager.state_dir)
        self.assertEqual(other.config_value(resolve=True)["draft"],str(self.root/"assets/draft.gguf"))

    def test_signal_refuses_changed_identity(self):
        record=dict(pid=os.getpid(),identity={"different":True})
        with patch.object(launcher.os,"kill",side_effect=AssertionError("must not signal")):
            self.assertFalse(launcher.signal_owned(record,signal.SIGTERM))

    def test_explicitly_clear_hot_budget_restores_variant_default(self):
        self.manager.save_config(dict(hot_bytes=123456))
        self.manager.save_config(dict(hot_bytes=None))
        self.assertNotIn("hot_bytes", self.manager.config_value())

    def test_invalid_startup_deadlines_do_not_spawn(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "timeout"):
                self.manager.start(timeout=timeout, check=False)
        self.assertFalse((self.manager.state_dir / "service.json").exists())

    def test_reset_paths_selects_correct_original_model(self):
        self.manager.save_config()
        self.manager.save_config(dict(variant="UD-IQ3_XXS"), reset_paths=True)
        plan = self.manager.plan()
        self.assertEqual(len(plan["assets"]), 10)
        self.assertIn("UD-IQ3_XXS-00001-of-00007", plan["assets"][0]["destination"])

    def test_identity_mismatch_never_signals_an_unrelated_process(self):
        self.manager.save_config()
        downloads.private_json(self.manager.state_dir / "service.json", dict(state="ready",
            supervisor=dict(pid=os.getpid(), identity={"wrong": True}), worker=dict(pid=os.getpid(), identity={"wrong": True})))
        with patch.object(launcher.os, "kill", side_effect=AssertionError("must not signal")):
            self.assertTrue(self.manager.stop()["ok"])
        self.assertFalse(self.manager.status()["running"])

    def test_current_process_identity_is_stable(self):
        record = dict(pid=os.getpid(), identity=launcher.process_identity(os.getpid()))
        time.sleep(.03)
        self.assertTrue(launcher.owned(record))


class Logs(ManagerFixture):
    def test_service_logs_include_native_loading_progress_and_bounded_tails(self):
        self.manager.save_config(dict(log="native.log"))
        frontend = self.manager.state_dir / "service.log"
        frontend.write_text("f" * 200 + "frontend-ready\n")
        native = self.root / "native.log"
        native.write_text("n" * 200 + "loading-experts\n")
        result = self.manager.logs(limit=128)
        self.assertIn("frontend-ready", result)
        self.assertIn("loading-experts", result)
        self.assertLessEqual(len(result), 160)
        native.unlink()
        self.assertIn("frontend-ready", self.manager.logs())
        self.manager.save_config(dict(log=str(frontend)))
        self.assertEqual(self.manager.logs().count("frontend-ready"), 1)
        (self.manager.state_dir / "install.log").write_text("build-progress")
        self.assertEqual(self.manager.logs("install"), "build-progress")


FAKE_SERVICE = r'''
import argparse,json,os,signal,threading,time,subprocess,sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--ready-file');a=p.parse_args()
c=json.loads(Path(a.config).read_text());stop=threading.Event()
signal.signal(signal.SIGTERM,lambda *_:stop.set())
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*_):pass
 def do_GET(self):
  b=json.dumps({'service':'strata','loaded':True,'model':c.get('alias','deepseek-v4.1')}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
if c.get('fixture_crash'):raise SystemExit(7)
if c.get('fixture_child'):
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
 Path(c['fixture_child']).write_text(str(child.pid))
 if c.get('fixture_crash_after_child'):
  time.sleep(.3);raise SystemExit(9)
s=ThreadingHTTPServer(('127.0.0.1',c['port']),Handler)
threading.Thread(target=s.serve_forever,daemon=True).start()
if not c.get('fixture_no_ready'):Path(a.ready_file).write_text(json.dumps({'pid':os.getpid()}))
print('fixture service ready',flush=True)
while not stop.wait(.02):pass
s.shutdown();s.server_close()
'''


class Lifecycle(ManagerFixture):
    def setUp(self):
        super().setUp()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.manager.save_config(dict(port=self.port))
        self.service = self.root / "fake_service.py"
        self.service.write_text(FAKE_SERVICE)
        self.manager.python.parent.mkdir(parents=True)
        self.manager.python.symlink_to(sys.executable)
        self.driver = self.root / "supervise.py"
        self.driver.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0," + repr(str(SOURCE_ROOT)) + ")\n"
                              "from tools.deepseek41 import launcher\nlauncher.ROOT=Path(" + repr(str(self.root)) + ")\n"
                              "launcher.ENTRY=Path(" + repr(str(self.service)) + ")\n"
                              "a=launcher.parser().parse_args()\nlauncher.supervise(launcher.Manager(a.config),a.instance,a.timeout)\n")
        self.real_popen = subprocess.Popen
        def popen(command, **kwargs):
            if "_supervise" in command:
                command = [sys.executable, str(self.driver), *command[2:]]
            return self.real_popen(command, **kwargs)
        self.proc_patch = patch.object(launcher.subprocess, "Popen", side_effect=popen)
        self.proc_patch.start()
        self.addCleanup(self.proc_patch.stop)
        self.addCleanup(self.stop)

    def stop(self):
        try:
            self.manager.stop(timeout=5)
        except RuntimeError:
            state = self.manager.status()
            for key in ("worker", "supervisor"):
                r = state.get(key)
                if launcher.owned(r):
                    os.kill(r["pid"], signal.SIGTERM)

    def test_ready_stop_restart_and_edit_lock(self):
        started = self.manager.start(timeout=8, check=False)
        self.assertTrue(started["ready"])
        self.assertEqual(launcher.health(self.manager.config_value())["service"], "strata")
        with self.assertRaisesRegex(RuntimeError, "stop this managed"):
            self.manager.save_config(dict(port=self.port + 1))
        self.assertEqual(self.manager.stop(timeout=8)["state"], "stopped")
        self.assertIsNone(launcher.health(self.manager.config_value()))
        second = self.manager.start(timeout=8, check=False)
        self.assertNotEqual(second["instance"], started["instance"])
        self.assertTrue(second["ready"])

    def test_busy_port_never_starts_worker_or_stops_its_owner(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", self.port));listener.listen()
            with self.assertRaisesRegex(RuntimeError, "failed to start"):
                self.manager.start(timeout=3, check=False)
            self.assertEqual(self.manager.status()["state"], "failed")
            self.assertNotIn("worker", self.manager.status())
            listener.getsockname()

    def test_nonzero_exit_is_failure_not_ready(self):
        cfg=self.manager.config_value();cfg["fixture_crash"]=True;downloads.private_json(self.manager.config,cfg)
        with self.assertRaisesRegex(RuntimeError, "failed to start"):
            self.manager.start(timeout=4, check=False)
        self.assertEqual(self.manager.status()["state"], "failed")
        self.assertIn("code 7", self.manager.status()["error"])

    def test_health_without_worker_ready_receipt_is_not_success(self):
        cfg=self.manager.config_value();cfg["fixture_no_ready"]=True;downloads.private_json(self.manager.config,cfg)
        with self.assertRaisesRegex(RuntimeError, "failed to start"):
            self.manager.start(timeout=1, check=False)
        deadline=time.monotonic()+5
        while self.manager.status()["running"] and time.monotonic()<deadline:time.sleep(.1)
        self.assertEqual(self.manager.status()["state"], "failed")
        self.assertIn("timed out", self.manager.status()["error"])
        self.assertIsNone(launcher.health(self.manager.config_value()))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux child subreaper")
    def test_frontend_crash_cleans_adopted_native_child(self):
        child_file=self.root/"child.pid"
        cfg=self.manager.config_value();cfg.update(fixture_child=str(child_file),fixture_crash_after_child=True)
        downloads.private_json(self.manager.config,cfg)
        with self.assertRaisesRegex(RuntimeError,"failed to start"):
            self.manager.start(timeout=3,check=False)
        child_pid=int(child_file.read_text())
        deadline=time.monotonic()+8
        while self.manager.status()["running"] and time.monotonic()<deadline:time.sleep(.1)
        self.assertIsNone(launcher.process_identity(child_pid))
        self.assertIn("code 9",self.manager.status()["error"])


class Web(ManagerFixture):
    def setUp(self):
        super().setUp()
        self.server=make_server(self.manager,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.close)
        self.url=f"http://127.0.0.1:{self.server.server_port}"
        html=urllib.request.urlopen(self.url,timeout=3).read().decode()
        self.token=re.search(r'name="strata-token" content="([^"]+)"',html)[1]

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join()

    def request(self,path,body=None,headers=None):
        request=urllib.request.Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,
            headers={"X-Strata-Token":self.token,"Content-Type":"application/json",**(headers or {})})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=5) as response:return json.load(response)

    def test_page_config_plan_and_no_secret_exposure(self):
        self.request("/api/config",dict(set=dict(api_key="must-stay-private")))
        state=self.request("/api/state")
        self.assertTrue(state["config"]["has_api_key"])
        self.assertNotIn("must-stay-private",json.dumps(state))
        self.assertEqual(len(self.request("/api/plan")["assets"]),8)

    def test_foreign_origin_host_and_missing_token_rejected(self):
        for headers in ({"Origin":"https://foreign.example"},{"Host":"foreign.example"},{"X-Strata-Token":""}):
            with self.subTest(headers=headers),self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/config",dict(set=dict(port=12345)),headers)
            self.assertEqual(error.exception.code,403)
            error.exception.close()
        self.assertFalse(self.manager.config.exists())

    def test_bad_config_and_non_json_cannot_change_configuration(self):
        for body,headers,code in ((dict(set=dict(port=0)),{},400),(dict(set=dict(port=12345)),{"Content-Type":"text/plain"},415)):
            with self.subTest(body=body),self.assertRaises(urllib.error.HTTPError) as error:
                self.request("/api/config",body,headers)
            self.assertEqual(error.exception.code,code)
            error.exception.close()
        self.assertFalse(self.manager.config.exists())


if __name__ == "__main__":
    unittest.main()
