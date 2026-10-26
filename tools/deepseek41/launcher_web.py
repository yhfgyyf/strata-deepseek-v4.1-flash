"""Local browser control for the DeepSeek launcher; no model is needed to open it."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import threading
import urllib.parse
import webbrowser

from tools.deepseek41.launcher import ROOT


def make_server(manager, port=30010):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, value, content_type="application/json; charset=utf-8"):
            body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, api=False):
            host = self.headers.get("Host", "")
            allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed_hosts:
                self.reply(403, {"error": "unrecognized Host"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + host:
                self.reply(403, {"error": "foreign Origin"})
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Strata-Token", ""), token):
                self.reply(403, {"error": "open this launcher's own page"})
                return False
            return True

        def do_GET(self):
            parsed = urllib.parse.urlsplit(self.path)
            path = parsed.path
            if not self.allowed(api=path.startswith("/api/")):
                return
            try:
                if path == "/":
                    html = (ROOT / "serve/web/launcher.html").read_text().replace("%%TOKEN%%", token)
                    self.reply(200, html, "text/html; charset=utf-8")
                elif path in ("/launcher.js", "/launcher.css"):
                    self.reply(200, (ROOT / "serve/web" / path[1:]).read_text(),
                               "text/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8")
                elif path == "/api/state":
                    self.reply(200, dict(config=manager.public_config(), service=manager.status(), job=manager.job()))
                elif path == "/api/plan":
                    self.reply(200, manager.plan())
                elif path == "/api/logs":
                    kind = urllib.parse.parse_qs(parsed.query).get("kind", ["service"])[0]
                    self.reply(200, dict(text=manager.logs(kind)))
                else:
                    self.reply(404, {"error": "not found"})
            except (OSError, ValueError, RuntimeError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            if not self.allowed(api=True):
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.reply(415, {"error": "send application/json"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if not 0 <= length <= 65536:
                    self.reply(413, {"error": "invalid request size"})
                    return
                self.connection.settimeout(5)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request")
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                if self.path == "/api/config":
                    value = manager.save_config(body.get("set"), body.get("setup"),
                                                body.get("reset_paths", False), body.get("clear_api_key", False))
                elif self.path == "/api/install":
                    value = manager.prepare_background(launch=body.get("launch") is True)
                elif self.path == "/api/start":
                    value = manager.start(wait=False)
                elif self.path == "/api/cancel":
                    value = manager.cancel_prepare()
                elif self.path == "/api/stop":
                    # Signal and wait in a worker so status polling continues while native shutdown completes.
                    result = {}
                    def stop():
                        try:
                            result.update(manager.stop())
                        except (OSError, ValueError, RuntimeError) as error:
                            result.update(error=str(error))
                    worker = threading.Thread(target=stop, daemon=True)
                    worker.start()
                    worker.join(.25)
                    value = result or dict(ok=True, state="stopping")
                elif self.path == "/api/doctor":
                    value = manager.doctor(body.get("stage", "run"))
                else:
                    self.reply(404, {"error": "not found"})
                    return
                self.reply(200, value)
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                self.reply(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def run(manager, port=30010, open_browser=True):
    server = make_server(manager, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print("DeepSeek manager: " + url, flush=True)
    print("Closing this page or manager leaves the managed model running; use Stop to end it.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        server.server_close()
