# Launcher and browser manager

Run from the repository root:

```bash
./start.sh
```

The management page opens at `http://127.0.0.1:30010/` even before a model is installed. The page is in Chinese. Select the model, download source and storage directory. **按模型和目录填写路径** fills and saves the selected pack's paths; advanced fields also accept existing files. Inspect **查看准备计划**, then use **准备并启动** or **仅准备文件与引擎**. Once installed, **启动模型** starts the service without rebuilding or downloading. **打开聊天** opens the existing chat/monitor page after the service is ready.

The runtime target remains Linux x86-64, Python 3.10+, CMake 3.24+, a C++20 compiler, CUDA 13.2 and one RTX PRO 6000 Blackwell 96GB with the documented host-memory configuration. The manager checks prerequisites; it does not install system tools/drivers, change huge-page policy or stop competing workloads. Native startup still requires CPU 8, 40 registered expert blocks and 100% huge-page backing. This launcher does not change chunk size or model arithmetic.

## Commands

All commands accept `--config FILE` (default `config.local.json`) and `--json`. JSON results go to stdout; preparation progress and failures go to stderr. Paths entered into the model JSON are relative to that file. Download/build directories supplied on the command line are resolved from the current directory.

```bash
./start.sh configure --model GSQ-RCO-IQ2_XS --data-dir /path/to/models
./start.sh plan --json
./start.sh launch --yes --json
./start.sh status --json
./start.sh logs
./start.sh stop --json
```

| Command | Behavior |
|---|---|
| `gui` | Open the local management page. `--port 30010 --no-browser` supports a headless host. |
| `configure` | Save paths/settings with a private timestamped backup. `--model UD-Q2_K_XL` / `UD-IQ3_XXS` selects the other packs. `--reset-paths` replaces existing asset paths using the selected model and data directory. |
| `plan` | Read-only file/disk/download plan; no hashing, writes, installation or GPU loading. |
| `install --yes` | Create `.venv`, install the existing pinned Python requirements, build the three native entries, resume/verify assets and check config. Leaves the model stopped. |
| `launch --yes` | Install, then start and wait for readiness. |
| `start` | Start the existing installation. The supervisor checks the port, owns the process and enforces a startup deadline. Default `--timeout 1800` is in seconds. |
| `status` | Report starting/ready/stopping/stopped/failed and the associated process identities, log and endpoint. |
| `stop` | Stop only this config's managed service; wait for frontend/native cleanup. |
| `cancel` | Cancel a background preparation job, retaining partial downloads and logs. For installation started in another foreground terminal, use Ctrl+C there. |
| `logs` | Show bounded tails of the frontend and configured native-engine logs, including model-loading progress; `--kind install` selects preparation output. |
| `doctor` | Check runtime requirements without loading a model. `--stage build` checks compiler/build prerequisites. |

`start`/`launch --no-wait` returns after dispatch with `ready: false` while loading. It is not an inference success; keep polling `status`, inspect failures, and verify a request. Closing the page or exiting `gui` leaves an already managed service running. Use `stop` explicitly.

## Download and build settings

`configure`, `plan`, `install`, `launch` and `gui` accept `--source modelscope|huggingface|hf-mirror`, `--data-dir`, `--build-dir`, `--cuda`, `--ggml-dir`, `--jobs` and `--no-proxy`. ModelScope is the default. Explicit source selection is retained; failures never silently switch providers. The Hugging Face sources use the pinned revisions in `data/`; ModelScope uses its `master` path but must match the same file sizes and publisher SHA256 values, including the official tokenizer.

The downloader preserves `.part` on interruptions and hash failures. A resume requires a valid HTTP byte range; an endpoint that ignores ranges is rejected rather than appended to. Files are published under their final names only after full SHA256 succeeds. Existing files are never replaced. Unchanged verified file identities can reuse their local verification receipt; modified files are rechecked. Some hosts limit very large regular HTTP downloads; choose another source explicitly when necessary.

`--no-proxy` bypasses proxy settings for model HTTP transfers and clears proxy environment variables in build/install subprocesses. Dependencies use the existing `requirements.txt`; there are no extra production packages for the manager. CMake uses the same pinned ggml revision as manual builds. If `--ggml-dir` supplies an extracted copy, its revision/content must already have been verified by the operator. An existing custom `engine` path is reused; the default engine in `build/` is built incrementally.

## Existing models and private configuration

An existing `serve_experimental.py` config can be opened directly:

```bash
./start.sh gui --config examples/local.json
./start.sh start --config examples/local.json --json
```

The managed runtime uses the repository's `.venv/bin/python`. For an existing installation, create/populate this environment as described in [manual installation](INSTALL.md), or use `install --yes` to prepare it. Original calibrated `profile`/`heat`, explicit `integrity`, cost/vocabulary files and unknown existing JSON fields are retained. A blank API-key input keeps the current key; clearing requires the explicit checkbox. Configs and backups created by the manager have mode 0600; defaults and backup patterns are Git-ignored.

`configure --set /private/changes.json` applies an object of allowed model fields, including custom model/draft/tokenizer paths. Keep credentials in the file, not command-line arguments or published logs. Non-loopback API binding requires a key; the manager itself always binds to loopback and rejects foreign Host/Origin and missing control tokens.

## Access from another computer

On the inference host:

```bash
./start.sh gui --no-browser
```

From your own computer, forward the manager and chat ports through your existing SSH host alias:

```bash
ssh -N -L 30010:127.0.0.1:30010 -L 30011:127.0.0.1:30011 YOUR_HOST
```

Open `http://127.0.0.1:30010/`. If you change the API port, forward that port as well. This browser manager is not an Electron/native desktop installer and does not add a public administration endpoint.

## State and recovery

Each absolute config path has its own records under `runtime/launcher/<config-id>/`: setup, service identity, background job, download receipts, `service.log` and `install.log`. A second launch of the same config is refused; a port used by any existing service is refused before native startup. PID reuse is checked before signalling. On Linux the supervisor also reaps the native child after a frontend crash.

Do not remove ownership records while the service runs. If the supervisor was killed externally and an owned worker remains, `status` reports `orphaned` and ordinary stop refuses to guess which process group is safe. Inspect the exact recorded processes. Failed builds, partial weights, previous configuration backups and logs are retained for diagnosis and retry.
