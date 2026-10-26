# Set up this DeepSeek repository with an AI agent

This guide applies to **strata-deepseek-v4.1-flash**, its fixed Linux/CUDA/RTX PRO 6000 profile and the three model variants listed in the README. Use this repository's `./start.sh`; upstream Strata's `strata launch`, desktop installers and model identifiers are different products/workflows.

## Inspect before acquiring files

Read the user's selected model, destination, download source and proxy constraints. Reuse existing files when requested. Check OS/architecture, free disk, CPU 8, available RAM/VRAM and competing GPU workloads. Do not install drivers, alter huge-page settings, stop existing services or substitute a different model as a hidden preparation step.

```bash
./start.sh doctor --stage build --json
./start.sh configure --model GSQ-RCO-IQ2_XS --data-dir /path/to/models --source modelscope --json
./start.sh plan --json
```

`configure` writes a private config with a backup; `plan` does not write, download, hash or load a GPU. Use `--config FILE` on **every** command when the default is not intended. Inspect file destinations, missing bytes, disk margins, selected source, build/engine and URL. Existing matching sizes are not integrity verification.

Use `UD-Q2_K_XL` or `UD-IQ3_XXS` only when that is the chosen pack. `configure --set /private/fields.json` accepts existing asset paths and other supported fields. `--reset-paths` deliberately resets paths to the selected model/data directory; do not use it on a calibrated config merely to change one setting. Keep API keys out of shell arguments and answers.

## Prepare and start

After the plan fits the user's authorized setup/download scope:

```bash
./start.sh launch --yes --json
```

This creates the Python environment, installs the repository requirements, builds the native entries, downloads/verifies the selected files, checks the config, and starts the supervised service. The command waits for actual model/frontend readiness by default. Progress is on stderr; the final JSON result is on stdout. Large initial downloads and SHA256 scans take time. Preserve the job and monitor its logs; do not claim completion solely because files are present.

If installation and startup should be separate:

```bash
./start.sh install --yes --json
./start.sh doctor --json
./start.sh start --json
```

`--source hf-mirror` and `--no-proxy` honor an explicitly selected mirror/direct model-download path. No automatic source fallback is performed. Size/SHA mismatch is a failure, with the original or `.part` file retained; never invent a successful proof or disable verification to continue. A native huge-page admission failure must be reported and diagnosed within the user's existing authorization.

## Verify the actual service

```bash
./start.sh status --json
curl --fail http://127.0.0.1:30011/health
curl --fail http://127.0.0.1:30011/v1/models
```

Use the **reported** address and configured authentication. Confirm `ready: true`, the intended model, loaded health, and the expected API protocol with a small real request from [API.md](API.md). `--no-wait` only reports dispatch; it requires subsequent polling and request verification. A failed or timed-out launch is not a serving success. Operator accuracy, model quality and measured performance remain separate from installation and readiness.

## Handoff and stop

Report the exact model/config, model storage, service URL, readiness/request result and remaining limitations, without printing credentials. The user can open `./start.sh` to manage the service and chat, or use:

```bash
./start.sh status --json
./start.sh logs
./start.sh stop --json
```

The stop command targets only the recorded owned service. Do not kill processes by name or stop another service to free a port. Do not start a benchmark, modify model arithmetic, reboot or publish local logs as part of setup. For browser access to a headless host, use the SSH forwarding instructions in [LAUNCHER.md](LAUNCHER.md).
