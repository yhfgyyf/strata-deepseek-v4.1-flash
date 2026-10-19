# Runtime profiles

## Fixed experimental starter

`tools/deepseek41/serve_experimental.py` launches `strata-deepseek41-experimental` using [the example JSON](../examples/rtxpro6000.experimental.json). Relative paths are resolved against the configuration file's directory. `--check-config` checks files and the engine's executable bit without loading a model or accessing a GPU. Full model SHA proofs are enforced by the native loader during startup.

The C++ recipe is [experimental_server.cpp](../tools/deepseek41/experimental_server.cpp). It is intentionally a fixed, reviewable recipe for the test machine:

| Setting | Value |
|---|---|
| Context / request slots | 32768 / 1 |
| Prefill chunk | 8192; automatic chunk selection disabled |
| Work / state / GPU headroom | 6720 / 1024 / 700 MiB |
| GPU expert budget | 72,596,914,176 bytes; 5649 experts with the original frozen ranking |
| RAM expert budget / host headroom | 144 / 16 GiB |
| Engram host row cache | 8 GiB |
| Expert migration scratch | 3072 MiB |
| Host expert backing | 40 segments, 100% 2 MiB pages before and after registration |
| Main short-row MoE fusion | Mode 2 |
| Indexer scores / paired expert expansion | Fast paths enabled |
| DSpark | 5 draft tokens, batched target verification, incremental KV, prefix commit |
| Adaptive verification / shared Q8 pair | Enabled / enabled |
| Sampling | Greedy, temperature 0, top_k 1, seed 1234 |
| Residency | Format groups, accepted-row update period, saved startup heat |
| Routed expert arithmetic | GPU; rejected CPU overlap path is not enabled |
| Images | Optional matching mmproj; omit `mmproj` or set it to an empty string for text only |

The GPU budget is a byte budget, not a guaranteed expert count on a different profile. Starting with a different profile or heat distribution changes execution and may change speed. The cost file is a historical calibration, not an automatically tuned setting.

The starter keeps a process-owned runtime, matching the tested native harness, and exits the native IPC process after service shutdown. It does not apply OS huge-page policy, clear global caches, stop unrelated processes or download assets. Insufficient huge pages cause startup to fail with an explicit error.

## Generic launcher

`python -m serve.deepseek41 --config examples/basic.json` uses the original general CLI wrapper. Edit the asset paths first. It is a basic non-speculative configuration; it does not enable every experimental runtime switch. Its paths follow the working directory, as in the inherited launcher.

The source still contains `auto` chunk parsing and other research options. The supplied profiles deliberately use **8192** and do not select `auto` or `auto:N`.

## Asset identity

The experimental starter expects the tested derived draft SHA256 `9269e73cbd04c493d2510d3513da85429441949f6f10ed8130392ae0de2a949b`. Its input/output vocabulary maps, startup ranking and heat must be supplied locally. The maps contain token IDs; profile/heat contain expert ranking and counters. None of these files is silently inferred from the current request or replaced by a generic placeholder.

The loader checks model file identity against a locally created full-hash proof, including device, inode, size and timestamps. Moving or changing a file can require regenerating that proof. The proof is a startup integrity check, not a claim that subsequent hardware reads can never be inconsistent.
