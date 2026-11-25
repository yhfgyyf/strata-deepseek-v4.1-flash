# Runtime configuration

`./start.sh` manages `config.local.json` by default. Select an existing file with `./start.sh gui --config FILE`; the same `--config FILE` works for `plan`, `install`, `start`, `status`, `stop`, `logs` and `doctor`. The management page can edit DeepSeek paths and settings while its service is stopped. It preserves unedited fields and creates a timestamped private backup before saving. The chat page's inherited generic `/config` editor is separate and is not used for DeepSeek configuration. See [management commands](LAUNCHER.md).

[UD-Q2_K_XL](../examples/rtxpro6000.ud-q2-k-xl.json), [UD-IQ3_XXS](../examples/rtxpro6000.ud-iq3-xxs.json), [GSQ-RCO-IQ2_XS](../examples/rtxpro6000.gsq-rco-iq2-xs.json) and [GSQ-RCO-IQ2_XS-Uncensored](../examples/rtxpro6000.gsq-rco-iq2-xs-uncensored.json) use `tools/deepseek41/serve_experimental.py --config FILE`. `rtxpro6000.experimental.json` is a compatibility copy of the Q2_K_XL example.

## Fixed starter settings

| Setting | Value |
|---|---|
| Context / request slots | 1048576 / 1 |
| Default maximum output | 32768; bounded by remaining context |
| Prefill chunk | 8192, no automatic chunk selection |
| Work / state / GPU headroom | 6720 / 1024 / 700 MiB |
| GPU expert byte budget | UD-Q2_K_XL: 68664950784; UD-IQ3_XXS: 67636039680; both GSQ variants: `auto` from available VRAM after runtime reservations |
| RAM expert budget / host headroom | 160 / 16 GiB |
| Engram host cache / migration scratch | 8 GiB / 3072 MiB |
| Host expert backing | 40 registered blocks, 100% 2 MiB page coverage before and after registration |
| Main arithmetic | Existing BF16/route/reduction contracts; accepted mixed-format and short-batch kernels |
| DSpark | Original 1017 draft; at most 5 proposals, adaptive greedy verification, incremental KV, prefix commit |
| Shared Q8 pair / main MoE fusion | Enabled / mode 2 |
| Decode Graph segments | Output/router mask 2 |
| Sampling | Greedy, temperature 0, top_k 1, seed 1234 |
| Residency | Format groups and online Maya accepted-row adaptation |
| Request/control thread | CPU 8, pinned after runtime workers are created |
| Text prefix cache | Example configs: 4 completed checkpoints, up to 1024 MiB |
| Vision | Enabled in all four profiles. Four-shard GSQ models load their verified F32 image-router rows beside the BF16 mmproj. |

## JSON fields

Required: `engine`, `variant`, `model` (first shard), `draft`, `tokenizer`. `variant` is `UD-IQ3_XXS`, `UD-Q2_K_XL`, `GSQ-RCO-IQ2_XS` or `GSQ-RCO-IQ2_XS-Uncensored`. GSQ-RCO requires all four main shards and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` beside the first shard. The example paths must be edited before use.

All four profiles set `vision: true` and supply the 1017 BF16 `mmproj`. Older GSQ configs that omit `vision` keep their text-only behavior. For UD, omitting `vision` enables images when `mmproj` is set. An explicit `vision: false` disables image input for any variant. The two UD models read their image-router rows from the main shards. The two GSQ models read the publisher’s 61,440-byte `exp_probs_b_vl.f32` beside the configured mmproj. Startup verifies its SHA256 and writes a separate proof in `runtime_dir`. With image input disabled, keeping `mmproj` still reserves the projector memory.

Optional `runtime_dir` contains auto-generated `integrity.txt`, `starter.strp` and `starter-heat.txt`. The default is `runtime/<config-stem>` beside the JSON. Bundled `costs`, `draft_input_vocab` and `draft_output_vocab` are used unless explicit paths are supplied.

To reuse calibrated startup assets, provide both `profile` and `heat`. An explicit `integrity` path uses your existing native-loader proof. Explicit files must exist and are not overwritten. `hot_bytes` overrides the variant's byte budget; it accepts a positive integer or `"auto"`. Automatic budgeting preserves the work, state, expert-ring and GPU-headroom reservations.

`host`, `port`, `alias`, `log` and private `api_key` control the frontend. The defaults are localhost, 30011 and `deepseek-v4.1`. Non-loopback binding requires authentication. Paths are relative to the config file, while the supplied first-shard directory spelling is preserved for intentional symlink assemblies.

`prefix_cache_mib` enables exact-token text-prefix reuse in the single request slot. The examples use 1024; an omitted value or 0 disables it, and valid integer budgets are 0–4096 MiB. Up to four completed checkpoints are retained in a private temporary directory, with the matching conversation root kept while other entries follow LRU. The byte/count limits take priority; one pending checkpoint can temporarily use additional space. Image requests bypass automatic caching, and normal shutdown removes the cache. Clients still send the full conversation history.

The launcher sets `STRATA_DS41_BLOCK_SEGMENTS=40`, `STRATA_DS41_BLOCK_HUGE_MIN_PERCENT=100` and `STRATA_DS41_DECODE_SEGMENTS=2`, and defaults `CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX=0`, matching the adopted profile. It does not change system configuration. `--check-config` only checks paths and executable permissions; startup checks generated metadata and full model identity.

It also defaults `STRATA_DS41_ADAPTIVE_POLICY=maya`, `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=1`. Set the policy to `legacy` to select the earlier residency policy. It defaults `STRATA_DS41_OPT_MASK=0xFF7FFFFF` and `STRATA_DS41_OPT_MASK2=0xFEF`. The first mask enables tensor-core routed MoE (`0x8000`), the q_b projection (`0x40000`), Q5_K tensor cores for the shared-expert down projection and q_b (`0x1000000`) and the fused tensor-core prefill MoE (`0x20000000`); the second selects the BF16 fused prefill MoE (`0x400`, weights dequantized exactly as the BF16 cuBLAS path, which it matches up to accumulation order) and keeps the byte-identical int8 kernel v2 (`0x800`) as the fallback. `STRATA_DS41_OPT_MASK=0x7B7FFF` selects the bit-exact decode path with the cuBLAS prefill MoE. `STRATA_DS41_ENGRAM_MIRRORS` (colon-separated paths, unset by default) names byte-identical copies of the Engram GGUF on other drives; large Engram read batches are then spread over all copies, one AIO context per drive. `start-l20-mask479.sh` sets `0`. The experimental starter requires CPU 8 to be available in the process CPU set.

## Explicit dual-L20 profile

The [L20 example](../examples/l20-dual.gsq-rco-iq2-xs.json) uses the separate [mask479 entry](INSTALL.md#dual-l20-mask479), port10002 and `vision: false`. `pipeline_peer: 1` selects the second visible CUDA device; omitting it or setting `-1` retains the single-device path. `pipeline_contiguous: false` retains the CED stage ownership 0:10 / 10:20 / 20:32 / 32:40. `decode_cpu_misses: false` keeps cold experts on GPU. `huge_pinned_memory: false` explicitly selects ordinary-page block-pinned memory with 40 registrations per bank; the PRO 6000 default remains `true` with 100% huge-page coverage. These fields do not change drivers, system page settings or another service.

Use the L20 script and binary together. Merely selecting two GPUs with the original experimental binary does not activate the complete mask479 dispatch set. The six-shard model path selects its pinned PLE identity; the existing four-shard GSQ and Uncensored identities remain distinct.
