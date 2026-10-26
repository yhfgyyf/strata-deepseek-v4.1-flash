# Runtime configuration

[UD-Q2_K_XL](../examples/rtxpro6000.ud-q2-k-xl.json), [UD-IQ3_XXS](../examples/rtxpro6000.ud-iq3-xxs.json) and [GSQ-RCO-IQ2_XS](../examples/rtxpro6000.gsq-rco-iq2-xs.json) use `tools/deepseek41/serve_experimental.py --config FILE`. `rtxpro6000.experimental.json` is a compatibility copy of the Q2_K_XL example.

## Fixed starter settings

| Setting | Value |
|---|---|
| Context / request slots | 1048576 / 1 |
| Default maximum output | 32768; bounded by remaining context |
| Prefill chunk | 8192, no automatic chunk selection |
| Work / state / GPU headroom | 6720 / 1024 / 700 MiB |
| GPU expert byte budget | UD-Q2_K_XL: 68664950784; UD-IQ3_XXS: 67636039680; GSQ-RCO-IQ2_XS: `auto` from available VRAM after runtime reservations |
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
| Vision | Matching BF16 mmproj; optional for UD models. GSQ-RCO is text-only. |

## JSON fields

Required: `engine`, `variant`, `model` (first shard), `draft`, `tokenizer`. `variant` is `UD-IQ3_XXS`, `UD-Q2_K_XL` or `GSQ-RCO-IQ2_XS`. GSQ-RCO requires all four main shards and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` beside the first shard. The example paths must be edited before use.

Optional `mmproj` enables images for the UD models. Boolean `vision: false` disables the image API while retaining the supplied mmproj allocation. GSQ-RCO defaults to `vision: false` and rejects `vision: true` because this checkpoint lacks the required vision-routing tensors. Its example retains mmproj to match the tested memory allocation; removing it changes the available expert budget.

Optional `runtime_dir` contains auto-generated `integrity.txt`, `starter.strp` and `starter-heat.txt`. The default is `runtime/<config-stem>` beside the JSON. Bundled `costs`, `draft_input_vocab` and `draft_output_vocab` are used unless explicit paths are supplied.

To reuse calibrated startup assets, provide both `profile` and `heat`. An explicit `integrity` path uses your existing native-loader proof. Explicit files must exist and are not overwritten. `hot_bytes` overrides the variant's byte budget; it accepts a positive integer or `"auto"`. Automatic budgeting preserves the work, state, expert-ring and GPU-headroom reservations.

`host`, `port`, `alias`, `log` and private `api_key` control the frontend. The defaults are localhost, 30011 and `deepseek-v4.1`. Non-loopback binding requires authentication. Paths are relative to the config file, while the supplied first-shard directory spelling is preserved for intentional symlink assemblies.

`prefix_cache_mib` enables exact-token text-prefix reuse in the single request slot. The examples use 1024; an omitted value or 0 disables it, and valid integer budgets are 0–4096 MiB. Up to four completed checkpoints are retained in a private temporary directory, with the matching conversation root kept while other entries follow LRU. The byte/count limits take priority; one pending checkpoint can temporarily use additional space. Image requests bypass automatic caching, and normal shutdown removes the cache. Clients still send the full conversation history.

The launcher sets `STRATA_DS41_BLOCK_SEGMENTS=40`, `STRATA_DS41_BLOCK_HUGE_MIN_PERCENT=100` and `STRATA_DS41_DECODE_SEGMENTS=2`, and defaults `CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX=0`, matching the adopted profile. It does not change system configuration. `--check-config` only checks paths and executable permissions; startup checks generated metadata and full model identity.

It also defaults `STRATA_DS41_ADAPTIVE_POLICY=maya`, `OMP_NUM_THREADS=6` and `OPENBLAS_NUM_THREADS=1`. Set the policy to `legacy` to select the earlier residency policy. The experimental starter requires CPU 8 to be available in the process CPU set.
