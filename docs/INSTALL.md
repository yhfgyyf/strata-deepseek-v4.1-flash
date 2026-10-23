# Build and model setup

## Requirements

Tested on Linux x86-64, GCC 11.4, Python 3.10, CMake 3.27.8, CUDA Toolkit 13.2.51, driver 595.104.02 and one RTX PRO 6000 Blackwell 96GB, with i7-14700KF and 192GB DDR5-4800. Python runtime packages are in [requirements.txt](../requirements.txt). Other inherited Strata backends are not qualified by this DeepSeek release.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DEEPSEEK41_TESTS=ON
cmake --build build --parallel 2 --target \
  strata-deepseek41 strata-deepseek41-experimental strata-deepseek41-inspect
```

CMake uses llama.cpp/ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. To reuse that revision, pass `-DSTRATA_GGML_DIR=/path/to/llama.cpp`. Architecture-specific indexer/CAKE objects use SM120a. Prebuilt bounded GEMM cubins are included; Triton 3.6.0 is only needed to regenerate them, not to build or serve normally. No new runtime dependency is required for the two-model launcher.

## Model assets

Download from [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/60ad3651641b5c97679c807e064ab1d7a3ca1455) or [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files). The loader pins file content, not a moving repository branch; the approved files and all SHA256 values are in [models-1017.json](../data/models-1017.json).

| Asset | Required files |
|---|---|
| Main model, choose one | All seven `UD-IQ3_XXS` shards **or** all seven `UD-Q2_K_XL` shards. Keep names/directory together and configure shard 00001. |
| DSpark | Original `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, SHA256 `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` |
| Image encoder, optional | Original `mmproj-BF16.gguf`, SHA256 `9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9` |
| Tokenizer | [Official tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/tokenizer.json), SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

No GGUF merge, offline conversion, separate Engram extraction, or draft requantization is needed. The last main shard contains large SSD-backed data and must not be omitted. Keep weights on local NVMe. Runtime BF16 expansion and selected draft vocabulary rows do not produce a converted checkpoint or modify original files.

The included draft mappings contain 114801 input IDs and 90000 output IDs, derived only from the matching tokenizer's characters/BPE ranks. They contain no weights or request token streams. The main model keeps all 129280 IDs; if a proposal's first token is outside the input map, DSpark falls back to main-model decoding. See [mapping provenance](../data/draft-vocab-provenance.json).

## One launch script for both models

```bash
cp examples/rtxpro6000.ud-q2-k-xl.json examples/local.json
# For IQ3, copy examples/rtxpro6000.ud-iq3-xxs.json instead.
# Fill in the model, draft, tokenizer and optional mmproj paths.
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

The script prepares metadata automatically:

1. Validate all seven shard paths and configured assets before starting the engine.
2. Create a complete 40×384 starter expert ranking, interleaved across layers, and zero initial heat. This is explicitly uncalibrated. Online adaptive residency remains enabled.
3. Hash all main/draft/vision files in full and atomically write the native proof only if every expected SHA256 matches. Later starts reuse it only while device, inode, size, mtime and ctime still match. A first launch can spend several minutes reading hundreds of GB.
4. Start the native engine and HTTP frontend with 1,048,576 context, chunk 8192, 32768 default maximum output and DSpark.

`runtime_dir` holds generated metadata. Existing calibrated `profile` and `heat` may be supplied together; an explicit `integrity` proof is also accepted. Explicit asset paths are validated and not regenerated. Relative paths are relative to the JSON file. Generated metadata is not a substitute for the benchmark's calibrated ranking and does not inherit its throughput claim.

Optional preflight, without writes, hashing or GPU access:

```bash
python tools/deepseek41/serve_experimental.py --config examples/local.json --check-config
```

Default endpoint: `http://127.0.0.1:30011/v1`. Stop with `Ctrl+C`. Set a private `api_key` for a non-loopback bind. Remove `mmproj` for text-only operation. The process requires sufficient available host RAM/VRAM and 100% huge-page backing for its registered expert blocks; it does not alter system huge-page policy or stop other processes. [Configuration details](CONFIGURATION.md).

For an existing advanced setup, the standalone proof utility remains available. Use `--sidecars-1017` with either model, plus `--main-q2-k-xl` for Q2_K_XL. The unified starter selects these identities automatically. The generic `python -m serve.deepseek41` CLI remains a lower-level interface; its basic example does not reproduce the optimized starter.

## Checks without GPU inference

```bash
python -m unittest serve.test_deepseek41 serve.test_deepseek41_images \
  serve.test_responses tools.deepseek41.test_release_config
cmake --build build --parallel 2 --target \
  deepseek41_contract_test deepseek41_engram_test deepseek41_sampling_test \
  deepseek41_integrity_test deepseek41_residency_test deepseek41_prefill_chunk_test
CUDA_VISIBLE_DEVICES= ctest --test-dir build --output-on-failure \
  -R '^deepseek41_(contract|engram|sampling|integrity|residency|prefill_chunk)_test$'
```

These selected CPU/configuration checks do not execute the GPU operator tests, establish full-model quality or remeasure performance. See [publication evidence](SOURCE_PROVENANCE.md).
