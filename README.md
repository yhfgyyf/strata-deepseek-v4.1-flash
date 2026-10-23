# strata-deepseek-v4.1-flash

[简体中文](README.zh-CN.md)

DeepSeek-V4.1-Flash on a home PC: **Intel Core i7-14700KF, one RTX PRO 6000 Blackwell 96GB, and 192GB DDR5-4800**.

Built on [Strata](https://github.com/Niko1221/Strata) and [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4), with a native DeepSeek runtime, SSD-backed Engram, GPU/RAM expert residency, and **DSpark speculative decoding**. Both supported Unsloth models use the same launch script and their original GGUF files: **no offline weight conversion, shard merging, or extra quantization is required**.

The service supports **1M context capacity (1,048,576 tokens)**, **multimodal text + image input**, tool calls, OpenAI Chat Completions / Responses, and Anthropic Messages. The starter uses one request slot, chunk 8192, and a default maximum output of 32768 tokens. The 1M capacity was allocated and short API requests passed; measured long inputs reach 512K. Full 1M input and continuous 32K output have not been stress-tested. Image support does not imply audio/video support.

This is an experimental source release. Operator checks and functional tests have defined coverage; full-model quality, repeatability and long-run stability remain open. See [validation and limitations](docs/PERFORMANCE.md#validation-and-limitations).

## Supported models

Both variants are from **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**. Download all seven files in the chosen directory and point the configuration at `00001-of-00007.gguf`. Keep the original names and all siblings together, including the large final shard used for Engram.

| Main model | Hugging Face | ModelScope | Launch configuration |
|---|---|---|---|
| **UD-IQ3_XXS** | [UD-IQ3_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [Model files — select UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [Model files — select UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |

Use `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` and, for images, `mmproj-BF16.gguf` from the same repository: [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files). The DSpark file retains its released mixed MXFP4/Q8/BF16 storage; its filename does not mean every tensor is Q8. Runtime dequantization and vocabulary-row selection leave the downloaded files unchanged.

The two mirrors' 14 main shards and two sidecars had matching sizes and SHA256 on October 23, 2026; the loader accepts the recorded content identities in [models-1017.json](data/models-1017.json). Also obtain the matching [official tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/tokenizer.json). Model files and the tokenizer are downloaded separately. [Detailed asset requirements](docs/INSTALL.md#model-assets).

## Machine

| Component | Tested configuration |
|---|---|
| GPU | One NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96GB, SM120 |
| CPU | Intel Core i7-14700KF, 20 cores / 28 threads |
| RAM | 192GB DDR5, configured at 4800 MT/s |
| Storage | Local NVMe SSD for GGUF and Engram reads |
| Software | Ubuntu Linux, CUDA Toolkit 13.2, NVIDIA driver 595.104.02 |
| GPU link | PCIe 5.0 ×16 observed in the October 23 runs |

## Throughput

**Maximum measured prefill: 6,080.745 tokens/s**, at 131,072 input tokens, chunk 32768, with UD-Q2_K_XL and original DSpark. This is one formal, profiler-free run; the corresponding three-run mean is **6,037.613 tokens/s**.

October 23 measurements below use one request, one warmup plus three formal runs per length, greedy decoding and exactly 512 output tokens. The input is a repeated/truncated LongSWE token stream, not a natural long-document quality benchmark. Rates are means ± sample standard deviations.

| Input tokens | Chunk | Resident experts | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 5571 | 3,619.978 ± 11.526 | 34.312 ± 0.027 |
| 32,768 | 32,768 | 4221 | 5,769.141 ± 25.338 | 34.936 ± 0.010 |
| 131,072 | 32,768 | 4221 | **6,037.613 ± 40.617** | 35.048 ± 0.026 |
| 524,288 | 32,768 | 4221 | 4,499.192 ± 3.861 | 42.216 ± 0.038 |

Prefill uses native prompt time; decode includes DSpark generation and excludes the separately recorded final drain. These are not client TTFT measurements. The 32K chunk benchmark reserves more workspace and fewer resident experts than the 8K starter. The benchmark also uses a calibrated ranking/heat snapshot; the automatic starter metadata is uncalibrated and has not been separately benchmarked. The rebuilt release binaries have no new GPU performance measurement. [All 12 formal runs](data/performance-20261023.csv) · [conditions, historical results and limits](docs/PERFORMANCE.md).

## Build and launch

The tested target is Linux x86-64, Python 3.10+, a C++20 compiler, CMake 3.24+, and CUDA 13.2 / SM120. The indexer and CAKE objects use SM120a. Existing CUDA/huge-page prerequisites remain necessary; the launcher does not change system settings.

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
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

Choose either model's configuration, edit its model/draft/tokenizer/mmproj paths, and run **the same script**:

```bash
cp examples/rtxpro6000.ud-q2-k-xl.json examples/local.json
# Or copy examples/rtxpro6000.ud-iq3-xxs.json instead.
# Edit the /path/to entries. Remove mmproj for text-only service.
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

On first launch the script hashes every model file, writes an integrity proof and a complete starter expert ranking with zero heat, and uses the included tokenizer-derived draft ID maps. Later launches reuse the proof only while file identities match. No separate conversion/calibration command is required. To use an existing calibrated setup, provide its `profile` and `heat` paths together; they are preserved. `--check-config` checks paths without creating metadata, hashing weights or loading a GPU.

Default API: `http://127.0.0.1:30011/v1`; stop with `Ctrl+C`. A non-loopback bind requires `api_key` in your private config. [Installation](docs/INSTALL.md) · [configuration](docs/CONFIGURATION.md) · [API examples](docs/API.md).

## DeepSeek-specific additions to Strata

| Addition | Implementation |
|---|---|
| Native DeepSeek runtime | Compressed/indexed attention, KV ownership, hyper-connections, routed experts and CED prompt/tail replay. |
| Original Unsloth GGUF support | Seven-shard loading, mixed IQ3/Q2_K/Q6_K expert formats, shared main/draft vocabulary tensors and original BF16 vision files. |
| DSpark | Five-token proposals, confidence-adaptive greedy verification, incremental KV, accepted-prefix commit and selected vocabulary rows. |
| SM120 operators | Mixed-format expert dequantization, bounded grouped GEMM, short-row IQ verification, boundary fusions, indexer and CAKE attention paths. |
| CUDA Graphs | Accepted decode output/router segments, with dynamic routing and migration boundaries retained. |
| Residency and Engram | Online adaptive format groups, expert lookahead, SSD row cache, asynchronous I/O and 40-segment block-pinned memory with 100% huge-page admission. |
| APIs and images | DeepSeek message/tokenizer integration, thinking/tool encoding, image preprocessing and native vision encoder. |

Expert caching, offload, speculation, image input and compatible APIs already exist in upstream Strata. These are DeepSeek-specific implementations and integrations. [Source entry points and provenance](docs/CHANGES.md) · [third-party notices](docs/THIRD_PARTY.md) · [MIT license](LICENSE).
