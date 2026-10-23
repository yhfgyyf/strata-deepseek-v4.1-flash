# strata-deepseek-v4.1-flash

[简体中文](README.zh-CN.md)

DeepSeek-V4.1-Flash on a home PC: **Intel Core i7-14700KF, one RTX PRO 6000 Blackwell 96GB, and 192GB DDR5-4800**.

Built on [Strata](https://github.com/Niko1221/Strata) and [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4), with a native DeepSeek runtime, SSD-backed Engram, GPU/RAM expert residency, and **DSpark speculative decoding**. Both supported Unsloth models use the same launch script and their original GGUF files: **no offline weight conversion, shard merging, or extra quantization is required**.

The service supports **1M context capacity (1,048,576 tokens)**, **multimodal text + image input**, tool calls, OpenAI Chat Completions / Responses, and Anthropic Messages. The starter uses one request slot, chunk 8192, and a default maximum output of 32768 tokens.

## Supported models

Both variants are from **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**.

| Main model | Hugging Face | ModelScope | Launch configuration |
|---|---|---|---|
| **UD-IQ3_XXS** | [UD-IQ3_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [Model files — select UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [Model files — select UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |

| Model | Total weights (GB) | Main model excluding Engram (GB) | Engram (GB) |
|---|---:|---:|---:|
| **UD-IQ3_XXS** | **394.737** | 233.452 | 161.285 |
| **UD-Q2_K_XL** | **334.794** | 199.622 | 135.172 |

- **DSpark**: [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files)
- **mmproj**: [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/mmproj-BF16.gguf) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files)

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

October 23 measurements below use one request, one warmup plus three formal runs per length, greedy decoding and exactly 512 output tokens. The input is a repeated/truncated LongSWE token stream for length testing. Rates are means ± sample standard deviations.

| Input tokens | Chunk | Resident experts | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 5571 | 3,619.978 ± 11.526 | 34.312 ± 0.027 |
| 32,768 | 32,768 | 4221 | 5,769.141 ± 25.338 | 34.936 ± 0.010 |
| 131,072 | 32,768 | 4221 | **6,037.613 ± 40.617** | 35.048 ± 0.026 |
| 524,288 | 32,768 | 4221 | 4,499.192 ± 3.861 | 42.216 ± 0.038 |

Prefill uses native prompt time; decode uses DSpark generation time. The measurements use a calibrated expert ranking and heat, with chunk sizes and resident expert counts shown above.

## Expert profile and memory transfers

The calibrated startup profile was refined using expert-routing statistics from a broader mix of coding, agent and general-reasoning prompts: DeepSWE, SWE-bench Pro and Verified, R2E-Gym, LongSWE, LiveCodeBench, BigCodeBench, ProgramBench, Terminal-Bench, BFCL and HLE. These observations inform the initial expert ranking and heat, followed by online adaptive residency.

**Block-pinned memory + huge pages on PCIe 5.0** substantially improves expert-transfer bandwidth on the tested machine. Expert RAM uses 40 CUDA-registered blocks with 100% 2 MiB huge-page coverage. An isolated H2D comparison measured **23.952 → 45.104 GB/s (+88.31%)** versus one registration with 4 KiB pages. Faster expert uploads reduce transfer cost and can improve both prefill and decode speed when transfers limit execution. This measured gain is specific to the tested PCIe 5.0 ×16 setup.

On a server with **AVX-512, ample RAM and high memory bandwidth**, it is also worth testing **CPU execution of MoE experts during decode**, while keeping attention and other suitable work on the GPU.

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

Supports **OpenAI Chat Completions / Responses** and **Anthropic Messages** compatible APIs.

[Installation](docs/INSTALL.md) · [configuration](docs/CONFIGURATION.md) · [API examples](docs/API.md).

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

Expert caching, offload, speculation, image input and compatible APIs already exist in upstream Strata. These are DeepSeek-specific implementations and integrations. [third-party notices](docs/THIRD_PARTY.md) · [MIT license](LICENSE).
