<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/strata-lockup-dark.svg">
    <img src="docs/assets/readme/strata-lockup-light.svg" alt="Strata" width="220" align="middle">
  </picture>
  &nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/whale-girl.png" alt="DeepSeek Whale-chan" width="156" align="middle">
</p>

[简体中文](README.md)

The goal is practical use in agent harnesses such as [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness): make inference as fast as possible while maintaining a useful level of model reasoning quality.

DeepSeek-V4.1-Flash on a home PC: **Intel Core i7-14700KF, one RTX PRO 6000 Blackwell 96GB, and 192GB DDR5-4800**.

Built on [Strata](https://github.com/Niko1221/Strata) and [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4), with a native DeepSeek runtime, SSD-backed Engram, GPU/RAM expert residency, and **DSpark speculative decoding**. The supported models use the same launch script and their original GGUF files: **no offline weight conversion, shard merging, or extra quantization is required**.

The service supports **1M context capacity (1,048,576 tokens)**, **multimodal text + image input** with the Unsloth models, tool calls, OpenAI Chat Completions / Responses, and Anthropic Messages. GSQ-RCO currently supports text input. The starter uses one request slot, chunk 8192, and a default maximum output of 32768 tokens. For long contexts, the setup can be adjusted to **chunk 32K (32768 tokens)** and **a maximum output of 128K (131072 tokens)**.

**Recommended GPUs: one RTX PRO 6000 Blackwell 96GB, or two RTX 5090 cards.**

## Supported models

**GSQ-RCO-IQ2_XS** is from **ISTA-DASLab / DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF**, with a separate IQ4_NL Engram file. The UD variants are from **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**.

| Main model | Hugging Face | ModelScope | Launch configuration |
|---|---|---|---|
| **GSQ-RCO-IQ2_XS** | [GSQ-RCO-IQ2_XS + IQ4_NL Engram](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/cb02422c900df489cea5833e8a92171224059b3c) | [GSQ-RCO-IQ2_XS + IQ4_NL Engram](https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | [rtxpro6000.gsq-rco-iq2-xs.json](examples/rtxpro6000.gsq-rco-iq2-xs.json) |
| **UD-IQ3_XXS** | [UD-IQ3_XXS](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [Model files — select UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [Model files — select UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |

| Model | Total weights (GB) | Main model excluding Engram (GB) | Engram (GB) |
|---|---:|---:|---:|
| **GSQ-RCO-IQ2_XS** | **265.821** | 155.226 | 110.595 |
| **UD-IQ3_XXS** | **394.737** | 233.452 | 161.285 |
| **UD-Q2_K_XL** | **334.794** | 199.622 | 135.172 |

Sizes are decimal GB of the main GGUF files and Engram file, excluding DSpark and mmproj. GSQ-RCO uses four main shards plus the external Engram file; keep all five together.

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
| GPU link | PCIe 5.0 ×16 observed during throughput measurements |

## Throughput

**Maximum measured prefill: 6,261.998 tokens/s**, at 32,768 input tokens, chunk 32768, with GSQ-RCO-IQ2_XS and original DSpark. This is one formal, profiler-free run; the corresponding three-run mean is **6,245.142 tokens/s**.

**Maximum measured decode: 114.472 tokens/s**, with GSQ-RCO-IQ2_XS, original DSpark, 524,288 input tokens, chunk 32768 and exactly 512 output tokens. This profiler-free run includes generation and final adaptive drain/synchronization; the three-run mean is **112.323 tokens/s**. The peaks are selected independently from the formal runs.

The GSQ-RCO-IQ2_XS measurements keep the same machine, original DSpark, 1M context capacity and cumulative runtime. Each input length uses a fresh process, one warmup and three formal runs, a new session per request and no prefix reuse. Greedy decoding ignores EOS to produce exactly 512 output tokens. Maya expert heat carries across the four requests within each process.

| Input tokens | Output tokens | Chunk | Startup resident experts | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 512 | 32,768 | 5704 | 6,245.142 ± 23.250 | 104.548 ± 2.675 |
| 131,072 | 512 | 32,768 | 5704 | 6,186.661 ± 48.766 | 107.008 ± 2.448 |
| 524,288 | 512 | 32,768 | 5704 | 4,541.946 ± 7.958 | 112.323 ± 2.758 |

Rates are means ± sample standard deviations. Prefill uses native prompt time; decode includes generation and final adaptive drain/synchronization. The chunk-32768 workspace reserves 22.1875 GiB, with automatic expert budgeting, 40 block registrations and 100% huge-page coverage. Earlier results are preserved in the [performance history](docs/PERFORMANCE_HISTORY.md).

## Expert profile and memory transfers

The calibrated startup profile was refined using expert-routing statistics from a broader mix of coding, agent and general-reasoning prompts: DeepSWE, SWE-bench Pro and Verified, R2E-Gym, LongSWE, LiveCodeBench, BigCodeBench, ProgramBench, Terminal-Bench, BFCL and HLE. These observations inform the initial expert ranking and heat, followed by online adaptive residency.

The current runtime uses **block-pinned memory with huge pages** for expert RAM: 40 CUDA-registered blocks with 100% 2 MiB huge-page coverage. On PCIe 5.0 ×16, the isolated CPU→GPU (H2D) transfer comparison improved from **23.952 to 45.104 GB/s (+88.31%)** versus a single registration with 4 KiB pages. See the [performance history](docs/PERFORMANCE_HISTORY.md).

**Reaching this transfer rate requires a PCIe 5.0 GPU and a CPU/motherboard link operating at PCIe 5.0 ×16.** If a PCIe 4.0 or older link is already saturated, block pinning and huge pages provide little or no additional transfer throughput.

With this substantial increase in PCIe bandwidth, the project defaults to **GPU-only prefill and decode**, with no model-weight computation on the CPU. On the RTX PRO 6000 system described above, **GPU power draw is around 500 W** during inference.

On a server with **AVX-512, ample RAM and high memory bandwidth**, it is also worth testing **CPU execution of MoE experts during decode**, while keeping attention and other suitable work on the GPU.

## Build and launch

The tested target is Linux x86-64, Python 3.10+, a C++20 compiler, CMake 3.24+, and CUDA 13.2 / SM120. The indexer and CAKE objects use SM120a. Existing CUDA/huge-page prerequisites remain necessary; the launcher does not change system settings.

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
./start.sh
```

The local management page opens at **http://127.0.0.1:30010**. Choose a model and storage directory, click **Use model paths** to save its paths, inspect the download plan, then select **Set up and start**. It creates a private Python environment, builds the engine, resumes missing downloads and verifies SHA256. Existing files can be selected in the same page. Start/stop, progress, logs and the chat link remain available there.

For a terminal or AI agent:

```bash
./start.sh configure --model GSQ-RCO-IQ2_XS --data-dir /path/to/models
./start.sh plan --json
./start.sh launch --yes --json
# Later: ./start.sh start | status | stop | logs
```

Downloads default to ModelScope; `--source huggingface` and `--source hf-mirror` are explicit alternatives with the same required hashes. The manager stays on localhost, preserves interrupted downloads and controls only its own service. The API and existing web chat use port **30011** by default. The original `serve_experimental.py --config FILE` entry remains supported.

Supports **OpenAI Chat Completions / Responses** and **Anthropic Messages** compatible APIs.

[Launcher and GUI](docs/LAUNCHER.md) · [AI setup](docs/AI_SETUP.md) · [manual installation](docs/INSTALL.md) · [configuration](docs/CONFIGURATION.md) · [API examples](docs/API.md).

## DeepSeek-specific additions to Strata

| Addition | Implementation |
|---|---|
| Native DeepSeek runtime | Compressed/indexed attention, KV ownership, hyper-connections, routed experts and CED prompt/tail replay. |
| Original GSQ-RCO GGUF support | Four main shards, external IQ4_NL Engram, mixed IQ2/IQ3 experts, Q5_K projections and automatic GPU expert budgeting. |
| Original Unsloth GGUF support | Seven-shard loading, mixed IQ3/Q2_K/Q6_K expert formats, shared main/draft vocabulary tensors and original BF16 vision files. |
| DSpark | Five-token proposals, confidence-adaptive greedy verification, incremental KV, accepted-prefix commit and selected vocabulary rows. |
| SM120 operators | Mixed-format expert dequantization, bounded grouped GEMM, short-row IQ/MMVQ verification, grouped projections, boundary fusions, indexer and CAKE attention paths. |
| CUDA Graphs | Accepted decode output/router segments, shared-expert early submission and compact route handoff, with dynamic routing and migration boundaries retained. |
| Residency and Engram | Online Maya adaptation within format groups, expert lookahead, SSD row cache, asynchronous I/O and 40-segment block-pinned memory with 100% huge-page admission. |
| APIs and images | DeepSeek message/tokenizer integration, thinking/tool encoding, automatic text-prefix checkpoints, cancellation recovery, image preprocessing and native vision encoder. |
| Setup and management | Shared CLI/browser configuration, resumable verified downloads, local build, identified service processes, health checks, start/stop and logs. |

Expert caching, offload, speculation, image input and compatible APIs already exist in upstream Strata. These are DeepSeek-specific implementations and integrations. [third-party notices](docs/THIRD_PARTY.md) · [MIT license](LICENSE).
