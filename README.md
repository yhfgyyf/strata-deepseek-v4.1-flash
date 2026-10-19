# strata-deepseek-v4.1-flash

[简体中文](README.zh-CN.md)

DeepSeek-V4.1-Flash on a home PC with **one RTX PRO 6000 Blackwell 96GB and 192GB DDR5-4800**.

Built on [Strata](https://github.com/Niko1221/Strata) and [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4), with a native DeepSeek runtime, low-bit DSpark speculation, SSD-backed Engram, and GPU/RAM expert residency. The server supports text, images, tool calls, OpenAI Chat Completions and Responses, and Anthropic Messages.

**Experimental source snapshot.** Operator and functional checks have passed within their documented scope. Long-run output drift and intermittent weight-readback inconsistencies remain unresolved; this is not a production quality or stability certification. See [validation and limitations](docs/PERFORMANCE.md#validation-and-limitations).

## Model

The tested main checkpoint is **DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf**, from [the pinned GGUF repository](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF/tree/8270aceb26eb5590ea40f347256a1736999f551d), based on [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

- 40 layers, 384 routed experts per layer, 6 selected experts per token, hidden size 5120.
- Main routed weights use Q3_K and Q2_K. Long prefill expands selected expert weights to BF16; short decode/verification uses packed expert kernels.
- The optimized DSpark draft uses a mixed Q2_K/Q4/Q8 representation and separate input/output vocabulary maps.
- Engram tables remain in the main GGUF on SSD, with a host row cache and an asynchronous transfer pipeline.
- Optional image input uses the matching BF16 mmproj GGUF.

The SM120 MXFP4 optimization is for **indexer scoring**. It does not change the main expert weights to NVFP4. Model weights, tokenizer files and locally prepared profiles are separate assets; they are not included in this repository. See [asset requirements](docs/INSTALL.md#model-assets).

## Machine

| Component | Tested configuration |
|---|---|
| GPU | One NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96GB, SM120 |
| CPU | Intel Core i7-14700KF, 20 cores / 28 threads |
| RAM | 192GB DDR5, configured at 4800 MT/s |
| Storage | Local NVMe SSD for the GGUF and Engram reads |
| Software | Ubuntu Linux, CUDA Toolkit 13.2, NVIDIA driver 595.104.02 |
| GPU link capability | PCIe 5.0 ×16; individual request link-state traces were not collected |

## Throughput

October 19, 2026, without a profiler: one active request, one fixed **16,180-token** coding input, fixed prefill chunk **8192**, greedy decoding and natural EOS, at most 512 output tokens. Loading and warmup are excluded. Each row is the arithmetic mean of three candidate runs; all controls and individual runs are in the [measurement table](data/performance-20261019.csv).

| Configuration | Actual output tokens in the three runs | Prefill (tokens/s) | Decode (tokens/s) |
|---|---|---:|---:|
| Current cumulative experiment: 5-token adaptive DSpark + shared Gate/Up pair | 445 / 445 / 512 | **3,612.4** | **49.52** |
| Earlier adaptive DSpark experiment, before the shared-pair integration | 512 / 512 / 512 | 3,413.2 | 55.56 |

These are separate experiment processes, not a matched comparison between the two rows. The current experiment's third round drifted from its initial output, although each local A/B/A triple agreed internally. The earlier experiment had matching outputs but transient readback anomalies. Do not treat the difference between these rows as a regression caused by the paired projection, or combine their best values into one result.

The timed experiments used **5,792 resident experts** and a configured 1,048,576-token context ceiling; only the stated 16,180-token input was measured here. The later API/image regression and included experimental starter use **32K context, chunk 8192 and 5,649 resident experts** with the original frozen ranking, reserving more workspace for images. That functional configuration has no comparable throughput measurement. Rebuilt publication binaries have not been rebenchmarked. See [measurement conditions](docs/PERFORMANCE.md).

## DeepSeek-specific additions to Strata

| Addition | What this implementation does |
|---|---|
| Native DeepSeek V4.1 runtime | Implements the model's compressed/indexed attention, KV ownership, hyper-connections, routed experts and CED prompt/tail replay semantics. |
| DSpark draft and verification | Adds the three-layer low-bit draft, separate vocabulary maps, batched target verification, incremental draft KV and accepted-prefix commit without replay. |
| Confidence adaptive verification | Selects a 0–5-token draft prefix before target execution using confidence survival probabilities and a frozen cost table. The included profile is greedy-only. |
| SM120 indexer and expert kernels | Adds block-scaled MXFP4 score computation with guarded fallback, paired Q2_K/Q3_K expansion, and DeepSeek short-row Gate/Up/Down fusion. |
| Shared Gate/Up pairing | Connects the shared expert to the existing exact paired Q8 projection, reusing one Q8_1 input conversion. |
| Engram and host-memory pipeline | Adds asynchronous Engram reads, row caching, expert lookahead, format-group residency and checked 40-segment, 2 MiB host registration for this model. |
| DeepSeek API and vision integration | Adds the model's tokenizer/template, thinking controls, tool encoding, image preprocessing and the matching native image encoder. |

GPU expert caching, generic CPU/GPU offload, speculative decoding, image input and OpenAI/Anthropic APIs already exist in upstream Strata. The additions above are the **DeepSeek-specific paths and implementations**, not claims that these general ideas were invented here. The slower 7-token setting and rejected CPU cold-expert overlap experiment are not enabled. See [source entry points and upstream comparison](docs/CHANGES.md).

## Build

Linux x86-64, Python 3.10+, CMake 3.24+, a C++20 compiler, and **CUDA 13.2 / SM120** are the tested target. The architecture-specific indexer builds as SM120a. This is a source distribution with manual asset preparation, not a model-downloading installer.

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DEEPSEEK41_TESTS=ON
cmake --build build --parallel 2 --target \
  strata-deepseek41 strata-deepseek41-experimental strata-deepseek41-inspect
```

CMake fetches the pinned ggml dependency when `STRATA_GGML_DIR` is not set. See [INSTALL.md](docs/INSTALL.md) for existing dependencies, hash proofs and asset preparation boundaries.

## Run

After preparing the verified assets, copy [the experimental example](examples/rtxpro6000.experimental.json), fill in the file paths, then validate and start it:

```bash
cp examples/rtxpro6000.experimental.json examples/local.json
# Edit examples/local.json to point to your own verified local assets.
python tools/deepseek41/serve_experimental.py --config examples/local.json --check-config
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

The experimental starter fixes **prefill at 8192**, context at 32768 and one request slot. It enables 5-token greedy adaptive DSpark, the cumulative kernels and the shared pair. It requires full 2 MiB huge-page backing for all 40 registered expert segments and stops if that condition is unmet. The experimental cost table is specific to the tested hardware/residency, not a universal tuning preset. See [configuration details](docs/CONFIGURATION.md).

The generic launcher remains available for the basic, non-speculative example:

```bash
python -m serve.deepseek41 --config examples/basic.json
```

Stop a foreground server with `Ctrl+C`.

## API

The default local base URL is `http://127.0.0.1:30011/v1`; the model alias is `deepseek-v4.1`.

| Endpoint | Checked behavior |
|---|---|
| `/v1/chat/completions` | JSON/SSE text, images and standard automatic tool calls |
| `/v1/responses` | JSON/SSE text, images and automatic tool round trips |
| `/v1/messages` | JSON/SSE text, images, automatic tools and thinking enabled/disabled |

Forced/named tool selection and stateful `previous_response_id` are unsupported and return explicit errors. The latest regression covered **38 HTTP cases** (29 real generations and 9 expected errors), including Unicode, code, tool results, six image requests and a 13,037-token input. These checks are functional coverage, not a general model-quality benchmark.

```bash
curl http://127.0.0.1:30011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"Explain Python generators in three sentences."}],"max_tokens":256,"temperature":0,"reasoning_effort":"none"}'
```

For a non-loopback bind, set an `api_key`. See [API.md](docs/API.md) for limits and examples.

## License

The code retains Strata's MIT License and the original third-party notices. Model weights and model-derived assets follow their respective model licenses. See [THIRD_PARTY.md](docs/THIRD_PARTY.md) and [source provenance](docs/SOURCE_PROVENANCE.md).
