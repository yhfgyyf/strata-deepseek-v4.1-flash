# DeepSeek additions and inherited features

This repository starts from `yhfgyyf/strata-nvfp4` commit `67adfffe4fe969df68d599f13190fd14665816cc`, then assembles the locally developed DeepSeek runtime and cumulative experiment sources. It is a standalone source snapshot, not a claim to contain every later upstream Strata change.

For the phrase “new compared with Strata”, the upstream inspection was pinned to [Niko1221/Strata f57c0ad](https://github.com/Niko1221/Strata/tree/f57c0adcb31c00f3f216550c004dac4e713a9f70), inspected on October 20, 2026 (Asia/Taipei). Its complete 4,950-path tree had no DeepSeek/DSpark/Engram-named paths, and its README described Qwen3.8-Flash-Next. This bounds the comparison; it does not prove there is no similar primitive elsewhere or establish universal novelty.

## Added model-specific paths

| Feature | Source entry points | What changed |
|---|---|---|
| Model contracts and CED | [model.cpp](../src/models/deepseek41/model.cpp), [session.cpp](../src/models/deepseek41/session.cpp), [layer.cpp](../src/models/deepseek41/layer.cpp) | DeepSeek geometry, KV ownership, compressed/indexed attention, hyper-connections and prompt/tail replay. |
| Engram | [engram.cpp](../src/models/deepseek41/engram.cpp), [engram_pipeline.cpp](../src/models/deepseek41/engram_pipeline.cpp) | SSD row access, host row cache and asynchronous preparation/upload. |
| Expert residency | [adaptive_cache.cpp](../src/models/deepseek41/adaptive_cache.cpp), [adaptive_policy.hpp](../src/models/deepseek41/adaptive_policy.hpp), [expert_pipeline.cpp](../src/models/deepseek41/expert_pipeline.cpp) | Q3_K/Q2_K format groups, bounded migration, prefill lookahead and raw-cache handling. |
| Host backing | [host_memory.hpp](../src/models/deepseek41/host_memory.hpp) | Model-specific 40-segment registration and measured 2 MiB coverage checks; bounded collapse attempts on process-owned memory. |
| DSpark | [draft.cpp](../src/models/deepseek41/draft.cpp), [weights.cpp](../src/models/deepseek41/weights.cpp), [deepseek41_runtime.cu](../src/kernels/cuda/deepseek41_runtime.cu) | Low-bit draft, distinct input/output vocabularies, fused/batched projections, incremental KV and exact accepted-prefix commit. |
| Adaptive verification | [adaptive_verification.hpp](../include/strata/models/deepseek41/adaptive_verification.hpp), [deepseek41_confidence.cu](../src/kernels/cuda/deepseek41_confidence.cu) | Fused FP32 confidence head plus sigmoid; cost-based prefix selection before verification, currently greedy-only. |
| Indexer scores | [deepseek41_indexer_mx.cu](../src/kernels/cuda/deepseek41_indexer_mx.cu), [indexer.cpp](../src/models/deepseek41/indexer.cpp) | SM120 block-scaled MMA and guarded edge-case fallback. This is indexer math, not NVFP4 storage for the main experts. |
| Expert expansion and fusion | [deepseek41_ops.cu](../src/kernels/cuda/deepseek41_ops.cu), [moe.cpp](../src/models/deepseek41/moe.cpp) | Q2_K/Q3_K paired expansion and DeepSeek short-row Gate/Up/Down fused paths. |
| Shared pair | [moe.cpp](../src/models/deepseek41/moe.cpp), [native_mmvq.cu](../src/kernels/cuda/native_mmvq.cu) | Connects DeepSeek shared Gate/Up to the existing exact pair implementation and reuses input quantization. |
| APIs and images | [deepseek41_frontend.py](../serve/deepseek41_frontend.py), [deepseek41_images.py](../serve/deepseek41_images.py), [vision.cpp](../src/models/deepseek41/vision.cpp) | DeepSeek template/tokenizer, thinking/tool encoding and native image preprocessing/encoder integration. |

## Existing ideas reused

Strata already supplies expert caching, heterogeneous offload, chunked prefill, speculative/MTP infrastructure, quantized projection primitives, image handling and OpenAI/Anthropic-compatible APIs. The project reuses these foundations. Likewise, confidence scheduling and SM120 helper primitives have upstream references; see [third-party attribution](THIRD_PARTY.md).

## What is enabled

The fixed experimental starter enables the current 5-token adaptive/shared-pair recipe. Source support for 7 tokens remains for reproducibility, but the measured 7-token configuration regressed and is not selected. The separate rejected CPU cold-expert overlap candidate is not merged into this snapshot. Broad model quality and stability remain open as described in [PERFORMANCE.md](PERFORMANCE.md).

## Publication changes

Publication adds only portable path/configuration entry points, the missing standalone CMake wiring for the already-tested MX indexer, a derived-draft proof selection, documentation and source provenance. It does not introduce a new quantization algorithm or a new model-performance result. The core cumulative arithmetic sources are preserved byte-for-byte, with the packaging exceptions recorded separately.
