# Performance and validation

## October 23: UD-Q2_K_XL with original DSpark

The current performance table is derived from the frozen 21-item cumulative baseline, source manifest SHA256 `e0370fb6588f5ee93b8a7aa10e0afb1336e9437fa859fcd08207f89b3d1d768c`. It uses Unsloth DeepSeek-V4.1-Flash-1017 UD-Q2_K_XL and the original 1017 DSpark file, on i7-14700KF, one RTX PRO 6000 Blackwell 96GB and 192GB DDR5-4800. CUDA 13.2 / driver 595.104.02, one request, no profiler.

Each length ran in a separate process with one warmup and three formal requests, exactly 512 output tokens, greedy sampling, reset initial expert/adaptation state and no prefix reuse. The input repeats/truncates one LongSWE token stream to each length. Loading and warmup are excluded. These are synthetic length tests, not natural long-document quality evaluations.

| Input | Chunk | Initial residents | Prefill tok/s, mean ± sample SD | Decode tok/s, mean ± sample SD | Decode including drain, mean |
|---:|---:|---:|---:|---:|---:|
| 8192 | 8192 | 5571 | 3619.978 ± 11.526 | 34.312 ± 0.027 | 34.239 |
| 32768 | 32768 | 4221 | 5769.141 ± 25.338 | 34.936 ± 0.010 | 34.787 |
| 131072 | 32768 | 4221 | 6037.613 ± 40.617 | 35.048 ± 0.026 | 34.990 |
| 524288 | 32768 | 4221 | 4499.192 ± 3.861 | 42.216 ± 0.038 | 42.091 |

**Highest formal prefill: 6080.745399 tok/s**, case K128/M1, 131072 tokens / 21.555252094 seconds. Decode in that same request was 35.075439 tok/s. The four groups' highest decode was 42.250250 tok/s (K512/M3; use the CSV's full precision). The About `decode ~56 tok/s` figure is from the older model and configuration described below; it was not obtained together with the 6081 prefill figure.

[All 12 formal measurements](../data/performance-20261023.csv) and [derived summary/provenance](../data/performance-summary-20261023.json) are included. Raw prompts, outputs, token arrays, logits and traces remain private.

## Configuration and timing boundaries

- Prefill = input tokens / Runtime native `prefill_ms`. The CSV separately preserves request-side prefill wall time; neither is HTTP client TTFT.
- Decode = output tokens / generation wall time including draft and target verification. Final adaptive drain is recorded separately, with a second rate that includes it.
- The 8192 chunk uses 6720 MiB workspace and 68664950784 expert bytes. The 32768 chunk uses 23823646720 workspace bytes and 51887734784 expert bytes. The combined budget stays fixed, but resident counts differ. Do not attribute the difference solely to input length.
- All runs retained original DSpark, 5-token adaptive verification, Graph mask 2, mixed-format/BF16 contracts, online adaptive residency, 40 registered host blocks and 100% huge-page coverage. Sampled links were Gen5×16; no external CUDA compute processes were observed during measurements.
- The benchmark used its saved calibrated ranking and heat. The automatic launcher instead offers a complete, uncalibrated ranking and zero heat when no calibrated pair is supplied. It has CPU/configuration coverage, not a fresh full-model or performance qualification.
- No Nsight was requested for this sweep. There is no matching new timeline/Top10 or new bottleneck attribution for these four lengths. Historical operator/request diagnostics remain separate in the research archive.

## Host-memory transfer measurements

The README's bandwidth figures come from an October 21 isolated UD-IQ3_XXS H2D test on the same i7-14700KF / RTX PRO 6000 / DDR5-4800 machine, with CUDA 13.2, driver 595.104.02 and sampled PCIe 5.0 ×16 links. All variants used the same contiguous expert arena, payload, binary, transfer order and GPU ring. The two variables were the number of CUDA registration ranges and the CPU page size; block pinning means registering parts of that arena, not allocating separate arenas.

| Registration | Actual CPU pages | H2D GB/s |
|---|---|---:|
| One range | 4 KiB | 23.952108 |
| 40 ranges | 4 KiB | 23.915125 |
| One range | 100% 2 MiB | 24.258977 |
| 40 ranges | 100% 2 MiB | **45.104124** |

Each value is the mean of two process medians, with three timed full transfers after a warmup in each process. Decimal GB/s uses actual copied bytes divided by wall time, including layer submission/synchronization; model loading and registration are excluded. The combined configuration improved measured bandwidth by **88.31%** versus one range with small pages. Neither individual change alone produced a comparable gain, and the result does not establish the GPU/IOMMU mapping mechanism.

This was a bandwidth-only run without payload byte validation or full-model requests; earlier readback anomalies remain unresolved. Background host I/O was observed and the sequence included a service-restoration interval, although repeated process medians differed by less than 0.05%. No matched prefill/decode A/B or request timeline was collected for this comparison. Faster transfers can benefit transfer-limited inference, but the 88.31% figure must not be reported as a tokens/s gain or generalized to other PCIe generations, drivers or memory layouts.

Server CPUs with AVX-512, sufficient RAM and high memory bandwidth are a separate candidate for MoE decode execution. AVX-512 support alone does not establish an efficient kernel for the actual quantization/activation contract. Such a path needs its own numerical and end-to-end checks, including NUMA placement and CPU/GPU scheduling. The supplied launcher uses GPU experts; no server-CPU decode throughput is claimed here.

## 1M capacity and multimodal function

The deployed Q2_K_XL service was configured for 1048576 context, chunk 8192, one slot and a 32768 default output cap. Actual 1M session state occupied 1000.523 MiB within the existing 1 GiB state budget. The calibrated startup retained 5571 experts and 40 blocks / 100% huge pages; short HTTP and client requests passed. This is capacity and functional evidence, not a complete 1M input or continuous 32K generation test.

Both Unsloth quantizations have existing original-GGUF full-model execution evidence, and the IQ integration included original DSpark/BF16 vision API coverage. The new release shares the accepted runtime and one launcher, with variant-specific file hashes/budgets. This publication does not restart the running service or execute a new GPU request for either model. The newly packaged automatic metadata path is tested with CPU fixtures; its uncalibrated ranking has not been run as a new full-model benchmark. Image support means text + images, not audio/video qualification.

## Historical results

The [October 19 snapshot](PERFORMANCE-20261019.md) records the earlier GSQ-RCO model and locally quantized DSpark, with all consistency warnings retained. Its approximately 56 tok/s peak decode (55.600490) and 55.564 mean belong to that historical configuration. They are not current Unsloth-model speed claims. The supported-model list now contains only the two Unsloth variants.

## Validation and limitations

- Accepted operator changes retain their recorded numerical/semantic gates and single-operation measurements. This publication copies their cumulative sources; it adds launch/metadata packaging rather than a new arithmetic or quantization method.
- All four length groups produced identical output IDs across their three formal runs. The 32K group had a maximum prefill-logit relative L2 difference of 0.00176809541 (max absolute 0.0284321308); the other groups' prefill logits matched bitwise. All recorded logits were finite. Matching outputs are not a model-quality certification.
- Historical same/cross-process output/logit drift and intermittent readback anomalies remain unresolved. Later successful checks do not erase those findings.
- General task quality, full 1M input, continuous 32K output, long-running stability and other hardware/concurrency configurations have not been accepted.
- The historical cost table and selected draft maps are retained; the adaptive starter is greedy-only. Changes in ranking, output language, cache behavior or residency can change acceptance and speed.
- Publication verification consists of isolated Linux/CUDA builds, selected native CPU checks and Python API/configuration tests. Rebuilt binaries and the automatic starter do not acquire a GPU-performance claim from those checks. See [source provenance](SOURCE_PROVENANCE.md).
