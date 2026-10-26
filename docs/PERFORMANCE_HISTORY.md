# Historical performance measurements

[简体中文](PERFORMANCE_HISTORY.zh-CN.md)

These UD-Q2_K_XL measurements used earlier runtime revisions and are retained with their original input, capacity and timing conditions. They do not describe the current code. See the [README](../README.md#throughput) for the current GSQ-RCO measurements.

Archived source: [32f09f0 / README.md](https://github.com/yhfgyyf/strata-deepseek-v4.1-flash/blob/32f09f09f368d063deff561f12ea3e009347446d/README.md).

**Maximum measured prefill: 6,080.745 tokens/s**, at 131,072 input tokens, chunk 32768, with UD-Q2_K_XL and original DSpark. This is one formal, profiler-free run; the corresponding three-run mean is **6,037.613 tokens/s**.

**Maximum measured decode: 77.483 tokens/s**, with UD-Q2_K_XL, original DSpark, 16,180 input tokens, chunk 8192 and exactly 512 output tokens on October 25. This profiler-free run includes generation and final adaptive drain; the three-run mean is **76.760 tokens/s**. The prefill and decode peaks come from separate runs and input lengths.

October 23 measurements below use one request, one warmup plus three formal runs per length, greedy decoding and exactly 512 output tokens. The input is a repeated/truncated LongSWE token stream for length testing. Rates are means ± sample standard deviations.

| Input tokens | Chunk | Resident experts | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 5571 | 3,619.978 ± 11.526 | 34.312 ± 0.027 |
| 32,768 | 32,768 | 4221 | 5,769.141 ± 25.338 | 34.936 ± 0.010 |
| 131,072 | 32,768 | 4221 | **6,037.613 ± 40.617** | 35.048 ± 0.026 |
| 524,288 | 32,768 | 4221 | 4,499.192 ± 3.861 | 42.216 ± 0.038 |

Prefill uses native prompt time; decode uses DSpark generation time. The measurements use a calibrated expert ranking and heat, with chunk sizes and resident expert counts shown above.

The October 25 cumulative runtime uses the same machine, one warmup plus three formal LongSWE runs, greedy decoding, 5571 startup resident experts and online Maya residency adaptation. Each request starts a new session without conversation-prefix reuse; expert heat carries across runs.

| Input tokens | Output tokens | Chunk | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 16,180 | 512 | 8,192 | 4,231.216 ± 5.476 | **76.760 ± 1.054** |

These rates use native prefill time and generation plus final adaptive drain for decode. They describe the measured workload and its adapted expert cache.

## Historical memory-transfer comparison

**Block-pinned memory + huge pages on PCIe 5.0** substantially improves expert-transfer bandwidth on the tested machine. Expert RAM uses 40 CUDA-registered blocks with 100% 2 MiB huge-page coverage. An isolated H2D comparison measured **23.952 → 45.104 GB/s (+88.31%)** versus one registration with 4 KiB pages. Faster expert uploads reduce transfer cost and can improve both prefill and decode speed when transfers limit execution. This measured gain is specific to the tested PCIe 5.0 ×16 setup.
