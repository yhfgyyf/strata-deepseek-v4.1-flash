# Historical snapshot: October 19, 2026

The following text records the previous release. Its uses of “current” refer to that date and its model/configuration. For the supported Unsloth models and current release, see [PERFORMANCE.md](PERFORMANCE.md).

# Performance and validation

## Measurement conditions

The two published tables derive from October 19, 2026 measurements on one RTX PRO 6000 Blackwell 96GB, i7-14700KF and 192GB DDR5-4800. Main model: DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf; draft: the tested Q2_K/Q4 derivative. CUDA 13.2, driver 595.104.02.

Both studies used one 16,180-token coding input, concurrency 1, fixed chunk 8192, greedy sampling / seed 1234, natural EOS with a maximum of 512, and 5,792 GPU-resident experts. The configured context ceiling was 1,048,576; the table does not validate that full context length. Cold-prefix requests reset the same initial expert membership, slots, heat and raw-cache state. Model loading and per-arm warmup were excluded. Each study used three A/B/A triples with interleaved controls.

Main MoE fusion mode 2, fast scores and expert expansion, format-group adaptive residency, 40 registered host segments and 100% 2 MiB page coverage were retained. The current shared-pair study enabled 5-token adaptive verification in both arms and toggled only shared Gate/Up pairing. The earlier confidence study toggled adaptive verification while keeping its other settings fixed.

## Current cumulative experiment

| Round | A0 decode | B decode | A1 decode | B prefill | Output tokens, each arm |
|---|---:|---:|---:|---:|---:|
| 1 | 49.169 | 49.208 | 49.218 | 3611.406 | 445 |
| 2 | 49.195 | 48.881 | 48.859 | 3610.493 | 445 |
| 3 | 49.833 | 50.472 | 50.288 | 3615.252 | 512 |

Rates are tokens/s. Candidate B means: **3612.384 prefill / 49.520 decode**. All three local A/B/A triples matched outputs, prefill logits and acceptance trajectories internally. Round 3 differed from the initial reference, and a separate trace process differed again. No warmup was included in these means; no failed-consistency round was silently removed.

The paired shared-projection decode change was **+0.184%**, sample standard deviation **0.576 percentage points**: no material whole-request speedup was established. Its isolated projection benefit and numerical checks justified experimental adoption. This is not a claim that the current implementation is generally faster than the earlier row below.

## Earlier adaptive verification experiment

Three candidate requests all produced 512 tokens; candidate means were **3413.186 prefill / 55.564 decode**. The six surrounding controls averaged 3432.222 / 52.080. The mean paired decode improvement was **6.690%**, while prefill changed −0.556%. This result belongs to the earlier process/build, before shared-pair integration. All nine timed requests matched outputs, but transient immutable-weight readback anomalies remained.

Both studies' 18 timed requests, including all controls, are in [performance-20261019.csv](../data/performance-20261019.csv); [performance-summary.json](../data/performance-summary.json) records candidate means and hashes of the private source summaries. Only derived measurements are distributed, not prompts, output text, token arrays, logits or raw traces.

## Timing definitions

- Prefill throughput: actual input token count divided by native prefill time.
- Decode throughput: actual emitted output count divided by generate wall time, excluding the final drain. The CSV separately includes the rate with drain.
- Request time: prefill wall time plus generation time. It is not an HTTP client TTFT or complete cold-start duration.
- These are profiler-free experiment measurements. Separate diagnostic traces are not used as throughput samples.

The shared prompt does not make different processes or different output lengths identical controls. Speculative acceptance, expert access and platform drift can change request speed. No general quality benchmark or representative production workload suite was completed.

## Functional regression

The later API regression used fixed chunk 8192, context 32768, one slot, 6720 MiB work and 1024 MiB state, with a 72,596,914,176-byte GPU expert budget. The original frozen profile admitted 5649 experts. This leaves room for image processing and differs from the 5792-resident timing configuration.

All 38 HTTP cases met expectations: 29 real generations and nine expected HTTP 400 errors. Coverage included JSON and SSE on all three APIs, Unicode/code output, automatic tools and tool results, two image fixtures across all three APIs, thinking controls, malformed-input handling and a 13,037-token prompt crossing the 8192 chunk boundary. The supporting CPU checks comprised 75 tests in the original environment. Packaging checks are recorded separately in [source provenance](SOURCE_PROVENANCE.md).

Actual memory coverage in that regression was 100% huge pages over 40 segments. Its isolated loader allowed a 99% minimum, but the published cumulative source retains the stricter 100% rule. The included portable starter removes machine-specific paths and preserves the arithmetic/settings of that regression, with the stricter admission rule. Fresh publication binaries have only build/CPU validation unless explicitly noted otherwise; they were not used for another full-model or performance run.

## Validation and limitations

- Real-weight operator checks passed for the added scores, expansion, fusion, confidence and shared-pair paths within their recorded input/shape tolerances. The shared pair matched its original path bitwise.
- Some same-process paired generation checks passed, but long-run/cross-process output or logits drift and intermittent weight-readback anomalies remain unresolved. A later successful read does not erase an earlier anomaly.
- Functional API success does not establish broad model quality, production stability or equivalence to a high-precision checkpoint.
- Adaptive verification in the included starter is greedy-only. The frozen cost table came from the tested 5792-resident configuration; its transfer to the 5649-resident functional profile has not been performance-qualified.
- 7-token speculation was slower in its matched trials. CPU cold-expert overlap regressed decode and changed outputs. Neither is enabled by the supplied experimental starter.
- The full private event timelines, phase-specific Top10 tables and raw traces remain in the local research archive. They are not part of this source distribution. No new profiling was performed for publication.
