# Source provenance

Release update: October 23, 2026, Asia/Taipei. Previous published commit: `159aba79f139b368fccc64a636799acb4958cb84`.

## Cumulative source

The foundation remains `yhfgyyf/strata-nvfp4` commit `67adfffe4fe969df68d599f13190fd14665816cc`. This release overlays the frozen, 576-file October 23 cumulative snapshot, manifest SHA256 `e0370fb6588f5ee93b8a7aa10e0afb1336e9437fa859fcd08207f89b3d1d768c`, and the subsequently deployed 1M service entry. The source includes the accepted mixed-format/Unsloth/DSpark adaptations and 21-item experiment baseline. The original research history and evidence remain in their own locations.

`data/source-manifest.json` identifies each distributed program-source hash and the files changed for packaging. All numerical runtime/kernel sources match the frozen snapshot, except the embedded bounded GEMM header whose non-allocated debug path strings are sanitized. Every non-debug ELF section, including executable code and constants, is byte-identical to the tested cubins. The PTX change is confined to `.file` paths. Original and packaged cubin hashes are retained in [bounded_gemm_generated.json](../src/kernels/cuda/bounded_gemm_generated.json).

## Packaging changes

- Omit private, external-path experiment targets from CMake; keep the actual native runtime and operator-test targets.
- Carry forward the deployed context=1048576, chunk=8192, IQ short-batch service entry. The frontend adds portable import paths and sets the adopted 40-block, 100%-huge-page and Graph-mask-2 environment.
- Supply two explicit model configurations with variant-specific expert byte budgets and original 1017 draft/vision hashes.
- Prepare full model-hash proofs and a clearly identified uncalibrated expert ranking/zero heat when calibrated assets are absent. Existing calibrated paths remain supported. There is no offline model conversion, shard merge or requantization.
- Include two tokenizer-only ID-selection maps, with checked input/output subset semantics and hashes; no prompt or generated token streams are included.
- Refresh bilingual documentation and derived measurements, preserving the prior performance report and its unresolved findings. FlashInfer CAKE source notices and exact source hashes are retained.

## Verification scope

A fresh isolated Linux/CUDA build linked the three native entry points; all 6 selected native CPU checks and 78 Python API/configuration tests passed. Actual asset-path preflight passed for both models, with no GPU loading. Source parity, 16 model-mirror identities, artifact content and 85 local documentation links are recorded in [publication-checks.json](../data/publication-checks.json). The previous release's receipt is preserved as [publication-checks-20261020.json](../data/publication-checks-20261020.json).

Both 1017 variants use the same original-GGUF runtime. Previously recorded GPU operator/full-model, vision and capacity evidence is reused within its stated scope; no new GPU inference, profile collection or service restart is part of publication. The automatic uncalibrated metadata path has CPU tests and path validation, not a fresh full-model execution or throughput result. The 6080.745 prefill peak and 512K input measurement are from the separate October 23 benchmark, not rebuilt publication binaries.

No weights, private prompts/outputs, raw traces, calibrated heat snapshots, credentials or machine-private absolute paths are published. Repository visibility remains private. The existing running service and its original directories are not replaced.
