# Source provenance

Publication date: October 20, 2026, Asia/Taipei.

## Source composition

- Foundation: `yhfgyyf/strata-nvfp4` commit `67adfffe4fe969df68d599f13190fd14665816cc` plus the current DeepSeek adaptation sources.
- Cumulative experiment: the October 19 five-token adaptive/shared-Gate-Up-pair baseline. Its archived build receipt SHA256 is `c830d5f6e0b4e0bf400b31d1148b25c72f86d424c51eb88c53ff38f53adae3b3`.
- 80 cumulative source-manifest entries were verified. The inactive `moe.cpp.orig` backup is not distributed; the other 79 entries supply the cumulative source overlay.
- 141 supporting headers were verified against the same archived build receipt. Source overlays take precedence where manifests overlap.
- The 45 Python serving files match the post-fix source set used for the latest API regression.
- 314 selected source/tool files, including the 45 serving Python files, were checked against the current remote working tree with no differences. Manifest groups overlap; the published source manifest records each path once.

`data/source-manifest.json` records the exact published source hashes. It does not contain local home paths, host addresses or credentials. The repository is a new Git history containing this curated snapshot; the original source/workspace histories are retained separately.

## Packaging changes

- Add the missing standalone CMake object target for `deepseek41_indexer_mx.cu`, with SM120a compilation, and a portable experimental executable.
- Parameterize asset paths in the already-tested fixed native profile and add a local JSON launcher. The profile fixes chunk 8192, context 32768 and one slot.
- Keep the cumulative loader's 100% huge-page guard. The last functional experiment's private 99% admission relaxation is not adopted; its actual allocation had been 100%.
- Let the SHA-proof utility explicitly select the known Q2_K/Q4 draft digest.
- Add bilingual documentation, configuration examples, a frozen historical cost table and derived measurement summaries. Private logs, traces, raw model inputs/outputs and weights are excluded.

## Validation

A clean Linux/CUDA build linked all three native entry points; six selected native CPU tests and 69 Python API/configuration tests passed. The existing ggml checkout used for that build matched all 1376 files under the pinned upstream `ggml/` tree. Exact checks are in [publication-checks.json](../data/publication-checks.json). Build checks do not load a model. Existing operator and API evidence is described in [PERFORMANCE.md](PERFORMANCE.md), with its quality and configuration limits retained.

No new GPU inference, profiling, model download, OS setting change or production-service deployment is part of publication. The existing server source tree and previous deployed binaries are not replaced.
