# Build and manual setup

## Requirements

The tested environment is Linux x86-64, GCC 11.4, Python 3.10.12, CMake 3.27.8, CUDA Toolkit 13.2.51, driver 595.104.02 and an RTX PRO 6000 Blackwell. Runtime package versions are pinned in [requirements.txt](../requirements.txt). The repository also retains generic inherited Strata sources; that does not imply this DeepSeek profile is qualified on other GPUs or operating systems.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DEEPSEEK41_TESTS=ON
cmake --build build --parallel 2 --target \
  strata-deepseek41 strata-deepseek41-experimental strata-deepseek41-inspect
```

CMake fetches llama.cpp/ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` if no checkout is supplied. For an existing matching checkout, add `-DSTRATA_GGML_DIR=/path/to/llama.cpp` to the configure command. No model is downloaded or loaded by these build commands. The SM120 indexer has a separate SM120a compilation target because of its block-scaled MMA instruction.

## Model assets

Download model files separately from [the pinned GSQ-RCO GGUF revision](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF/tree/8270aceb26eb5590ea40f347256a1736999f551d). Keep the main GGUF on a fast local SSD: the Engram reader uses it at runtime. Use the matching official DeepSeek tokenizer, not a Qwen tokenizer or a tokenizer from another DeepSeek revision.

| Asset | Expected SHA256 |
|---|---|
| `DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf` | `11f46543370256ef616b6e458b6950e148625b5f8b7545173d124445d3393c53` |
| Original `DeepSeek-V4.1-Flash-DSpark-draft.gguf` | `d5de12076cac2378cf8cdbf0853d8e0afb0a0cd06e6ef2829d0aceee69ff94a9` |
| `DeepSeek-V4.1-Flash-mmproj-BF16.gguf` | `bd8a7593fc3721fad8ef84689beeb65437cd7a75326f005d6f741521a872d09e` |
| Locally prepared `draft-q2_k-q4.gguf` used by the experimental starter | `9269e73cbd04c493d2510d3513da85429441949f6f10ed8130392ae0de2a949b` |

The **basic launcher** can run the verified main model without the DSpark derivative, vocabulary maps or expert profile. Edit [basic.json](../examples/basic.json), then make its proof:

```bash
python tools/deepseek41/hash_proof.py \
  --main /path/to/DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf \
  --output /path/to/integrity.txt
python -m serve.deepseek41 --config examples/basic.json
```

The **optimized experimental starter** additionally requires the pre-existing verified Q2_K/Q4 draft, separate input and output token-ID maps, a startup STRP expert ranking and its heat file. This source release does not include an automated conversion/calibration pipeline or redistribute those locally prepared assets. Use your verified assets from the corresponding experimental setup; merely renaming the original draft does not create the derivative. The generic launcher is available when these assets are not present, but its configuration does not reproduce the published performance table.

Create a proof for the actual files on the destination host:

```bash
python tools/deepseek41/hash_proof.py \
  --main /path/to/DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf \
  --draft /path/to/draft-q2_k-q4.gguf --quantized-draft \
  --vision /path/to/DeepSeek-V4.1-Flash-mmproj-BF16.gguf \
  --output /path/to/integrity.txt
```

This streams all supplied files and only writes a new proof when the complete hashes match. It does not modify weights. Omit `--vision` for a text-only setup. Proofs, model files and local configuration should stay outside Git.

## Start the experimental profile

```bash
cp examples/rtxpro6000.experimental.json examples/local.json
# Edit every /path/to entry. Relative paths are relative to this JSON file.
python tools/deepseek41/serve_experimental.py --config examples/local.json --check-config
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

Default endpoint: `http://127.0.0.1:30011/v1`. Stop with `Ctrl+C`. For a non-loopback bind, supply a local `api_key`. The starter has a fixed 8192 prefill chunk, 32K context and one slot; see [CONFIGURATION.md](CONFIGURATION.md).

## CPU checks without inference

```bash
python -m unittest serve.test_deepseek41 serve.test_deepseek41_images \
  serve.test_responses \
  tools.deepseek41.test_release_config
cmake --build build --parallel 2 --target \
  deepseek41_contract_test deepseek41_engram_test deepseek41_sampling_test \
  deepseek41_integrity_test deepseek41_residency_test deepseek41_prefill_chunk_test
CUDA_VISIBLE_DEVICES= ctest --test-dir build --output-on-failure \
  -R '^deepseek41_(contract|engram|sampling|integrity|residency|prefill_chunk)_test$'
```

The CMake tree also contains GPU and model-dependent tests. Do not interpret the selected CPU checks as execution of those tests or as a new model-quality benchmark.
