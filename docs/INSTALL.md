# Build and model setup

For guided installation and service management, run `./start.sh` from the repository root. The browser manager handles a private Python environment, the build and resumable model downloads. See [Launcher and GUI](LAUNCHER.md) or [AI setup](AI_SETUP.md). The manual equivalent remains below.

## Requirements

Tested on Linux x86-64, GCC 11.4, Python 3.10, CMake 3.27.8, CUDA Toolkit 13.2.51, driver 595.104.02 and one RTX PRO 6000 Blackwell 96GB, with i7-14700KF and 192GB DDR5-4800. Python runtime packages are in [requirements.txt](../requirements.txt).

```bash
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

CMake uses llama.cpp/ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`. To reuse that revision, pass `-DSTRATA_GGML_DIR=/path/to/llama.cpp`. Architecture-specific indexer/CAKE objects use SM120a. Prebuilt bounded GEMM cubins are included; Triton 3.6.0 is only needed to regenerate them, not to build or serve normally.

## Model assets

Download the UD models and original DSpark/mmproj from [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/60ad3651641b5c97679c807e064ab1d7a3ca1455) or [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files). File sizes and SHA256 values are listed in [models-1017.json](../data/models-1017.json). Download GSQ-RCO from [ISTA-DASLab on Hugging Face](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/cb02422c900df489cea5833e8a92171224059b3c); its file sizes and publisher SHA256 values are in [models-gsq-rco-1017.json](../data/models-gsq-rco-1017.json).

| Asset | Required files |
|---|---|
| Main model, choose one | All seven `UD-IQ3_XXS` shards, all seven `UD-Q2_K_XL` shards, **or** all four `GSQ-RCO-IQ2_XS` shards plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`. Keep names/directory together and configure shard 00001. |
| DSpark | Original `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`, SHA256 `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` |
| Image encoder, optional | Original `mmproj-BF16.gguf`, SHA256 `9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9` |
| Tokenizer | [Official tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/314113f8a7dadb388215396ba5fbee2c610a04d6/tokenizer.json), SHA256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |

Keep model files on local NVMe storage.

## One launch script for all models

```bash
cp examples/rtxpro6000.ud-q2-k-xl.json examples/local.json
# For IQ3, copy examples/rtxpro6000.ud-iq3-xxs.json instead.
# For GSQ-RCO, copy examples/rtxpro6000.gsq-rco-iq2-xs.json.
# Fill in the model, draft, tokenizer and optional mmproj paths.
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

The script starts the native engine and HTTP frontend with 1,048,576 context, chunk 8192, 32768 default maximum output and DSpark. Relative asset paths are resolved from the JSON configuration file.

Optional preflight, without writes, hashing or GPU access:

```bash
python tools/deepseek41/serve_experimental.py --config examples/local.json --check-config
```

Default endpoint: `http://127.0.0.1:30011/v1`. Stop with `Ctrl+C`. Set a private `api_key` for a non-loopback bind. Remove `mmproj` for text-only operation. GSQ-RCO is text-only even with mmproj supplied; its example retains that allocation to match the tested setup. The process requires sufficient available host RAM/VRAM and 100% huge-page backing for its registered expert blocks; it does not alter system huge-page policy or stop other processes. [Configuration details](CONFIGURATION.md).
