# Licenses and sources

- **Strata** — [Niko1221/Strata](https://github.com/Niko1221/Strata), MIT. Original notices remain in [LICENSE](../LICENSE).
- **NVFP4 source foundation** — [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) and [yhfgyyf/strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4), retaining the original notices. Generic inherited kernels remain in the source tree even though the tested DeepSeek main model is GGUF Q3_K/Q2_K.
- **llama.cpp/ggml** — [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d), MIT. CMake pins this source revision. The bundled header license is in [third_party/ggml/LICENSE](../third_party/ggml/LICENSE).
- **DeepSeek message encoding/image reference** — model-supplied reference code, MIT. The retained notice is [serve/deepseek41_LICENSE](../serve/deepseek41_LICENSE).
- **SM120 helper primitives** — adapted from [vllm-project/DeepGEMM a7a9277](https://github.com/vllm-project/DeepGEMM/tree/a7a927731b109fdfec770df7c2a5be5d7e2216b5), MIT. The full notice and adaptation description are retained in [deepseek41_mx_support.cuh](../src/kernels/cuda/deepseek41_mx_support.cuh).
- **Marlin/SGLang-derived inherited primitives** — Apache-2.0, with notices in [nvfp4_marlin_primitives.cuh](../src/kernels/cuda/nvfp4_marlin_primitives.cuh) and [LICENSES/Apache-2.0.txt](../LICENSES/Apache-2.0.txt).
- **DSpark/confidence reference comparisons** — [vLLM dde9aca](https://github.com/vllm-project/vllm/tree/dde9aca41cd3ccceb1ffa902933b6678a09c83ce) and [SGLang 0025f93](https://github.com/sgl-project/sglang/tree/0025f9394421451aa09ec192a4a6e7928e14736b). They are reference implementations, not runtime dependencies or full-framework performance baselines.
- **Outfit web font** — SIL Open Font License, retained in [serve/web/fonts/OFL.txt](../serve/web/fonts/OFL.txt).

DeepSeek and GGUF model weights follow their model-card licenses. This repository does not redistribute weights, tokenizer/model assets, private prompts/outputs, token arrays or raw traces. The code license does not replace model licenses.
