# strata-deepseek-v4.1-flash

[English](README.md)

在家用电脑运行 DeepSeek-V4.1-Flash：**Intel Core i7-14700KF、单张 RTX PRO 6000 Blackwell 96GB、192GB DDR5-4800**。

基于 [Strata](https://github.com/Niko1221/Strata) 与 [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4)，提供原生 DeepSeek 推理、SSD Engram、GPU/内存专家缓存及 **DSpark 推测解码**。两种 Unsloth 模型共用一个启动脚本，直接读取下载的原始 GGUF，**无需离线转换权重、合并分片或额外量化**。

支持 **1M 上下文容量（1,048,576 tokens）**、**文本＋图片多模态输入**、工具调用，以及 OpenAI Chat Completions / Responses 和 Anthropic Messages。启动配置为单请求槽位、chunk 8192、默认最大输出 32768 tokens。已验证 1M 会话分配及短 API 请求，长输入测速到 512K；完整 1M 输入和连续 32K 输出尚未压测。图片支持不代表已适配音频或视频。

本项目是实验性源码发布。算子与功能检查有明确覆盖范围；整模型质量、重复性和长期稳定性仍有未决问题，见[验证与限制](docs/PERFORMANCE.md#validation-and-limitations)。

## 适配模型

以下两种均来自 **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**。下载所选目录全部 7 个分片，配置指向 `00001-of-00007.gguf`；保留原文件名和同级目录，包括用于 Engram 的大尺寸末分片。

| 主模型 | Hugging Face | ModelScope（魔搭） | 启动配置 |
|---|---|---|---|
| **UD-IQ3_XXS** | [UD-IQ3_XXS 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [文件列表，选择 UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [文件列表，选择 UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |

两种模型共用仓库内的 `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`；图片输入另需 `mmproj-BF16.gguf`：[Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files)。DSpark 保持发布文件中的 MXFP4/Q8/BF16 混合格式，文件名 Q8_0 不代表所有张量都是 Q8。运行时解量化与词表行选择不会改写下载文件。

2026-10-23 已核对两站的 14 个主模型分片及 2 个辅助模型文件，尺寸与 SHA256 全部一致；加载器接受的内容身份见 [models-1017.json](data/models-1017.json)。还需下载匹配的[官方 tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/tokenizer.json)。权重和 tokenizer 不包含在代码仓库中。[文件准备说明](docs/INSTALL.md#model-assets)。

## 测试机器

| 项目 | 配置 |
|---|---|
| GPU | 单张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96GB，SM120 |
| CPU | Intel Core i7-14700KF，20 核 / 28 线程 |
| 内存 | 192GB DDR5，配置为 4800 MT/s |
| 存储 | 本地 NVMe SSD，承载 GGUF 和 Engram 读取 |
| 软件 | Ubuntu Linux、CUDA Toolkit 13.2、NVIDIA 驱动 595.104.02 |
| 显卡链路 | 10 月 23 日测试期间采样为 PCIe 5.0 ×16 |

## 吞吐

**已测最大 prefill：6,080.745 tokens/s**，使用 UD-Q2_K_XL、原格式 DSpark、131,072 输入、chunk 32768。这是无 profiler 的单次正式结果；同配置三次均值为 **6,037.613 tokens/s**。

以下为 2026-10-23 测试：单请求，每档一次预热＋三次正式测量，贪心解码，固定完整 512 输出。输入由同一 LongSWE token 流重复拼接和截取，用于长度压测，不是自然长文质量评测。数值为均值 ± 样本标准差。

| 输入 tokens | Chunk | 常驻专家数 | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 5571 | 3,619.978 ± 11.526 | 34.312 ± 0.027 |
| 32,768 | 32,768 | 4221 | 5,769.141 ± 25.338 | 34.936 ± 0.010 |
| 131,072 | 32,768 | 4221 | **6,037.613 ± 40.617** | 35.048 ± 0.026 |
| 524,288 | 32,768 | 4221 | 4,499.192 ± 3.861 | 42.216 ± 0.038 |

Prefill 使用引擎原生 prompt 时间；decode 包含 DSpark 生成过程，最终 drain 单独记录，不冒充客户端 TTFT。32K chunk 需要更大工作区和更少常驻专家，与默认 8K 启动配置不同。测速采用既有校准排序和热度；脚本自动生成的是未校准的通用排序，尚未单独测速。本次重新构建的发布二进制也没有追加 GPU 性能测试。[12 条正式记录](data/performance-20261023.csv) · [条件、历史结果和限制](docs/PERFORMANCE.md)。

## 构建与启动

测试目标为 Linux x86-64、Python 3.10+、C++20、CMake 3.24+、CUDA 13.2 / SM120；indexer 与 CAKE 对象使用 SM120a。需事先准备 CUDA 和大页运行环境，脚本不会修改系统设置。

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
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

选择一个模型配置，填写 model/draft/tokenizer/mmproj 路径，然后用**同一脚本启动**：

```bash
cp examples/rtxpro6000.ud-q2-k-xl.json examples/local.json
# IQ3 版本改为复制 examples/rtxpro6000.ud-iq3-xxs.json。
# 编辑 /path/to 路径；纯文本可删除 mmproj 字段。
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

首次启动自动完整 SHA 校验、写入校验凭据与全专家通用排序/零初始热度，并使用仓库附带的 tokenizer 派生草稿 ID 映射；后续仅在文件身份不变时复用凭据。不需要额外权重转换或校准命令。已有校准方案可同时传入 `profile` 和 `heat`，脚本会保留这些文件。`--check-config` 只核对路径，不写元数据、不读取全量权重、不加载 GPU。

默认 API 为 `http://127.0.0.1:30011/v1`，`Ctrl+C` 停止。监听非本机地址时须在私有配置中设置 `api_key`。[安装说明](docs/INSTALL.md) · [配置说明](docs/CONFIGURATION.md) · [API 示例](docs/API.md)。

## 相对上游 Strata 的 DeepSeek 专项适配

| 适配 | 实现内容 |
|---|---|
| 原生 DeepSeek runtime | 压缩/索引注意力、KV 所有权、超连接、路由专家及 CED prompt/tail replay。 |
| 原始 Unsloth GGUF | 七分片读取、IQ3/Q2_K/Q6_K 混合专家格式、主模型与草稿共享词表张量、原 BF16 视觉文件。 |
| DSpark | 最多 5-token 草稿、置信度自适应贪心验证、增量 KV、接受前缀提交、词表行选择。 |
| SM120 算子 | 混合格式解量化、有界 grouped GEMM、IQ 短窗口验证、边界融合、indexer 和 CAKE 注意力。 |
| CUDA Graph | 已采用的 decode 输出/router 分段，保留动态路由与迁移边界。 |
| 专家与 Engram | 在线自适应格式分组、专家预取、SSD 行缓存、异步 I/O，以及 40 段 block-pinned / 100% 大页准入。 |
| API 与图片 | DeepSeek 消息/tokenizer、thinking/工具编码、图片预处理及原生视觉编码器。 |

专家缓存、异构卸载、推测解码、图片输入和兼容 API 已存在于上游 Strata；此处说明的是 DeepSeek 专项实现与接入。[源码入口与来源](docs/CHANGES.md) · [第三方声明](docs/THIRD_PARTY.md) · [MIT 许可证](LICENSE)。
