# strata-deepseek-v4.1-flash

[English](README.md)

在家用电脑运行 DeepSeek-V4.1-Flash：**Intel Core i7-14700KF、单张 RTX PRO 6000 Blackwell 96GB、192GB DDR5-4800**。

基于 [Strata](https://github.com/Niko1221/Strata) 与 [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4)，提供原生 DeepSeek 推理、SSD Engram、GPU/内存专家缓存及 **DSpark 推测解码**。适配模型共用一个启动脚本，直接读取下载的原始 GGUF，**无需离线转换权重、合并分片或额外量化**。

支持 **1M 上下文容量（1,048,576 tokens）**、Unsloth 模型的**文本＋图片多模态输入**、工具调用，以及 OpenAI Chat Completions / Responses 和 Anthropic Messages。GSQ-RCO 当前支持文本输入。启动配置为单请求槽位、chunk 8192、默认最大输出 32768 tokens。

## 适配模型

UD 两种版本来自 **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**；**GSQ-RCO-IQ2_XS** 来自 **ISTA-DASLab / DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF**，搭配独立的 IQ4_NL Engram 文件。

| 主模型 | Hugging Face | ModelScope（魔搭） | 启动配置 |
|---|---|---|---|
| **UD-IQ3_XXS** | [UD-IQ3_XXS 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [文件列表，选择 UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [文件列表，选择 UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |
| **GSQ-RCO-IQ2_XS** | [GSQ-RCO-IQ2_XS＋IQ4_NL Engram](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/cb02422c900df489cea5833e8a92171224059b3c) | — | [rtxpro6000.gsq-rco-iq2-xs.json](examples/rtxpro6000.gsq-rco-iq2-xs.json) |

| 模型 | 总权重（GB） | 主模型，不含 Engram（GB） | Engram（GB） |
|---|---:|---:|---:|
| **UD-IQ3_XXS** | **394.737** | 233.452 | 161.285 |
| **UD-Q2_K_XL** | **334.794** | 199.622 | 135.172 |
| **GSQ-RCO-IQ2_XS** | **265.821** | 155.226 | 110.595 |

容量按主模型 GGUF 和 Engram 文件的十进制 GB 统计，不含 DSpark 与 mmproj。GSQ-RCO 使用四个主分片和一个外置 Engram 文件，五个文件需放在同一目录。

- **DSpark**：[Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files)
- **mmproj**：[Hugging Face](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/mmproj-BF16.gguf) · [ModelScope](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files)

## 测试机器

| 项目 | 配置 |
|---|---|
| GPU | 单张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96GB，SM120 |
| CPU | Intel Core i7-14700KF，20 核 / 28 线程 |
| 内存 | 192GB DDR5，配置为 4800 MT/s |
| 存储 | 本地 NVMe SSD，承载 GGUF 和 Engram 读取 |
| 软件 | Ubuntu Linux、CUDA Toolkit 13.2、NVIDIA 驱动 595.104.02 |
| 显卡链路 | 10 月 23 日及 26 日测试期间采样为 PCIe 5.0 ×16 |

## 吞吐

**已测最大 prefill：6,261.998 tokens/s**，使用 GSQ-RCO-IQ2_XS、原格式 DSpark、32,768 输入、chunk 32768，测于 2026-10-26。这是无 profiler 的单次正式结果；同配置三次均值为 **6,245.142 tokens/s**。

**已测最大 decode：114.472 tokens/s**，来自 2026-10-26 的 GSQ-RCO-IQ2_XS、原格式 DSpark、524,288 输入、chunk 32768、固定完整 512 输出。该无 profiler 结果包含生成与最终自适应迁移收尾／同步；三次均值为 **112.323 tokens/s**。两个峰值分别从正式测量中选取。

以下为 2026-10-23 测试：单请求，每档一次预热＋三次正式测量，贪心解码，固定完整 512 输出。输入由同一 LongSWE token 流重复拼接和截取，用于长度测试。数值为均值 ± 样本标准差。

| 输入 tokens | Chunk | 常驻专家数 | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 5571 | 3,619.978 ± 11.526 | 34.312 ± 0.027 |
| 32,768 | 32,768 | 4221 | 5,769.141 ± 25.338 | 34.936 ± 0.010 |
| 131,072 | 32,768 | 4221 | **6,037.613 ± 40.617** | 35.048 ± 0.026 |
| 524,288 | 32,768 | 4221 | 4,499.192 ± 3.861 | 42.216 ± 0.038 |

Prefill 按引擎原生 prompt 时间统计；decode 按 DSpark 生成时间统计。测速使用校准后的专家排序与热度，chunk 和常驻专家数见上表。

2026-10-25 的累计优化版本使用同一台机器，LongSWE 一次预热＋三次正式测量，贪心解码，启动常驻 5571 个专家，并保留 Maya 在线自适应。每个请求新建 session，不复用对话前缀；专家热度在连续测量之间保留。

| 输入 tokens | 输出 tokens | Chunk | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 16,180 | 512 | 8,192 | 4,231.216 ± 5.476 | **76.760 ± 1.054** |

本组 prefill 使用原生时间，decode 包含生成与最终自适应迁移收尾，反映该输入和已适应专家缓存下的实测速度。

10 月 26 日的 GSQ-RCO-IQ2_XS 测量保持同一机器、原始 DSpark、1M 上下文容量和累计运行时。每档输入独立启动进程，1 次预热＋3 次正式测量；每次新建 session，不复用前缀，贪心解码忽略 EOS 以固定完整 512 输出。Maya 专家热度在同进程的四次请求间延续。

| 输入 tokens | 输出 tokens | Chunk | 启动常驻专家数 | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 512 | 32,768 | 5704 | 6,245.142 ± 23.250 | 104.548 ± 2.675 |
| 131,072 | 512 | 32,768 | 5704 | 6,186.661 ± 48.766 | 107.008 ± 2.448 |
| 524,288 | 512 | 32,768 | 5704 | 4,541.946 ± 7.958 | 112.323 ± 2.758 |

速度为平均值 ± 样本标准差。Prefill 使用原生 prompt 时间；decode 包含生成和最终自适应排空／同步。chunk32768 工作区预留 22.1875 GiB，专家预算自动计算，保持 40 段注册和 100% 大页覆盖。历史 UD 表格使用各自记录的运行时与容量；这些是独立测量，不构成受控提速对照。

## 初始专家 profile 与内存搬运

已使用更广泛的代码、agent 和通用推理 prompt 采集专家路由统计，来源包括 DeepSWE、SWE-bench Pro 与 Verified、R2E-Gym、LongSWE、LiveCodeBench、BigCodeBench、ProgramBench、Terminal-Bench、BFCL 和 HLE，据此调整初始 profile 的专家排序与热度，并保留运行期间的在线自适应。

**PCIe 5.0 上的 block-pinned memory（分块锁页内存）＋大页存储**显著提高了本机的专家搬运带宽：专家内存使用 40 段 CUDA 注册及 100% 的 2 MiB 大页覆盖，独立 H2D 对照从单次注册／4 KiB 页的 **23.952 GB/s 提升到 45.104 GB/s（+88.31%）**。更快的专家上传可减少搬运开销，从而在搬运受限时提升 prefill 和 decode 速度。该实测提升限定于已测试的 PCIe 5.0 ×16 配置。

如果使用支持 **AVX-512、配备大容量内存和高内存带宽的服务器 CPU**，也可以尝试在 **decode 阶段由 CPU 执行 MoE 专家**，GPU 继续处理注意力等适合的工作。

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
# GSQ-RCO 改为复制 examples/rtxpro6000.gsq-rco-iq2-xs.json。
# 编辑 /path/to 路径；纯文本可删除 mmproj 字段。
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

支持 **OpenAI Chat Completions / Responses** 和 **Anthropic Messages** 兼容接口。

[安装说明](docs/INSTALL.md) · [配置说明](docs/CONFIGURATION.md) · [API 示例](docs/API.md)。

## 相对上游 Strata 的 DeepSeek 专项适配

| 适配 | 实现内容 |
|---|---|
| 原生 DeepSeek runtime | 压缩/索引注意力、KV 所有权、超连接、路由专家及 CED prompt/tail replay。 |
| 原始 Unsloth GGUF | 七分片读取、IQ3/Q2_K/Q6_K 混合专家格式、主模型与草稿共享词表张量、原 BF16 视觉文件。 |
| 原始 GSQ-RCO GGUF | 四个主分片、外置 IQ4_NL Engram、IQ2/IQ3 混合专家、Q5_K 投影及自动 GPU 专家预算。 |
| DSpark | 最多 5-token 草稿、置信度自适应贪心验证、增量 KV、接受前缀提交、词表行选择。 |
| SM120 算子 | 混合格式解量化、有界 grouped GEMM、IQ/MMVQ 短窗口验证、分组投影、边界融合、indexer 和 CAKE 注意力。 |
| CUDA Graph | 已采用的 decode 输出/router 分段、共享专家提前提交与精简路由交接，保留动态路由与迁移边界。 |
| 专家与 Engram | 格式分组内的 Maya 在线自适应、专家预取、SSD 行缓存、异步 I/O，以及 40 段 block-pinned / 100% 大页准入。 |
| API 与图片 | DeepSeek 消息/tokenizer、thinking/工具编码、文本前缀自动检查点、取消后恢复、图片预处理及原生视觉编码器。 |

专家缓存、异构卸载、推测解码、图片输入和兼容 API 已存在于上游 Strata；此处说明的是 DeepSeek 专项实现与接入。[第三方声明](docs/THIRD_PARTY.md) · [MIT 许可证](LICENSE)。
