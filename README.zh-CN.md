<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/strata-lockup-dark.svg">
    <img src="docs/assets/readme/strata-lockup-light.svg" alt="Strata" width="220" align="middle">
  </picture>
  &nbsp;&nbsp;&nbsp;
  <img src="docs/assets/readme/whale-girl.png" alt="DeepSeek Whale-chan" width="156" align="middle">
</p>

[English](README.md)

本项目面向 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) 等 harness 环境的实际使用，目标是在保证模型一定推理质量的前提下，尽可能提高推理速度。

在家用电脑运行 DeepSeek-V4.1-Flash：**Intel Core i7-14700KF、单张 RTX PRO 6000 Blackwell 96GB、192GB DDR5-4800**。

基于 [Strata](https://github.com/Niko1221/Strata) 与 [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4)，提供原生 DeepSeek 推理、SSD Engram、GPU/内存专家缓存及 **DSpark 推测解码**。适配模型共用一个启动脚本，直接读取下载的原始 GGUF，**无需离线转换权重、合并分片或额外量化**。

支持 **1M 上下文容量（1,048,576 tokens）**、Unsloth 模型的**文本＋图片多模态输入**、工具调用，以及 OpenAI Chat Completions / Responses 和 Anthropic Messages。GSQ-RCO 当前支持文本输入。启动配置为单请求槽位、chunk 8192、默认最大输出 32768 tokens。长上下文场景可调整为 **chunk 32K（32768 tokens）**、**最大输出 128K（131072 tokens）**。

**推荐显卡：单张 RTX PRO 6000 Blackwell 96GB，或双张 RTX 5090。**

## 适配模型

**GSQ-RCO-IQ2_XS** 来自 **ISTA-DASLab / DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF**，搭配独立的 IQ4_NL Engram 文件。UD 两种版本来自 **Unsloth / DeepSeek-V4.1-Flash-1017-GGUF**。

| 主模型 | Hugging Face | ModelScope（魔搭） | 启动配置 |
|---|---|---|---|
| **GSQ-RCO-IQ2_XS** | [GSQ-RCO-IQ2_XS＋IQ4_NL Engram](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/tree/cb02422c900df489cea5833e8a92171224059b3c) | [GSQ-RCO-IQ2_XS＋IQ4_NL Engram](https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | [rtxpro6000.gsq-rco-iq2-xs.json](examples/rtxpro6000.gsq-rco-iq2-xs.json) |
| **UD-IQ3_XXS** | [UD-IQ3_XXS 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ3_XXS) | [文件列表，选择 UD-IQ3_XXS](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-iq3-xxs.json](examples/rtxpro6000.ud-iq3-xxs.json) |
| **UD-Q2_K_XL** | [UD-Q2_K_XL 目录](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-Q2_K_XL) | [文件列表，选择 UD-Q2_K_XL](https://modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/files) | [rtxpro6000.ud-q2-k-xl.json](examples/rtxpro6000.ud-q2-k-xl.json) |

| 模型 | 总权重（GB） | 主模型，不含 Engram（GB） | Engram（GB） |
|---|---:|---:|---:|
| **GSQ-RCO-IQ2_XS** | **265.821** | 155.226 | 110.595 |
| **UD-IQ3_XXS** | **394.737** | 233.452 | 161.285 |
| **UD-Q2_K_XL** | **334.794** | 199.622 | 135.172 |

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
| 显卡链路 | 吞吐测试期间采样为 PCIe 5.0 ×16 |

## 吞吐

**已测最大 prefill：6,261.998 tokens/s**，使用 GSQ-RCO-IQ2_XS、原格式 DSpark、32,768 输入、chunk 32768。这是无 profiler 的单次正式结果；同配置三次均值为 **6,245.142 tokens/s**。

**已测最大 decode：114.472 tokens/s**，来自 GSQ-RCO-IQ2_XS、原格式 DSpark、524,288 输入、chunk 32768、固定完整 512 输出。该无 profiler 结果包含生成与最终自适应迁移收尾／同步；三次均值为 **112.323 tokens/s**。两个峰值分别从正式测量中选取。

GSQ-RCO-IQ2_XS 测量保持同一机器、原始 DSpark、1M 上下文容量和累计运行时。每档输入独立启动进程，1 次预热＋3 次正式测量；每次新建 session，不复用前缀，贪心解码忽略 EOS 以固定完整 512 输出。Maya 专家热度在同进程的四次请求间延续。

| 输入 tokens | 输出 tokens | Chunk | 启动常驻专家数 | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 512 | 32,768 | 5704 | 6,245.142 ± 23.250 | 104.548 ± 2.675 |
| 131,072 | 512 | 32,768 | 5704 | 6,186.661 ± 48.766 | 107.008 ± 2.448 |
| 524,288 | 512 | 32,768 | 5704 | 4,541.946 ± 7.958 | 112.323 ± 2.758 |

速度为平均值 ± 样本标准差。Prefill 使用原生 prompt 时间；decode 包含生成和最终自适应排空／同步。chunk32768 工作区预留 22.1875 GiB，专家预算自动计算，保持 40 段注册和 100% 大页覆盖。历史测试数据保存在[性能记录](docs/PERFORMANCE_HISTORY.zh-CN.md)中。

## 初始专家 profile 与内存搬运

已使用更广泛的代码、agent 和通用推理 prompt 采集专家路由统计，来源包括 DeepSWE、SWE-bench Pro 与 Verified、R2E-Gym、LongSWE、LiveCodeBench、BigCodeBench、ProgramBench、Terminal-Bench、BFCL 和 HLE，据此调整初始 profile 的专家排序与热度，并保留运行期间的在线自适应。

当前运行时的专家内存使用 **block-pinned memory（分块锁页内存）＋大页存储**：40 段 CUDA 注册、100% 的 2 MiB 大页覆盖。在 PCIe 5.0 ×16 上，独立 CPU→GPU（H2D）传输对照从单次注册／4 KiB 页的 **23.952 GB/s 提升到 45.104 GB/s（+88.31%）**，详见[性能记录](docs/PERFORMANCE_HISTORY.zh-CN.md)。

**要达到上述传输速度，必须使用 PCIe 5.0 显卡，且 CPU／主板链路实际运行在 PCIe 5.0 ×16。** 如果 PCIe 4.0 或更低版本的链路已经饱和，分块锁页与大页带来的额外传输提升几乎为零。

如果使用支持 **AVX-512、配备大容量内存和高内存带宽的服务器 CPU**，也可以尝试在 **decode 阶段由 CPU 执行 MoE 专家**，GPU 继续处理注意力等适合的工作。

## 构建与启动

测试目标为 Linux x86-64、Python 3.10+、C++20、CMake 3.24+、CUDA 13.2 / SM120；indexer 与 CAKE 对象使用 SM120a。需事先准备 CUDA 和大页运行环境，脚本不会修改系统设置。

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
./start.sh
```

浏览器会打开 **http://127.0.0.1:30010** 管理页。选择模型和存放目录，点**Use model paths**保存，再查看准备计划并点**Set up and start**。工具会创建独立 Python 环境、编译引擎、续传缺少的文件并校验 SHA256；也可以直接填写已有文件路径。页面提供启停、进度、日志和聊天入口。

终端或 AI 可使用非交互命令：

```bash
./start.sh configure --model GSQ-RCO-IQ2_XS --data-dir /path/to/models
./start.sh plan --json
./start.sh launch --yes --json
# 后续：./start.sh start | status | stop | logs
```

默认从 ModelScope 下载；可明确指定 `--source huggingface` 或 `--source hf-mirror`，都必须通过同一组摘要校验。管理页仅监听本机，下载中断保留断点，启停仅针对本工具创建的服务。API 与原有 Web 聊天默认使用 **30011** 端口。原 `serve_experimental.py --config FILE` 入口继续可用。

支持 **OpenAI Chat Completions / Responses** 和 **Anthropic Messages** 兼容接口。

[启动与图形管理](docs/LAUNCHER.md) · [AI 自主部署](docs/AI_SETUP.md) · [手动安装](docs/INSTALL.md) · [配置说明](docs/CONFIGURATION.md) · [API 示例](docs/API.md)。

## 相对上游 Strata 的 DeepSeek 专项适配

| 适配 | 实现内容 |
|---|---|
| 原生 DeepSeek runtime | 压缩/索引注意力、KV 所有权、超连接、路由专家及 CED prompt/tail replay。 |
| 原始 GSQ-RCO GGUF | 四个主分片、外置 IQ4_NL Engram、IQ2/IQ3 混合专家、Q5_K 投影及自动 GPU 专家预算。 |
| 原始 Unsloth GGUF | 七分片读取、IQ3/Q2_K/Q6_K 混合专家格式、主模型与草稿共享词表张量、原 BF16 视觉文件。 |
| DSpark | 最多 5-token 草稿、置信度自适应贪心验证、增量 KV、接受前缀提交、词表行选择。 |
| SM120 算子 | 混合格式解量化、有界 grouped GEMM、IQ/MMVQ 短窗口验证、分组投影、边界融合、indexer 和 CAKE 注意力。 |
| CUDA Graph | 已采用的 decode 输出/router 分段、共享专家提前提交与精简路由交接，保留动态路由与迁移边界。 |
| 专家与 Engram | 格式分组内的 Maya 在线自适应、专家预取、SSD 行缓存、异步 I/O，以及 40 段 block-pinned / 100% 大页准入。 |
| API 与图片 | DeepSeek 消息/tokenizer、thinking/工具编码、文本前缀自动检查点、取消后恢复、图片预处理及原生视觉编码器。 |
| 安装与管理 | CLI 与浏览器共用配置、断点下载与摘要校验、本机构建、服务进程身份记录、健康检查、启停和日志。 |

专家缓存、异构卸载、推测解码、图片输入和兼容 API 已存在于上游 Strata；此处说明的是 DeepSeek 专项实现与接入。[第三方声明](docs/THIRD_PARTY.md) · [MIT 许可证](LICENSE)。
