# strata-deepseek-v4.1-flash

[English](README.md)

在家用 PC 上，用 **单张 RTX PRO 6000 Blackwell 96GB 和 192GB DDR5-4800** 运行 DeepSeek-V4.1-Flash。

基于 [Strata](https://github.com/Niko1221/Strata) 和 [strata-nvfp4](https://github.com/yhfgyyf/strata-nvfp4)，加入原生 DeepSeek 运行时、低比特 DSpark 推测解码、SSD Engram 和 GPU/RAM 专家常驻。支持文本、图片、工具调用、OpenAI Chat Completions / Responses 和 Anthropic Messages。

**这是实验源码快照。** 已完成文档所述范围的算子准确性与功能检查；长跑输出漂移、间歇性权重读回不一致仍未解决，不能视为生产质量或稳定性验收通过。详见[验证与限制](docs/PERFORMANCE.md#validation-and-limitations)。

## 模型

实际测试的主模型是 **DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf**，来自[固定版本的 GGUF 仓库](https://huggingface.co/kaushikvira/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF/tree/8270aceb26eb5590ea40f347256a1736999f551d)，原模型为 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)。

- 40 层，每层 384 个路由专家，每 token 选择 6 个，隐藏维度 5120。
- 主模型专家存储使用 Q3_K / Q2_K；长 prefill 展开选中的专家为 BF16，短 decode / verify 使用 packed 专家算子。
- 优化后的 DSpark 草稿采用 Q2_K / Q4 / Q8 混合表示，并分离输入、输出词表。
- Engram 表保留在 SSD 上的主 GGUF 中，通过主机行缓存和异步搬运流水线访问。
- 图片输入使用匹配的 BF16 mmproj GGUF。

SM120 的 MXFP4 优化用于 **indexer scores**，不表示主模型专家改成了 NVFP4。权重、tokenizer 和本地生成的 profile 不包含在代码仓库中，见[资产要求](docs/INSTALL.md#model-assets)。

## 测试机器

| 部件 | 配置 |
|---|---|
| GPU | 单张 NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96GB，SM120 |
| CPU | Intel Core i7-14700KF，20 核 / 28 线程 |
| 内存 | 192GB DDR5，实际设定 4800 MT/s |
| 存储 | 本地 NVMe SSD，存放 GGUF 和 Engram 数据 |
| 软件 | Ubuntu Linux，CUDA Toolkit 13.2，驱动 595.104.02 |
| GPU 链路能力 | PCIe 5.0 ×16；没有采集每次请求的实时协商速率轨迹 |

## 速度

2026-10-19 无 profiler 测量：单请求并发、同一份 **16,180-token** 代码输入，prefill chunk 固定 **8192**，greedy，自然 EOS，最多输出 512 token。排除模型加载和预热；每行是三个候选请求的算术平均。全部控制组和单次数据见[测量表](data/performance-20261019.csv)。

| 配置 | 三次实际输出 token | Prefill（token/s） | Decode（token/s） |
|---|---|---:|---:|
| 当前累计实验：5-token adaptive DSpark＋共享 Gate/Up pair | 445 / 445 / 512 | **3,612.4** | **49.52** |
| 较早的 adaptive DSpark 实验，共享 pair 接入前 | 512 / 512 / 512 | 3,413.2 | 55.56 |

两行来自不同实验进程，不能当作同条件前后对照。当前实验每个 A/B/A 三元组内部一致，但第三轮相对初始输出已经漂移；较早实验的输出一致，但出现瞬时读回异常。不能将两行速度之差归因于共享 pair，也不能拼接各自最佳值。

计时实验使用 **5,792 个常驻专家**，配置的上下文上限为 1,048,576；本表只测了所述 16,180-token 输入。随后接口/图片回归及仓库中的实验启动示例使用 **32K 上下文、chunk 8192**，在原冻结名单下为 **5,649 个常驻专家**，为图片保留更多工作区；这组功能配置没有可比的吞吐测量。发布打包后重新编译的二进制也未重新测速。见[完整测量条件](docs/PERFORMANCE.md)。

## 相对原 Strata 新增的 DeepSeek 技术

| 新增路径 | 实际内容 |
|---|---|
| 原生 DeepSeek V4.1 运行时 | 压缩/索引注意力、KV 所有权、超连接、专家路由及 CED prompt/tail replay 语义。 |
| DSpark 草稿与验证 | 三层低比特草稿、独立输入/输出词表、主模型批量验证、增量草稿 KV、已接受前缀直接提交而不重放。 |
| Confidence 自适应验证 | 在主模型执行前，根据 confidence 生存概率和冻结成本表选取 0–5 个草稿；示例仅支持 greedy。 |
| SM120 indexer 与专家算子 | 带边界回退的 MXFP4 block-scaled scores、Q2_K/Q3_K 成对展开、DeepSeek 短行 Gate/Up/Down 融合。 |
| 共享 Gate/Up 合并 | 将共享专家接到已有精确 Q8 pair 投影，一次 Q8_1 输入量化供 Gate/Up 共用。 |
| Engram 与主机内存流水线 | 异步 Engram 读取、行缓存、专家预取、按量化格式分组的常驻策略，以及此模型的 40 段、2 MiB 大页注册核对。 |
| DeepSeek 接口与视觉 | 模型专属 tokenizer/template、思考控制、工具编码、图片预处理和原生视觉编码器。 |

原 Strata 已有 GPU 专家缓存、CPU/GPU 卸载、推测解码、图片和 OpenAI/Anthropic 接口。这里新增的是 **DeepSeek 专属适配与实现**，不把通用概念称为本项目首创。负收益的 7-token 配置和被拒绝的 CPU 冷专家异步分流没有启用。见[源码入口和上游比较](docs/CHANGES.md)。

## 构建

目标环境为 Linux x86-64、Python 3.10+、CMake 3.24+、C++20 编译器和 **CUDA 13.2 / SM120**；架构专属 indexer 编译为 SM120a。这是需要手动准备模型资产的源码发布，不附带自动模型下载器。

```bash
git clone https://github.com/yhfgyyf/strata-deepseek-v4.1-flash.git
cd strata-deepseek-v4.1-flash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DEEPSEEK41_TESTS=ON
cmake --build build --parallel 2 --target \
  strata-deepseek41 strata-deepseek41-experimental strata-deepseek41-inspect
```

未指定 `STRATA_GGML_DIR` 时，CMake 拉取固定版本的 ggml。依赖、本地完整 SHA 校验和模型准备边界见[安装说明](docs/INSTALL.md)。

## 运行

准备好已校验资产，复制[实验配置示例](examples/rtxpro6000.experimental.json)，填入自己的文件路径：

```bash
cp examples/rtxpro6000.experimental.json examples/local.json
# 编辑 examples/local.json 中的文件路径。
python tools/deepseek41/serve_experimental.py --config examples/local.json --check-config
python tools/deepseek41/serve_experimental.py --config examples/local.json
```

实验入口固定 **prefill 8192、上下文 32768、单请求槽**，启用 5-token greedy adaptive DSpark、累计算子及共享 pair。它要求 40 段专家内存全部满足 2 MiB 大页条件；不满足即停止启动。附带成本表只对应测试硬件/常驻条件，不是通用调优参数。详见[配置说明](docs/CONFIGURATION.md)。

基础非推测配置也可通过原通用入口运行：

```bash
python -m serve.deepseek41 --config examples/basic.json
```

前台服务按 `Ctrl+C` 结束。

## API

默认本地地址为 `http://127.0.0.1:30011/v1`，模型别名 `deepseek-v4.1`。

| 接口 | 已检查功能 |
|---|---|
| `/v1/chat/completions` | JSON/SSE 文本、图片、标准自动工具调用 |
| `/v1/responses` | JSON/SSE 文本、图片、自动工具往返 |
| `/v1/messages` | JSON/SSE 文本、图片、自动工具、开启/关闭思考 |

强制/指定工具选择和带状态的 `previous_response_id` 尚不支持，会明确报错。最近一次回归 **38 个 HTTP 用例**符合预期，包括 29 次真实生成、9 次预期错误、Unicode、代码、工具结果、6 次图片请求及 13,037-token 输入；不等于通用模型质量评测。

```bash
curl http://127.0.0.1:30011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"用三句话解释 Python 生成器。"}],"max_tokens":256,"temperature":0,"reasoning_effort":"none"}'
```

监听非本机地址时需要配置 `api_key`。更多例子见[API 说明](docs/API.md)。

## 许可证

代码保留 Strata 的 MIT License 及第三方声明；模型权重和派生资产遵循各模型许可证。见[第三方来源](docs/THIRD_PARTY.md)和[源码来源](docs/SOURCE_PROVENANCE.md)。
