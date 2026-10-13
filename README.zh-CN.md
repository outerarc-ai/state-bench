# state-bench：面向 agent 推理的状态缓存评测工具

[![CI](https://github.com/outerarc-ai/state-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/outerarc-ai/state-bench/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[English](README.md) · [文档](docs/README.md) · [参与贡献](CONTRIBUTING.md) · [更新记录](CHANGELOG.md)

state-bench 研究推理引擎在 agent 负载下如何跨请求复用缓存状态。Qwen3.5、Nemotron-H 等混合模型除了注意力的 KV 缓存，还有递推（Mamba/GDN）状态；前缀只能从存在状态快照的位置恢复。引擎把快照放在哪里、保留哪些、驱逐哪些，决定了一个长 agent 会话里有多少内容要重算。

项目提供不含内容的 agent 会话轨迹格式、确定性的合成轨迹生成器，以及对现有引擎行为的说明文档。快照放置仿真器和面向 OpenAI 兼容服务的配对重放工具正在开发。性能测量以 AMD Instinct GPU（ROCm）为目标；轨迹工具可在任意 CPU 上运行。

## 核心能力

- **不含内容的轨迹：**请求以 64 token 链式块哈希、消息边界和时间记录。保留前缀复用关系而不保存提示词，录制的会话可以共享。
- **可复现的合成会话：**按种子生成追加、分叉、重试、改写四类模式。相同参数逐字节一致，所有参数都标注为假设。
- **可机器校验的产物：**轨迹按带版本号的 JSON Schema 和跨记录规则校验。所有命令结果确定、退出码有意义，供其他工具消费的输出均为 JSON。
- **引擎行为文档：**[SGLang Mamba 快照](docs/SGLANG_MAMBA_CHECKPOINTS.md) 说明 SGLang 目前在哪里放置、插入和驱逐递推状态快照，并给出源码位置。

合成轨迹用于检验策略的行为，不用于测量性能。性能结果只来自真实硬件上的受控运行，并遵循[测量规范](docs/MEASUREMENT.md)。

## 安装

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)：

```sh
git clone https://github.com/outerarc-ai/state-bench.git
cd state-bench
uv sync --locked --group dev
uv run state-bench --help
```

## 快速开始

```sh
uv run state-bench synth --preset mixed --sessions 8 --seed 0 -o mixed.jsonl
uv run state-bench validate mixed.jsonl
uv run state-bench stats mixed.jsonl
uv run state-bench schema > trace.schema.json
```

`stats` 给出的是“保留每个父请求的理想缓存”可复用的输入 token 比例，是由轨迹推出的上限，不是对任何引擎的测量。

## 进展

| 组件 | 状态 |
| --- | --- |
| 轨迹格式 v0.1 与校验 | 可用 |
| 合成生成器（追加、分叉、重试、改写） | 可用 |
| SGLang 快照行为说明 | 可用，基于 SGLang `166b4ae7f` |
| 公开 agent 轨迹导入 | 计划中，逐个数据集核实许可证后接入 |
| 本地 agent 会话录制代理 | 计划中 |
| 快照放置仿真器 | 开发中 |
| 针对真实服务的配对重放与统计 | 计划中 |

## 文档

| 文档 | 内容 |
| --- | --- |
| [总览](docs/README.md) | 文档索引 |
| [轨迹格式](docs/TRACE_FORMAT.md) | 记录类型、字段、哈希与校验规则 |
| [合成轨迹](docs/SYNTHETIC_TRACES.md) | 模式、参数与合成数据的局限 |
| [测量规范](docs/MEASUREMENT.md) | 结果如何产生、标注与报告 |
| [SGLang Mamba 快照](docs/SGLANG_MAMBA_CHECKPOINTS.md) | 当前的放置、插入与驱逐规则 |
| [相关工作](docs/RELATED_WORK.md) | 本项目参考的上游讨论与实现 |

欢迎提交 issue 和 pull request，开发环境与评审要求见[参与贡献](CONTRIBUTING.md)。state-bench 以 [Apache License 2.0](LICENSE) 发布。
