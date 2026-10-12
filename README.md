# AI Infra Research

**从实现机制到架构决策，而不是 GPU 规格表或每日新闻汇总。**

面向已有云/解决方案架构背景、希望深入 AI Infrastructure 的学习者。中文分析保留 English terminology，研究覆盖 GPU、HBM、互连/网络、存储、分布式训练、推理平台、调度运维以及容量/成本。优先深入 GPU/HBM/Network，再把组件连接到端到端工作负载。

## 阅读入口

| 入口 | 内容 |
|---|---|
| [研究框架](docs/framework.md) | 分层方法、组件深挖契约、来源和结论边界 |
| [全栈地图](docs/stack-map.md) | 11 个层面、技术栈、架构问题与研究进度 |
| [架构师学习路线](docs/learning/roadmap.md) | 云架构 → 单 GPU → 多 GPU → 多节点 → 平台架构 |
| [能力矩阵](docs/learning/competency-matrix.md) · [练习](docs/learning/exercises.md) | 可检查的知识/设计能力；不以文章数量推断掌握 |
| [GPU/HBM 深入研究](docs/components/gpu-hbm.md) | 执行模型、内存层级、容量账本、Roofline 与工作负载 |
| [互连与网络分层](docs/components/interconnect-network.md) | NVLink/NVSwitch、IB/RoCE/RDMA、NCCL 及固定源码分析 |
| [技术趋势](docs/trends/index.md) | 证据支持的变化、成熟度与采用条件 |
| [每日研究](docs/daily/2026-10-12.md) | 首轮研究及限制；后续按北京时间日期增量归档 |
| [原创离线实验](experiments/capacity_roofline/README.md) | 可重算的显存 payload 与 Roofline 理论界限 |
| [合成容量架构练习](docs/architectures/synthetic-inference-budget.md) | 将显存账本、数据流和失败边界连接到方案设计 |
| [研究执行契约](docs/research-operations.md) | 研究 queue、跨运行续接、分支/PR、防重叠与失败恢复 |

## 证据状态

`announcement`、`specification`、`implementation`、`author-benchmark`、`own-offline`、`own-hardware`、`unverified` 分开登记。原文读过不等于完整规范审阅，源码静态分析不等于运行，离线计算不等于 GPU benchmark。

来源、主张、研究问题和进度分别保存在 [sources](research/sources.json)、[evidence](research/evidence.json)、[queue](research/queue.json)、[progress](research/progress.json)。用户学习掌握单独评估，当前未评估。

## 运行

Python 3.12 标准库；不需要 GPU 或第三方 Python 包。在仓库根目录运行：

```powershell
python -m unittest discover -s tests
python scripts\validate_research.py
python experiments\capacity_roofline\run.py --check
python scripts\source_monitor.py
```

来源监控联网，只取得 allowlist 中的公开页面并保留哈希/状态，不归档第三方全文。失败返回非零；网页哈希变化不是新技术发布。结构验证也不能代替人工事实、版权与隐私审阅。

## 公开与实验边界

这是干净历史的公开研究仓库，只保存原创分析、公开来源链接和合成实验。研究通过分支/PR 更新，**不自动合并**；PR 在合并前也已经公开。首轮实质成果位于研究 PR，不应把初始化的 `main` 当成全部内容。

默认只做静态阅读和 CPU/offline 实验。GPU、付费云、模型/API 调用需单独批准；不自动安装驱动、不占用现有集群、不执行上游安装器/训练脚本。

每日研究沿用北京时间 22:30 的 Local Copilot 自动化。运行需要本机、App 与网络可用；不能保证关机、休眠或退出后继续。每次从远端有效研究分支恢复，不每天重建基线。

第三方材料保留原作者和原有权利，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。未声明覆盖全仓库的开源许可证；公开可读不等于取得第三方再分发权。
Evidence-driven AI infrastructure research: GPU, HBM, interconnects, distributed systems, and solution architecture
