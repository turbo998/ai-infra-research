# Third-Party Notices, Provenance & Public Boundaries / 来源、版权与公开边界声明

## 1. Original Authorship & No Mirroring Policy (原创性与非镜像声明)

All Solution Architect decision guides (`docs/components/`), reference architectures (`docs/architectures/`, `docs/alibaba-cloud-ai-infra/`), interactive HTML calculators and 7-card visual reports (`docs/reports/`), SVG architecture diagrams (`assets/svg/`), and deterministic Python 3.12 analytical models (`experiments/`) in this repository are independently authored original works.

Third-party technical documentation, academic papers, open-source repositories, and industry specifications referenced in `research/sources.json` and inline citations are used strictly as factual engineering references. No third-party full-text articles, proprietary benchmarks, model weights, datasets, or upstream source files are mirrored or redistributed in this repository.

本仓库中的所有解决方案架构师（SA）选型指南（`docs/components/`）、全栈参考架构（`docs/architectures/`、`docs/alibaba-cloud-ai-infra/`）、自包含交互式 HTML 容量计算器与 7-Card 视觉报告（`docs/reports/`）、SVG 架构总结图（`assets/svg/`）以及基于 Python 3.12 标准库的确定性离线算例（`experiments/`）均为独立编写的原创作品。所引用的外部官方文档、学术论文、行业规范与开源项目仅作为事实与架构机理参考，仓库内不镜像任何第三方全文、模型权重、数据集或上游源码文件。

---

## 2. Upstream Trademarks, Specifications & Open-Source Attributions (第三方商标与开源权利归属)

All product names, trademarks, and open-source projects discussed in architectural comparisons remain the property of their respective owners under their upstream licenses:

- **NVIDIA Ecosystem**: CUDA, TensorRT-LLM, NCCL (Apache-2.0 / BSD-3-Clause), NVIDIA Dynamo (Apache-2.0), NIXL (Apache-2.0), FlashInfer (Apache-2.0), Megatron-LM / Megatron-Core, and DCGM documentation/specifications belong to NVIDIA Corporation and respective contributors.
- **AMD Ecosystem**: Instinct MI300X/MI325X/MI355X, ROCm, RCCL, and AITER documentation belong to Advanced Micro Devices, Inc. and respective contributors.
- **Open-Source LLM Serving, Training & KV Systems**: PyTorch / TorchTitan (BSD-3-Clause), vLLM (Apache-2.0), SGLang (Apache-2.0), Mooncake (Apache-2.0), LMCache (Apache-2.0), DeepSeek FlashMLA / DeepEP / DeepGEMM / 3FS (MIT), veRL (Apache-2.0), and Ray / KubeRay (Apache-2.0) retain their respective upstream licenses and copyrights.
- **Networking, Cloud-Native Orchestration & Observability**: Linux `rdma-core` (GPLv2 / OpenIB.org BSD), Ultra Ethernet Consortium (UEC) public specifications, Kubernetes / Kueue / DRA / KAITO / KEDA (Apache-2.0), Slurm (SchedMD), Cilium (Apache-2.0), and OpenTelemetry (Apache-2.0) retain their respective upstream rights.
- **Cloud Provider Architecture References**: Public product documentation and published conference papers from Microsoft Azure, AWS, Google Cloud, and Alibaba Cloud (T-Head, PAI-Lingjun, PAI, Bailian / Model Studio, ACM SIGCOMM 2024 HPN 7.0) are cited with explicit provenance boundaries.

---

## 3. Provenance & Benchmark Boundary Disclosure (数据溯源与评测边界声明)

1. **Analytical & Calibrated Models (`own-offline`)**: Unless explicitly labeled otherwise, quantitative tables, interactive sizers, and 7-card visual reports are driven by deterministic Python 3.12 analytical roofline, queueing, memory, network, storage, and TCO models under `experiments/*/results.json`. They are designed for Solution Architect capacity planning and architectural trade-off evaluation rather than claiming live physical multi-thousand-GPU customer cluster telemetry.
2. **Confidentiality & Sanitization**: Zero customer-identifiable data, private internal URLs, credentials, signed tokens, or local machine paths are permitted in tracked repository artifacts (enforced automatically by `scripts/validate_research.py`).
