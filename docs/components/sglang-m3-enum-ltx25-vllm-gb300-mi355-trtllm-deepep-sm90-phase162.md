# SGLang M3 推理强度枚举校验与 LTX-2.5 / Pika-H3-Flex 扩散模型、vLLM GB300 AgentX 与全卡 MI355 MoE、TRT-LLM PyExecutor 及 DeepEP SM90 兼容（Phase 162）

> **证据分类**：上游源码级提交分析（`implementation`：`140bb6dc8d70`、`dc79ae47ca74`、`37fc17b8d60a`、`d90e35cffa21`、`59d58fd261b1`、`d2d550c3fc12`、`7969af8decf9`、`d870ded367b1`、`01dc3aaac820`、`60d44037a702`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162/run.py`）。

## 1. 核心架构问题与代码实现路径

1. **SGLang M3 `reasoning_effort` 枚举拦截（`140bb6dc8d70`）、LTX-2.5 双阶段默认 `1920x1088` 画布（`dc79ae47ca74`）与 Pika-H3-Flex 6 步 LoRA 支持（`37fc17b8d60a`）**：
   - 严格校验并拦截 M3 解析器中越界的 `reasoning_effort` 值（拦截率 `100.0%`），将 LTX-2.5 双阶段生成的默认分辨率对齐至原生 `1920x1088`，并集成 MiniMax-H3 + 6-step LoRA 的 `Pika-H3-Flex` 架构。
2. **vLLM MiniMax-M3.1-Flash GB300 InferenceX AgentX 预设（`d90e35cffa21`）与 ROCm MI355 全卡 8-XCD MoE 内核验证（`59d58fd261b1`、`d2d550c3fc12`）**：
   - 将 ROCm MI355 MoE 内核验证从受限的 DPX（`4` XCD 分区）切换至完整 MI355 设备（`8` XCDs，可用 HBM 与计算带宽提升 `2.0x`），并新增 GB300 InferenceX AgentX 生产配置。
3. **TensorRT-LLM `pyexecutor` 与 Connector 测试解耦（`d870ded367b1`、`7969af8decf9`）及 DeepEP SM90 Hopper 兼容性修复（`60d44037a702`、`01dc3aaac820`）**：
   - 消除 DeepEP 在 SM90 Hopper 架构上的 PTX 编译不兼容错误（`0` 错误）。

## 2. 确定性实验验证

```bash
python experiments/sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162/run.py --check
```
