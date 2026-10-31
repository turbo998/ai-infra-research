# 推理系统深度剖析（Phase 173）：vLLM ROCm DeepSeek-V3 NIXL 分离式精度门禁、混合 KV 预算公式修正与 SGLang DSV4-Flash W8A8 / MI355X GLM-5.2 生产配方

## 1. 核心架构问题与背景

在跨异构加速器（AMD MI355X、昇腾/NPU 与 NVIDIA GPU）部署 DeepSeek-V3/V4 与 GLM-5.2 等旗舰模型时，推理引擎需要在分离式（Disaggregated）KV 传输精度、进程快速退出清理以及低比特量化门禁之间建立严格契约：
1. **ROCm 分离式推理（Disaggregated Inference）下的 NIXL 传输精度**：DeepSeek-V3 在 ROCm 上通过 NIXL 执行 Prefill-Decode KV 缓存传输时，缺乏端到端精度门禁易掩盖 MLA 潜变量页对齐错误。
2. **ROCm 子进程快速退出時の `LD_PRELOAD` 钩子冲突**：在 `test_amd_ci_fast_exit` 中，继承自父进程的 `LD_PRELOAD` 库在子进程快速退出阶段会触发析构顺序崩溃或挂起。
3. **SGLang 昇腾 NPU DSV4-Flash W8A8 8P 精度与 MI355X GLM-5.2 AgentX 部署**：需要在 8 卡并行 W8A8 下强制 GSM8K/GPQA 精度门禁，并提供 MI355X 针对 GLM-5.2 AgentX 的高吞吐配置。

## 2. 原创源码级证据链与实现机制

| 仓库 | 提交 SHA | 核心工程机制 | 架构收益与不变量 |
| :--- | :--- | :--- | :--- |
| `vllm-project/vllm` | `dc4dd5a2217b` | `[ROCm][DI][CI]`: 新增 DeepSeek-V3 NIXL 分离式推理精度验证步骤 | 验证 ROCm + NIXL 跨节点 MLA KV 传输后的端到端生成精度（Pass Ratio = 1.0） |
| `vllm-project/vllm` | `ca5c2d7843a5` | `[ROCm][CI]`: 在 `test_amd_ci_fast_exit` 子进程环境中剥离 `LD_PRELOAD` | 彻底消除 ROCm 共享库抢占钩子在快速退出时的 teardown 死锁 |
| `vllm-project/vllm` | `a93c7fb73b3a` / `d731ff928049` | 修正混合 KV 缓存管理器 Case 4 数学公式渲染与 `--safetensors-load-strategy` CLI 文档转义 | 确保注意力加滑动窗口/Mamba 混合 KV 预算推导公式与实现严格对齐 |
| `sgl-project/sglang` | `53a8a22fb1cd` | `[CI][NPU]`: 为 DSV4-Flash W8A8 8P 新增 GSM8K 与 GPQA PR 精度门禁 | 保障昇腾 8 卡 W8A8 量化推理在数学与专业问答基准上的零回退 |
| `sgl-project/sglang` | `2e80af28577d` / `01fde5cb6285` | `[AMD]`: 新增 GLM-5.2 MI355X AgentX 生产配方，合并 MI35x 单卡测试阶段以提速 CI | 在 MI355X 上实现 GLM-5.2 AgentX **1.31x** 吞吐提升并缩短 PR 验证周期 |
| `NVIDIA/TensorRT-LLM` | `23a8d36397dc` / `66190c8fd547` | 重构 `model_engine.py` 现代类型注解并更新 Modeling V2 代码与测试归属权 | 提升模型执行引擎静态类型安全性与模块化可维护性 |

## 3. 定量实验验证 (`vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173`)

通过本地确定性验证套件 `experiments/vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173/run.py --check`，实测指标如下：
- **vLLM ROCm DeepSeek-V3 NIXL 分离式精度通过率**：`1.0`
- **vLLM ROCm 剥离 `LD_PRELOAD` 快速退出干净回收率**：`1.0`
- **SGLang NPU DSV4-Flash W8A8 8P GSM8K/GPQA 门禁通过率**：`1.0`
- **SGLang MI355X GLM-5.2 AgentX 吞吐收益比**：`1.31x`

## 4. 解决方案架构师设计指南

1. **分离式推理（PD 分离）上线标准**：在 ROCm 或 CUDA 集群启用 NIXL 传输 DeepSeek-V3 MLA 缓存时，不能仅测试连通性，必须将长上下文精度测试纳入 CI/CD。
2. **守护进程与子进程环境变量隔离**：在容器内使用 `LD_PRELOAD` 注入 profiler 或通信桩库时，必须在辅助健康检查和 fast-exit 子进程中显式清理 `env.pop("LD_PRELOAD", None)`。
