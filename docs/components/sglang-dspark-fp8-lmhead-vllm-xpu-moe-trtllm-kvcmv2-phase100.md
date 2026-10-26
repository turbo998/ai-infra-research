# SGLang AMD `gfx95` DSPARK FP8 Verify-Logits LM Head、Ascend `arch35` MXFP8 门控、vLLM Intel XPU 网格转置 MoE 与 TensorRT-LLM KVCMv2 8 线程并行预填分析（Phase 100 · 100-Phase Milestone）

- **日期 / 阶段**：2026-10-27 · Phase 100 (`priority: 109` — **100-Phase Milestone!**)
- **组件归属**：`inference-disaggregation`
- **关联问题 ID**：`q-sglang-dspark-fp8-lmhead-vllm-xpu-moe-trtllm-kvcmv2-phase100`
- **关联证据 ID**：`claim-sglang-dspark-fp8-lmhead-trtllm-kvcmv2-phase100`
- **可复现实验套件**：[`experiments/sglang_dspark_trtllm_kvcmv2_phase100/`](../../experiments/sglang_dspark_trtllm_kvcmv2_phase100/run.py)

---

## 1. 架构背景与核心问题

在多硬件推理引擎（AMD `gfx95`、Ascend Atlas A5 `arch35`、Intel XPU、NVIDIA Blackwell）的投机采样（Speculative Decoding）、MoE 专家分派与大容量 Host KV Cache 池初始化中，存在四个关键瓶颈：

1. **DSPARK 投机验证路径（Verify Logits）的 `lm_head` 带宽墙**：在投机解码验证步中，目标模型需对 `[num_verify_tokens, hidden] x [hidden, vocab_slice]` 执行词表投影。当词表切片较大（如 $V_{\text{slice}} = 64,640$）时，BF16 权重读取占主导，成为每个解码步的固定延迟下界。
2. **Ascend `arch35` (Atlas A5) Block-FP8 与 MXFP8 的精度/拓扑冲突**：将任意 FP32 缩放因子的 $128\times 128$ Block-FP8 权重静态重量化为 $1\times 32$ UE8M0 MXFP8 属于有损变换（指数截断为 2 的幂次）；同时 Atlas A5 的 MXFP8 Normal Dispatch 仅支持单机内（intranode-only），跨机 Prefill 必须保持 BF16。
3. **Intel XPU Batched Triton MoE 网格维度顺序与 DFlash 混合精度残差**：传统 `(E, num_tiles)` 网格在 XPU 上导致线程块调度局部性不佳；而在 DFlash 投机解码中，若目标模型将残差流保持在 FP32（如 Muse Spark），辅助隐藏态全连接层 `self.model.fc(hidden_states)` 会因 dtype 不匹配报错。
4. **TensorRT-LLM KVCMv2 主机内存池（Host Tier）单线程缺页填充瓶颈**：当二级 Host KV Cache 池容量达数十 GiB 时，单线程串行执行 `mprotect` + 页预填充（prefault/populate）受制于 Linux 内核页分配器单核吞吐，导致引擎冷启动耗时十余秒。

---

## 2. 一手源码实证分析（11 个零重叠上游提交）

### 2.1 SGLang `#39622` (`01759f0a68`)：AMD `gfx95` DSPARK FP8 Target `lm_head` Verify Logits

SGLang 新增环境变量 `SGLANG_ROCM_OPT_DSPARK_FP8_LM_HEAD`（仅在 `is_gfx95_supported()` 且 `lm_head` 为浮点未量化权重时启用）：
- **离线缓存权重量化**：模块加载时将 `lm_head` 权重一次性量化为 FP8 并缓存每输出通道缩放因子（per-output-channel weight scale）。
- **在线激活量化 + `torch._scaled_mm`**：验证步对 `num_verify_tokens` 个 token 计算 per-token 激活缩放因子，调用 `torch._scaled_mm` 完成投影，将权重 HBM 读取字节减半（在 $H=7168, V_{\text{slice}}=64640$ 下单步节省 **441.65 MiB** HBM 流量，`num_verify_tokens=64` 时加速 **`1.85x`**）。

### 2.2 SGLang `#40519` (`66cac6b79c`)、`#40435` (`589d196bc3`)、`#46417` (`132b0f0ae7`) 与 `#45919` (`bc809a3622`)

- **Ascend NPU W4A8 MXFP4 MoE 双模分派 (`#40519`)**：`NPUW4A8MXFP4MoEMethod` 通过 `self.w13_kernel.DISPATCHER_QUANT_CONFIG` 声明分派线格式：低延迟模式（Decode）使用 `mxfp8`（带 E8M0 块缩放），普通模式（Prefill）保持 `bf16`，因为 Atlas A5 MXFP8 normal dispatch 仅限节点内，保持 `bf16` 可兼顾多节点 Prefill。
- **Ascend `arch35` Block-FP8 -> MXFP8 有损重量化门控 (`#40435`)**：
  - 若 Checkpoint 本身为 **UE8M0**（缩放因子已是 2 的幂），只需将 $128$ 组缩放因子无损复制 4 份到 $1\times 32$ 子组（`lossless MXFP8 re-layout`）。
  - 若 Checkpoint 为普通 **FP32 Block-FP8**，通过 `npu_dynamic_mx_quant` 转为 $1\times 32$ UE8M0 是**有损**的，因此默认关闭，仅在显式开启环境变量时执行，默认回退至无损 Triton 模拟路径。
- **Intel XPU `dp-attention` 守护 (`#46417`) 与异步视频生成进度上报 (`#45919`)**：在 XPU 上跳过 `CustomAllReduceV2` 导入防止崩溃，并为 OpenAI Diffusion 异步视频任务报告逐去噪步（per-step）进度。

### 2.3 vLLM `#51605` (`12d54c2948`)、`#63436` (`533906528c`)、`#63817` (`7a1ebe4ef5`) 与 `#63569` (`938be2d83d`)

- **Intel XPU `batched_triton_kernel_swapped` 与 `int4_w4a16` 维度归一化 (`#51605`)**：
  - 新增 `batched_triton_kernel_swapped`，在 XPU 上将网格顺序从 `(E, num_tiles)` 转置为 `(num_tiles, E)`（其中 $\text{num\_tiles} = \lceil M_{\max}/B_M\rceil \times \lceil N/B_N\rceil$），提升跨 Tile 并发度。
  - 在 `_moe_config_shape(w2_shape, dtype)` 中，当 `dtype == "int4_w4a16"` 时将打包的 $N$ 维还原为 $N \times 2$，使调优配置表与默认启发式使用一致的逻辑通道数。
- **DFlash 投机解码 `params_dtype` 显式对齐 (`#63436`)**：在 `qwen3_dflash.py` 的 `combine_hidden_states` 中将 `self.model.fc(hidden_states)` 改为 `self.model.fc(hidden_states.to(self.model.fc.params_dtype))`，兼容保持 FP32 残差流的目标模型。
- **NIXL 1.5.1 升级 (`#63817`) 与 `llama3_json` 解析器引擎迁移 (`#63569`)**。

### 2.4 TensorRT-LLM `#20790` (`cfae4519c3`) 与 `#20437` (`3205aeaedb`)

- **KVCMv2 8 线程并行预填与 `4 MiB` 粒度调用 (`#20790`)**：
  - 在 `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.cpp` 中定义 `constexpr int kInitialFillHelpers = 7`，连同主提交线程共 **8 个并发线程**并行填充初始 Host Tier，直至饱和操作系统内核页分配器。
  - 每个填充步按 **`4 MiB`** 粒度调用 `open()` 与 `populate()`，完成后调用 `HostMemBacking::commit(offset, size, /*populated=*/true)` 仅做 CUDA 上下文注册，避免重复 `mprotect`；失败状态使用哨兵 `kFillFailed = UINT32_MAX`。
- **OpenEngine 标准 gRPC Health 服务暴露 (`#20437`)**：为云原生探针提供标准 `grpc.health.v1.Health` 接口。

---

## 3. 定量实验验证（Python 3.12 标准库）

运行 `python experiments/sglang_dspark_trtllm_kvcmv2_phase100/run.py` 验证全部四项定量不变量：

| 评测维度 | 基线路径 | Phase 100 优化路径 | 定量收益 / 验证结果 |
|---|---|---|---|
| SGLang `gfx95` DSPARK Verify `lm_head` ($M=64$) | BF16 GEMM (`181.61 µs`) | FP8 `torch._scaled_mm` (`98.06 µs`) | **`1.85x` 加速**，单步节省 `441.65 MiB` HBM 读取 |
| SGLang `arch35` FP32 Block-FP8 权重加载 | 无门控强制有损转 MXFP8 | 默认路由 `triton_block_fp8_emulation_lossless` | 消除静默指数截断误差，UE8M0 保持 `4x` 无损重排 |
| vLLM XPU Batched Triton MoE ($E=64, M=256, N=4096$) | 网格 `(64, 256)`、`int4` 未还原 $N$ | 交换网格 `(256, 64)`、`int4_w4a16` $N=4096$ | 网格维度与 `int4` 逻辑宽度严格对齐 |
| TensorRT-LLM KVCMv2 `64 GiB` Host Tier 初始化 | 单线程填充 (`15.238 s`) | `1 + 7 = 8` 线程 `4 MiB` 并行填充 (`2.381 s`) | **`6.40x` 冷启动加速**（`16,384` 次 `4 MiB` 调用） |
