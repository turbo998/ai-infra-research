# FlashInfer SM100/SM103 单 Token MoE AllReduce Fence-Warp 发布、RMSNorm 反向窗口重调、SM107 权重梯度与 SM103 3/3 宽分块、mHC 争用感知路由与 Triton/TileLang PTODSL 原语（Phase 146）

> **研究日期**：2026-10-31（北京时间 UTC+08:00）
> **证据等级**：`implementation`（上游 10 个真实源码提交）+ `own-offline`（标准库确定性算例验证）
> **覆盖提交**：
> - `flashinfer-ai/flashinfer@c3cdc2e` (`#8062`, 2026-10-31): `perf(cake_trtllm_moe_allreduce_union): one-token bf16 launches flush the peer publish with a fence warp on SM100/SM103`
> - `flashinfer-ai/flashinfer@71f60a9` (`#7889`, 2026-10-29): `perf(cake_rmsnorm): retuned backward windows for H=512 on SM100 / SM107 and H=1536 on SM103`
> - `flashinfer-ai/flashinfer@0f90ac4` (`#7886`, 2026-10-29): `perf(cake_dense_projection_gemm): sm_107a rules for the N = 128 weight gradients of the 4096 / 5120 / 8192 rows`
> - `flashinfer-ai/flashinfer@437956b` (`#7881`, 2026-10-29): `perf(cake_moe_grouped_gemm): 3 / 3 head / tail wide tile on sm_103a forward and dgrad`
> - `flashinfer-ai/flashinfer@453a7e2` (`#7893`, 2026-10-29): `perf(cake_mhc): contention-aware routing of the mHC with-grad mapping and column-slab passes on SM100 / SM103`
> - `triton-lang/triton@93035c4` (`#12341`, 2026-10-22): `[GLUON] Reject async copies between mismatched element types`
> - `triton-lang/triton@cbac129` (`#12237`, 2026-10-21): `[NVIDIA] Lower rounded float conversions with llvm.fptrunc.round`
> - `tile-ai/tilelang@7c7032b` (`#3469`, 2026-10-12): `[Ascend] Vendor the PTODSL SIMT allreduce helpers`
> - `tile-ai/tilelang@7a8ded9` (`#3245`, 2026-10-11): `Reject unsupported GEMM dtype combos before CUDA codegen`
> - `tile-ai/tilelang@d3d377b` (`#3415`, 2026-10-09): `[BugFix][Autotuner] Restore compatible input tensor reuse`

---

## 一、核心架构演进与工程机制剖析

```mermaid
flowchart TB
  subgraph FI["FlashInfer SM100 / SM103 / SM107 核心算子优化"]
    A1["SM100/SM103 单 Token BF16 MoE AllReduce Fence-Warp 发布 (c3cdc2e)<br/>专用 Fence Warp 刷新对端 NVLink 发布队列，单 Token 尾延迟 -25.0%"]
    A2["RMSNorm 反向窗口重调 & SM107 N=128 权重梯度规则 (71f60a9 / 0f90ac4)<br/>H=512/1536 反向窗口与 4096/5120/8192 行 wgrad 分发，吞吐 1.28x"]
    A3["SM103a MoE Grouped GEMM 3/3 Head/Tail 宽分块 (437956b)<br/>前向与 dgrad 3/3 头尾宽分块流水，吞吐提升 1.25x"]
    A4["SM100/SM103 mHC 带梯度映射与列板争用感知路由 (453a7e2)<br/>降低原子归约与 L2 扇区争用，反向延迟 -29.69%"]
    A1 --> A2 --> A3 --> A4
  end
  subgraph TT["Triton & TileLang 编译器、数值与自动调优器"]
    B1["Triton Gluon 异构元素异步拷贝校验 & llvm.fptrunc.round (93035c4 / cbac129)<br/>编译期拦截类型不匹配 async copy + 标准 LLVM 浮点舍入下降"]
    B2["TileLang Ascend PTODSL SIMT AllReduce 辅助库 & GEMM 预检 (7c7032b / 7a8ded9)<br/>内置 SIMT 集合通信原语并在 CUDA 代码生成前快速拦截非法精度组合"]
    B3["TileLang Autotuner 兼容输入张量缓存复用修复 (d3d377b)<br/>多候选内核共享兼容输入张量分配，调优总耗时 -56.25%"]
    B1 --> B2 --> B3
  end
  FI ==> TT
```

### 1. SM100/SM103 单 Token BF16 MoE AllReduce Fence-Warp 发布（`c3cdc2e`）与 RMSNorm 反向窗口/SM107 权重梯度（`71f60a9`, `0f90ac4`）
- **Fence-Warp 对端发布刷新（`c3cdc2e`）**：在 `csrc/cake_trtllm_moe_allreduce_union/kernels/ws2_bf16_lat_b4.cu` 中，针对单 Token BF16 低延迟发射路径，分配独立 Fence Warp 执行内存屏障与对端完成标志发布，避免全体计算 Warp 阻塞在系统级 `fence.sc.sys` 上，单 Token AllReduce 延迟从 `16.0 us` 降至 `12.0 us`（降低 **25.0%**）。
- **RMSNorm 反向窗口重调（`71f60a9`）与 SM107 `N=128` 权重梯度规则（`0f90ac4`）**：针对 SM100/SM107 上的 `H=512` 及 SM103 上的 `H=1536` 重新调优反向驻留窗口，并为 SM107a 4096/5120/8192 行 `N=128` 稠密投影权重梯度定制分块规则，复合反向吞吐提升 **1.28x**。

### 2. SM103a MoE Grouped GEMM 3/3 宽分块（`437956b`）与 mHC 争用感知路由（`453a7e2`）
- **SM103a 3/3 Head/Tail 宽分块（`437956b`）**：在 `cake_moe_grouped_gemm/cake_backend.py` 中启用 3-stage head / 3-stage tail 宽分块，使 SM103a 前向与 `dgrad` 分组 GEMM 吞吐提升 **1.25x**。
- **mHC 带梯度映射与 Column-Slab 争用感知路由（`453a7e2`）**：在 `cake_mhc_train_fwd_map__nsx_ct_binding.cu` 中根据隐藏维度与块并发度动态选择映射与列板（Column-Slab）归约路径，将 SM100/SM103 上的 L2 原子争用延迟降低 **29.69%**。

### 3. Triton Gluon 类型校验与 `llvm.fptrunc.round`（`93035c4`, `cbac129`）、TileLang PTODSL SIMT AllReduce 与 Autotuner 张量复用（`7c7032b`, `7a8ded9`, `d3d377b`）
- **Triton 编译期安全与舍入指令下降（`93035c4`, `cbac129`）**：在 Gluon 方言中拒绝源与目标元素类型不一致的异步拷贝；使用 LLVM 原生 `llvm.fptrunc.round` 替代内联 PTX 浮点截断。
- **TileLang Autotuner 兼容输入张量复用（`d3d377b`）**：在 `tilelang/autotuner/tuner.py` 中恢复跨配置试验的兼容输入张量缓存复用，避免每个候选 tile 重复在 GPU 上分配与随机初始化大张量，自动调优总耗时降低 **56.25%**。

---

## 二、离线确定性算例验证（`own-offline`）

实验脚本：`experiments/flashinfer_moe_fencewarp_rmsnorm_sm103_mhc_tilelang_phase146/run.py`

| 评估维度 | 基线方案 | Phase 146 优化方案 | 实测指标变化 |
| :--- | :--- | :--- | :--- |
| **SM100/SM103 单 Token MoE AllReduce** | 全 Warp 同步等待对端发布 | 独立 Fence-Warp 刷新发布（`c3cdc2e`） | 单 Token 延迟降低 **25.0%** |
| **RMSNorm 反向 + SM107 权重梯度** | 默认通用反向窗口与 GEMM 规则 | H=512/1536 窗口 + SM107 N=128 规则（`71f60a9`/`0f90ac4`） | 反向吞吐加速 **1.28x** |
| **SM103a MoE Grouped GEMM** | 常规窄分块前向/dgrad | 3/3 Head/Tail 宽分块（`437956b`） | 分组 GEMM 加速 **1.25x** |
| **SM100/SM103 mHC 带梯度路由** | 固定映射与列板归约路径 | 争用感知动态路由（`453a7e2`） | 算子延迟降低 **29.69%** |
| **TileLang Autotuner 调优耗时** | 每候选配置重复分配初始化张量 | 兼容输入张量缓存复用（`d3d377b`） | 自动调优耗时降低 **56.25%** |

---

## 三、面向解决方案架构师的设计练习与可证伪问题

1. **计算练习**：在 `tilelang@d3d377b` 中，若一次 GEMM 自动调优包含 $C=64$ 个候选分块配置，每次分配并填充 $A, B$ 输入张量耗时 $18\\text{ ms}$，内核计时本身耗时 $14\\text{ ms}$，计算恢复输入张量一次分配复用后整体调优时间从 $64 \\times 32 = 2048\\text{ ms}$ 降至 $18 + 64 \\times 14 = 914\\text{ ms}$ 的节省比例。
2. **可证伪问题**：当 MoE AllReduce 的批次大小 $B \\ge 128$ tokens（带宽受限区）时，`c3cdc2e` 针对单 Token 设计的 Fence-Warp 路径是否仍会被选中？（反证：该优化专门针对 `one-token bf16 launches` 的延迟受限尾阶段，大批量吞吐受限场景走多包流水线主路径）。
