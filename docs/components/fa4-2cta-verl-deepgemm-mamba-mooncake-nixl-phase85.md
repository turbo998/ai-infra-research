# FA4 SM100/SM103 2-CTA dQaccum 与 hd256 统一、veRL DeepGEMM QAT 统一低精度 Rollout、Megatron Mamba 局部残差态、Mooncake Soft-Pin 与 NIXL Trace V2 深度解析（Phase 85）

> **关联实验**：`experiments/fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85/`  
> **关联问题**：`q-fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85` (`priority: 94`)  
> **English Version**: [`fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85.en.md`](./fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85.en.md)

---

## 1. 背景与跨栈协同架构

在 Phase 85 (`priority: 94`) 的第三条主线中，我们审计了来自 FlashAttention-4、veRL、Megatron-Core、Mooncake Store、NIXL、Dynamo 与 TransformerEngine 的 10 项一手上游提交，涵盖 Blackwell/Ultra 注意力内核、RL 训练-推理解耦低精度对齐、Mamba 混合残差流、分布式 KV 缓存生命周期管理与传输追踪：

1. **FlashAttention-4 SM100/SM103 2-CTA `dQaccum` 寄存器直载、`hd128` 寄存器上限截断与 `hd256` 前向统一（PR #2960 `47e91f1f`、PR #2950 `037e808b`、PR #2962 `85d6dfe3`）**。
2. **veRL Megatron ModelOpt QAT 与 DeepGEMM 低精度 Rollout 统一策略（PR #8308 `619c397d` & PR #8287 `8f016846`）**。
3. **Megatron-Core `ResidualBranchInput` Mamba 局部残差态与动态 `dp_cp` 通信组超时传递（PR #8344 `85bf0a66` & PR #7736 `3630e44a`）**。
4. **Mooncake Store 独立批量 Soft-Pin HTTP 控制面与单键游标 Standby 恢复（PR #4858 `e764353f` & PR #4844 `5872ee87`）**。
5. **NIXL Trace Plugin API V2 全量 `TraceContext` 下推、Dynamo `/health` 按需实例序列化与 TE JAX MoE FSDP 布局验证（PR #2445 `56c1c2f7`、PR #16454 `a789babb`、PR #3638 `eb3c860a`）**。

---

## 2. 核心源码机制与定量公式

### 2.1 FlashAttention-4：2-CTA `dQaccum` 寄存器直载、`hd128` 崩溃修复与 `hd256` 统一 (`47e91f1f`, `037e808b`, `85d6dfe3`)
- **2-CTA Backward Postprocess `dQaccum` 寄存器直载 (`47e91f1f`)**：在 `M=128` 的 2-SM MMA 中，结果行按 `m // 64 = CTA rank` 拆分在两个 CTA 上，CTA 内 `M=64` TMEM Atom 将列 `[0, hdim/2)` 放在 lanes `0–63`、`[hdim/2, hdim)` 放在 lanes `64–127`。`flash_bwd_postprocess.py` 构造显式 GMEM 布局 `_dQaccum_tile_layout_2cta()`，直接将 GMEM 中的 FP32 `dQaccum`（`128 x 128 x 4 B = 65,536 B`）加载进寄存器，不再通过 SMEM 中转（SMEM 仅保留 `32,768 B` 的 BF16 `sdQ` 缓冲区，节省 **`65,536 B`** SMEM）。
- **SM103 (GB300) `hd128` Decode `setmaxnreg.dec` 非法指令修复 (`037e808b`)**：当 `q_stage = 1`、`num_regs_softmax = 176`、`num_regs_correction = 64`、`threads_per_cta = 512` 时，原本推导公式计算非 Softmax/Correction Warp Group 的寄存器数为：
  $$\text{num\_regs\_other}_{\text{uncapped}} = \left\lfloor \frac{512 - 176 \times 1 - 64}{3 - 1} \right\rfloor_{/8 \times 8} = 136$$
  但 `512` 线程 CTA 的启动寄存器上限为 $65536 / 512 = 128$，导致 `setmaxnreg.dec 136` 在 GB300 上触发非法指令陷阱（Illegal Instruction Trap）。修复公式加入硬件上限钳制：
  $$\text{num\_regs\_other} = \min\left(\left\lfloor \frac{512 - R_{\text{softmax}} \cdot q_{\text{stage}} - R_{\text{corr}}}{3 - q_{\text{stage}}} \right\rfloor_{/8 \times 8},\; \frac{65536}{\text{threads\_per\_cta}}\right) = \min(136, 128) = 128$$
- **`hd256` 前向统一与非对齐 NamedBarrier (`85d6dfe3`)**：移除独立的 `2,007` 行 `sm100_hd256_2cta_fmha_forward.py`，将 `hd256` 统一并入 `flash_fwd_sm100.py`（支持 2-CTA / S ping-pong 与分块 SplitKV），并在 `NamedBarrier` 中新增 `arrive_and_wait_unaligned`（调用 `nvvm.barrier_cta_sync(..., aligned=False)`）解决 TMEM 非对齐屏障同步问题。

### 2.2 veRL：Megatron QAT 与 DeepGEMM 低精度 Rollout 统一标准 (`619c397d`, `8f016846`)
- **Trainer 侧预量化与 DeepGEMM 统一后端**：PR #8287 与 PR #8308 将低精度权重导出（`FP8_BLOCK`、`MXFP8`、`MXFP4`、`NVFP4`）统一收口至 Megatron Trainer 侧生成 `compressed-tensors` 格式，并将 vLLM 与 SGLang 的非 NVFP4 低精度算子统一锁定为 **`deep_gemm`**（避免训练与推理在激活量化、缩放因子布局与 MoE Epilogue 上漂移；Hopper FP8 Block 保留 FP32 缩放因子不强转 E8M0）。
- **Blackwell 实测 KL 散度（`Qwen3.5-35B-A3B`，Megatron `EP=8`，vLLM Rollout `TP=2`）**：
  - `MXFP8 Linear + MoE`（FlashInfer CuteDSL Linear + DeepGEMM MoE）：无 QAT `k3_kl = 0.00387` $\rightarrow$ 匹配 QAT **`0.00325`**（降低 **`1.191x`**）。
  - `BF16 Linear + MXFP4 MoE`（DeepGEMM MoE + 动态 FP8 group-128 激活）：无 QAT `k3_kl = 0.00790` $\rightarrow$ 匹配 QAT **`0.00140`**（降低 **`5.643x`**）。
  - `BF16 Linear + NVFP4 MoE`（FlashInfer CUTLASS W4A4 MoE）：无 QAT `k3_kl = 0.00808` $\rightarrow$ 匹配 QAT **`0.00722`**（降低 **`1.119x`**）。

### 2.3 Megatron-Core：`ResidualBranchInput` 与动态 `dp_cp` 超时 (`85bf0a66`, `3630e44a`)
- **`ResidualBranchInput[StateT]` (`85bf0a66`)**：定义泛型 `NamedTuple(branch_input, state)`，使普通 `MambaLayer`（保存 `residual` 张量）与 `WideResidualMambaLayer`（保存 `(wide_residual,)` 及重计算上下文）共享同一套前向/CUDA Graph 分支调度，无需在模块实例上挂载可变侧信道属性，并为未来 Token 相关宽残差（WR）与多流超连接（mHC）预留局部状态契约。
- **动态 `dp_cp` / `expt_dp` 超时传递 (`3630e44a`)**：将 `distributed_timeout_minutes`（如 `45 min = 2,700 s`）显式传入 `create_dynamic_dp_cp_groups(..., timeout=timeout)` 及 `expt_dp` Gloo 组，消除长序列 CP 动态切分时沿用 PyTorch 默认 `10 min` 超时导致的假死中断。

### 2.4 Mooncake Store、NIXL Trace V2、Dynamo 与 TE JAX MoE (`e764353f`, `5872ee87`, `56c1c2f7`, `a789babb`, `eb3c860a`)
- **Mooncake 独立批量 Soft-Pin (`e764353f`) 与单键游标恢复 (`5872ee87`)**：在 Admin HTTP 端口暴露 `POST /api/v1/soft_pin` 与 `/api/v1/soft_unpin`（单次支持 `1..4096` 个 Key），无需重新 `Put/Upsert` 即可更新对象淘汰截止期，并将默认最大 Soft-Pin TTL 从 `24 小时 (86,400,000 ms)` 提升 **`7x` 至 `7 天 (604,800,000 ms)`**；同时以 `PublishRestoredObjects` 原子发布恢复对象，移除阻塞分片的 `kStaleHandleCleanupBatchSize = 64` 扫表锁。
- **NIXL Trace Plugin API V2 (`56c1c2f7`)**：将 `TraceBackend` 接口从 64-bit `pushCorrelationId` 升级为 `pushContext(const TraceContext &)`（插件 ABI 版本升至 `V2`），在 `postXferReq` 与 `xfer.complete` 跨线程传递完整 `trace_id`、`span_id` 与 `sampled` 标志。
- **Dynamo `/health` 探针减负 (`a789babb`) 与 TE JAX MoE (`eb3c860a`)**：Dynamo 默认 `/health` 仅返回 `{"status": "healthy", "stage": "ready"}`，仅当显式传入 `?instances=true` 时才序列化全集群实例列表（消除大集群每次存活探针数十毫秒的序列化开销）；TE JAX MoE 将 `NCCL_EP_HT_EM_AG_SCAN_MODE=1` 严格限定在非融合 Split-Dispatch 逐位对齐测试中。

---

## 3. 架构师设计练习与可证伪问题

1. **计算练习（FA4 SM103 寄存器预算）**：当 `threads_per_cta = 512`、`q_stage = 1`、`num_regs_softmax = 176`、`num_regs_correction = 64` 时，验证剩余寄存器预算公式给出的 `136` 为何会触发 `setmaxnreg.dec` 非法指令，以及当 `q_stage = 2` 时 `num_regs_other = (512 - 352 - 64) // 1 // 8 * 8 = 96 <= 128` 为何不会崩溃。
2. **可证伪假设**：在 `Qwen3.5-35B-A3B` 的 `BF16 Linear + MXFP4 MoE` RL Rollout 中，启用匹配的 ModelOpt QAT + DeepGEMM 动态 FP8 激活可将训练-推理 `k3_kl` 从 `0.00790` 降至 `0.00140`（`5.643x` 降幅），显著优于未开启 QAT 的直接后训练量化（PTQ）导出。
