# Megatron-LM LayerShardedMuon 流式 Newton-Schulz、TE cuBLASMp NVFP4 通信计算重叠与 Mooncake RDMA 拥塞控制 / NIXL 追踪深挖

> **English Companion**: [megatron-layersharded-muon-stream-te-cublasmp-nvfp4-mooncake-rdma-cc-nixl-xfer.en.md](megatron-layersharded-muon-stream-te-cublasmp-nvfp4-mooncake-rdma-cc-nixl-xfer.en.md)  
> **Phase 41 (`priority: 50`) · 确定性复现实验**: [`experiments/megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl/run.py`](../../experiments/megatron_muon_stream_te_cublasmp_nvfp4_mooncake_cc_nixl/run.py)  
> **7-Card 4:5 视觉报告**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 一、核心架构结论（Executive Summary）

1. **Megatron-LM `LayerShardedMuon` 流式分批 Newton-Schulz 与单 Rank 域加速（`PR #8379` / `6c956376e3`）**：
   - 将 `LayerShardedMuon._run_ns` 重构为惰性生成器 `_iter_ns(full_by_k, load=ns_input)`，按同形状 `ns_batch_size` 分块按需构造 Nesterov FP32 输入（`p.grad.lerp(buf, beta).float()`），并在正交化后执行原地两次连乘 `orth.mul_(scale).mul_(self.extra_scale_factor)` 立即消费并释放 `ns_output`。
   - **单 Rank 域（如 `TP=1, Expt_TP=1, GTP_remat=1` 的 Layer-Wise 分布式优化器）不再回退至逐参数循环**：对单 Rank 持有 `1` 个 Dense 矩阵（`4096x4096`）+ `8` 个本地专家（`16` 个 `4096x1536` / `1536x4096` 矩阵，共 `17` 个矩阵）的典型配置，`ns_batch_size=4` 将 Newton-Schulz Kernel 启动次数从 **`17` 次降至 `5` 次（缩减 `3.4x`）**，并将峰值瞬态 FP32 显存从 **`1,088.0 MiB` 压缩至 `192.0 MiB`（降低 `5.6667x`）**，每次 NS 调用前存活的 Nesterov `lerp` 临时张量数严格受限于 `<= ns_batch_size + 1`（`5` 个）。
2. **TransformerEngine `cuBLASMp` (`>= 0.11.0`) NVFP4 通信+GEMM 重叠（`PR #3634` / `6b5cd7e2e4`）**：
   - 在 `comm_gemm.cpp` 中为 `cuBLASMp`（`CUBLASMP_VERSION >= 1100`）解锁 NVFP4（`VEC16_UE4M3` 1D Block Scaling，要求 TN 布局）的 `nvte_all_gather_gemm` 与 `nvte_gemm_reduce_scatter` 重叠。
   - **12 字节设备端指针模式标量与单次双向上下文量化**：通过 `DeviceScalarsCreate` 分配 **`12 B`**（3 个 FP32：`{alpha, 0.0f, 1.0f}`）设备缓冲区并启用 `CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_POINTER_MODE`，在 GPU Stream 上由 `nvte_nvfp4_compute_per_tensor_scale` 直接融合 `amax_a * amax_b / ((448*6)^2)`，消除 Host 同步；当每 Rank 切分列数 `(N / tp_size) % 128 != 0` 时自动回退至 `CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT`；同时在反向传播 `dgrad` 前对 `grad_output` 执行**单次 `rowwise + columnwise` 联合量化**（量化调用从 `2` 次减为 `1` 次，节省 `16.0 MiB` BF16 读取并保证 `dgrad` 与 `wgrad` 随机舍入种子严格一致）。
3. **Mooncake RDMA 自适应拥塞控制核心（`PR #4197` / `3b08b77168`）与 NIXL 完成追踪闭环（`PR #2462` / `f8248873bd`）**：
   - Mooncake Transfer Engine 新增 `adaptive_congestion_control` 核心状态机（默认窗口 `[4 MiB, 64 MiB]`，目标排空时间 `target_drain_time_ns = 2,000,000 ns` 即 `2000 us`）：连续 **`2` 个**高压周期（`drain_ns > 2000 us` 或 Route 级 RNR `receiver_pressure`）将窗口减半 `max(min_win, win / 2)` 并标记 `kCongested`；连续 **`3` 个**低压且有成功完成周期（`drain_ns < 1000 us`）按 `+min_win / 4`（`+1 MiB`）线性增窗；空域（`inflight == 0`）允许单个不可分割超大 Slice（如 `8 MiB > 4 MiB`）准入，且通过代数围栏（`generation`）在 `resetGeneration` 后对旧 `Permit` 返回 `("kAvoid", "kStaleGeneration")`。
   - NIXL 在 `postXferReq`（同步立即完成与 Repost 检查）、`getXferStatus`、`releaseXferReq` 全部 **4 个完成观测点**通过 `exchange(true)` 幂等发射 `nixl::xfer.complete` NVTX/Trace 标记，将 UCX 50 次传输测试中的完成标记覆盖率从 **`33/50`（`66%`）提升至 `50/50`（`100%`）**。

---

## 二、一手上游提交与证据映射（First-Hand Upstream Ledger）

| 上游仓库 | 提交 / PR | 日期 | 核心改动模块 | 证据 ID |
|---|---|---|---|---|
| `NVIDIA/Megatron-LM` | [`6c956376e3` (`#8379`)](https://github.com/NVIDIA/Megatron-LM/commit/6c956376e3ef22528413ff35412ad8b938272986) | `2026-10-21` | `megatron/core/optimizer/layer_sharded_muon.py` | `EV-2026-361` |
| `NVIDIA/TransformerEngine` | [`6b5cd7e2e4` (`#3634`)](https://github.com/NVIDIA/TransformerEngine/commit/6b5cd7e2e43cee6da5dff9b9f86bc2ec05d7b1ad) | `2026-10-21` | `transformer_engine/common/comm_gemm/comm_gemm.cpp`, `linear.py` | `EV-2026-362` |
| `kvcache-ai/Mooncake` & `ai-dynamo/nixl` | [`3b08b77168` (`#4197`)](https://github.com/kvcache-ai/Mooncake/commit/3b08b77168870d265e32b6cfeab661f037426b9c) / [`f8248873bd` (`#2462`)](https://github.com/ai-dynamo/nixl/commit/f8248873bd5b8d3609ba1448fa2f623aefb361cf) | `2026-10-21` | `adaptive_congestion_control.cpp`, `nixl_Agent.cpp` | `EV-2026-363` |

---

## 三、架构机制深挖（Architectural Deep Dive）

### 3.1 `LayerShardedMuon._iter_ns` 流式分块与原地缩放流水线

```
Legacy Eager _run_ns:
  1. Build local_ns_inputs[0..16] upfront (17 FP32 Nesterov tensors = 448 MiB live)
  2. Accumulate ns_by_k[0..16] in dict (another 448 MiB live) + 2x chunk temporaries (192 MiB)
     => Peak transient memory = 1,088.0 MiB

PR #8379 Streaming _iter_ns(replicated, load=ns_input):
  For each shape bucket -> For each chunk of up to ns_batch_size (4):
    1. x = torch.stack([ns_input(p) for p in chunk])   # at most 4 * 24 MiB = 96 MiB
    2. orth = newton_schulz(x, ...)                    # batched 3D SYRK + baddbmm (96 MiB)
    3. orth.mul_(scale).mul_(extra_scale_factor)       # in-place sequential multiplies (0 extra alloc)
    4. yield (k, o) -> _apply_update -> del ns_output, del orth
     => Peak transient memory = 192.0 MiB (5.67x reduction!), NS launches: 17 -> 5 (3.4x reduction!)
```

### 3.2 Mooncake RDMA 自适应拥塞控制 AIMD 状态机与代数围栏

- **双层准入域（`Device` + `Route`）**：`tryAcquire` 先申请本地 RDMA 网卡 `Device` 窗口配额，再申请目标对端 `Route` 窗口配额；注意 `OutcomeClass::kReceiverPressure`（IBV RNR 重试耗尽）仅计入 `route=true` 的对端路由域，避免单个慢接收端误杀本地 NIC 上发往其他健康节点的并发路由。
- **排空时间驱动的 AIMD 窗口演化**：
  $$\text{drain\_ns} = \left\lfloor \frac{\text{backlog\_bytes} \times 10^9}{\text{delivery\_rate\_bytes\_per\_sec}} \right\rfloor$$
  当 `drain_ns > 2,000,000` 持续 `2` 个控制周期，窗口执行乘法减小 $W \leftarrow \max(W_{\min}, \lfloor W/2 \rfloor)$；当 `drain_ns < 1,000,000` 且 `successes > 0` 持续 `3` 个控制周期，窗口执行加法增大 $W \leftarrow \min(W_{\max}, W + \lfloor W_{\min}/4 \rfloor)$。

---

## 四、标准库确定性复现与验证指标（Deterministic Reproduction Metrics）

| 评估维度 | 指标名称 | 确定性复现值 | 验证状态 |
|---|---|---|---|
| Megatron Muon | 单 Rank 17 矩阵峰值瞬态显存 (`Eager -> Streaming`) | `1088.0 MiB -> 192.0 MiB` (`5.6667x` 缩减) | `PASS` |
| Megatron Muon | Newton-Schulz Kernel 启动次数 / 每块最大 `lerp` 数 | `17 -> 5` (`3.4x` 缩减) / `<= 5` | `PASS` |
| TE cuBLASMp NVFP4 | 设备端标量缓冲区字节 / `(N/tp)%128!=0` 算法回退 | `12 B` / `SPLIT_P2P_OVERLAP -> DEFAULT` | `PASS` |
| TE cuBLASMp NVFP4 | `grad_output` 反向传播量化次数 (`Before -> After`) | `2 -> 1` (节省 `16.0 MiB` 读取，保持随机舍入一致) | `PASS` |
| Mooncake RDMA CC | 空域 `8 MiB` 超大 Slice 准入 / 12 低压+2 高压周期窗口演化 | `kAllow` / `4 MiB -> 8 MiB -> 4 MiB (kCongested)` | `PASS` |
| NIXL Trace Marks | 50 次 UCX 传输 `nixl::xfer.complete` 标记覆盖数 | `33/50 -> 50/50 (100%)` | `PASS` |

---

## 五、生产采纳建议（Production Guidance）

1. **MoE Layer-Wise Muon 训练配置**：在 Megatron-LM 开启 `--use-layer-wise-distributed-optimizer` 与 `--optimizer muon` 时，显式设置 `--muon-ns-batch-size 4`（或 `8`），既能在单 Rank 专家域获得 Batched SYRK `3.4x` 启动缩减，又不会因 Nesterov 预分配撑爆 HBM。
2. **NVFP4 `cuBLASMp` TP 列对齐**：使用 TransformerEngine `--tp-comm-overlap` 跑 NVFP4 训练时，确保隐藏层输出维度满足 `(N / tp_size) % 128 == 0`，否则 `cuBLASMp` 将回退至非分片默认算法。
