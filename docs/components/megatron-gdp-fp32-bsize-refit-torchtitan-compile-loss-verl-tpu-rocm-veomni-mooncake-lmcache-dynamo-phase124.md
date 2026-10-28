# Megatron-LM GDP FP32 Sigmoid 与 Batch 阶跃/Refit 均衡、TorchTitan LocalCompileConfig 与泛型 Loss、veRL TPU v7x / ROCm DSV4 / VeOmni FSDP2 卸载、Mooncake EFA 停滞下限 / WeakPtr 句柄 / NUMA Cpuset / HCCS 2MiB 对齐、LMCache 与 Dynamo 深挖（Phase 124）

> **English Edition**: [megatron-gdp-fp32-bsize-refit-torchtitan-compile-loss-verl-tpu-rocm-veomni-mooncake-lmcache-dynamo-phase124.en.md](megatron-gdp-fp32-bsize-refit-torchtitan-compile-loss-verl-tpu-rocm-veomni-mooncake-lmcache-dynamo-phase124.en.md)
> **证据等级**：`implementation` + `author-benchmark` + `own-offline`（零 GPU 硬件重测声明）

---

## 1. 背景与核心架构问题

在大规模混合线性注意力（GatedDeltaProduct）预训练、RLHF/GRPO 后训练（TPU v7x、ROCm MI300X/MI350X、VeOmni FSDP2）与 RDMA/EFA/HCCS 分布式 KV 传输平面中，Phase 124 Track C 解决了九个关键问题：

1. **Megatron-LM `GatedDeltaProductMixer` BF16 Sigmoid 饱和导致梯度归零 (`a296523510`)**：`_compute_gating` 直接在 BF16 上调用 `b.sigmoid()`，当门控输入 $b \in [6.25, 9.0]$ 时，$\sigma(6.25) \approx 0.99807328$ 按 IEEE-754 最近偶数舍入到 BF16 精确变为 `1.0`，导致反向传播 $\beta(1-\beta)$ 完全归零（死神经元），而 FP32 下 $2\sigma(6.25)(1-\sigma(6.25)) = 0.00384602$ 仍保持显著梯度。
2. **Megatron-LM 超大 DP 下 Batch-Size 阶跃漏检与单 DP 组 Refit 热点 (`d1eec51b85`, `4a55e6fee6`)**：当 `DP = 8192`、`global_batch_size` 从 `8192` 阶跃至 `12288` 时，整数整除 `num_microbatches` 保持 `1 == 1`，导致 `is_batch_size_transition` 漏检；`_round_robin_dp` 在 `len(grouped_by_dp) == 1` 时直接返回 `src_meta_list[0]`，使全部张量挤在 replica 0 发送。
3. **TorchTitan 树外编译后端注入与 Loss 类型硬编码 (`0f9fd2bd64`, `740d5f755a`)**：缺乏可配置的 `LocalCompileConfig`，且 `MTPLoss` / `ChunkedLossWrapper` 硬编码绑定 `CrossEntropyLoss`。
4. **veRL TPU v7x Qwen3.5 GRPO 精度、ROCm DeepSeek-V4 B-Preshuffle 与 VeOmni FSDP2 CPU Offload 峰值主机内存 (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)**：VeOmni FSDP2 开启 CPU offload 时若未透传 `override_optimizer_config`（fused AdamW），会在每次 step 分配与完整模型参数等大的主机 FP32 临时缓冲区（35B 模型高达 **`130.39 GiB`**）。
5. **Mooncake EFA 瞬态拥塞误触发 QP 重置、OffsetAllocator 原子引用计数争用、容器 `cpuset.mems` 下 `mbind` 失败及 Ascend HCCS 未对齐访问 (`3a05eec5ef`, `9df448399b`, `a69bdd41cf`, `179ffb21c8`, `e70c09986d`, `5ef59b0c6e`)**：EFA `kMaxStalledWaves = 1024` 在高频轮询下仅数毫秒即可触发昂贵的 QP 重建；`OffsetAllocationHandle` 每次分配拷贝 `shared_ptr` 带来 2 次原子 CAS；RDMA `mbind(MPOL_BIND)` 未按容器 `cpuset.mems` 过滤 NIC NUMA 节点引发 `EINVAL`；Ascend HCCS 缺少 `2 MiB` 设备内存对齐校验。
6. **LMCache 全零 KV 假阳性校验与 Dynamo DP 队列膨胀 (`58c2ace2c2`, `0108f5b634`, `cd3b977edb`, `08907b7ad7`)**：全零数据无法区分真实命中与未初始化清零缓冲区；Dynamo 每个 DP rank 独立设置 `20,000` 排队上限，在 `DP=8` 下累积 `160,000` 请求导致 TTFT 失控。

---

## 2. 核心实现路径与定量证据分析

```
+---------------------------------------------------------------------------------------------------+
| Phase 124 Track C: Megatron-LM / TorchTitan / veRL / Mooncake / LMCache / Dynamo Full-Stack Fix   |
+---------------------------------------------------------------------------------------------------+
| 1. Megatron-LM (`a296523510`, `d1eec51b85`, `4a55e6fee6`) & TorchTitan (`0f9fd2bd64`, `740d5f75`)  |
|    - GDP gating: `b.float().sigmoid().to(b.dtype)` preserves grad=0.003846 at b=6.25 (vs 0 in BF16)|
|    - Batch ramp: checks `get_current_global_batch_size() != global_batch_size` (catches 8192->12288)|
|    - `_round_robin_dp`: drops `len(grouped_by_dp) == 1` fast path -> 8.0x replica load balance     |
+---------------------------------------------------------------------------------------------------+
| 2. veRL (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)                                  |
|    - TPU v7x Qwen3.5-35B-A3B GRPO (`PROMOTE_F32`) + ROCm DSV4 (`VLLM_ROCM_DSV4_BPRESHUFFLE=0`)    |
|    - VeOmni FSDP2 CPU offload honors fused AdamW (avoids 130.39 GiB host FP32 temp on 35B model)  |
+---------------------------------------------------------------------------------------------------+
| 3. Mooncake (`3a05eec5ef`..`5ef59b0c6e`), LMCache (`58c2ace2c2`) & Dynamo (`cd3b977edb`)          |
|    - EFA stall requires `kMaxStalledWaves=1024` AND `efa_stall_min_ms=1000` (50 ms floor)         |
|    - `OffsetAllocationHandle` stores `weak_ptr` (2 -> 0 atomic refcount ops/alloc)                |
|    - NUMA `mbind` filters NIC nodes by `cpuset.mems` & `MPOL_BIND`; HCCS enforces 2 MiB alignment |
|    - LMCache SHA-256 `[1, 127]` KV verification; Dynamo `40000 / rank_count` DP queue cap (4.0x)  |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 Megatron-LM FP32 Sigmoid 保梯度、Batch 阶跃检测与 Refit 负载均衡
- **FP32 Sigmoid 避免 BF16 饱和死梯度 (`a296523510`)**：将 `b.sigmoid()` 改为 `b.float().sigmoid().to(b.dtype)`。由于 `float()` 与 `to(b.dtype)` 构成围绕 `aten::sigmoid` 的可微类型转换节点，PyTorch Autograd 保存的是 **FP32 sigmoid 输出**并以此计算反向梯度：
  - 当 $b = 6.25$ 时，$\sigma_{\text{FP32}}(6.25) = 0.99807328$，但 $\sigma_{\text{BF16}}(6.25) = 1.0$；旧实现梯度为 $0.0$，新实现保留梯度 $2\sigma(1-\sigma) = 0.00384602$（$b = 9.0$ 时保留 $0.00024679$ vs 旧实现 $0.0$）。
- **Batch 阶跃与 Refit Round-Robin (`d1eec51b85`, `4a55e6fee6`)**：`is_batch_size_transition` 增加 `get_current_global_batch_size() != global_batch_size` 判定，修复 `DP=8192` 下 `8192 -> 12288` 阶跃漏检；`_round_robin_dp` 删除单 DP 组短路分支，将 8 副本单组 refit 的单卡发送峰值降低 **`8.0x`**（16 张量从 replica 0 独占变为每副本 2 张量）。

### 2.2 TorchTitan 扩展与 veRL 跨硬件（TPU v7x / ROCm / VeOmni FSDP2）强化
- **TorchTitan `LocalCompileConfig` 与泛型 `Loss[LossInputT]` (`0f9fd2bd64`, `740d5f755a`)**：通过 `model.local_compile.apply()` 支持树外后端编译，并用逆变 `TypeVar("LossInputT", contravariant=True)` 解耦 `MTPLoss` 与 `ChunkedLossWrapper`。
- **veRL TPU v7x、ROCm DSV4 与 VeOmni FSDP2 (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)**：
  - TPU v7x Qwen3.5-35B-A3B GRPO 默认开启 `--xla_tpu_bf16_emission_mode=PROMOTE_F32`；ROCm (`gfx942`/`gfx950`) DeepSeek-V4 禁用 `deep_gemm` 并支持 `VLLM_ROCM_DSV4_BPRESHUFFLE=0`。
  - VeOmni FSDP2 CPU offload 正确透传 `override_optimizer_config`（启用 fused AdamW 消除 35B 模型 **`130.39 GiB`** 主机 FP32 临时内存分配），并集成 `ep_sharded_stream_load` 与训练端 `Fp8ConfigWeightQuantizer`。

### 2.3 Mooncake 传输引擎、LMCache 确定性校验与 Dynamo 队列治理
- **Mooncake 六项核心加固 (`3a05eec5ef`..`5ef59b0c6e`)**：
  1. **EFA 双重停滞门控 (`3a05eec5ef`)**：`submitSlicesOnPeer` 必须同时满足 `stalled_waves >= 1024` **且**持续时间达到 `efa_stall_min_ms`（默认 `1000 ms`，下限 `50 ms`，可通过 `MC_EFA_STALL_MIN_MS` 配置）才触发 QP 重置。
  2. **`OffsetAllocationHandle` `weak_ptr` (`9df448399b`)**：句柄改存 `std::weak_ptr<OffsetAllocator>`（`weak_from_this()`），将每次分配/释放的原子引用计数操作从 `2` 次降至 **`0` 次**。
  3. **容器 NUMA `mbind` 过滤 (`a69bdd41cf`) 与 HCCS `2 MiB` 对齐 (`e70c09986d`)**：将 NIC 首选 NUMA 节点与 `/sys/fs/cgroup/cpuset.mems` 及当前线程 `get_mempolicy(MPOL_BIND)` 求交集，消除容器内 `mbind` `EINVAL`；Ascend HCCS 在注册时强制校验 `kHccsDeviceMemAlignment = 2 MiB`。
  4. **Conductor 上下文块上限配置 (`179ffb21c8`) 与 PG 视图原子快照 (`5ef59b0c6e`)**：解耦 `max_blocks_per_context`（默认 `200,000`）并消除 `applyGroupState` 中 `activeRanks` 的撕裂读取。
- **LMCache SHA-256 确定性非零 KV (`58c2ace2c2`, `0108f5b634`) 与 Dynamo DP 队列上限 (`cd3b977edb`, `08907b7ad7`)**：LMCache 将 `SHA-256(req_id:layer_id)` 映射至闭区间 `[1, 127]`（排除 `0`）验证冷热存取并封装不可变 `ObjectKeyIdentity`；Dynamo 将全局 `40,000` 排队预算按 `40000 / rank_count` 分摊（`DP=8` 下每 rank 从 `20,000` 降至 `5,000`，全集群积压缩减 **`4.0x`**）。

---

## 3. 确定性实验与架构师练习

- **离线验证程序**：`experiments/megatron_gdp_fp32_bsize_refit_torchtitan_compile_loss_verl_tpu_rocm_veomni_mooncake_lmcache_dynamo_phase124/run.py`
- **架构师核算练习**：验证 IEEE-754 单精度浮点数 $\sigma(6.25) = 0.99807328$ 的十六进制表示为 `0x3F7F81BF`，其低 16 位 `0x81BF > 0x8000` 进位后使高 16 位从 `0x3F7F` 变为 `0x3F80`（即 BF16 `1.0`），证明在 BF16 中直接计算 `sigmoid(b)` 在 $b \ge 6.25$ 时必然产生零导数。
