# FlashAttention-4 Paged-KV CUDA Graph 与 SM100 HD256 SplitKV、TorchTitan GB300 4D RL 拓扑、Mooncake NoF 解耦与 LMCache 紧凑 Token IPC（Phase 87）

> **证据类型**：First-hand Implementation Audit (`implementation` + `author-benchmark` + `own-offline` 确定性 Python 3.12 标准库算例 `experiments/fa4_verl_mcore_mooncake_phase87/run.py`)。

## 1. 核心架构发现与第一手源码路径

| 上游提交 / PR | 关键源码路径 | 架构改动与定量指标 |
|---|---|---|
| `Dao-AILab/flash-attention#2984` (`73019affed`, 2026-10-15), `#2917` (`e9cf2c1651`, 2026-09-25), `#2891` (`7f4592e596`, 2026-09-24) | `csrc/flash_attn/flash_api.cpp`, `flash_attn/cute/{block_info,flash_fwd_sm100,sm100_hd256_2cta_fmha_forward,sm100_hd256_2cta_fmha_backward*}.py` | `#2984` 修复 `mha_fwd_kvcache` 在启用 `paged_KV` 时因执行 `seqlens_k.max().item<int>()` 触发 Device-to-Host 同步而导致 CUDA Graph Capture 崩溃的缺陷：通过 `c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None` 仅在非捕获模式下运行边界断言，使捕获期 Host Sync 次数从 **1 降至 0**；`#2917` 为 SM100 `hd256` 2CTA 前向内核补齐 `SplitKV` 支持与 `BlockInfo.has_kv_work(n_block_min, n_block_max)` 谓词（消除 `ceil_div` 分块下空 Split 产生负数块计数或不同 Warp Role 分支不一致的问题），并修正 `_get_fwd_config` 中 2CTA Cluster 的 `total_mblocks = 2 * B * H_kv * qhead_per_kvhead * ceil_div(S_q, 2 * tile_m)` 计数；`#2891` 为 SM100 `hd256` 2CTA 反向内核支持变长 `seqused_q` / `seqused_k`。 |
| `pytorch/torchtitan#5581` (`43d1bb9801`, 2026-10-22) | `torchtitan_recipes/rl/dapo_math.py`, `tests/rl/unit_tests/cpu/test_dapo_math.py` | 重构 GB300 上 `muse_spark_816b_a42b_dapo_math_gb300_config` RL 训练拓扑：从 **256 GPUs (`FSDP 64 x TP 4 x EP 4`)** 切换为 **128 GPUs (`FSDP 8 x TP 4 x CP 4 x EP 16`)**（Generator 保持 16 GPUs `DP 4 x TP 4 x EP 16` + Full CUDA Graphs）。将每个路由专家的 FSDP 切片组从 64 缩减 **8.0x** 至 8（使每个 `EP=16` 组恰好置于 4 台 GB300 主机组成的同一 NVLink 域内，消除跨机架专家 All-Gather/Reduce-Scatter 瓶颈）；同时引入 `ContextParallelTransform` 将训练端全局与滑动窗口注意力统一映射为 `UlyssesVarlenInnerAttention`（单卡激活与 Dist-MoE 暂存按 $1/(TP \times CP) = 1/16$ 缩放）。在 GB300 131K 上下文实测中，峰值显存 **186 GiB/GPU**，Dist-MoE 暂存峰值仅占 16 倍预留的 **22%**（约均衡路由的 3.5x），训练 MFU 从 **15% 爬升至 26% -> 28%**，权重同步耗时 `11–13 s`。 |
| `volcengine/verl#8284` (`d430157d15`, 2026-10-23) & `#8ced5d2f4b` (2026-10-18) | `verl/utils/kernel/{linear_cross_entropy,kernels}.py`, `verl/experimental/agent_loop/agent_loop.py` | `#8284` 将 Linear Cross-Entropy 内核 Host 代码与 18 个测试文件中的硬编码 `"cuda"` / `"nccl"` / `torch.cuda.nvtx` 替换为 `verl.utils.device` 与 `get_platform().nvtx_range(...)` 抽象，并修复 CPU-only 环境下 `get_device_name()` 默认回退 `"cuda"` 导致缺少 `is_device_available()` 守卫时触发 Ray/NCCL 崩溃的问题；`#8ced5d2f4b` 在 Agent Loop 中严格校验 sampler support receipts 的布尔源类型，拒绝数值/NaN 混入。 |
| `NVIDIA/Megatron-LM#6502` (`2052769557`, 2026-10-23) & `#7926` (`7d3f0fb3b5`, 2026-10-23) | `megatron/training/training.py`, `megatron/core/inference/engines/dynamic_engine.py` | `#6502` 修复 Checkpoint 格式转换路径中的完整性校验配置缺失；`#7926` 移除 Dynamic Inference Engine 中已废弃的 `disable_ep_consensus` 标志及 64 行冗余协调器分支，统一 EP 动态推理共识路径。 |
| `kvcache-ai/Mooncake#3424` (`0afc417657`, 2026-10-23) | `mooncake-store/include/nof/*`, `mooncake-store/src/nof/*` | 将单例 `SpdkWrapper` 解耦为 `NVMeoFInitiator`（基于不透明 `NofSegmentHandle` 的字节段 I/O）与 `DmaBufferAllocator`（2MB 大页引用计数表 `NofPageRegistry`）两大角色接口，并将 SPDK/DPDK 静态库隔离进 `mooncake_nof_spdk` pimpl 实现；引入引用计数 `SpdkEnvGuard` 保持 SPDK 环境直至进程退出，根治 SPDK v23.01.1 在 `spdk_env_fini` 后再次调用 `spdk_env_init` 返回 `-EINVAL` 导致客户端重连失败的顽疾，且在 `spdk_env_init` 前后保存/恢复 CPU 亲和性掩码防止 `rte_eal_init` 锁死单核。 |
| `LMCache/LMCache#5150` (`6913259264`, 2026-10-24) & `#4410` (`00f6d71ed1`, 2026-10-24) | `lmcache/integration/vllm/lmcache_mp_metadata.py`, `lmcache/v1/kv_layer_groups.py`, `csrc/sycl/mp_mem_kernels_sycl.cpp` | `#5150` 将多进程（MP）缓存键与元数据中的 `token_ids` 从 Python `tuple[int, ...]` 替换为紧凑 `uint32` 字节缓冲区（`IPCCacheServerKey.from_token_ids` + `num_packed_tokens`），对 32,768-token 长上下文请求将单次键载荷从 `~1,179,696 B` 压缩至 **`131,072 B`（缩减 9.0x）**；`#4410` 通过 `KVLayerGroupsManager` 实现异构混合 KV 层组（如 Qwen3.5 GDN + Attention）的引擎驱动多组传输，并在 SYCL 聚集/散播描述符中保留真实 Padded Block Stride。 |

## 2. 数学模型与架构师计算法则

### 2.1 RL 训练中 Expert FSDP 组规模与跨机架通信复杂度
在总训练卡数 $N_{\text{gpu}}$ 下，混合专家层的并行度满足：
$$N_{\text{gpu}} = \text{FSDP}_{\text{expert}} \times \text{TP} \times \text{EP}$$
而注意力层（非专家参数）的并行度满足：
$$N_{\text{gpu}} = \text{FSDP}_{\text{attn}} \times \text{TP} \times \text{CP}$$
在 TorchTitan `#5581` 中：
- **旧布局（256 GPUs）**：$\text{TP}=4, \text{EP}=4, \text{CP}=1 \implies \text{FSDP}_{\text{expert}} = 256 / (4 \times 4) = 64$。每个路由专家的 BF16 All-Gather 与 FP32 Reduce-Scatter 必须跨越 64 张卡（跨多个机架），通信完全主导步耗时。
- **新布局（128 GPUs）**：$\text{TP}=4, \text{CP}=4, \text{EP}=16 \implies \text{FSDP}_{\text{expert}} = 128 / (4 \times 16) = \mathbf{8}$。专家 FSDP 组缩小 **8.0x**，且每个 $\text{EP}=16$ 组恰好落在 4 台 GB300 主机组成的单一 NVLink 域内；同时单卡 token 激活分片因子从 $\text{TP} \times \text{CP} = 4$ 增至 **16**，使训练硬件减少一半（256 $\to$ 128 GPUs）的同时 MFU 达 **28%**。

### 2.2 SM100 HD256 SplitKV 的非空块一致性谓词
当 KV 序列长度 $S_k$ 被切分为 $B_{\text{total}} = \lceil S_k / B_N \rceil$ 个块并分配给 $K_{\text{split}}$ 个 Split 时，每个 Split 分得块步长 $B_{\text{step}} = \lceil B_{\text{total}} / K_{\text{split}} \rceil$。第 $i$ 个 Split 的范围为：
$$n_{\min}(i) = i \cdot B_{\text{step}}, \quad n_{\max}(i) = \min\big((i+1) \cdot B_{\text{step}}, B_{\text{total}}\big)$$
当 $S_k = 384, B_N = 128, K_{\text{split}} = 4$ 时，$B_{\text{total}} = 3, B_{\text{step}} = 1$，第 3 号 Split ($i=3$) 得到 $n_{\min}=3, n_{\max}=3$。通过统一谓词 `has_kv_work = (n_block_min < n_block_max)`，Load、MMA、Softmax 与 Correction 四类 Warp Role 同步跳过空 Split，消除 Warp Specialization 流水线挂起。

## 3. 可复现实验验证（`experiments/fa4_verl_mcore_mooncake_phase87`）

运行 `python experiments/fa4_verl_mcore_mooncake_phase87/run.py --check` 通过全部确定性断言：
- `paged_kv_cudagraph_guard.host_syncs_during_capture_after == 0`；
- `sm100_hd256_2cta_splitkv` 在 $S_k=384, K_{\text{split}}=4$ 下 `active_splits == 3, skipped_empty_splits == 1`；
- `torchtitan_muse_spark_gb300` 训练卡数缩减 `2.0x`（`256 -> 128`），专家 FSDP 组缩减 `8.0x`（`64 -> 8`），Token 分片因子增至 `16`；
- `lmcache_packed_uint32` 在 32K tokens 下载荷缩减 `9.0x`（`1,179,696 B -> 131,072 B`）。
