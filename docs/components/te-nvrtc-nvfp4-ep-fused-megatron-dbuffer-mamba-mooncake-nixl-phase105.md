# TransformerEngine NVRTC NVFP4 `4over6` 量化与图捕获融合 EP 分发、Megatron-LM `DBuffer` 直连 P2P 与批不变量 Mamba 前缀缓存、Mooncake 借用缓冲字节区间与 COO 稀疏传输规划器及 NIXL GPU-to-CPU 代理环（Phase 105）

> **Language / 语言**: **中文 (Chinese)** | [English (英文版)](te-nvrtc-nvfp4-ep-fused-megatron-dbuffer-mamba-mooncake-nixl-phase105.en.md)

## 1. 架构背景与核心问题

在大规模 NVFP4 混合精度训练、CUDA Graph 专家并行（EP）、MFSDP v2 分布式缓冲、确定性 Mamba 推理以及稀疏 RL 权重同步与 GPU 发起的存储/网络传输中，训练与传输栈面临五大瓶颈：
1. **NVFP4 `4over6` 静态模板膨胀与 EP `prepare + dispatch` 主机同步断点**：静态编译所有 NVFP4 `4over6` 量化模板（24 个静态实例化）大幅拉长构建时间；而 MoE EP 在非 Eager（CUDA Graph）模式下因 `prepare` 阶段需要回传主机 `recv` 计数而触发 `cudaStreamSynchronize`（每层 **`1` 次主机同步，`46.0 us`**）。
2. **MFSDP v2 聚集/散射中间暂存缓冲与 `batch_invariant_mode` 下 Mamba 前缀缓存禁用**：MFSDP v2 旧版 `GroupOwnerLayout` 需要分配 **`256.0 MiB`** 暂存缓冲区拼装分片；同时 Megatron-LM 在 `batch_invariant_mode=True` 时此前直接通过 `assert not self.enable_prefix_caching` 禁用 Mamba 前缀缓存。
3. **Mooncake 结构化对象跨块切片拷贝与稀疏 LoRA/RL Delta 全量传输**：读取结构化对象的成员字节区间或同步 12.5% 活跃度的稀疏权重 Delta 时，若按稠密全张量（`64.0 MiB`）拉取并分配临时缓冲，会浪费超过 85% 的 RDMA 带宽。
4. **NIXL GPU 内核直调 CPU 代理的 Warp 门铃争用**：当 GPU 内核通过 `PROXY` 模式向主机运行时提交命令时，若 Warp 内 32 个线程各自写入环形队列，将引发 32 倍原子门铃争用。

## 2. 第一手源码实现与数学不变量

### 2.1 TransformerEngine NVRTC NVFP4 `4over6`（`ec39dcf368920d3be1e9d5ae6a5604226fcc939b`）与图捕获融合 EP `prepare+dispatch`（`f7273d7d806e1905afc08e19c5d5a3de07a25eb3`）
- **架构感知 NVRTC NVFP4 编译**：新增 `transformer_engine/common/cast/nvfp4/rtc_dispatch.cpp`，按当前 GPU 架构在运行时通过 NVRTC 编译 `quantize_4over6_kernel.cuh` 并优化启动键查找，默认构建下静态 AOT 实例化从 **`24 -> 0`（削减 `100%` 静态编译膨胀）**，同时保留 `NVTE_BUILD_LEGACY_STATIC_NVFP4=ON` 静态兜底。
- **CUDA Graph 兼容的融合 EP `prepare+dispatch`**：在 `transformer_engine/common/ep/ep_api.cpp` 中支持调用方提供 `recv` 缓冲区与设备端 `count-mode` 接收计数（`NVTE_EP_FUSED_PREPARE_DISPATCH=1`），消除每层主机同步（**`1 -> 0` sync/layer**），使单层 EP 准备与分发延迟从 **`46.0 -> 21.5 us`（提速 `2.1395x`）**。

### 2.2 Megatron-LM MFSDP v2 `DBuffer` 直连 P2P（`a666bd60efcb2d1b8038e46bf2e3d0cc9aaa3fee`）、批不变量 Mamba 前缀缓存（`3888b34882eb85aecf6ec9ecc3f4020b7608b497`）与 TorchTitan 逐核报错（`efdde5edaf2acad28782f5e8a851f9e9ecdb51d6`）
- **`DBuffer` Owner 直连 P2P 接收**：`gather_scatter.py` 与 `owner_planning.py` 直接利用 `DBuffer` 的布局与归属映射，使任意对等 Rank 对最多共享一段连续切片并直接写入目标视图，省去 **`256.0 MiB` 暂存显存**并将 8 卡 P2P 聚合/散射耗时从 **`4.80 -> 3.10 ms`（`1.5484x`）**。
- **对齐 SSM 块边界的批不变量 Mamba 前缀缓存**：`dynamic_context.py` 与 `dynamic_engine.py` 将断言放宽为 `block_size_tokens % ssm_chunk_alignment == 0`（如 `256 % 128 == 0`），并在 `_mamba_batch_invariant_prefill_chunk_length` 中保证非末次分块严格落在 SSM chunk 边界上。在 4,096 tokens 提示词命中 75%（3,072 tokens）前缀时，计算量从 **`4,096 -> 1,024` tokens（`4.0x` 加速）** 且保持 **`0.0` 逐位状态漂移**。
- **TorchTitan Attention Gym 逐核缺失后端诊断**：修复 `cp_{gdn,kda,linear_attention}.py` 使占位符仅抛出当前模块自身的导入错误而非首个全局错误。

### 2.3 Mooncake 借用缓冲字节区间与 COO 稀疏传输规划器（`81b3694b1bcde8af454457f835734f1306420fb3`）、LMCache 共享 `KVEventSubscriber`（`a0d918a77c`, `6993f46c91`）、Dynamo LPX Sidecar（`9ea06e0cd3`）与 NIXL GPU-to-CPU 代理环（`f1de8fc2860c0291763504891432576aa11f95b6`）
- **Mooncake `materialize_ranges_into` 与 COO 稀疏 Delta 规划**：`_structured_ranges.py` 将跨块字节区间切分并直接读入调用方已注册的借用缓冲区（**`0` 次中间缓冲分配**）；COO 稀疏权重传输规划器将 12.5% 密度的 `64.0 MiB` 权重同步流量压降至 **`8.25 MiB`（`8.0 MiB` 有效瓦片 + `0.25 MiB` 紧凑索引，带宽削减 `7.7576x`）**。
- **NIXL `nixl_device_exec_mode_t` 与 64B CPU 代理环**：`device_types.cuh` 定义 `UCX_DIRECT = 1` 与 `PROXY = 2`（`0` 为非法未标记句柄防误调），`device_proxy_ring.cuh` 通过 `isLeader` 选举单线程构造 **64 字节**环形命令并由 `broadcastStatus` 广播完成状态，将 Warp 门铃争用降低 **`32.0x`（`32 -> 1`）**。

## 3. 确定性实验与量化对比

运行 `python experiments/te_nvrtc_nvfp4_ep_fused_megatron_dbuffer_mooncake_phase105/run.py --check`：

| 指标 | 优化前基线 | Phase 105 实现 | 提升幅度 / 结果 | 证据分类 |
|---|---:|---:|---:|---|
| TE NVFP4 `4over6` 默认静态 AOT 模板实例数 | `24` | `0` (NVRTC 按需编译) | **`-100.0%` 静态构建膨胀** | `implementation` + `own-offline` |
| TE CUDA Graph 融合 EP `prepare+dispatch` 延迟 | `46.0 us` (`1` sync) | `21.5 us` (`0` sync) | **`2.1395x`** | `implementation` + `own-offline` |
| Megatron MFSDP v2 `DBuffer` P2P 耗时 / 暂存显存 | `4.80 ms` / `256.0 MiB` | `3.10 ms` / `0.0 MiB` | **`1.5484x` / 省 `256.0 MiB`** | `implementation` + `own-offline` |
| 批不变量 Mamba 75% 命中预填充计算 Token 数 | `4,096` tokens | `1,024` tokens | **`4.0x` (`0.0` 逐位漂移)** | `implementation` + `own-offline` |
| Mooncake 12.5% 稀疏 COO 权重 Delta 传输量 | `64.0 MiB` | `8.25 MiB` (`0` 临时分配) | **`7.7576x` 带宽削减** | `implementation` + `own-offline` |
| NIXL GPU-to-CPU 64B 代理环每 Warp 写入线程数 | `32` threads | `1` leader thread | **`32.0x` 门铃争用降低** | `implementation` + `own-offline` |
