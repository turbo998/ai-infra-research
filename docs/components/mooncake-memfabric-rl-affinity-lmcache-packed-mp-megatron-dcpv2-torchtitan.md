# Mooncake MemFabric 批量注册与 RL HCA 亲和性、LMCache 紧凑 uint32 Token 编码与混合 KV 分组、Megatron-LM DCP V2 及 TorchTitan FA3/FA4 校验

> English version: [mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan.en.md](mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan.en.md)

## 1. 背景与核心问题

在跨节点 KV 缓存传输（Mooncake、LMCache）与超大规模分布式训练（Megatron-LM、TorchTitan、TransformerEngine、Dynamo）中，系统架构师面临四类关键基础设施瓶颈：

1. **NPU 大页对齐下的相邻内存块重复注册冲突与 RL 跨 NUMA/HCA 远端传输 (`Mooncake #4870, #4631`)**：在昇腾/NPU MemFabric 传输层中，若对 `std::vector<BufferDesc>` 逐个调用单缓冲区注册，底层按 NPU 大页（如 2 MiB）向上/向下对齐后会导致相邻缓冲区产生页重叠报错；同时在强化学习 (RL) 权重/KV 高频 `put`/`get` 中，随机段分配器若忽略客户端优选 RDMA HCA 列表，会导致跨 PCIe root complex / QPI 的慢路径传输。
2. **多进程 KV 缓存服务中 Python Token 元组的序列化膨胀与混合 GDN/Attention 窗口传输 (`LMCache #5150, #4410`)**：在 LMCache 多进程模式下，token ID 序列每步跨越 3 次进程边界（Scheduler → Client → Server），以 `tuple[int, ...]` 传递时每个 token 需实例化 28 字节 `PyLongObject` 加 8 字节元组槽指针（共 `36 B/token`）；同时在 Qwen3-Next 等 GDN 线性注意力 + 全注意力混合模型中，不同层组具有截然不同的跨块窗口需求。
3. **动态上下文并行 (DCP V2) 独立调度器维护成本与非 2 的幂 `DPxCP` 拓扑切分 (`Megatron-LM #7047, #6502`)**：第一代混合 CP 依赖 687 行独立的 `hybrid_cp_schedule.py` 和 `HybridCPDataLoaderWrapper`，且在检查点格式转换 (`--ckpt-convert-format`) 时过早覆写 `args.ckpt_format` 会导致源 `torch_dist` 检查点跳过完整性校验 (`verify_integrity`)。
4. **Hopper/Blackwell 变长注意力隐式降级风险与 `attn_gym` 惰性导出定位困难 (`TorchTitan #5546, #5565`)**：在 `SM90` (FA3) 或 `SM100` (FA4) 上执行 varlen 注意力时，若缺少对应 `flash_attn_interface` 或 `flash_attn.cute` wheel，必须显式报错拦截而非静默运行错误路径；而在普通 SDPA 回退路径上，需严格验证 PyTorch FA2 的前向 Loss 与反向梯度范数零漂移，并显式拒绝 MLA 非对称 `QK != V` head dim。

---

## 2. 第一手源码实现路径审计

### 2.1 Mooncake TENT `SmemTransBatchRegisterMem` 批量注册与三层 HCA 亲和性 (`Mooncake@6a835cbf934a`, #4870; `868a6d05abf9`, #4631)

- **MemFabric 批量内存注册避免 NPU 页对齐重叠 (#4870)**：`MemFabricTransport::addMemoryBuffer(std::vector<BufferDesc>& desc_list, ...)` 不再循环调用单缓冲区 `addMemoryBuffer`，而是收集全部 `addrs` 与 `caps` 后一次性调用 `MemFabricUnderApi::SmemTransBatchRegisterMem(handle_, addrs.data(), caps.data(), count, 0)`。例如两个紧邻的 `1.5 MiB` 缓冲区（`0x10000000..0x10180000` 与 `0x10180000..0x10300000`）共享同一个 `2 MiB` NPU 页（页号 `128`），单次批量调用允许 MemFabric 内部先合并相邻区间再执行页对齐注册，彻底消除了页重叠失败。
- **Mooncake Store 请求级三层 HCA 亲和度分配与副本选择 (#4631)**：通过 `MC_STORE_HCA_AFFINITY=1` 与 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`，`AllocatorManager::SetHcaAffinity` 与 `RandomAllocationStrategy` 将候选内存段划分为 **3 个优先级层 (`tiers = 3`)**：
  - **Tier 0**：段分配器 `LocalityHandle` 匹配请求方优选 HCA 列表（同 PCIe/NUMA 域）；
  - **Tier 1**：未上报局部性拓扑 (`unknown locality`) 的内存段；
  - **Tier 2**：已知 HCA 不匹配但仍有剩余容量的远端段（保底不因亲和性导致分配失败）。

### 2.2 LMCache 大端序 `uint32` Token 紧凑编码与混合 KV 层组窗口管理 (`LMCache@6913259264db`, #5150; `00f6d71ed1af`, #4410)

- **`pack_token_ids` 大端序 `uint32` 零拷贝传输与 Blake3 哈希兼容 (#5150)**：`lmcache/v1/multiprocess/token_codec.py` 将 token 序列打包为 `TOKEN_STRIDE = 4` 字节的大端序 `uint32` `bytes`（在小端主机上利用 `array.array("I", token_ids)` + `words.byteswap()` 实现比 `struct.pack` 快 ~2x 的批量转换）。因为 `_make_blake3_hash_func` 原本就接收大端序 `uint32` 字节流，打包后的 `token_bytes` 可直接送入 Blake3 哈希而无需解包，且与既有缓存键 **100% 哈希兼容**。在 `32,768` token 序列上，单跳内存占用从 `1,179,648 B` (`36 B/token`) 降至 `131,072 B` (`4 B/token`)，缩减 **`9.0x`**。
- **`KVLayerGroupsManager` 混合 GDN/Attention 窗口块计算与 SYCL 跨距扩展 (#4410)**：`get_kernel_group_window_chunks(kernel_group_idx)` 按内核组独立计算尾部所需 LMCache 块数：全注意力组返回 `-1`（全序列）；滑动窗口或循环状态组（如 GDN）返回 $\max(1, \lceil \text{sw\_size\_tokens} / \text{chunk\_size} \rceil)$。当开启 `full_sw_kv=True` 时，滑动窗口注意力组拓宽为 `-1`，但 `recurrent_state=True` 的 GDN 组仍严格保留 `1` 个尾部状态块 (`[-1, -1, 1]`)。同时 SYCL 内存核支持 `NL_X_NB_BSV_BSS`、`NL_X_NB_BS_NH_HS` 及 `is_fused_packed` 格式的显式填充 `block_stride_elems`。

### 2.3 Megatron-LM DCP V2 标准调度集成与检查点完整性保护 (`Megatron-LM@ea99fdfce50a`, #7047; `2052769557dc`, #6502)

- **DCP V2 拓扑子组划分与移除独立调度器 (#7047)**：`get_valid_dynamic_context_parallel_group_sizes(total_size)` 枚举所有能整除 `total_size` (`DP x CP`) 的 2 的幂子组大小，并始终将完整的 `total_size` 追加在末尾以复用既有通信组。例如当 `DP x CP = 24`（非 2 的幂）时，合法运行时 CP 组大小为 **`[1, 2, 4, 8, 24]`**。同时将 DCP V2 直接整合进标准 `data_schedule.py` 与训练/评估循环，删除了 687 行独立的 `hybrid_cp_schedule.py`。
- **检查点转换期间的源完整性校验保护 (#6502)**：在 `_build_model_wrapper` 处理 `--ckpt-convert-format` 时，确保在加载源 `torch_dist` 检查点阶段保留用户配置的 `verify_integrity`，仅在切换至非 `torch_dist` 目标输出格式（如 `torch`）准备保存前才将 `args.verify_integrity` 设为 `False`。

### 2.4 TorchTitan FA3/FA4 显式模块核验、FA2 精确回退与 `attn_gym` 隔离 (`torchtitan@89e4339c963c`, #5546; `3ed5aa976fdf`, #5565)

- **Varlen Attention 的 FA3/FA4 `find_spec` 门控与 SDPA FA2 零误差回退 (#5546)**：`get_cuda_flash_attention_impl()` 在 `SM100+` 上要求 `importlib.util.find_spec("flash_attn.cute")` (`FA4`)，在 `SM90` 上要求 `find_spec("flash_attn_interface")` (`FA3`)，缺失时直接抛出含安装指引的 `ValueError`。在标准 SDPA 回退至 PyTorch FA2 的回归测试中，前向 Loss 精确匹配 `0.19502778351306915` (`atol=0, rtol=0`)、反向梯度 L2 范数精确匹配 `0.012680011801421642`，并显式拦截 MLA 非对称 `QK != V` head dim。
- **`attn-gym==0.0.17` 逐导出项构建失败隔离 (#5565)**：通过 `_BUILD_FAILURE_CASES` 9 组单点故障注入（分别破坏 `short_conv`、`causal_conv1d`、`chunk_kda`、`bound_gate`、`l2norm`、`chunk_gdn`、`recurrent_gdn`、`context_parallel_kda`、`context_parallel_gdn`），验证 `require_attn_gym` 声明列表中的每一项均为承载式依赖 (load-bearing)。

---

## 3. 可复现 CPU/Offline 验证实验与量化汇总

```bash
python experiments/mooncake_memfabric_rl_affinity_lmcache_packed_mp_megatron_dcpv2_torchtitan/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| 组件 / 仓库 | 关键机制 / 函数 | 修复前痛点 / 基线 | 修复后定量指标 |
|---|---|---|---|
| Mooncake MemFabric (#4870) | `SmemTransBatchRegisterMem` | 逐 buffer 注册在 2 MiB NPU 页边界重叠报错 | **1 次批量注册调用**，0 页重叠冲突 |
| Mooncake Store HCA 亲和 (#4631) | 3-Tier `HcaAffinityTier` | 随机段分配忽略客户端 RDMA HCA 局部性 | **Tier 0 -> Tier 1 -> Tier 2** 三级无损回退 |
| LMCache Token Codec (#5150) | 大端序 `uint32` `pack_token_ids` | `tuple[int]` 每 token 占 `36 B` 且每跳重分配 | **`4 B/token`** (**`9.0x`** 内存缩减)，Blake3 零解包直算 |
| Megatron-LM DCP V2 (#7047) | `get_valid_dynamic_context_parallel_group_sizes` | 687 行独立 `hybrid_cp_schedule.py` | 删除 **687 行** 冗余调度器；`total=24` -> `[1, 2, 4, 8, 24]` |
| TorchTitan FA3/FA4 & `attn_gym` (#5546, #5565) | `find_spec` + 9 组 `_BUILD_FAILURE_CASES` | 缺失 FA3/FA4 wheel 时行为不透明 | SDPA FA2 回退 `atol=0, rtol=0`；**9 组** 导出依赖 100% 校验 |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **架构设计练习（非 2 的幂集群下的 DCP V2 序列分桶）**：在 3 节点 × 8 卡 (`DP x CP = 24`) 的 H100/B200 集群上训练最长 `192K` 上下文模型时，合法 DCP 组大小为 `[1, 2, 4, 8, 24]`（不存在 `12` 或 `16`）。请设计一套序列长度到 `cp_size` 的阈值映射表，使得单卡激活显存不超过 `cp_size = 8` 时的上限，并计算 `cp_size` 从 `8` 跳变到 `24` 时的 All-to-All / Ring-Attention 通信域跨节点跳数变化。
2. **可证伪后续问题**：在 Mooncake Store 的 3 层 HCA 亲和分配中，当高并发 RL Rollout 导致 Tier 0（同 HCA 本地内存段）满载达 90% 时，继续坚持 Tier 0 顺序探测相比直接溢流至 Tier 2（同机跨 NUMA HCA）的尾部 P99 分配延迟交叉点在哪里？
