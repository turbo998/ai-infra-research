# Megatron-LM 视觉 FP32 RoPE 精度保持与确定性 NVLS AllReduce、TransformerEngine ZeroTensor 虚拟权重梯度与 SM12x FA4 确定性防护、Mooncake EFA 过期描述符重取与动态 PeerMem 探测、LMCache 请求标签键、Dynamo 按需 Token 解码与 NIXL 共享锁探针（Phase 120）

> **模块定位**：面向负责大规模分布式训练（Megatron-LM / TransformerEngine）与多级 KV / 传输底座（Mooncake / LMCache / Dynamo / NIXL）的解决方案架构师。本文档基于 10 个核心提交，系统剖析视觉 RoPE 在 `Float16Module` 下的 FP32 频率保护、Blackwell 确定性 `Ring,NVLS` 规约、共享 MTP/LM Head 的 `ZeroTensor` 零开销累加、EFA 动态挂载段描述符单次刷新、GB300 开源内核模块 `dma_buf` 自动回退及 `flock(LOCK_SH)` 并发探针无锁冲突设计。
> **证据等级**：`implementation`（源码固定 commit 级核验）+ `author-benchmark`（上游 PR 实测数据）+ `own-offline`（本仓库 Python 标准库可复现建模验证）。

---

## 1. 核心架构问题与 10 项上游证据总览

| 序号 | 仓库与 Commit | 日期 | 核心机制与架构变更 | 证据分类 |
| :--- | :--- | :--- | :--- | :--- |
| C1 | `NVIDIA/Megatron-LM` [`3ca52d9`](https://github.com/NVIDIA/Megatron-LM/commit/3ca52d95d6f4ce8cec12103af40cd774be43ee30) | 2026-10-27 | `vit_model.py`：`Pixtral2DRotaryEmbedding` 将 `self.freqs` 移出 `register_buffer`，防止 `Float16Module` 将频率截断为 BF16/FP16 | `implementation` |
| C2 | `NVIDIA/Megatron-LM` [`62c063f`](https://github.com/NVIDIA/Megatron-LM/commit/62c063f2b731b57b1c2031b2b4f4c6a769271ccb) | 2026-10-18 | `determinism.py`：`--deterministic-mode` 在非 Hopper 架构上允许 `NCCL_ALGO=Ring,NVLS` 与空字符串 `""`，并通过 `libc.getenv` 校验 | `implementation` |
| C3 | `NVIDIA/TransformerEngine` [`fc48751`](https://github.com/NVIDIA/TransformerEngine/commit/fc48751409f2d0bd8c07260efcf9803e84eb52de) | 2026-10-13 | `module/base.py`：引入 `zero_tensor_dummy_wgrad`（`torch._efficientzerotensor`）并按 `(*shape, dtype, device)` 缓存虚拟梯度 | `implementation` + `author-benchmark` |
| C4 | `NVIDIA/TransformerEngine` [`6d6a500`](https://github.com/NVIDIA/TransformerEngine/commit/6d6a500860d0fb60bd4438a0df0dd4d5d0846f1b) | 2026-10-06 | `dot_product_attention`：在 SM12x (`arch == 12`) 且 `deterministic and requires_backward` 时禁用 FA4，并按张量所属 `device` 查询算力 | `implementation` |
| C5 | `kvcache-ai/Mooncake` [`c596310`](https://github.com/kvcache-ai/Mooncake/commit/c596310fbb184d30e2ff557876ad41a29069089a) | 2026-10-28 | `efa_transport.cpp` / `efa_context.cpp`：当目标地址不在已缓存描述符中时，按每个 `target_id` 单次强制刷新描述符并重试 | `implementation` |
| C6 | `kvcache-ai/Mooncake` [`c8fe63e`](https://github.com/kvcache-ai/Mooncake/commit/c8fe63e52cbb9e1af6c3ed760d6ce1b7262ba0d4) | 2026-10-28 | `ascend_direct_transport`：`InjectNicTopoPathFromCustomTopoEnvTe` 将 `MC_CUSTOM_TOPO_JSON` 注入 HIXL `comm_resource_config.nic_topo_path` | `implementation` |
| C7 | `kvcache-ai/Mooncake` [`cb670d5`](https://github.com/kvcache-ai/Mooncake/commit/cb670d5847b128fea7d8606327911475f48ec8ac) | 2026-10-28 | `config.cpp`：`loadGlobalConfig()` 通过 `/sys/module/nvidia_peermem` 或 `nv_peer_mem` 动态决定默认使用 `ibv_reg_mr` 或 `dma_buf` | `implementation` |
| C8 | `LMCache/LMCache` [`e45d4d5`](https://github.com/LMCache/LMCache/commit/e45d4d5a85bdb0f3b1343b561a8b3f616774b0d2) | 2026-10-22 | `mooncake_lookup_client.py`：`lookup()` 将 `request_configs` 传给 `token_database.process_tokens()` 以包含请求级隔离标签 | `implementation` |
| C9 | `ai-dynamo/dynamo` [`cffc4fa`](https://github.com/ai-dynamo/dynamo/commit/cffc4fa75c15017bcd32e62f7d0fe4bad468c35d) | 2026-10-27 | `lib/llm/src/backend.rs`：仅当 `data.log_probs.is_some()` 时在 `process_token_ids_with_token_text` 中解码逐 Token 文本 | `implementation` |
| C10 | `ai-dynamo/nixl` [`842eedf`](https://github.com/ai-dynamo/nixl/commit/842eedf0117fe19d864414df980576e55c939c6b) | 2026-10-12 | `prometheus_mp`：`storeWriterAlive()` 与 `heldAddressesExcept()` 改用 `flock(LOCK_SH \| LOCK_NB)` 消除并发探针互斥误判 | `implementation` |

---

## 2. 深度技术剖析与公式推导

### 2.1 Megatron-LM 视觉 RoPE FP32 频率保持与确定性 NVLS AllReduce（`3ca52d9`, `62c063f`）

1. **Pixtral 2D RoPE FP32 频率保护（`3ca52d9`）**：
   - 原 `Pixtral2DRotaryEmbedding` 使用 `self.register_buffer("freqs", freqs, persistent=False)`。当模型被包裹进 `Float16Module` 时，模块递归转换会将所有 buffer 降精度为 BF16/FP16。在大尺寸视觉网格（如 `110×110`，最大坐标 $p = 109$）下，低频项 $\omega_k = 10000^{-2k/d}$ 被截断为 BF16（7 位尾数）后再乘以 $p$，最大相位角误差达到 `0.029879 rad`，余弦值误差达 `0.025852`；
   - 修复后将 `self.freqs` 作为普通 Python 属性持有并在 `forward` 中按需迁移到目标 `device`，确保 `freqs` 始终保持 FP32。
2. **确定性模式支持 `Ring,NVLS` 与空 `NCCL_ALGO`（`62c063f`）**：
   - NVLS（NVLink SHARP）树形多播规约在 Blackwell (`sm_100`+) 上具备逐位确定性，但在 Hopper (`arch == 9`) 老驱动下会因进程内先分配的多播对象改变归约顺序（`NVIDIA/nccl#2360`）；
   - `apply_determinism_to_args` 现允许 `NCCL_ALGO="Ring,NVLS"`（强制要求 `Ring` 同时存在以支持 Broadcast/Reduce）或 `NCCL_ALGO=""`，并在 `get_device_arch_version() == 9` 时显式断言拒绝；
   - 由于 PyTorch C++ 在 `torch.use_deterministic_algorithms(True)` 下仅当 `NCCL_ALGO` 未设置时写入 `NCCL_ALGO=^NVLS`，`assert_nccl_algo_still_empty()` 在首次进程组初始化后通过 `ctypes.CDLL(None).getenv(b"NCCL_ALGO")` 校验底层 C 环境变量仍为空字符串 `b""`。

### 2.2 TransformerEngine `ZeroTensor` 虚拟权重梯度与 SM12x FA4 确定性防护（`fc48751`, `6d6a500`）

1. **ZeroTensor 虚拟权重梯度（`fc48751`）**：
   - 开启 `fuse_wgrad_accumulation=True` 时，权重梯度已直接累加进 `main_grad`，算子向 PyTorch Autograd 返回一个 dummy `wgrad`。当同一权重在一次反向传播中被使用两次（如 Megatron-LM 中主 LM Head 与 MTP Head 共享 `[vocab=129280, hidden=7168]` 权重）时，Autograd 会对两个 dummy 张量执行一次无意义的逐元素加法（读取 2 个输入 + 写出 1 个输出，共 $3 \times 129280 \times 7168 \times 2\text{ B} = 5308.42\text{ MiB}$ HBM 流量，在 VR200 上耗时 **0.44 ms**）；
   - 设置 `weight.zero_tensor_dummy_wgrad = True` 后，`get_dummy_wgrad` 返回 `torch._efficientzerotensor(shape, dtype=dtype, device=device)`（零显存分配、加法 O(1) 空操作），同时将普通 dummy 缓存键扩展为 `(*shape, dtype, device)` 修复多 GPU 同形状设备冲突。
2. **SM12x 确定性反向传播跳过 FA4（`6d6a500`）**：
   - CuTe SM12x（`device_compute_capability[0] == 12`）FlashAttention-4 反向内核尚不支持确定性执行。TE 新增 `requires_backward = torch.is_grad_enabled() and any(t.requires_grad ...)`，仅当 `deterministic and requires_backward and arch == 12` 时回退禁用 FA4，保留纯前向推理的 FA4 加速。

### 2.3 Mooncake EFA 过期描述符重取、HIXL 拓扑注入与动态 PeerMem 探测（`c596310`, `c8fe63e`, `cb670d5`）

- **EFA 过期段描述符按目标单次重取（`c596310`）**：Mooncake Store 节点以 `max_mr_size` 分块挂载大内存段时，客户端首次通信缓存的 `SegmentDesc` 不包含后续注册的分块。修复后在 `EfaTransport::submitTransferTask` 与 `EfaContext::submitPostSend` 中维护 `refreshed_targets` / `refreshed_descs`，每个 `target_id` 每次调用最多执行 **1 次** `getSegmentDescByID(target_id, /*force_update=*/true)`，使指向新挂载块的 32 个 Slice 从全部报 `TRANSFER_FAIL (-800)` 变为 **0 失败**。
- **Ascend HIXL `MC_CUSTOM_TOPO_JSON` 注入（`c8fe63e`）**：`InjectNicTopoPathFromCustomTopoEnvTe` 将 `MC_CUSTOM_TOPO_JSON` 自动写入 HIXL `comm_resource_config.nic_topo_path`（优先级高于 `MC_TENT_CONF`）。
- **GB300 动态 PeerMem 探测（`cb670d5`）**：`peerMemoryModuleLoaded()` 检查 `/sys/module/nvidia_peermem` 或 `/sys/module/nv_peer_mem`。在搭载 NVIDIA 开源内核模块（如 610.57.04）与内核原生 `rdma-core` 的 GB300 主机上，未设置 `WITH_NVIDIA_PEERMEM` 时自动默认走 `dma_buf` 导出，彻底消除启动时 `ibv_reg_mr()` 触发的 `Bad address [14] (EFAULT)`。

### 2.4 LMCache 请求标签、Dynamo 按需 Token 解码与 NIXL 共享锁探针（`e45d4d5`, `cffc4fa`, `842eedf`）

- **LMCache `MooncakeLookupClient` 标签透传（`e45d4d5`）**：在 `lookup()` 中向 `self.token_database.process_tokens(token_ids, request_configs=request_configs)` 传入 `request_configs`，保证 `lmcache.tag.*` 隔离标签在 Lookup 与 Retrieve 阶段生成的 `CacheEngineKey` 完全一致。
- **Dynamo 跳过无 Logprobs 的逐 Token 字符串解码（`cffc4fa`）**：在 Rust 流式后端中令 `needs_token_text = data.log_probs.is_some()`，当请求未开启 `logprobs` 时跳过每个生成步的 `String` 分配与 UTF-8 解码（512 输出 Token 减少 **512 次** 堆分配）。
- **NIXL `prometheus_mp` 共享锁活性探针（`842eedf`）**：Writer 持有排他锁 `LOCK_EX | LOCK_NB`；当两个 Scraper 同时探测已退出的进程文件时，旧版探针使用 `LOCK_EX | LOCK_NB` 会互相阻塞并误判僵尸文件仍存活。新版将探针改为共享锁 `LOCK_SH | LOCK_NB`：存活 Writer 的 `LOCK_EX` 仍会拒绝 `LOCK_SH`，而并发探针之间不再互斥。

---

## 3. 本仓库离线建模验证（`own-offline`）

对应实验目录：`experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/`

- **运行命令**：`python experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/run.py --check`
- **关键输出校验**：
  - `megatron_vit_and_nvls.vit_rope_max_angle_rad_err_if_cast_to_bf16 = 0.029879`，`ring_nvls_on_blackwell_sm100 = true`，`ring_nvls_on_hopper_sm90 = false`；
  - `te_zerotensor_and_fa4.avoided_dummy_add_hbm_mib = 5308.42`，`vr200_saved_ms_per_backward = 0.44`，`sm120_det_bwd_fa4_enabled = false`；
  - `mooncake_efa_hixl_peermem.efa_new_failed_slices = 0`，`gb300_open_kmod_unset_env_path = "dma_buf_export"`；
  - `lmcache_dynamo_nixl.dynamo_token_string_allocs_without_logprobs = 0`，`nixl_probe_lock_mode = "LOCK_SH | LOCK_NB"`。
