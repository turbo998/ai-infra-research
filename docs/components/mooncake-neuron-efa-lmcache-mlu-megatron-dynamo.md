# Mooncake AWS Neuron (Trainium) HBM-over-EFA 直传与滚动升级陈旧对端驱逐及 HCA/NUMA 亲和存储、LMCache 寒武纪 MLU 后端与独占块本地校验、TorchTitan FlashAttention 安全回退与惰性导出延迟报错、Megatron-LM GDP 预填充准备融合与 EP 共识收敛，以及 Dynamo KV-Router 事件过滤、Worker GC 冻结与批量 Token 指标计数

> **语言 / Language**: [中文](mooncake-neuron-efa-lmcache-mlu-megatron-dynamo.md) | [English](mooncake-neuron-efa-lmcache-mlu-megatron-dynamo.en.md)
> **阶段 / Phase**: Phase 73 (`priority: 82`, Track C)
> **关联问题 / Question ID**: `q-phase73-mooncake-neuron-efa-lmcache-mlu-megatron-dynamo`
> **证据等级 / Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. 概述与核心架构洞察

在 2026-10-22 至 2026-10-25 的上游提交中，`kvcache-ai/Mooncake`、`LMCache/LMCache`、`pytorch/torchtitan`、`NVIDIA/Megatron-LM` 与 `ai-dynamo/dynamo` 针对跨云异构加速器（AWS Trainium Neuron HBM、Cambricon MLU、Huawei Ascend MemFabric）、大规模 PD 分离滚动升级稳定性、混合线性注意力（Gated Delta Product）以及高并发推理控制面完成了五组系统级演进：

1. **Mooncake AWS Neuron (Trainium `trn2.48xlarge`) HBM-over-EFA 零拷贝直传、滚动升级陈旧对端驱逐与 HCA/NUMA 亲和存储**（`Mooncake@f88d94d1bc`, `487da6af82`, `868a6d05ab`, `6a835cbf93`, `44094f2e01`）：
   - **AWS Neuron HBM 直接经 EFA 传输（无需 Neuron SDK 头文件编译期依赖）**：通过解析 `/proc/self/maps` 中的 `/dev/neuron<N>` 字符设备映射，在运行时自动识别设备内存地址与设备序号（强制覆盖调用方传入的 `"cpu:N"` 提示，防止触发 `preTouchMemory()` CPU 写页损坏设备内存）；在首次 `fi_getinfo()` 前以 `RTLD_GLOBAL` 动态加载 `/opt/aws/neuron/lib/libnrt.so.1`，设置 `FI_HMEM=neuron` 与 `FI_MR_HMEM` 并调用 `fi_mr_regattr(iface=FI_HMEM_NEURON)`。同时显式设置 `FI_EFA_ENABLE_SHM_TRANSFER=0` 禁用同机 SHM 子提供者（因为 libfabric `FI_HMEM_NEURON` 未实现 `copy_from_hmem` 和 IPC，SHM 会退化为在未缓存 PCIe BAR 上 `memcpy`，吞吐仅 **`0.07 GB/s`**，而走 EFA 设备环回可达 **`19.23 GB/s`**，相差 **`274.7x`**）。更重要的是，通过 `nec_get_device_pci_bdf()` 将每个 Neuron 设备与其同 PCIe Switch 下的 2 张 EFA 网卡绑定：跨 PCIe Switch 网卡仅 **`2.5 GB/s`**，同 Switch 网卡达 **`19.2 GB/s`（`7.68x`）**；16 卡总聚合带宽从 **`12.31 GB/s` 飙升至 `231.16 GB/s`（`18.78x`）**，同时 `Topology::resolve()` 改为按排序后的存储键遍历以保证收发双方 HCA 编号一致。
   - **120P+120D 大规模滚动升级陈旧对端驱逐（Stale-Peer Eviction）**：当对端重启导致 `ip:port:ts@nic` 中的临时端口或时间戳变化时，旧代码在 `peer_map_` 中无限累积条目，在远未达到 `MC_MAX_EP_PER_CTX (65536)` 前就耗尽 libfabric 内部地址句柄缓存并使 `fi_av_insert` 返回 `0`（静默拒绝）。新代码提取稳定键 `stableHostNicKey("ip:port:ts@nic") -> "ip@nic"`，在重连时通过 `disconnectIfIdle()`（使用非阻塞 `tryLock()` 防止与等待 CQ credit 的发送线程死锁）安全移除旧 AV 槽位，并识别 `prov_errno in {4, 8, 9, 14}` 自动失效陈旧句柄。在 2×`p5en.48xlarge` 5000 次对端抖动压测中，`peer_map_` 最终大小从 **`5000` 降至 `1`**。
   - **RL Put/Get 的请求级 HCA/NUMA 亲和性与 C API `getBatchTransferStatus`**：`MC_STORE_HCA_AFFINITY=1` 支持按每个 Slice 所在源/目的内存的优先 HCA 选择同 NUMA 段副本，C API 新增 $O(1)$ `getBatchTransferStatus()` 替代逐 Task 轮询。
2. **LMCache 寒武纪 MLU 引擎驱动后端与 `lmcache bench server` 独占块本地校验**（`LMCache@a2cd25df48`, `6b2b754654`）：
   - **Cambricon MLU (`MluDeviceSpec` / `MluDeviceOps`)**：支持五维 K/V-first NHD/HND 与三维 MLA 布局（自动将旧版 vLLM MLU 查询返回的 NHD 归一化为物理分配的 HND），D2H 逐层填充单个 Chunk 以限制设备临时显存。针对 `torch_mlu 1.26.1 + torch 2.6.0` 在 `skip_prefix_n_blocks > 0` 产生非连续（Strided）主机切片时 `copy_(..., non_blocking=True)` 即使同步后也会返回错误数据的硬件/驱动缺陷，专门对非连续主机切片强制使用阻塞拷贝，而连续切片保持异步拷贝。
   - **`lmcache bench server` 请求独占 `BlockAllocator` 与本地 MD5 校验**：新增支持 `sequential` / `random` 模式的无前缀共享独占块分配器，Warm Pass 先将目标块清零（`set_blocks_to_zero`）再 `RETRIEVE`，直接在本地 Worker 缓冲区计算跨层逻辑块 MD5 与 Cold Pass 对比，移除对外部 HTTP Checksum 接口的依赖。
3. **TorchTitan FlashAttention 缺失 Wheel 安全回退与 `attn_gym` 惰性导出延迟报错**（`torchtitan@89e4339c96`, `3ed5aa976f`）：
   - 在激活 Hopper FA3 (`flash_attn_interface`) 或 Blackwell FA4 (`flash_attn.cute`) 前通过 `importlib.util.find_spec` 探测可用性；若缺失且 Q/K/V 头维度相等则告警并安全回退至 PyTorch 内置 FA2（在 L4 上 Eager 与 `torch.compile(fullgraph=True)` 对比显式 FA2 的输出、Loss `0.19502778` 与梯度 L2 范数 `0.01268001` 差异均为 `0`），而对 DeepSeek-V3 / Kimi-K3 等 Q/K 与 V 头维度不等的 MLA 布局则在构造期立即抛出包含 SM 与所需包名的明确异常。
   - 修复 `attn_gym` 惰性属性导出（如 `causal_conv1d` 首次访问触发 `cutlass` `ModuleNotFoundError`）在模块导入期逃逸的问题，使无 `cutlass`/`triton` 环境下的 CPU 单元测试收集从 `1371 collected / 23 errors` 提升至 **`1640 collected / 5 errors`**（修复 18 个测试模块）。
4. **Megatron-LM Gated Delta Product (GDP) 预填充准备融合与移除 `disable_ep_consensus`**（`Megatron-LM@b291235082`, `7d3f0fb3b5`）：
   - 新增 Triton 内核 `gdp_prefill_prepare`，将 GDP 预填充在因果短卷积与变长分块递推之间的约 **14 个独立 PyTorch 逐元素/跨步拷贝算子**（切分 `V/K/Q`、GQA 扩展、折叠 Householder 副本、计算 $\beta = \sigma(b)$ 与 $g = -\exp(A_{\text{log}})\cdot\text{softplus}(a + dt_{\text{bias}})$）融合为 **1 次 Kernel 启动**（40 层模型从 `560` 次启动降至 `40` 次），并通过 `triton.language.extra.libdevice` 保证与 Eager 路径逐位一致（`torch.equal`）。
   - 移除危险的 `--inference-disable-ep-consensus` 标志，统一使用 `ep_consensus_interval = 20` 步周期性协调，避免無共识下空闲 Rank 盲目空转或死锁。
5. **Dynamo 1.7.0 KV-Router 非主注意力组预过滤、`/health` 轻量化、Worker GC 冻结与批量 Token 原子计数**（`dynamo@8bf042187a`, `a789babbe3`, `45cc9e8a1a`, `906e89d3a3`）：
   - **KV-Router `normalize_raw_batch` 顺序修复**：先执行 `preprocess_residency_with_reason` 过滤掉 `NonMainAttentionKind` / `NonMainAttentionGroup`（如滑动窗口或 Mamba/SSM 混合状态组），再对主注意力组校验 `block_size`，防止混合模型中非主注意力组的异构块尺寸误触发 `KVCR block size is incompatible` 隔离故障。
   - **控制面与数据面去抖动**：`/health` 默认仅返回 `{"status": "healthy", "stage": "ready"}`，将序列化全量发现实例放入可选参数 `?instances=true`；新增 `--freeze-gc-after-init` 在模型注册前调用 `freeze_gc_heap()` 冻结约 **100 万个**静态 Python 对象（EngineClient、Tokenizer、Config），消除高负载下 Gen2 GC 导致的数百毫秒请求毛刺；`RequestMetrics` 将每 Token 递增全局共享原子变量 `output_tokens` 改为本地 `pending_output_tokens` 每 **`LOCAL_FLUSH_TOKENS = 64`** 个 Token（或请求结束时）批量刷写一次，将多核缓存行争用降低 **`64x`**。

---

## 2. 关键架构数据流图：Mooncake Trainium HBM-over-EFA 与 EFA 槽位生命周期

```
[User Buffer Addr]
       │
       ▼
neuronProbeAddress(/proc/self/maps -> /dev/neuron<N>)
       ├── 命中 Neuron HBM: 忽略 "cpu:N" 提示 -> 命名为 "neuron:<N>" (跳过 preTouchMemory CPU 写页!)
       │     ├── dlopen("libnrt.so.1", RTLD_GLOBAL) (在首次 fi_getinfo 前满足 libfabric 符号需求)
       │     ├── setenv("FI_HMEM", "neuron") + setenv("FI_EFA_ENABLE_SHM_TRANSFER", "0")
       │     │     └── 禁用同机 SHM (0.07 GB/s 未缓存 BAR memcpy) -> 强制走 EFA 环回 (19.23 GB/s, 274.7x!)
       │     └── nec_get_device_pci_bdf() 匹配同 PCIe Switch 的 2 张 EFA NIC
       │           └── 单卡 2.5 -> 19.2 GB/s (7.68x), 16 卡聚合 12.31 -> 231.16 GB/s (18.78x!)
       │
       ▼
[EfaContext Reconnect / Rolling Upgrade (120P + 120D)]
  新路径 "10.0.1.12:12500:1700000500@rdmap201s0"
       └── stableHostNicKey -> "10.0.1.12@rdmap201s0"
             └── 发现旧临时路径 -> disconnectIfIdle() (tryLock 防 CQ credit 死锁) -> fi_av_remove 释放旧槽位
```

---

## 3. 可复现实验验证 (`own-offline` vs `author-benchmark`)

运行本仓库原创标准库仿真脚本验证全部 6 项不变量：

```powershell
python experiments\mooncake_neuron_efa_lmcache_mlu_megatron_dynamo_sim\run.py --check
```

| 系统组件 | 硬件 / 场景 | 基线表现 | 优化后表现 | 提升幅度 / 结果 | 证据分类 |
|---|---|---|---|---|---|
| Mooncake EFA Neuron HBM 直传与拓扑亲和 | AWS `trn2.48xlarge` (16× Trainium2 + EFA) | 跨 Switch `2.5 GB/s` (单卡) / `12.31 GB/s` (16卡)；同机 SHM `0.07 GB/s` | 同 Switch `19.2 GB/s` (单卡) / **`231.16 GB/s` (16卡)**；EFA 环回 **`19.23 GB/s`** | 单卡 **`7.68x`**、16卡 **`18.78x`**、同机 **`274.7x`** | `author-benchmark` + `own-offline` |
| Mooncake EFA 陈旧对端驱逐 | 2× `p5en.48xlarge` 5000 次对端重启抖动 | `peer_map_` 线性膨胀至 `5000` 并触发 `fi_av_insert == 0` 静默拒绝 | `stableHostNicKey` + `disconnectIfIdle` 驱逐旧句柄 | `peer_map_` 稳定为 **`1`**，零 AV 泄漏 | `author-benchmark` + `own-offline` |
| LMCache Cambricon MLU 后端 | `torch_mlu 1.26.1` + `torch 2.6.0` | `skip_prefix_n_blocks > 0` 非连续主机切片异步 D2H 数据损坏 | 非连续主机切片强制同步拷贝，连续切片保持异步 | **消除前缀跳过场景下的静默数据损坏** | `implementation` + `own-offline` |
| TorchTitan FA 回退与 `attn_gym` 惰性导出 | NVIDIA L4 (CUDA 13.4) / macOS 15 CPU | 缺失 FA3/FA4 Wheel 或 `cutlass` 时导入/构造直接崩溃（23 个 CPU 测试收集错误） | 安全回退 FA2（Loss/Grad 0 误差）；惰性导出错误延后至构造期 | CPU 测试收集 **`1371 -> 1640`**（修复 18 模块） | `author-benchmark` + `own-offline` |
| Megatron-LM GDP 预填充准备融合 | 40 层 Gated Delta Product 动态推理 | 每层 ~14 个逐元素/跨步拷贝 PyTorch 内核（共 `560` 次启动） | 单次 `gdp_prefill_prepare` Triton 内核（`libdevice` 位级对齐） | 启动次数 **`560 -> 40` (`14x` 削减)** | `implementation` + `own-offline` |
| Dynamo 1.7.0 运行时优化 | 高并发 vLLM Worker + Prometheus 指标 | Gen2 GC 反复扫描 ~100 万静态对象；每生成 1 Token 触发 1 次共享原子写 | `--freeze-gc-after-init` + `LOCAL_FLUSH_TOKENS = 64` 批量刷写 + `/health` 轻量化 | 共享原子缓存行争用降低 **`64x`** | `implementation` + `own-offline` |

---

## 4. 架构师练习与可证伪问题

1. **故障排查练习（Mooncake EFA 滚动升级死锁预防）**：为什么 `EfaEndPoint::disconnectIfIdle()` 在 CQ 轮询线程中必须调用 `lock_.tryLock()` 而绝对不能使用阻塞式 `WriteGuard`？
   - *解答提示*：发送线程在 `submitPostSend()` -> `submitSlicesOnPeer()` 中持有 `lock_` 的**读锁（ReadGuard）**自旋等待 CQ credit 释放，而只有 CQ 轮询线程能消费完成事件并归还 credit。如果 CQ 轮询线程在清理陈旧对端时阻塞等待 `lock_` 的**写锁（WriteGuard）**，读写自旋锁会阻止新的读者并等待现有读者退出，从而形成经典的“发送者握读锁等 CQ 线程放 credit、CQ 线程等写锁等发送者退读锁”的雙线程死锁。
2. **可证伪假设（Dynamo KV-Router 混合模型块大小校验顺序）**：在包含主注意力（`block_size=64`）与 Mamba/滑动窗口状态组（`block_size != 64`）的混合模型中，若在过滤非主注意力组之前对所有 `KvEventOwnership::Kvcr` 事件校验 `block_size == kv_block_size`，首个状态组块存储事件就会将整个 `CacheOwner` 流永久置为隔离故障状态（Quarantine）。
