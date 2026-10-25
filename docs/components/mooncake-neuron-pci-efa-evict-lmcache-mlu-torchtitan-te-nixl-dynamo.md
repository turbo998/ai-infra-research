# Mooncake AWS Trainium PCIe-Switch EFA 亲和与陈旧对端驱逐、LMCache 寒武纪 MLU HND 归一化与校验器、TorchTitan Muse Spark 816B GB300 RL 重切分与 Qwen3 MXFP8 融合 Rollout、TE 越界钳位及 NIXL v6 256B 遥测环

> **English Version**: [mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo.en.md](mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo.en.md)  
> **关联问题 ID**: `mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo` (Phase 74, Priority 83)  
> **关联可复现实验**: [`experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py`](../../experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py)  
> **证据分类**: `implementation` / `author-benchmark` (上游源码与 PR 实测) + `own-offline` (确定性标准库仿真器)

---

## 1. 背景与架构痛点

在多云异构加速器（AWS Trainium、寒武纪 MLU、ARM Grace/Neoverse、NVIDIA GB300/B200）上的分布式 KV 传输、强化学习（RL）训练与控制面遥测中，Mooncake、LMCache、TorchTitan、TransformerEngine、NIXL 与 Dynamo 在 `2026-10-22` 至 `2026-10-25` 期间解决了六项核心生产工程问题：

1. **AWS Trainium 上 Mooncake EFA 网卡跨 PCIe Switch 错配导致吞吐暴跌 94%**（[`Mooncake@f88d94d1bce8`](https://github.com/kvcache-ai/Mooncake/commit/f88d94d1bce85e959d8cfb5fdd02c840c791d632)）：Trainium 实例没有 `libcudart.so`，此前 Mooncake 回退到主机级拓扑，将全部 16 张 NeuronCore 绑定到同一张跨 PCIe Switch 的 EFA 网卡上。在 Trainium 上，同 PCIe Switch 下的 EFA 带宽为 **`19.2 GB/s`**，而跨一级 PCIe Switch 仅 **`2.5 GB/s`**（相差 `7.68x`），导致 16 卡总吞吐被压制在 **`12.31 GB/s`**。
2. **Mooncake EFA 滚动升级时陈旧对端句柄泄露耗尽 Libfabric 地址向量 (`fi_av_insert`)**（[`Mooncake@487da6af8205`](https://github.com/kvcache-ai/Mooncake/commit/487da6af8205b131b845d535cbe513152051a6d6)、[`Mooncake@44094f2e01a0`](https://github.com/kvcache-ai/Mooncake/commit/44094f2e01a05bde80a5a7440caa6d324aa02c12)）：在 `2 × p5en.48xlarge`（120 Decode + 120 Prefill）集群中，每次对端实例重启都会生成带新端口/时间戳的 `peer_nic_path`（`"ip:port:ts@nic"`），旧条目在 `peer_map_` 与 Libfabric AV 表中永不释放，累积数千次滚动升级后 `fi_av_insert` 返回 `0` 导致新节点静默失联。此外 C API 缺少批量状态查询，调用方被迫对每个子任务逐一轮询。
3. **LMCache 寒武纪 MLU 多进程后端布局误判与显存暂存区膨胀**（[`LMCache@a2cd25df48b7`](https://github.com/LMCache/LMCache/commit/a2cd25df48b741cde7776971b38f9c476f48bdd3)、[`LMCache@6b2b75465484`](https://github.com/LMCache/LMCache/commit/6b2b75465484f38ee7d7d7e0949e239444e1c564)）：`vllm-mlu` 历史接口对物理 `HND`（`[2, NB, NH, BS, HS]`）张量返回 `"NHD"` 字符串，且多 chunk 并发传输若为每个 chunk 预留整层暂存区会使峰值显存随 chunk 数线性膨胀。
4. **TorchTitan Muse Spark 816B-A42B GB300 RL 训练跨机架 AllGather 通信瓶颈与 Qwen3 Rollout 延迟**（[`torchtitan@43d1bb98014e`](https://github.com/pytorch/torchtitan/commit/43d1bb98014e0dab19a83e5220ec7ea1ab06d49e)、[`torchtitan@df4e4dc22a33`](https://github.com/pytorch/torchtitan/commit/df4e4dc22a335031a9952cbe599b206c410f9d60)）：旧版 `256` 卡训练拓扑（`FSDP=64 × TP=4 × EP=4`）的专家 FSDP 组高达 `64` 卡，跨机架 AllGather 阻塞 `131,072` 长序列 RL 训练；同时 Qwen3-30B-A3B DAPO-Math 的 BF16 生成器解码成为训练步等待瓶颈。
5. **TransformerEngine Triton `mask_map` MoE 重排在开启 Capacity Factor 时越界写内存**（[`TransformerEngine@c9c92255465b`](https://github.com/NVIDIA/TransformerEngine/commit/c9c92255465b4518380561b1bbb0aebb177c0b6d)）：当 `num_out_tokens < num_tokens * topK` 时，`_row_id_map_pass_2_kernel` 未将超出 `num_out_tokens` 的 `row_id` 置为 `-1`，导致后续 permute/unpermute 写穿目标张量边界。
6. **NIXL SPSC 共享内存遥测环在 ARM 上的伪共享 (`False Sharing`) 与 UCX 跨节点 Span 断链**（[`nixl@a9f1f7a76f46`](https://github.com/ai-dynamo/nixl/commit/a9f1f7a76f46f0ee78ec7448936d48ca52073fe3)、[`nixl@f3673f6be3f4`](https://github.com/ai-dynamo/nixl/commit/f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4)、[`dynamo@a789babbe37a`](https://github.com/ai-dynamo/dynamo/commit/a789babbe37a53175b962085f8a7f89909425e9b)）：NIXL 原遥测环将 `write_pos` 与 `read_pos` 放在相邻 8 字节，在 ARM 256B 缓存独占粒度下引发剧烈缓存行乒乓；且 Dynamo `/health` 每次探针默认序列化全量实例列表。

---

## 2. 核心实现机制与源码走读

### 2.1 Mooncake Trainium PCIe-Switch EFA 亲和与陈旧对端驱逐

- **Neuron `/proc/self/maps` 发现与 PCIe Switch 分组**（`f88d94d1bce8`）：扫描 `/proc/self/maps` 中的 `/dev/neuron<N>` 映射推导进程可见的 Neuron 设备序号，以 `RTLD_LAZY | RTLD_GLOBAL` 动态加载 `libnrt.so.1` 调用 `nec_get_device_pci_bdf()` 获取 PCI BDF，并按上级 PCIe Switch 桥接器将每张 Neuron 卡绑定到同 Switch 的 EFA 网卡；同时在 `Topology::resolve()` 中对 HCA 名称排序，防止远端网卡枚举顺序反转。16 卡多线程总吞吐从 `12.31 GB/s` 跃升至 **`231.16 GB/s`（`18.78x`）**，单线程从 `6.07 GB/s` 升至 **`18.30 GB/s`（`3.01x`）**。
- **EFA 陈旧对端自动驱逐与批量状态查询**（`487da6af8205`、`44094f2e01a0`）：提取 `stableHostNicKey("ip:port:ts@nic") -> "ip@nic"` 维护 `stable_key_to_path_`；当同一主机网卡以新端口重连时，若旧 endpoint 引用计数为 0（`disconnectIfIdle()`），立即调用 `fi_av_remove` 释放 AV 表项，使 `5,000` 轮滚动升级后 `peer_map_` 条目数恒定为 **`1`**。同时新增 C API `getBatchTransferStatus` 实现 $O(1)$ 批量完成轮询。

### 2.2 LMCache 寒武纪 MLU `MluDeviceSpec` 与请求级校验分配器

- **`MluDeviceSpec` 布局归一化与有界暂存**（`a2cd25df48b7`）：将 `vllm-mlu` 报告的 `"NHD"` 统一归一化为物理 `"HND"`，拒绝裸整数指针输入，并在传输流上按 chunk 即时释放临时张量，将 61 层 16 chunks 的峰值设备暂存从 `976.0 MiB` 压降至 **`62.0 MiB`（缩减 `15.74x`）**。
- **`BlockAllocator` 端到端 Checksum 校验**（`6b2b75465484`）：为冷写 (`STORE`) 与热读 (`RETRIEVE`) 分配互不重叠的物理 `GroupBlocks`，在本地 worker 缓冲区逐 chunk 比对校验和。

### 2.3 TorchTitan Muse Spark 816B GB300 RL 重切分、Qwen3 MXFP8 Rollout 与 TE 钳位

- **Muse Spark 816B-A42B GB300 RL 128 卡重切分**（`43d1bb98014e`）：将训练器从 `256` 卡（`FSDP=64 × TP=4 × EP=4`）重构为 **`128` 卡（`FSDP=8 × TP=4 × CP=4 × EP=16`）**，全层启用 `UlyssesVarlenInnerAttention`（含全局层与滑动窗口层），使专家 FSDP AllGather 域缩小 **`8x`（`64 -> 8`）**、训练卡数减半（`-50%`），在 `131,072` 上下文下峰值显存 `186 GiB/GPU`、达到 `28%` MFU。
- **Qwen3-30B-A3B B200 MXFP8 Mega-Fused Rollout**（`df4e4dc22a33`）：开启 `mxfp8_mega_fused_inference=True` 后，生成器解码耗时从 `20.2 s` 降至 **`17.8 s`（`1.14x`）**，训练器等待耗时从 `15.8 s` 降至 **`13.6 s`（`1.16x`）**，总步时提升 **`1.12x`（`22.7 -> 20.3 s`）**。
- **TE Triton Over-Capacity 钳位**（`c9c92255465b`）：在 `_row_id_map_pass_2_kernel` 中对 `num_out_tokens > 0` 执行 `tl.where(row_id >= num_out_tokens, -1, row_id)`，安全丢弃超容 token。

### 2.4 NIXL v6 256B 对齐 SPSC 遥测环、UCX `treq` 关联与 Dynamo 轻量 `/health`

- **NIXL `TELEMETRY_VERSION = 6` 256B 缓存行隔离**（`a9f1f7a76f46`）：将 `write_pos` 与 `read_pos` 隔离在独立的 256 字节块中（头部总长 `512 B`，保持 `version` 在偏移 `16 B`），并缓存对端游标与读者共享 `flock`，在 ARM 同 Socket 上实现 **`51.9M vs 11.9M events/s`（`4.36x`）**、跨 Socket **`9.2M vs 2.5M events/s`（`3.68x`）**，往返周期从 `1,022 ns` 降至 **`555 ns`（`1.84x`）**。
- **UCX 16B `treq` 追踪标签与 Dynamo 轻量健康检查**（`f3673f6be3f4`、`a789babbe37a`）：NIXL 在 UCX 通知尾部追加 16 字节小端 `treq`（`requestId` + `execution`）打通跨节点收发 Span；Dynamo `/health` 默认仅返回 `{"status":"healthy","stage":"ready"}`，仅在显式携带 `?instances=true` 时才枚举发现实例。

---

## 3. 架构师定量汇总表

运行 `python experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py --check` 验证：

| 系统组件 / 提交 | 关键架构动作 | 优化前指标 | 优化后指标 | 提升幅度 |
|---|---|---|---|---|
| **Mooncake Trainium EFA** (`f88d94d1bce8`) | `/dev/neuron` 发现 + PCIe Switch 分组 + HCA 排序 | `12.31 GB/s` (16 卡总和) | **`231.16 GB/s`** | **18.78x** (单卡同 Switch **7.68x**) |
| **Mooncake EFA AV 驱逐** (`487da6af8205`) | `stableHostNicKey` + `disconnectIfIdle` + `fi_av_remove` | `5,000` 泄露条目 (AV 耗尽) | **`1` 活跃条目** | 彻底消除滚动升级断连 |
| **LMCache MLU 后端** (`a2cd25df48b7`) | `"NHD" -> "HND"` 归一化 + 逐 chunk 释放设备暂存 | `976.0 MiB` 暂存 | **`62.0 MiB` 暂存** | **15.74x** 暂存显存缩减 |
| **TorchTitan Muse Spark RL** (`43d1bb98014e`) | `FSDP=8 × TP=4 × CP=4 × EP=16` + Ulysses 全层变长注意力 | `256` GPUs (`FSDP=64`) | **`128` GPUs (`FSDP=8`)** | **8x** 缩小专家 FSDP 域 / **-50% GPUs** |
| **TorchTitan Qwen3 RL** (`df4e4dc22a33`) | B200 `mxfp8_mega_fused_inference=True` | `15.8 s` Rollout 等待 | **`13.6 s` Rollout 等待** | **1.16x** (解码 **1.14x**) |
| **NIXL v6 SPSC 遥测环** (`a9f1f7a76f46`) | 256B 独立缓存块 (`512 B` header) + 游标缓存 | `11.9M events/s` (`1,022 ns`) | **`51.9M events/s` (`555 ns`)** | **4.36x** 吞吐 / **1.84x** 低延迟 |
