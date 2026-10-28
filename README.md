# AI Infra Research — 全栈智算与系统工程研究工作台

> 🌐 **Language / 语言切换**: **[中文](README.md)** | **[English](README.en.md)** | **[📚 完整 124 阶段全量目录归档 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** | **[📚 Full 124-Phase Catalog (`docs/full-catalog-en.md`)](docs/full-catalog-en.md)** | **[🗺️ 11 层全栈地图 (`docs/stack-map.md`)](docs/stack-map.md)**

---

## 📌 Executive Overview Summary（执行摘要与核心架构结论）

> **核心研究基线（Phase 1 – Phase 124 全量合入 `main`，Priority 133）**
> 本仓库围绕 **11 层 AI 算力与分布式系统工程栈** 展开基于一手固定版本源码（`implementation`）、架构规范（`specification`）与零依赖 Python 3.12 确定性仿真模型（`own-offline`）的深度技术研究，累计完成 **100+ 篇双语深度专题报告**，所有定量结论均通过自动化校验测试，**零未核验主观臆断**：
> - **研究规模与可复现账本**：累计完成 **`387` 个核心工程研究问题**（`385 done` + `2 blocked` 外部闭源标准，公开可复现实验 100% 覆盖）、**`382` 套零依赖 Python 3.12 标准库实验**（全部通过 `--check`）、**`2,031` 条固定 Commit/Blob 第一手源码引用**（[`research/sources.json`](research/sources.json)）、**`688` 条分级证据主张**（[`research/evidence.json`](research/evidence.json)）、**`630` 项全绿单元测试** 以及 **`124` 份 7-Card 4:5 视觉评测报告**（`docs/reports/`，最新 **[Phase 124 7-Card 评测报告](docs/reports/ai-infra-phase124-benchmark-cards.html)**）。
> - **L01 GPU / SM100 Blackwell 算子与低精度数值**：剖析 `tcgen05`、512 列 `TMEM` 原位混叠（消除 `64 KiB` SMEM 溢出）、`FlashAttention-4`（`FFMA2` 多项式仿真突破 `MUFU.EX2` 瓶颈，在 B200 达 **`2,790.1 TFLOPS`**、SM103 B300 达 **`3,200.0 TFLOPS`**）、`FlashMLA V4.1`（`+32B` TMA 行填充消除 `LDS.32` 4 路冲突，`get_mla_metadata` 削减 **`89.88%`** 偏斜尾波）、`CUTLASS CLC` 动态窃取及 `NVFP4/MXFP8` 分块缩放。
> - **L02 & L04 分层 KV 缓存与 Prefill-Decode 分离推理**：解构四级 KV 缓存层次（VRAM → Host Pinned DRAM → CXL 3.0 `Device-DAX` `2 MiB` 对齐 → GDS/3FS），结合 `Mooncake Transfer Engine`（ConnectX-8 `Data Direct` 零锁批处理、7 级 Auto-GID 重探测与 `W=4` 有界 Scatter 将有效带宽从 **`313.0 -> 357.85 Gbps`** 提升）、`NVIDIA Dynamo`（二级 `kvcr` 连接器与 `NIXL` UCX/Libfabric 显存类型探测）、`vLLM V1`（`HiSparse` 稀疏 MLA 与拒绝采样）及 `SGLang`（`HiRadixCache` 与 `EAGLE-3` 投机树）。
> - **L03 & L09 超节点互连、`IBGDA` 直发 RDMA 与共封装光学（CPO/OCS）**：验证 `PyTorch CUDASymmetricMemory` + `NVLS multimem`（流量较 Unicast 1-Shot 降低 **`6.22x`**）、`NCCL Gin` 对称集合通信与 `PAT` 并行聚合二项式树（1,024 卡跳数从 `1,023 -> 7`）、`DeepEP V2.5` GPU 直发 RDMA（Warp 门铃聚合削减 **`10.67x`** MMIO 开销）、`UEC 1.0` 64B 包裁剪 + 逐包喷洒（32-to-1 Incast 下 P99 尾延迟降低 **`69.9%`**），以及 `1.6T CPO + 3D MEMS OCS` 十万卡光互连功耗与故障半径模型。
> - **L05 & L08 6D 分布式预训练、RL 后训练引擎与容错恢复**：建立 `TorchTitan` 6D 并行与 `Megatron-FSDP`（`HFSDP` 将 120B 峰值显存从 **`244.20 -> 51.01 GiB/GPU`** 压降，Copy Engine AG 实现 **`1.344x`** 单步加速）、`verl` 原位零拷贝 `MXFP4` 权重重分片、`Slime/Miles` 有界陈旧度异步 RL Rollout（逐 Token `TIS` 重要性采样）、`PyTorch DCP AsyncStager`（持久化锁页共享内存池将保存阻塞从 **`3.737 s -> 0.048 s`** 压缩 **`77.8x`**）及 `FlightRecorder + NCCL RAS` **`< 650 ms`** 静默死锁定位。
> - **L06, L07, L10 & L11 安全沙箱、拓扑感知 Gang 调度与 800VDC 液冷智算中心**：覆盖 `Firecracker` / `Kata` / `eBPF` Agentic RL 沙箱协同调度、Kubernetes `TopologyManager` 按位与掩码 / `Kueue TAS` / `DRA` / Slurm `topology/block & tree`、`±400V/800VDC` 高压直流母线 + `130kW+` 冷板液冷（DLC）热力学与 `dP/dt` 浪涌平滑，以及全套确定性基准评测框架。

---

## 🏗️ 11 层 AI 算力与系统工程架构图谱（11-Layer AI Compute & Systems Stack）

![AI Infrastructure 11-Layer Compute & Systems Stack](assets/svg/ai-infra-11-layer-stack.svg)

### 架构分层详解（Architectural Walkthrough）

| 层级 | 核心技术栈与代表开源项目 | 解决的核心物理瓶颈与定量收益 |
| :--- | :--- | :--- |
| **L01 · GPU/SM100 算子与微格式** | Blackwell `tcgen05` · `512-Col TMEM` · `FlashAttention-4` · `FlashMLA V4.1` · `FlashInfer Cake/Frost` · `DeepGEMM MegaMoE` · `CUTLASS CLC` | 消除 256 维注意力反向与双缓冲 Ping-Pong 的 `64 KiB` SMEM 溢出；`FFMA2` 突破超越函数 `MUFU.EX2` 吞吐墙（达 `2,790–3,200 TFLOPS`）；`NVFP4/MXFP8` 512B 原子块缩放实现近无损训练与解码。 |
| **L02 · HBM3e/HBM4 与 CXL KV 分层** | `HBM3e` (`8 TB/s`) / `HBM4` (`2048-bit`, `13.5 TB/s`) · `CXL 3.0 Device-DAX` · `LMCache` · `KIVI / NVFP4 KV` | 建立四级 KV 缓存层次（VRAM → Host Pinned DRAM → CXL DAX → GDS/3FS），通过 `/sys/dev/char` `2 MiB` 硬件对齐检查与 APC 前缀去重消除 `SIGBUS` 崩溃及 `37.5%` 假 L1 命中率。 |
| **L03 · NVLink / IBGDA / UEC 集合通信** | `NVLS multimem` · `NCCL Gin & PAT` · `DeepEP V2.5 IBGDA/GDAKI` · `ConnectX-8 Data Direct` · `UEC 1.0 Packet Trimming` | GPU SM 直接写 NIC Doorbell 并以 Warp 聚合削减 `10.67x` PCIe 门铃；`ConnectX-8 Data Direct` 消除跨 PCI Root `ENOTSUPP`；UEC 64B 包裁剪在 800G 浅缓冲交换机上压降 `69.9%` P99 尾延迟。 |
| **L04 · Prefill-Decode 分离推理系统** | `Mooncake Transfer Engine` · `NVIDIA Dynamo (kvcr + NIXL)` · `vLLM V1 (HiSparse)` · `SGLang (HiRadixCache + EAGLE-3)` | 解耦计算密集型 Prefill 与访存密集型 Decode；Mooncake 7 级 Auto-GID + 有界 `W=4` Scatter 流水将跨机 KV 传输推至 `357.85 Gbps`；配合稀疏注意力与投机树保障百毫秒级 TTFT 与穩定 TPOT。 |
| **L05 · RL 后训练与混合并行引擎** | `verl` · `THUDM/slime` · `Miles` · `TorchTitan FlexShard` · `Megatron-Core DualPipe & HFSDP` | `verl` 原位 `_mxfp4_staging_data` 重分片消除 Actor→Rollout 双份权重显存峰值；`Slime` 异步 Rollout 配合逐 Token 重要性采样（TIS）消除长尾推理生成空泡，训练有效吞吐提升 `1.4x–1.8x`。 |
| **L06 · MicroVM 与 Agentic 沙箱隔离** | `Firecracker UFFD/NBD` · `Kata Containers (AKSClaw)` · `eBPF Cilium` · `Agentic Sandbox Co-Scheduler` | 在代码执行与工具调用型 RL/Agent 负载中提供硬件虚拟化边界（`80 ms` 冷启动 / `108 ms` UFFD 快照恢复），并通过 GPU Rollout Worker 与 CPU 沙箱池协同配比消除 I/O 饥饿。 |
| **L07 · K8s / DRA / Kueue / Slurm 拓扑调度** | `K8s TopologyManager` · `Kueue TAS` · `Dynamic Resource Allocation (DRA)` · `Slurm topology/block & tree` · `NVIDIA Device Plugin` | 通过 CPU/GPU/RDMA NIC 按位与亲和掩码及 NVLink Clique / Leaf 交换机紧致分配，杜绝跨 QPI/UPI 与跨 Spine 通信降速；基于 `<UUID, GI, CI>` 隔离 MIG 切片级 XID 故障。 |
| **L08 · Checkpoint 容错与万卡自愈** | `PyTorch DCP AsyncStager` · `NVRx In-Process Warm Standby` · `FlightRecorder + NCCL RAS` · `Grey-Failure Straggler Detector` | `WeakIdKeyDictionary` 持久化锁页共享内存池将每次保存的 `cudaHostRegister` 开销从 `3.737 s` 降至 `0.048 s`；跨机架半世界双副本与 `< 650 ms` 零号病人死锁定位将万卡有效训练时间（Goodput）保持在 `93%+`。 |
| **L09 · 硅光互连（CPO/LPO）与光交换（OCS）** | `1.6T Co-Packaged Optics (9 pJ/bit)` · `Linear Pluggable Optics (LPO)` · `3D MEMS OCS` · `ELSFP External Laser` | 将可插拔 DSP 光模块功耗降低 `65%+`，消除热敏激光器贴近 ASIC 引发的热失效爆炸半径，并通过 3D MEMS OCS 实现跨 Pod 拓扑动态重配置与故障绕行。 |
| **L10 · 800VDC 高压供电与冷板液冷** | `±400V / 800VDC Busway` · `130kW+ NVL72 Cold-Plate DLC` · `CDU Heat Exchanger` · `Slurm Token-Bucket & DCGM Power Brake` | 建模 CDU 二次侧流量/温升、MNNVL `IMEX` 72-GPU 机柜级热耦合故障域，以及通过硬件 `Power Smoothing`、Slurm 令牌桶与 BESS 储能削峰抑制千卡同步迭代 `dP/dt` 电网冲击。 |
| **L11 · 确定性物理仿真与证据核验底座** | `scripts/validate_research.py` · `382 Python 3.12 Experiments` · `630 Unit Tests` · `124 7-Card 4:5 Benchmark Decks` | 纯 Python 3.12 标准库实现，100% 离线确定性复现所有公式、协议状态机与性能边界，严格区分 `implementation`、`author-benchmark` 与 `own-offline`。 |

---

## 🗂️ 六大主题域核心研究报告索引（6 Thematic Domain Summary Tables）

> 💡 **完整百科全量表入口**：本页精选展示六大主题域的代表性旗舰专题。如需查阅全部 **124 个研究阶段、387 个研究问题与 382 套实验的完整 650KB+ 百科对照表**，请直接访问：
> 👉 **[📚 中文全量组件与阶段百科表 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** | **[📚 English Complete Component & Phase Catalog (`docs/full-catalog-en.md`)](docs/full-catalog-en.md)**

### 域 1：GPU 架构、SM90/SM100 Blackwell 内核与低精度数值（L01）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[GPU/HBM 执行模型与显存层次基础](docs/components/gpu-hbm.md)** | SM/Warp 调度、寄存器压力、L2 持久化缓存、HBM3e/HBM4 屋脊线与张量核数据通路 | [`capacity_roofline`](experiments/capacity_roofline/README.md) |
| **[SM100 Blackwell FMHA 与 FlashAttention-4 (`CuTe DSL`)](docs/components/sm100-blackwell-fmha-tmem-fa4.md)** | `512` 列 `TMEM` 原位混叠（消除 `64 KiB` SMEM 溢出）、16-Warp 寄存器重分配、`FFMA2` 突破 `MUFU.EX2` 墙（`2,790 TFLOPS`） | [`sm100_blackwell_fmha_fa4`](experiments/sm100_blackwell_fmha_fa4/README.md) |
| **[FA4 Blackwell SM100 `hdim=256` 反向 `M=64` TMEM 打包](docs/components/fa4-sm100-hd256-bwd-m64-tmem-accum-layout.md)** · **[EN](docs/components/fa4-sm100-hd256-bwd-m64-tmem-accum-layout.en.md)** | 消除 `hdim=256` 反向传播 `768/640` 列 TMEM 溢出与 `sdS_xchg` 共享内存交换开销 | [`sm100_blackwell_fmha_fa4`](experiments/sm100_blackwell_fmha_fa4/README.md) |
| **[DeepSeek MLA 与 FlashMLA V4.1 分页解码内核](docs/components/deepseek-mla-flashmla-decode-kernels.md)** | SM90/SM100 2-CTA 协同、`W_UK/W_UV` 矩阵吸收（算术强度 `496–910 FLOP/B`）与 `get_mla_metadata` 消除 `89.9%` 尾波 | [`flashmla_paged_decode_scheduling`](experiments/flashmla_paged_decode_scheduling/README.md) |
| **[FlashMLA SM100 V4.1 融合 `Norm+RoPE+Attn` 与零冲突反量化](docs/components/flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.md)** · **[EN](docs/components/flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.en.md)** | `+32B` TMA 行填充消除 `LDS.32` 4 路 Bank 冲突、`e2m1*e4m3` 精确 BF16 转换与 `512/480` 列 TMEM 布局 | [`flashinfer_ds41_mixed_decode_sim`](experiments/flashinfer_ds41_mixed_decode_sim/README.md) |
| **[Blackwell SM100 块缩放微格式 (`NVFP4` vs `MXFP4`) 与 512B 原子](docs/components/sm100-block-scaled-nvfp4-kernels.md)** | `Sm1xxBlockScaledBasicChunk` 512B 雙射、UTCCP `MMA_N=256` 对齐填充、随机 Hadamard 变换 (`RHT+SR`) 预训练收敛性 | [`sm100_block_scaled_nvfp4_gemm`](experiments/sm100_block_scaled_nvfp4_gemm/README.md) · [`fp8_nvfp4_numerics`](experiments/fp8_nvfp4_numerics/README.md) |
| **[CUTLASS SM100 Ptr-Array / Grouped GEMM 与 `CLC` 动态窃取](docs/components/cutlass-sm100-grouped-gemm-clc-scheduler.md)** | 异步 `TensorMapUpdater`（`tensormaps_perform_update<false>`）与 Cluster Launch Control (`CLC`) 双级任务窃取 | [`cutlass_sm100_grouped_clc_gemm`](experiments/cutlass_sm100_grouped_clc_gemm/README.md) |
| **[DeepGEMM SM100 `MegaMoE` 融合专家池与 `MegaMHC` 超连接](docs/components/deepgemm-sm100-mega-moe-mhc-hyperconnection.md)** · **[EN](docs/components/deepgemm-sm100-mega-moe-mhc-hyperconnection.en.md)** | Sinkhorn-Knopp 双随机流形超连接（MHC）、CUDA Graph 纪元屏障（`Epoch 64B`）与分组 MoE 设备端 Swizzle | [`deepgemm_mega_moe_mhc_sim`](experiments/deepgemm_mega_moe_mhc_sim/README.md) |
| **[Hopper → Blackwell → Vera Rubin (`VR200 NVL144` / `HBM4`) 代际演进](docs/components/vera-rubin-nvl144-hbm4-architecture.md)** | `HBM4 2048-bit` (`13.5 TB/s`)、`NVLink-6` (`3.6 TB/s`) 与Groq TSP / Cerebras WSE 晶圆级 SRAM 异构投机级联对比 | [`vera_rubin_hbm4_nvl144_roofline`](experiments/vera_rubin_hbm4_nvl144_roofline/README.md) · [`wafer_sram_gpu_cascaded_serving`](experiments/wafer_sram_gpu_cascaded_serving/README.md) |
| **[批大小不变性确定性推理算子 (`Batch-Invariant Kernels`)](docs/components/batch-invariant-deterministic-inference.md)** | Split-KV / Split-K 固定归约树消除浮点非结合律引起的跨 Batch 1-ULP 漂移，实现 `0 ULP` 逐比特复现 | [`batch_invariant_deterministic_inference`](experiments/batch_invariant_deterministic_inference/README.md) |

---

### 域 2：HBM3e/CXL/LMCache 分层 KV 缓存与 Prefill-Decode 分离推理（L02 & L04）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[四级 KV 缓存卸载层次（VRAM → Host DRAM → CXL → NVMe/3FS）](docs/components/tiered-kv-cache-offload-hierarchy.md)** | 多级缓存命中率帕累托边界、PCIe Gen5/CXL 3.0 带宽争用与异步换入掩盖窗口 | [`tiered_kv_cache_hierarchy`](experiments/tiered_kv_cache_hierarchy/README.md) · [`kv_cache_quant_kivi_nvfp4`](experiments/kv_cache_quant_kivi_nvfp4/README.md) |
| **[LMCache Device-DAX/CXL L1 偏移对齐与 Mooncake OpLog 去重](docs/components/lmcache-dax-cxl-l1-offset-grpc-codec-mooncake-oplog-dedup.md)** · **[EN](docs/components/lmcache-dax-cxl-l1-offset-grpc-codec-mooncake-oplog-dedup.en.md)** | `/sys/dev/char` `2 MiB` 硬件对齐预检、多 Arena 不交叠分区与 APC 前缀去重（消除 `37.5%` 假命中率） | [`megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120`](experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/run.py) |
| **[Prefill/Decode 分离与分层 KV 缓存智能路由](docs/components/disaggregated-kv-routing.md)** | Mooncake / Dynamo KV 感知前缀路由、跨节点 RDMA Write 零拷贝传输与 TTFT/TPOT 联合 SLO 约束 | [`disaggregated_kv_routing`](experiments/disaggregated_kv_routing/README.md) · [`dynamo_kv_aware_router`](experiments/dynamo_kv_aware_router/README.md) |
| **[Mooncake ConnectX-8 `Data Direct` DMA 与 `BatchRdmaDeviceCache`](docs/components/mooncake-transfer-engine-cx8-data-direct-rdma.md)** | `mlx5dv_get_data_direct_sysfs_path` 顶层 PCI Root 配对，消除 4,096 次哈希锁竞争与跨 Root `ENOTSUPP` | [`mooncake_cx8_data_direct_rdma`](experiments/mooncake_cx8_data_direct_rdma/README.md) |
| **[Mooncake 7 级 Auto-GID 重探测、双端点确认与有界 Scatter](docs/components/mooncake-rdma-auto-gid-reprobe-bounded-reshard.md)** · **[EN](docs/components/mooncake-rdma-auto-gid-reprobe-bounded-reshard.en.md)** | 规避 Docker `172.17/12` GID 误选，`RdmaCompletionToken` 纪元过滤与 `W=4` 有界重分片（`357.85 Gbps`） | [`mooncake_cx8_data_direct_rdma`](experiments/mooncake_cx8_data_direct_rdma/README.md) |
| **[NIXL UCX VRAM `mem_type` 探测与 `cuMemCreate` Libfabric 回退](docs/components/nixl-ucx-vram-memtype-hint-libfabric-vmm-triton-cdna5-barrier.md)** · **[EN](docs/components/nixl-ucx-vram-memtype-hint-libfabric-vmm-triton-cdna5-barrier.en.md)** | `resolveMemoryTypeConfig` 显存类型提示、`cudaUpdateCtxPtr` 主上下文回退与 UCX `8.32x` 轮询偏差剔除 | [`megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121`](experiments/megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121/run.py) |
| **[vLLM V1 统一 Token Budget 调度、稀疏 MLA (`HiSparse`) 与 SLO](docs/components/inference-slo.md)** | Chunked Prefill 干扰边界、100% Prefix Cache 末块重算与 `vLLM HiSparse` 分层稀疏解码 | [`inference_slo_scheduler`](experiments/inference_slo_scheduler/README.md) · [`vllm_v1_sparse_mla_hisparse_sim`](experiments/vllm_v1_sparse_mla_hisparse_sim/README.md) |
| **[SGLang `HiRadixCache`、PD 动态角色切换与 `EAGLE-3` 投机树](docs/components/ai-infra-innovators-v2.md)** | 前缀基数树三级淘汰策略、无损 One-Hot 拒绝采样与硬件感知自适应投机步长 | [`sglang_hiradix_cache_controller`](experiments/sglang_hiradix_cache_controller/README.md) · [`eagle3_speculative_tree`](experiments/eagle3_speculative_tree/README.md) |

---

### 域 3：Scale-Up / Scale-Out 集合通信、直发 RDMA (`IBGDA`) 与光互连 (`UEC` / `CPO` / `OCS`)（L03 & L09）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[互连与网络分层及 RDMA Verbs 生命周期](docs/components/interconnect-network.md)** · **[RDMA 生命周期](docs/components/rdma-lifecycle.md)** | PCIe / NVLink / InfiniBand / RoCEv2 帧封装、MR 锁页注册、QP 状态机（`RESET→INIT→RTR→RTS`）与 CQ 轮询 | [`ring_cut`](experiments/ring_cut/README.md) · [`rail_channel_cut`](experiments/rail_channel_cut/README.md) |
| **[GPU 直发 RDMA (`IBGDA` / `GDAKI`) 与 `DeepEP V2.5` 门铃聚合](docs/components/ibgda-gpu-initiated-rdma.md)** | CTA 独占 RC QP、GPU SM 直接写 `BlueFlame`/MMIO 门铃、Warp 级 `AggregateRequests` 削减 `10.67x` 门铃流量 | [`ibgda_gpu_initiated_rdma`](experiments/ibgda_gpu_initiated_rdma/README.md) · [`moe_ep_alltoall`](experiments/moe_ep_alltoall/README.md) |
| **[DeepEP Engram 表 GPU 直发 RDMA Get 与异构对称虚拟内存](docs/components/deepep-engram-hybrid-elastic-symmetric-memory.md)** | `HybridElasticSymmetricMemory` `pidfd_getfd` VMM 跨进程映射，用 Host NUMA 内存池节省 `1,024 GiB` HBM | [`deepep_engram_hybrid_symmetric`](experiments/deepep_engram_hybrid_symmetric/README.md) |
| **[PyTorch `CUDASymmetricMemory` 与 NVLS `multimem` 硬件多播](docs/components/symmetric-memory-multimem-collectives.md)** | `multimem.ld_reduce.acc::f32` / `multimem.st` 机柜级 NVSwitch 硬件归约，较 1-Shot P2P 削减 `6.22x` 流量 | [`symm_mem_multimem_collective`](experiments/symm_mem_multimem_collective/README.md) |
| **[NCCL Device API (`ncclGin`)、Copy Engine `P2P_CUMEM` 与 `PAT` 算法](docs/components/nccl-device-gin-symmetric-collectives.md)** · **[EN](docs/components/nccl-device-gin-symmetric-collectives.en.md)** | `0 SM CTA` 硬件 Copy Engine DMA（消除 `10.5%–21.1%` GEMM 尾波膨胀）与 1,024 卡 `PAT` 对数跳数（`7` 跳） | [`nccl_gin_symmetric_sim`](experiments/nccl_gin_symmetric_sim/README.md) · [`nccl_ce_p2p_ras_plane`](experiments/nccl_ce_p2p_ras_plane/README.md) · [`nccl_pat_binomial_allgather_rs`](experiments/nccl_pat_binomial_allgather_rs/README.md) |
| **[UEC 1.0 数据包裁剪 (`Packet Trimming`) 与多轨喷洒拥塞控制](docs/components/uec-packet-trimming-switch-buffering.md)** · **[UEC 1.1](docs/components/uec-transport-congestion.md)** | 800G 浅缓冲交换机动态 Alpha 缓冲池、64B 裁剪报头 `0.5-RTT` 快速重传，32-to-1 Incast P99 尾延迟降低 `69.9%` | [`uec_packet_trimming_buffer`](experiments/uec_packet_trimming_buffer/README.md) · [`uec_fabric_congestion`](experiments/uec_fabric_congestion/README.md) |
| **[网内计算 SHARP 集合通信与共封装光学 (`CPO` / `LPO` / `OCS`)](docs/components/in-network-collectives-sharp.md)** · **[光学互连](docs/components/optical-interconnect-cpo-ocs.md)** | Quantum-2/3 SHARP 交换机归约树饱和与位级非确定性边界；1.6T CPO (`9 pJ/bit`) 与 3D MEMS OCS 重配置 | [`sharp_in_network_collectives`](experiments/sharp_in_network_collectives/README.md) · [`optical_cpo_ocs_fabric`](experiments/optical_cpo_ocs_fabric/README.md) |

---

### 域 4：6D 分布式预训练、RL 后训练引擎 (`verl` / `Slime` / `Miles`) 与 Checkpoint 容错（L05 & L08）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[分布式训练 6D 并行与 FSDP2/ZeRO 联合状态与通信机制](docs/components/training-parallelism.md)** | TorchTitan `ParallelDims`、FSDP2 逐参数 `DTensor` 双缓冲预取、2D `USP (Ulysses × ZigZag Ring)` 长序列注意力 | [`training_parallelism`](experiments/training_parallelism/README.md) · [`fsdp2_async_tp_composability`](experiments/fsdp2_async_tp_composability/README.md) · [`context_parallel_usp_zigzag`](experiments/context_parallel_usp_zigzag/README.md) |
| **[Megatron-FSDP (`HFSDP`)、`nccl_ub` 零 SM 抢占与 `DualPipe` 零气泡流水](docs/components/training-parallelism.md)** | `HFSDP` 将 120B 模型峰值显存从 `244.20 -> 51.01 GiB/GPU` 压降；`DualPipe` / `ZB-V` 消除流水线气泡 | [`megatron_fsdp_nccl_ub_hfsdp`](experiments/megatron_fsdp_nccl_ub_hfsdp/README.md) · [`zero_bubble_dualpipe`](experiments/zero_bubble_dualpipe/README.md) |
| **[Megatron-Core MoE 免辅助损失路由与 DeepEP EPLB 专家重平衡](docs/components/megatron-moe-auxfree-routing-token-drop.md)** | `expert_bias` 整数符号更新、`qb_dual_update` 与动态热门专家副本复制（EPLB）消除 All-to-All 木桶效应 | [`megatron_moe_auxfree_token_drop`](experiments/megatron_moe_auxfree_token_drop/README.md) · [`deepep_eplb_expert_rebalancing`](experiments/deepep_eplb_expert_rebalancing/README.md) |
| **[RL 后训练引擎：`verl` 零拷贝 FP4 重分片与 `Slime` 异步陈旧度控制](docs/components/async-rl-staleness-weight-sync.md)** | Actor→Rollout 原位 `_mxfp4_staging_data` 重分片、Partial Rollout 截断续生与逐 Token 重要性采样 (`TIS`) | [`rl_rollout_resharding`](experiments/rl_rollout_resharding/README.md) · [`async_rl_staleness_pareto`](experiments/async_rl_staleness_pareto/README.md) |
| **[分布式 Muon 优化器（五次 Newton-Schulz）与 `FlexShard`](docs/components/distributed-muon-newton-schulz-optimizer.md)** | 5 步 BF16 Newton-Schulz 正交化（条件数 $\kappa: 250 \to 1.02$）与 `FSDP-Owner + TP-Gram` 低通信分片 | [`distributed_muon_optimizer`](experiments/distributed_muon_optimizer/README.md) · [`torchtitan_flexshard_hybridep_sim`](experiments/torchtitan_flexshard_hybridep_sim/README.md) |
| **[PyTorch DCP `AsyncStager`、3FS/GDS 存储管线与万卡灰故障容错](docs/components/pytorch-dcp-async-stager-replication.md)** · **[GDS/DCP](docs/components/storage-checkpoint.md)** | 锁页共享内存池将 Checkpoint 停顿从 `3.737s -> 0.048s` 压缩；`FlightRecorder + NCCL RAS` `< 650ms` 死锁定位 | [`async_dcp_checkpoint_stager`](experiments/async_dcp_checkpoint_stager/README.md) · [`megascale_fault_tolerance`](experiments/megascale_fault_tolerance/README.md) · [`nccl_flight_recorder_hang_triage`](experiments/nccl_flight_recorder_hang_triage/README.md) |

---

### 域 5：MicroVM 安全沙箱 (`Firecracker` / `Kata`) 与 K8s / DRA / Kueue / Slurm 拓扑调度（L06 & L07）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[Agentic RL 工具沙箱（Firecracker / Kata / eBPF）与 GPU Rollout 协同调度](docs/components/agentic-sandbox-coscheduling.md)** | 硬件虚拟化 MicroVM 快照克隆（UFFD/NBD）、工具执行长尾超时隔离与 GPU-CPU 并发池配比寻优 | [`agentic_sandbox_coscheduler`](experiments/agentic_sandbox_coscheduler/README.md) |
| **[集群调度、拓扑感知与多租户故障隔离（K8s / Kueue TAS / Slurm）](docs/components/cluster-orchestration.md)** | Kubelet `TopologyManager` 按位与掩码、Kueue TAS 层级域、Slurm `topology/block & tree` 与 MIG XID 隔离 | [`cluster_orchestration`](experiments/cluster_orchestration/README.md) · [`heterogeneous_cluster_scheduler`](experiments/heterogeneous_cluster_scheduler/README.md) |
| **[GPU ECC 预测性排空（Predictive Drain）与万卡慢节点（Straggler）检测](docs/components/gpu-ecc-row-remapping-predictive-failure.md)** | NVML Row-Remapper / Page Retirement 阈值预警、双端遥测交叉定位灰故障节点与零中断热备替换 | [`gpu_ecc_predictive_drain`](experiments/gpu_ecc_predictive_drain/README.md) · [`grey_failure_straggler_detector`](experiments/grey_failure_straggler_detector/README.md) |
| **[机密计算 GPU（NVIDIA H100/Blackwell Confidential Computing）开销建模](docs/components/confidential-gpu-tee.md)** | PCIe Bounce Buffer 加密拷贝、SPDM/TDX 远程证明与多卡 NVLink 机密域 (`PPCIE`) 吞吐损耗边界 | [`confidential_gpu_overhead`](experiments/confidential_gpu_overhead/README.md) |

---

### 域 6：800VDC 供电、冷板液冷、AI Factory 成本经济学与评测基准（L10 & L11）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[机架功率密度、液冷/风冷边界与 MNNVL 机柜级故障域](docs/components/rack-power-cooling.md)** | 130kW+ NVL72 冷板液冷（DLC）与 CDU 热力学、DCGM `IMEX` 72-GPU 爆炸半径、Slurm 令牌桶浪涌限流 | [`rack_power_cooling`](experiments/rack_power_cooling/README.md) · [`rack_power_transients`](experiments/rack_power_transients/README.md) |
| **[容量、成本与吞吐经济学（TorchTitan MFU、vLLM Goodput 与 PD 配比）](docs/components/capacity-economics.md)** | 单 Token 真实训练/推理成本核算、Gamma 突发到达下的排队冗余、Prefix Cache 命中经济学与能耗积分 | [`capacity_economics`](experiments/capacity_economics/README.md) · [`ai_factory_tco_simulator`](experiments/ai_factory_tco_simulator/README.md) |
| **[AI Infra 创新公司深挖（Fireworks / Together / Groq / Cerebras）](docs/components/ai-infra-innovators.md)** · **[v2 进阶](docs/components/ai-infra-innovators-v2.md)** | 块稀疏 M3 Attention、Sequoia 投机卸载、Groq TSP 片上 SRAM 确定性流水与 Cerebras WSE 晶圆级 2D Mesh | [`innovator_arch_tradeoffs`](experiments/innovator_arch_tradeoffs/README.md) · [`innovator_pareto_v2`](experiments/innovator_pareto_v2/README.md) |
| **[阿里云 AI Infra（“通-云-哥”全栈架构）深度测评报告](docs/alibaba-cloud-ai-infra/README.md)** | 平头哥真武 PPU / 镇岳 SSD / 倚天 CPU、磐久 AL128 超节点、HPN 7.0 双平面网络与 PAI 灵骏潮汐调度 | [`docs/alibaba-cloud-ai-infra/evidence.json`](docs/alibaba-cloud-ai-infra/evidence.json) |
| **[端到端综合架构设计案例](docs/architectures/end-to-end-ai-infra-case.md)** · **[合成容量预算](docs/architectures/synthetic-inference-budget.md)** · **[架构师实战题库](docs/learning/exercises.md)** | 将显存账本、通信拓扑、PD 分离配比、机架功耗与故障恢复串联为可落地的集群架构方案与自测练习 | **[📚 完整 124 阶段目录 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** |

---

## 🧪 验证与可复现运行命令（Verification & Reproducibility）

本仓库所有验证脚本与 382 套实验均仅使用 **Python 3.12 标准库**，无需 GPU 或第三方依赖：

```powershell
# 1. 运行全量单元测试套件（630 项单元测试）
python -m unittest discover -s tests

# 2. 校验研究队列、证据台账、一手信源引用、隐私边界与全部 Markdown 相对链接
python scriptsalidate_research.py

# 3. 抽检代表性确定性物理仿真实验
python experiments\capacity_rooflineun.py --check
python experiments\sm100_blackwell_fmha_fa4un.py --check
python experiments\disaggregated_kv_routingun.py --check
python experiments\ibgda_gpu_initiated_rdmaun.py --check
python experimentsi_factory_tco_simulatorun.py --check
```

## 🛡️ 证据分级与公开边界（Evidence & Publication Boundaries）

- **严格证据分级**：[`research/sources.json`](research/sources.json)、[`research/evidence.json`](research/evidence.json)、[`research/queue.json`](research/queue.json) 与 [`research/progress.json`](research/progress.json) 将证据严格划分为 `announcement`、`specification`、`implementation`、`author-benchmark`、`own-offline`、`own-hardware` 与 `unverified`，绝不将静态源码分析或离线解析算例伪装为实机硬件跑分。
- **版权与第三方声明**：仓库仅保存原创技术分析、公开来源链接与自研标准库合成实验；第三方项目与商标权利归原作者所有，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`docs/research-operations.md`](docs/research-operations.md)。
