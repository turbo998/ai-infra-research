# AI Infra Research — 全栈智算与系统工程研究工作台

> 🌐 **Language / 语言切换**: **[中文](README.md)** | **[English](README.en.md)** | **[📚 完整 133 阶段全量目录归档 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** | **[📚 Full 133-Phase Catalog (`docs/full-catalog-en.md`)](docs/full-catalog-en.md)** | **[🗺️ 11 层全栈地图 (`docs/stack-map.md`)](docs/stack-map.md)**

---

## 📌 Executive Overview Summary（执行摘要与核心架构结论）

> **核心研究基线（Phase 1 – Phase 186 全量合入 `main`，Priority 195 · 567 题 & 563 套实验基线）**
> 本仓库围绕 **11 层 AI 算力与分布式系统工程栈** 展开基于一手固定版本源码（`implementation`）、架构规范（`specification`）与零依赖 Python 3.12 确定性仿真模型（`own-offline`）的深度技术研究，累计完成 **155+ 篇双语深度专题报告与架构师选型 FAQ 指南**，所有定量结论均通过自动化校验测试，**零未核验主观臆断**：
> - **研究规模与可复现账本**：累计完成 **`567` 个核心工程研究问题**（`565 done` + `2 blocked` 外部闭源标准，公开可复现实验 100% 覆盖）、**`563` 套零依赖 Python 3.12 标准库实验**（全部通过 `--check`）、**`3,814` 条固定 Commit/Blob 第一手源码引用**（[`research/sources.json`](research/sources.json)）、**`868` 条分级证据主张**（[`research/evidence.json`](research/evidence.json)）、**`813` 项全绿单元测试** 以及 **`186` 份 7-Card 4:5 视觉评测报告**（`docs/reports/`，最新 **[训练 Checkpoint 与推理 KV 分层卸载选型 7-Card 评测报告](docs/reports/ai-storage-checkpoint-kv-offload-selection-cards.html)**、**[超大 MoE 推理量化与并行选型 7-Card 评测报告](docs/reports/moe-quantization-parallelism-selection-cards.html)** 与 **[LLM 推理引擎与 PD 分离选型 7-Card 评测报告](docs/reports/llm-inference-engine-pd-selection-cards.html)**）。
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
| **L11 · 确定性物理仿真与证据核验底座** | `scripts/validate_research.py` · `409 Python 3.12 Experiments` · `657 Unit Tests` · `134 7-Card 4:5 Benchmark Decks` | 纯 Python 3.12 标准库实现，100% 离线确定性复现所有公式、协议状态机与性能边界，严格区分 `implementation`、`author-benchmark` 与 `own-offline`。 |

---

## 🗂️ 六大主题域核心研究报告索引（6 Thematic Domain Summary Tables）

> 💡 **完整百科全量表入口**：本页精选展示六大主题域的代表性旗舰专题。如需查阅全部 **133 个研究阶段、414 个研究问题与 409 套实验的完整 650KB+ 百科对照表**，请直接访问：
> 👉 **[📚 中文全量组件与阶段百科表 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** | **[📚 English Complete Component & Phase Catalog (`docs/full-catalog-en.md`)](docs/full-catalog-en.md)**

### 域 1：GPU 架构、SM90/SM100 Blackwell 内核与低精度数值（L01）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[GPU/HBM 执行模型与显存层次基础](docs/components/gpu-hbm.md)** | SM/Warp 调度、寄存器压力、L2 持久化缓存、HBM3e/HBM4 屋脊线与张量核数据通路 | [`capacity_roofline`](experiments/capacity_roofline/README.md) |
| **[SM100 Blackwell FMHA 与 FlashAttention-4 (`CuTe DSL`)](docs/components/sm100-blackwell-fmha-tmem-fa4.md)** | `512` 列 `TMEM` 原位混叠（消除 `64 KiB` SMEM 溢出）、16-Warp 寄存器重分配、`FFMA2` 突破 `MUFU.EX2` 墙（`2,790 TFLOPS`） | [`sm100_blackwell_fmha_fa4`](experiments/sm100_blackwell_fmha_fa4/README.md) |
| **[FA4 Blackwell SM100 `hdim=256` 反向 `M=64` TMEM 打包](docs/components/fa4-sm100-hd256-bwd-m64-tmem-accum-layout.md)** · **[EN](docs/components/fa4-sm100-hd256-bwd-m64-tmem-accum-layout.en.md)** | 消除 `hdim=256` 反向传播 `792/640` 列 TMEM 溢出与 `sdS_xchg` 共享内存交换开销 | [`sm100_blackwell_fmha_fa4`](experiments/sm100_blackwell_fmha_fa4/README.md) |
| **[DeepSeek MLA 与 FlashMLA V4.1 分页解码内核](docs/components/deepseek-mla-flashmla-decode-kernels.md)** | SM90/SM100 2-CTA 协同、`W_UK/W_UV` 矩阵吸收（算术强度 `496–910 FLOP/B`）与 `get_mla_metadata` 消除 `89.9%` 尾波 | [`flashmla_paged_decode_scheduling`](experiments/flashmla_paged_decode_scheduling/README.md) |
| **[FlashMLA SM100 V4.1 融合 `Norm+RoPE+Attn` 与零冲突反量化](docs/components/flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.md)** · **[EN](docs/components/flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.en.md)** | `+32B` TMA 行填充消除 `LDS.32` 4 路 Bank 冲突、`e2m1*e4m3` 精确 BF16 转换与 `512/483` 列 TMEM 布局 | [`flashinfer_ds41_mixed_decode_sim`](experiments/flashinfer_ds41_mixed_decode_sim/README.md) |
| **[Blackwell SM100 块缩放微格式 (`NVFP4` vs `MXFP4`) 与 512B 原子](docs/components/sm100-block-scaled-nvfp4-kernels.md)** | `Sm1xxBlockScaledBasicChunk` 512B 雙射、UTCCP `MMA_N=256` 对齐填充、随机 Hadamard 变换 (`RHT+SR`) 预训练收敛性 | [`sm100_block_scaled_nvfp4_gemm`](experiments/sm100_block_scaled_nvfp4_gemm/README.md) · [`fp8_nvfp4_numerics`](experiments/fp8_nvfp4_numerics/README.md) |
| **[CUTLASS SM100 Ptr-Array / Grouped GEMM 与 `CLC` 动态窃取](docs/components/cutlass-sm100-grouped-gemm-clc-scheduler.md)** | 异步 `TensorMapUpdater`（`tensormaps_perform_update<false>`）与 Cluster Launch Control (`CLC`) 双级任务窃取 | [`cutlass_sm100_grouped_clc_gemm`](experiments/cutlass_sm100_grouped_clc_gemm/README.md) |
| **[DeepGEMM SM100 `MegaMoE` 融合专家池与 `MegaMHC` 超连接](docs/components/deepgemm-sm100-mega-moe-mhc-hyperconnection.md)** · **[EN](docs/components/deepgemm-sm100-mega-moe-mhc-hyperconnection.en.md)** | Sinkhorn-Knopp 双随机流形超连接（MHC）、CUDA Graph 纪元屏障（`Epoch 64B`）与分组 MoE 设备端 Swizzle | [`deepgemm_mega_moe_mhc_sim`](experiments/deepgemm_mega_moe_mhc_sim/README.md) |
| **[Hopper → Blackwell → Vera Rubin (`VR200 NVL144` / `HBM4`) 代际演进](docs/components/vera-rubin-nvl144-hbm4-architecture.md)** | `HBM4 2048-bit` (`13.5 TB/s`)、`NVLink-6` (`3.6 TB/s`) 与Groq TSP / Cerebras WSE 晶圆级 SRAM 异构投机级联对比 | [`vera_rubin_hbm4_nvl144_roofline`](experiments/vera_rubin_hbm4_nvl144_roofline/README.md) · [`wafer_sram_gpu_cascaded_serving`](experiments/wafer_sram_gpu_cascaded_serving/README.md) |
| **[批大小不变性确定性推理算子 (`Batch-Invariant Kernels`)](docs/components/batch-invariant-deterministic-inference.md)** | Split-KV / Split-K 固定归约树消除浮点非结合律引起的跨 Batch 1-ULP 漂移，实现 `0 ULP` 逐比特复现 | [`batch_invariant_deterministic_inference`](experiments/batch_invariant_deterministic_inference/README.md) |

---

### 域 2：HBM3e/CXL/LMCache 分层 KV 缓存与 Prefill-Decode 分离推理（L02 & L04）

| 专题研究报告（中文 / English） | 核心覆盖机制与源码锚点 | 配套确定性实验 (`experiments/`) |
| :--- | :--- | :--- |
| **[大模型 RLHF / GRPO 后训练基础设施选型 FAQ（`原生 FSDP 采样` vs `Ray 静态双池` vs `HybridEngine 3D 重分片` vs `异步 Partial Rollout`）](docs/components/rlhf-post-training-infra-selection-faq.md)** · **[EN](docs/components/rlhf-post-training-infra-selection-faq.en.md)** | 64×H100 (`$160/hr`) 70B GRPO 推理模型四档后训练架构选型（`Tier 3 HybridEngine verl` 同池复用 + `3.2 s` 零拷贝 3D 权重重分片 + `cuMemUnmap` Sleep 将单步耗时从 `612.0 s` 压至 `158.2 s (3.87x)`；针对 `24K Long-CoT` `52.0%` 同步长尾气泡，`Tier 4 异步流式 + Partial Rollout` 跨步复用 KV 前缀并配合 `τ<=2` Dual-Clip 重要性采样将气泡降至 `8.5%`、单步压至 `96.4 s`、吞吐达 `174,037.5 tok/s (6.35x)`、MFU `73.0%`，千步成本从 `$27,200` 降至 `$4,284.80` 立省 `$22,915.20 (-84.2%)`） | [`rlhf_post_training_infra_selection`](experiments/rlhf_post_training_infra_selection/run.py) |
| **[AI 智算集群光互连与铜缆架构选型 FAQ（`DAC/AEC 铜缆` vs `DSP 可插拔` vs `800G LPO` vs `冷板 CPO + 3D MEMS OCS`）](docs/components/ai-cluster-optical-interconnect-selection-faq.md)** · **[EN](docs/components/ai-cluster-optical-interconnect-selection-faq.en.md)** | 2,048 卡 (`8,192` 个 800G 端点，3 年 `26,280 小时`) 四档光/铜互连架构选型（`Tier 2 柜内 DAC/AEC 铜缆 <=7m` 将下联 4,096 端口功耗从 `16.0W` 压至 `2.5W`、立省 `$3.56M CapEx`；`Tier 3 800G LPO <=50m` 拔除光模块 PAM4 DSP 使脊层 PHY 延迟从 `115.0 ns` 降至 `8.0 ns (14.4x)`、3 年总 TCO 降至 `$3,923,443.71 (-58.5%)`；`Tier 4 冷板硅光 CPO + 外置热插拔 ELS + 3D MEMS OCS` 消灭全部 Spine 光模块与 O-E-O 转换，互连功耗降至 `20.89 kW (-84.1%)`、3 年链路闪断从 `442.4 次` 压至 `36.9 次 (降 11.99x)`） | [`ai_cluster_optical_interconnect_selection`](experiments/ai_cluster_optical_interconnect_selection/run.py) |
| **[AI 智算中心高密机柜供电与液冷散热架构选型 FAQ（`强制风冷` vs `背门热交换 RDHx` vs `单相冷板液冷 DLC` vs `两相/浸没式液冷`）](docs/components/ai-datacenter-cooling-power-selection-faq.md)** · **[EN](docs/components/ai-datacenter-cooling-power-selection-faq.en.md)** | 1,152 卡（16×72 卡域）四档散热与供电架构选型边界（`单相冷板液冷 1-Phase DLC + CDU` + `ORv3 54V DC` 母排将机柜从 `64 柜` 压至 `16 柜`（`128.34 kW/柜`、占地缩减 `4.0x`），`PUE` 从 `1.48` 降至 `1.12`（3 年省电费 `$3.15M`），更解锁单柜 100% 无源铜缆 NVLink 背板立省 `$4.48M` 跨柜光模块，3 年综合基建 TCO 净省 `$6.59M (-46.0%)`；对比浸没式液冷 `$3.10M` CapEx 与 `88 分钟` 换件 MTTR） | [`ai_datacenter_cooling_power_selection`](experiments/ai_datacenter_cooling_power_selection/run.py) |
| **[多租户 GPU 切分共享与硬件隔离选型 FAQ（`Time-Slicing` vs `CUDA MPS` vs `硬件 MIG` vs `Green Contexts / Multi-LoRA`）](docs/components/gpu-sharing-multi-tenant-isolation-selection-faq.md)** · **[EN](docs/components/gpu-sharing-multi-tenant-isolation-selection-faq.en.md)** | H100 80GB 四档 GPU 切分与隔离边界（硬件 `MIG 7×1g.10gb` 将跨租户故障爆炸半径锁在 `0.0%`、噪音邻居 P99 稳定 `25.0 ms (+4.2%)`，7 合 1 月省 `$10,950 (-85.7%)`；区分 `CUDA MPS` `100.0%` Xid 31 致命连坐域；`CUDA 12.4+ Green Contexts` 同进程切 `108+24 SMs` 将 Decode P99 压降 `2.76x`；`Multi-LoRA` 将 28 个 70B 部门模型缩至 2 卡省 `$47,450/月`） | [`gpu_sharing_isolation_selection`](experiments/gpu_sharing_isolation_selection/run.py) |
| **[大模型投机解码与多 Token 预测选型 FAQ（`Autoregressive` vs `Prompt Lookup N-Gram` vs `7B Draft` vs `EAGLE-3 / DeepSeek-MTP`）](docs/components/speculative-decoding-selection-faq.md)** · **[EN](docs/components/speculative-decoding-selection-faq.en.md)** | 70B FP8 (TP=8) 四档投机解码选型边界（`EAGLE-3 / DeepSeek-MTP` 仅 `+1.2 GB` 显存、`1.95 ms` 草稿，代码/推理接受长度 `3.42 / 2.95`，`BS=4` 有效 TPOT 从 `18.40 ms` 压至 `6.10 ms` 达 `163.9 tok/s` 提速 `3.02x`；零显存 `Prompt Lookup N-Gram` 代码提速 `2.24x`；动态门限 `BS>=48` 自动旁路防高并发劣化） | [`speculative_decoding_selection`](experiments/speculative_decoding_selection/run.py) |
| **[分布式训练 Checkpoint 架构选型 FAQ](docs/components/training-checkpoint-architecture-selection-faq.md)** · **[EN](docs/components/training-checkpoint-architecture-selection-faq.en.md)** · **[长上下文 KV Cache 分层卸载选型 FAQ](docs/components/kv-cache-tiered-offload-selection-faq.md)** · **[EN](docs/components/kv-cache-tiered-offload-selection-faq.en.md)**（**[合集](docs/components/ai-storage-checkpoint-kv-offload-selection-faq.md)**） | 512 卡 `9,000 GB` 训练 Checkpoint 两阶段异步 DCP（`3.125 s` 冻结 vs 同步 POSIX `225.0 s`，压缩 `72.0x`，有效吞吐 `87.50% -> 99.83%`，月省 `$181.8K`）与 64K 长上下文 `2.048 GB` KV 卸载（Host Pinned DRAM `51.2 GB/s` 恢复 TTFT `52.0 ms` 加速 `15.0x`、400GbE RDMA NVMe+GDS `40.96 GB/s` 达 `62.0 ms` 加速 `12.58x`） | [`ai_storage_checkpoint_kv_offload`](experiments/ai_storage_checkpoint_kv_offload/README.md) |
| **[大模型推理量化选型 FAQ（`BF16` vs `W4A16` vs `FP8` vs `NVFP4/MXFP4`）](docs/components/llm-quantization-selection-faq.md)** · **[EN](docs/components/llm-quantization-selection-faq.en.md)** · **[超大 MoE 并行选型 FAQ（`TP` vs `DP+EP`）](docs/components/moe-parallelism-selection-faq.md)** · **[EN](docs/components/moe-parallelism-selection-faq.en.md)**（**[合集](docs/components/moe-quantization-parallelism-selection-faq.md)**） | 671B/37B MoE 4 种量化精度选型边界（`FP8 1×128/128×128` 保留 `99.6%` 精度、`2.65x` 吞吐、`62.0%` 降本；Blackwell `NVFP4` `4.34x` 吞吐、`77.2%` 降本）、`Attention DP + MoE EP + DeepEP` 消除跨机 AllReduce 与 MoE 小矩阵碎片、以及 `EPLB` 冗余专家复制压降 `37.1%` P99 尾延迟（Goodput `+53.1%`） | [`moe_quantization_parallelism_selection`](experiments/moe_quantization_parallelism_selection/README.md) |
| **[LLM 推理引擎选型 FAQ（vLLM vs. SGLang vs. TensorRT-LLM）](docs/components/llm-inference-engine-selection-faq.md)** · **[EN](docs/components/llm-inference-engine-selection-faq.en.md)** · **[PD 分离与 KV 卸载 FAQ](docs/components/pd-disaggregation-kv-offload-selection-faq.md)** · **[EN](docs/components/pd-disaggregation-kv-offload-selection-faq.en.md)**（**[合集](docs/components/llm-inference-engine-pd-disaggregation-selection-faq.md)**） | 三大开源推理引擎选型边界、Prefill（算力受限 `512.0 FLOP/B`）vs Decode（显存带宽受限 `31.2 FLOP/B`）屋脊线拆解、SGLang `RadixAttention`（`3.45x` TTFT）与 NVIDIA Dynamo + Mooncake 400G RDMA PD 分离（`1.024 GB @ 44.91 GB/s = 22.8 ms`，`3.76x` P99 TPOT 降低、`2.38x` Goodput、`-34.2%` TCO） | [`llm_inference_engine_pd_selection`](experiments/llm_inference_engine_pd_selection/README.md) |
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
| **[AI Infra 网络选型与 RoCEv2 架构 FAQ：InfiniBand vs. Ethernet](docs/components/ib-vs-ethernet-roce-selection-faq.md)** · **[EN](docs/components/ib-vs-ethernet-roce-selection-faq.en.md)** | IB vs. Ethernet/RoCEv2 选型边界、多租户/训推存融合/跨机房五大以太网优势场景、RoCEv1/v2 协议栈与 PFC/ECN/DCQCN/UEC 演进 | [`uec_packet_trimming_buffer`](experiments/uec_packet_trimming_buffer/README.md) · [`uec_fabric_congestion`](experiments/uec_fabric_congestion/README.md) |
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
| **[DeepSeek DeepInfer 与“单卡时薪 $38 / 日吞吐 155T”生产经济学](reports/2026-11-01/ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.md)** · **[EN](reports/2026-11-01/ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.en.md)** | `V4.1-Flash` `1:1` E/P/D 分离、`DSpark` 动态投机、`140-token` SWA 重放、`100 PiB` 3FS (`890 B/tok`) 与 `$38/GPU-hr` 拆账 | [`simulate_deepinfer_economics.py`](experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py) · [7-Card 报告](reports/2026-11-01/ai-infra-phase167-benchmark-cards.html) · [视觉工作台](reports/2026-11-01/ai-infra-phase167-visual-workbench.html) |
| **[AI Infra 创新公司深挖（Fireworks / Together / Groq / Cerebras）](docs/components/ai-infra-innovators.md)** · **[v2 进阶](docs/components/ai-infra-innovators-v2.md)** | 块稀疏 M3 Attention、Sequoia 投机卸载、Groq TSP 片上 SRAM 确定性流水与 Cerebras WSE 晶圆级 2D Mesh | [`innovator_arch_tradeoffs`](experiments/innovator_arch_tradeoffs/README.md) · [`innovator_pareto_v2`](experiments/innovator_pareto_v2/README.md) |
| **[阿里云 AI Infra（“通-云-哥”全栈架构）深度测评报告](docs/alibaba-cloud-ai-infra/README.md)** | 平头哥真武 PPU / 镇岳 SSD / 倚天 CPU、磐久 AL128 超节点、HPN 7.0 双平面网络与 PAI 灵骏潮汐调度 | [`docs/alibaba-cloud-ai-infra/evidence.json`](docs/alibaba-cloud-ai-infra/evidence.json) |
| **[端到端综合架构设计案例](docs/architectures/end-to-end-ai-infra-case.md)** · **[合成容量预算](docs/architectures/synthetic-inference-budget.md)** · **[架构师实战题库](docs/learning/exercises.md)** | 将显存账本、通信拓扑、PD 分离配比、机架功耗与故障恢复串联为可落地的集群架构方案与自测练习 | **[📚 完整 131 阶段目录 (`docs/full-catalog-zh.md`)](docs/full-catalog-zh.md)** |

---

## 🧪 验证与可复现运行命令（Verification & Reproducibility）

本仓库所有验证脚本与 409 套实验均仅使用 **Python 3.12 标准库**，无需 GPU 或第三方依赖：

```powershell
# 1. 运行全量单元测试套件（657 项单元测试）
python -m unittest discover -s tests

# 2. 校验研究队列、证据台账、一手信源引用、隐私边界与全部 Markdown 相对链接
python scriptsalidate_research.py

# 3. 抽检代表性确定性物理仿真实验
python experiments\capacity_roofline
un.py --check
python experiments\sm100_blackwell_fmha_fa4
un.py --check
python experiments\disaggregated_kv_routing
un.py --check
python experiments\ibgda_gpu_initiated_rdma
un.py --check
python experimentsi_factory_tco_simulator
un.py --check
```

## 🛡️ 证据分级与公开边界（Evidence & Publication Boundaries）

- **严格证据分级**：[`research/sources.json`](research/sources.json)、[`research/evidence.json`](research/evidence.json)、[`research/queue.json`](research/queue.json) 与 [`research/progress.json`](research/progress.json) 将证据严格划分为 `announcement`、`specification`、`implementation`、`author-benchmark`、`own-offline`、`own-hardware` 与 `unverified`，绝不将静态源码分析或离线解析算例伪装为实机硬件跑分。
- **版权与第三方声明**：仓库仅保存原创技术分析、公开来源链接与自研标准库合成实验；第三方项目与商标权利归原作者所有，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 与 [`docs/research-operations.md`](docs/research-operations.md)。

### Phase 148（2026-10-31 · priority: 157）
- [FlashInfer Cake VSA / QSA / SM120 GDN / MSA / MoE EP 与 Triton / TileLang（中文）](docs/components/flashinfer-cake-vsa-qsa-gdn-sm120-msa-triton-tilelang-phase147.md)
- [SGLang AMD 确定性 / AITER 与 vLLM Qwen4Exp Int64 / Rust 及 TRT-LLM LTX-2.3（中文）](docs/components/sglang-amd-aiter-ct-fp8-vllm-qwen4-rust-trtllm-glm-ltx23-phase147.md)
- [Megatron CG+CP / 无 Fork 清理与 PyTorch NEON/MTIA 及 Mooncake S3 / LMCache / NIXL（中文）](docs/components/megatron-cg-cp-nofork-rl-pytorch-mooncake-s3-lmcache-nixl-phase147.md)
- [Phase 148 7-Card 视觉基准报告卡](docs/reports/ai-infra-phase147-benchmark-cards.html)

### Phase 148（2026-10-31 · priority: 157）
- [FlashInfer LM-Head Loss / FP4 RMSNorm / GDN / Cake FMHA 与 Triton / TileLang（中文）](docs/components/flashinfer-lmhead-rmsnorm-fp4-gdn-fmha-triton-tilelang-phase148.md)
- [SGLang AMD DSv4.1 FP8 wo_a 与 vLLM Anthropic/Rust 及 TRT-LLM Responses（中文）](docs/components/sglang-dsv4-woa-fp8-vllm-anthropic-rust-trtllm-responses-phase148.md)
- [Megatron RL/TrainState 与 PyTorch Metal/Dynamo 及 Mooncake/LMCache/Dynamo/NIXL GDS（中文）](docs/components/megatron-rl-trainstate-pytorch-mooncake-part-lmcache-nixl-gds-phase148.md)
- [Phase 148 7-Card 视觉基准报告卡](docs/reports/ai-infra-phase148-benchmark-cards.html)

### Phase 149（2026-10-31 · priority: 158）
- [FlashInfer MXFP8 Band-Path / MoK Rubin EP8 / KG1 128-Row / MiniMax-H3 与 Triton / TileLang（中文）](docs/components/flashinfer-mxfp8-band-mok-rubin-kg1-minimax-triton-tilelang-phase149.md)
- [SGLang AMD AITER FAv3 / XPU page_size=1 / Ascend FIAS-v2 与 vLLM Rust / TRT-LLM TPOT（中文）](docs/components/sglang-aiter-fav3-xpu-ascend-vllm-rust-trtllm-tpot-phase149.md)
- [Megatron WandB / MFSDP 与 PyTorch Metal/Dynamo 及 Mooncake / LMCache / Dynamo / NIXL（中文）](docs/components/megatron-wandb-mfsdp-pytorch-mooncake-lmcache-dynamo-nixl-phase149.md)
- [Phase 149 7-Card 视觉基准报告卡](docs/reports/ai-infra-phase149-benchmark-cards.html)

### Phase 150（2026-10-31 · priority: 159 · 150 阶段里程碑）
- [FlashInfer Rubin sm_107a SSD / Warp Decode / GB200 NVFP4 MLA / SM120 MQA 与 Triton / TileLang（中文）](docs/components/flashinfer-ssd-sm107a-warpdecode-nvfp4-mla-sm120-mqa-triton-tilelang-phase150.md)
- [SGLang NPU DSpark / MLA k_scale=1.0 旁路与 vLLM Rust UMBP / Fast-Start 及 TRT-LLM TF32（中文）](docs/components/sglang-npu-dspark-mla-kscale-vllm-umbp-preload-trtllm-openengine-phase150.md)
- [Megatron 检查点保留 / Shortcut MoE 与 PyTorch Metal/Dynamo 及 Mooncake HA / Dynamo / NIXL（中文）](docs/components/megatron-ckpt-retain-shortcut-moe-pytorch-mooncake-ha-dynamo-nixl-phase150.md)
- [Phase 150 7-Card 视觉基准报告卡](docs/reports/ai-infra-phase150-benchmark-cards.html)

### Phase 151 (`priority: 160`, 2026-10-31) — Cake LM-Head 有效行预算、SM120 NVFP4 变长重叠、ROCm W4A16 64 位偏移、Muse Spark CUDA 路由与 Mooncake/NIXL 并发加固
- **Track 1 (GPU 内核与编译器栈)**：[中文](docs/components/flashinfer-lmhead-minimax-kda-mxfp8-fmha-triton-gsan-tilelang-phase151.md) | [EN](docs/components/flashinfer-lmhead-minimax-kda-mxfp8-fmha-triton-gsan-tilelang-phase151.en.md) — FlashInfer Cake LM-Head 有效行与峰值预算（算力节省 `37.5%`，激活封顶 `512 MB`）、SM120 NVFP4 变长重叠 (`1.2912x`)、KDA N32 直发 (`1.3122x`)、Grouped MXFP8 预取 (`1.2842x`)、GB300 FP8 FMHA 跨阶段 `S` 列复用 (`1.1964x`)、Triton GSan 剖析器与类 `constexpr` 预加载序列化。
- **Track 2 (推理与服务引擎)**：[中文](docs/components/sglang-paged-experts-mhc-vllm-w4a16-muse-trtllm-b12x-phase151.md) | [EN](docs/components/sglang-paged-experts-mhc-vllm-w4a16-muse-trtllm-b12x-phase151.en.md) — SGLang Paged-Experts 精确锁页（节省 `37.5%` 锁页内存）、`deep_select` 3-CTA 档位 (`1.3082x`)、vLLM ROCm Triton W4A16 64 位行偏移、Muse Spark CUDA 路由 (`3.8222x`)、GLM-5.3-Flash EPD 仅传元数据（缩减 `400x`）、TensorRT-LLM DSpark 保持 Target 页大小与 B12x MoE 参数强校验。
- **Track 3 (分布式训练与 KV 传输网)**：[中文](docs/components/megatron-gdn-fp64-pytorch-mooncake-uring-dynamo-nixl-phase151.md) | [EN](docs/components/megatron-gdn-fp64-pytorch-mooncake-uring-dynamo-nixl-phase151.en.md) — Megatron 槽位原地 GDN 解码 (`1.4661x`) 与 FP64 批次不变 MoE 合并（`0.0` 漂移）、PyTorch cuBLAS 句柄重试与 ROCm Kineto 退出防崩、Mooncake TENT 拒绝短 `io_uring` 完成事件与稠密索引 (`7.5897x`)、Dynamo 移出 Actor 构建候选行（锁持有削减 `83.81%`）、NIXL Libfabric Rail EP 锁串行化。
- **7-Card 4:5 视觉基准报告卡**：[`docs/reports/ai-infra-phase151-benchmark-cards.html`](docs/reports/ai-infra-phase151-benchmark-cards.html)（`2,850` 来源 / `769` 证据条目 / `468` 问题 / `463` 实验 / `711` 单元测试）。

### Phase 152 (`priority: 161`, 2026-10-31) — Muse Spark RMSNorm 路由调优、SM120 GDN 元素对槽位读写、NIXL Mamba 跨距修复、Megatron 最小显存 DSA 与 Dynamo 全局路由
- **Track 1 (GPU 内核与编译器栈)**：[中文](docs/components/flashinfer-muse-norm-sm120-gdn-dproj-sampling-triton-tilelang-phase152.md) | [EN](docs/components/flashinfer-muse-norm-sm120-gdn-dproj-sampling-triton-tilelang-phase152.en.md) — FlashInfer Cake LM-Head `valid_rows` 边界校验、Muse Spark RMSNorm 路由调优 (`1.3396x`)、SM120 GDN 元素对槽位读写 (`1.3609x`)、SM100a 8-Warp 稠密投影 GEMM (`1.3393x`)、GB300/B200 采样启动调优 (`1.3113x`)、TileLang 缓存式 IR 遍历 (`7.069x`) 与惰性变长参数保留。
- **Track 2 (推理与服务引擎)**：[中文](docs/components/sglang-strided-moe-vllm-nixl-mamba-trtllm-minimax-indexk-phase152.md) | [EN](docs/components/sglang-strided-moe-vllm-nixl-mamba-trtllm-minimax-indexk-phase152.en.md) — SGLang 跨步网格 MoE 填充 (`1.9676x`) 与统一 `Req` 完成状态写入、vLLM NIXL Mamba `block-outermost` 跨距修复（消除 `526/512` 块覆写）、Tokenizer 快照跳过权重 (`-99.86%` 流量)、Tool Call Delta Token 统计、TensorRT-LLM MiniMax-M3 VL 视频时间戳与稀疏 Index-K 层推导 (`-75.0%` 显存)。
- **Track 3 (分布式训练与 KV 传输网)**：[中文](docs/components/megatron-dsa-minmem-routing-mooncake-heal-dynamo-ucx-sgl-phase152.md) | [EN](docs/components/megatron-dsa-minmem-routing-mooncake-heal-dynamo-ucx-sgl-phase152.en.md) — Megatron 最小显存 DSA 注意力（激活节省 `58.33%`）、异步调度路由重放与 MTP 归约组保留、PyTorch Dynamo 字典长度 Guard 与预编译自定义算子导入、Mooncake 本地磁盘悬空副本自愈与备库恢复未挂载段跳过、Dynamo 负载/会话感知 Global Frontend (`1.5850x` TTFT)、NIXL UCX SGL `split_batch_size=4096` (`1.4850x`)。
- **7-Card 4:5 视觉基准报告卡**：[`docs/reports/ai-infra-phase152-benchmark-cards.html`](docs/reports/ai-infra-phase152-benchmark-cards.html)（`2,871` 来源 / `772` 证据条目 / `471` 问题 / `466` 实验 / `714` 单元测试）。
| Phase 156 (`priority: 164`) | [KDA TMEM & KCP 重计算](docs/components/flashinfer-kda-tmem-kcp-recompute-triton-hoist-tilelang-ue4m3-phase153.md) · [DSv4.1 EPLB & MI355X Mono](docs/components/sglang-dsv41-eplb-mamba-conv-vllm-mi355x-mono-trtllm-minimax-h3-phase153.md) · [MIMO CP & NIXL CPU Proxy](docs/components/megatron-mimo-cp-mtp-argmax-mooncake-repl-ctrl-dynamo-nixl-proxy-phase153.md) | [Phase 153 7-Card 报告](docs/reports/ai-infra-phase153-benchmark-cards.html) |

- **Phase 154 (`priority: 163`，2026-10-31）**：
  - [`docs/components/flashinfer-ulysses-gdn-kcp-kda-clc-triton-remat-tilelang-pto-phase154.md`](docs/components/flashinfer-ulysses-gdn-kcp-kda-clc-triton-remat-tilelang-pto-phase154.md)（英文版：[`.en.md`](docs/components/flashinfer-ulysses-gdn-kcp-kda-clc-triton-remat-tilelang-pto-phase154.en.md)）
  - [`docs/components/sglang-lora-eplb-cp-vllm-nvfp4-engram-symmmem-trtllm-dsml-phase154.md`](docs/components/sglang-lora-eplb-cp-vllm-nvfp4-engram-symmmem-trtllm-dsml-phase154.md)（英文版：[`.en.md`](docs/components/sglang-lora-eplb-cp-vllm-nvfp4-engram-symmmem-trtllm-dsml-phase154.en.md)）
  - [`docs/components/megatron-mla-rope-nvls-symmmem-mooncake-dfs-lmcache-nixl-redis-phase154.md`](docs/components/megatron-mla-rope-nvls-symmmem-mooncake-dfs-lmcache-nixl-redis-phase154.md)（英文版：[`.en.md`](docs/components/megatron-mla-rope-nvls-symmmem-mooncake-dfs-lmcache-nixl-redis-phase154.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase154-benchmark-cards.html`](docs/reports/ai-infra-phase154-benchmark-cards.html)

- **Phase 155 (`priority: 164`，2026-10-31）**：
  - [`docs/components/flashinfer-vsa-sdsa-fp4-minimax-triton-gluon-tti-tilelang-sm120-phase155.md`](docs/components/flashinfer-vsa-sdsa-fp4-minimax-triton-gluon-tti-tilelang-sm120-phase155.md)（英文版：[`.en.md`](docs/components/flashinfer-vsa-sdsa-fp4-minimax-triton-gluon-tti-tilelang-sm120-phase155.en.md)）
  - [`docs/components/sglang-pd-poll-rust-cpuset-vllm-dsv41-kv-gather-trtllm-mamba-phase155.md`](docs/components/sglang-pd-poll-rust-cpuset-vllm-dsv41-kv-gather-trtllm-mamba-phase155.md)（英文版：[`.en.md`](docs/components/sglang-pd-poll-rust-cpuset-vllm-dsv41-kv-gather-trtllm-mamba-phase155.en.md)）
  - [`docs/components/megatron-gtp-remat-owner-p2p-mhc-mooncake-mnnvl-lmcache-apc-nixl-phase155.md`](docs/components/megatron-gtp-remat-owner-p2p-mhc-mooncake-mnnvl-lmcache-apc-nixl-phase155.md)（英文版：[`.en.md`](docs/components/megatron-gtp-remat-owner-p2p-mhc-mooncake-mnnvl-lmcache-apc-nixl-phase155.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase155-benchmark-cards.html`](docs/reports/ai-infra-phase155-benchmark-cards.html)

- **Phase 156 (`priority: 165`，2026-10-31）**：
  - [`docs/components/flashinfer-megamoe-sm103a-mhc-kda-ulysses-triton-tilelang-phase156.md`](docs/components/flashinfer-megamoe-sm103a-mhc-kda-ulysses-triton-tilelang-phase156.md)（英文版：[`.en.md`](docs/components/flashinfer-megamoe-sm103a-mhc-kda-ulysses-triton-tilelang-phase156.en.md)）
  - [`docs/components/sglang-pp-kv-minimax-glm5-vllm-hisparse-muse-trtllm-qwen38-phase156.md`](docs/components/sglang-pp-kv-minimax-glm5-vllm-hisparse-muse-trtllm-qwen38-phase156.md)（英文版：[`.en.md`](docs/components/sglang-pp-kv-minimax-glm5-vllm-hisparse-muse-trtllm-qwen38-phase156.en.md)）
  - [`docs/components/megatron-fsdp-layout-rl-logprobs-dsa-mooncake-efa-lmcache-nixl-phase156.md`](docs/components/megatron-fsdp-layout-rl-logprobs-dsa-mooncake-efa-lmcache-nixl-phase156.md)（英文版：[`.en.md`](docs/components/megatron-fsdp-layout-rl-logprobs-dsa-mooncake-efa-lmcache-nixl-phase156.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase156-benchmark-cards.html`](docs/reports/ai-infra-phase156-benchmark-cards.html)

- **Phase 157 (`priority: 166`，2026-10-31）**：
  - [`docs/components/flashinfer-tns-detloss-mhc-sm107a-allreduce-megamoe-triton-tilelang-phase157.md`](docs/components/flashinfer-tns-detloss-mhc-sm107a-allreduce-megamoe-triton-tilelang-phase157.md)（英文版：[`.en.md`](docs/components/flashinfer-tns-detloss-mhc-sm107a-allreduce-megamoe-triton-tilelang-phase157.en.md)）
  - [`docs/components/sglang-page-envelope-qwen3-dsa-vllm-skinny-reload-trtllm-minimax-phase157.md`](docs/components/sglang-page-envelope-qwen3-dsa-vllm-skinny-reload-trtllm-minimax-phase157.md)（英文版：[`.en.md`](docs/components/sglang-page-envelope-qwen3-dsa-vllm-skinny-reload-trtllm-minimax-phase157.en.md)）
  - [`docs/components/megatron-mxfp8-refit-sigmoid-moe-rocm-mooncake-fmix64-lmcache-nixl-phase157.md`](docs/components/megatron-mxfp8-refit-sigmoid-moe-rocm-mooncake-fmix64-lmcache-nixl-phase157.md)（英文版：[`.en.md`](docs/components/megatron-mxfp8-refit-sigmoid-moe-rocm-mooncake-fmix64-lmcache-nixl-phase157.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase157-benchmark-cards.html`](docs/reports/ai-infra-phase157-benchmark-cards.html)

- **Phase 158 (`priority: 167`，2026-11-01）**：
  - [`docs/components/flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.md`](docs/components/flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.md)（英文版：[`.en.md`](docs/components/flashinfer-mqa-pdl-tinygemm2-conv1d-mhc-triton-tilelang-phase158.en.md)）
  - [`docs/components/sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.md`](docs/components/sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.md)（英文版：[`.en.md`](docs/components/sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.en.md)）
  - [`docs/components/megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.md`](docs/components/megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.md)（英文版：[`.en.md`](docs/components/megatron-pretrain-inductor-avx2-mooncake-tls-lmcache-nixl-phase158.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase158-benchmark-cards.html`](docs/reports/ai-infra-phase158-benchmark-cards.html)

- **Phase 159 (`priority: 168`，2026-11-01）**：
  - [`docs/components/flashinfer-kimi-k3-fmha-fp8-sm120-mqa-triton-fp4-phase159.md`](docs/components/flashinfer-kimi-k3-fmha-fp8-sm120-mqa-triton-fp4-phase159.md)（英文版：[`.en.md`](docs/components/flashinfer-kimi-k3-fmha-fp8-sm120-mqa-triton-fp4-phase159.en.md)）
  - [`docs/components/sglang-diffusion-subtile-vllm-weight-clone-trtllm-gdn-phase159.md`](docs/components/sglang-diffusion-subtile-vllm-weight-clone-trtllm-gdn-phase159.md)（英文版：[`.en.md`](docs/components/sglang-diffusion-subtile-vllm-weight-clone-trtllm-gdn-phase159.en.md)）
  - [`docs/components/nccl-gin-coop-rccl-dmabuf-nixl-ep-mooncake-lmcache-phase159.md`](docs/components/nccl-gin-coop-rccl-dmabuf-nixl-ep-mooncake-lmcache-phase159.md)（英文版：[`.en.md`](docs/components/nccl-gin-coop-rccl-dmabuf-nixl-ep-mooncake-lmcache-phase159.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase159-benchmark-cards.html`](docs/reports/ai-infra-phase159-benchmark-cards.html)

- **Phase 160 (`priority: 169`，2026-11-01）**：
  - [`docs/components/flashinfer-gdn-bwd-rubin-megamoe-fa4-swapab-triton-gsan-phase160.md`](docs/components/flashinfer-gdn-bwd-rubin-megamoe-fa4-swapab-triton-gsan-phase160.md)
  - [`docs/components/sglang-ttft-fill-vllm-mori-ep-trtllm-nvfp4-deepep-gin-phase160.md`](docs/components/sglang-ttft-fill-vllm-mori-ep-trtllm-nvfp4-deepep-gin-phase160.md)
  - [`docs/components/nccl-gin-mcst-rccl-bf16-nixl-obj-mooncake-tent-lmcache-phase160.md`](docs/components/nccl-gin-mcst-rccl-bf16-nixl-obj-mooncake-tent-lmcache-phase160.md)
  - 7-Card 报告：[`docs/reports/ai-infra-phase160-benchmark-cards.html`](docs/reports/ai-infra-phase160-benchmark-cards.html)

- **Phase 161 (`priority: 170`，2026-11-01）**：
  - [`docs/components/flashinfer-cake-mqa-mhc-fa-sm100-cutlass-48-triton-proton-phase161.md`](docs/components/flashinfer-cake-mqa-mhc-fa-sm100-cutlass-48-triton-proton-phase161.md)（英文版：[`.en.md`](docs/components/flashinfer-cake-mqa-mhc-fa-sm100-cutlass-48-triton-proton-phase161.en.md)）
  - [`docs/components/sglang-joint-lru-dp-abort-vllm-nixl152-trtllm-own-frag-phase161.md`](docs/components/sglang-joint-lru-dp-abort-vllm-nixl152-trtllm-own-frag-phase161.md)（英文版：[`.en.md`](docs/components/sglang-joint-lru-dp-abort-vllm-nixl152-trtllm-own-frag-phase161.en.md)）
  - [`docs/components/nccl-2323-rccl-dmabuf-amdsmi-nixl-gpunetio-mooncake-lmcache-phase161.md`](docs/components/nccl-2323-rccl-dmabuf-amdsmi-nixl-gpunetio-mooncake-lmcache-phase161.md)（英文版：[`.en.md`](docs/components/nccl-2323-rccl-dmabuf-amdsmi-nixl-gpunetio-mooncake-lmcache-phase161.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase161-benchmark-cards.html`](docs/reports/ai-infra-phase161-benchmark-cards.html)

- **Phase 162 (`priority: 171`，2026-11-01 · 501 题里程碑）**：
  - [`docs/components/flashinfer-det-loss-rmsnorm-fa-aiter-triton-mir-phase162.md`](docs/components/flashinfer-det-loss-rmsnorm-fa-aiter-triton-mir-phase162.md)（英文版：[`.en.md`](docs/components/flashinfer-det-loss-rmsnorm-fa-aiter-triton-mir-phase162.en.md)）
  - [`docs/components/sglang-m3-enum-ltx25-vllm-gb300-mi355-trtllm-deepep-sm90-phase162.md`](docs/components/sglang-m3-enum-ltx25-vllm-gb300-mi355-trtllm-deepep-sm90-phase162.md)（英文版：[`.en.md`](docs/components/sglang-m3-enum-ltx25-vllm-gb300-mi355-trtllm-deepep-sm90-phase162.en.md)）
  - [`docs/components/nccl4py-2323-rccl-nixl-mooncake-tent-tone-lmcache-xpu-phase162.md`](docs/components/nccl4py-2323-rccl-nixl-mooncake-tent-tone-lmcache-xpu-phase162.md)（英文版：[`.en.md`](docs/components/nccl4py-2323-rccl-nixl-mooncake-tent-tone-lmcache-xpu-phase162.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase162-benchmark-cards.html`](docs/reports/ai-infra-phase162-benchmark-cards.html)

- **Phase 163 (`priority: 172`，2026-11-01 · 504 题基线）**：
  - [`docs/components/flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.md`](docs/components/flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.md)（英文版：[`.en.md`](docs/components/flashinfer-dw-fp32-wcx-fa-packgqa-flashmla-abi-phase163.en.md)）
  - [`docs/components/sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.md`](docs/components/sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.md)（英文版：[`.en.md`](docs/components/sglang-late-staging-ulysses-3wave-vllm-aiter0127-deepep-v2-phase163.en.md)）
  - [`docs/components/nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.md`](docs/components/nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.md)（英文版：[`.en.md`](docs/components/nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase163-benchmark-cards.html`](docs/reports/ai-infra-phase163-benchmark-cards.html)

- **Phase 164 (`priority: 173`，2026-11-01 · 507 题与 502 实验里程碑）**：
  - [`docs/components/flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.md`](docs/components/flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.md)（英文版：[`.en.md`](docs/components/flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.en.md)）
  - [`docs/components/sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.md`](docs/components/sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.md)（英文版：[`.en.md`](docs/components/sglang-pd-sampler-blake3-vllm-mxfp4-deepep-internode-phase164.en.md)）
  - [`docs/components/nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.md`](docs/components/nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.md)（英文版：[`.en.md`](docs/components/nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase164-benchmark-cards.html`](docs/reports/ai-infra-phase164-benchmark-cards.html)

- **Phase 165 (`priority: 174`，2026-11-01 · 510 题与 505 实验基线）**：
  - [`docs/components/flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.md`](docs/components/flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.md)（英文版：[`.en.md`](docs/components/flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.en.md)）
  - [`docs/components/sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.md`](docs/components/sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.md)（英文版：[`.en.md`](docs/components/sglang-muse-2shot-nomas-kv-vllm-umbp-mori-deepep-v2-phase165.en.md)）
  - [`docs/components/nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.md`](docs/components/nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.md)（英文版：[`.en.md`](docs/components/nccl-niin-tma-local-rocjitsu-py-nixl-gpunetio-mooncake-phase165.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase165-benchmark-cards.html`](docs/reports/ai-infra-phase165-benchmark-cards.html)

- **Phase 167 (`priority: 176`，2026-11-01 · DeepSeek DeepInfer & "$38/GPU-Hour / 155T Tokens/Day" 专题）**：
  - 专题研究报告：[`reports/2026-11-01/ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.md`](reports/2026-11-01/ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.md)（英文版：[`.en.md`](reports/2026-11-01/ai-infra-phase167-deepinfer-v41-pd-economics-2026-11-01.en.md)）
  - 7-Card 4:5 视觉评测卡片：[`reports/2026-11-01/ai-infra-phase167-benchmark-cards.html`](reports/2026-11-01/ai-infra-phase167-benchmark-cards.html)
  - 交互式视觉工作台：[`reports/2026-11-01/ai-infra-phase167-visual-workbench.html`](reports/2026-11-01/ai-infra-phase167-visual-workbench.html)
  - 确定性经济学仿真套件：[`experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py`](experiments/phase167_deepinfer_v41_economics/simulate_deepinfer_economics.py)（汇总数据：[`deepinfer_economics_summary.json`](experiments/phase167_deepinfer_v41_economics/deepinfer_economics_summary.json)）

- **Phase 178 (`priority: 181`，2026-11-01 · 549 题与 526 实验基线）**：
  - [`docs/components/flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.md`](docs/components/flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.md)（英文版：[`.en.md`](docs/components/flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.en.md)）
  - [`docs/components/sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.md`](docs/components/sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.md)（英文版：[`.en.md`](docs/components/sglang-mori-aiter-megamoe-sym-vllm-logprobs-deepep-rdc-phase166.en.md)）
  - [`docs/components/nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.md`](docs/components/nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.md)（英文版：[`.en.md`](docs/components/nccl-niin-multictx-rccl-dda-prefetch-nixl-prom-mooncake-phase166.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase172-benchmark-cards.html`](docs/reports/ai-infra-phase172-benchmark-cards.html)

- **Phase 179 (`priority: 188`，2026-11-01 · 552 题与 547 实验基线 · 13 仓 31 项突破）**：
  - ![Phase 179 Visual Summary](assets/svg/ai-infra-phase179-visual-summary.svg)
  - [`docs/components/gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.md`](docs/components/gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.md)（英文版：[`.en.md`](docs/components/gpu-software-stack-flashinfer-gdn-sm107a-kimi-k3-cutlass-sm110-triton-nvfp4-phase179.en.md)）
  - [`docs/components/inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.md`](docs/components/inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.md)（英文版：[`.en.md`](docs/components/inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.en.md)）
  - [`docs/components/distributed-training-megatron-mfsdp-v2-mxfp8-te-thd-cp-nccl-mooncake-urma-phase179.md`](docs/components/distributed-training-megatron-mfsdp-v2-mxfp8-te-thd-cp-nccl-mooncake-urma-phase179.md)（英文版：[`.en.md`](docs/components/distributed-training-megatron-mfsdp-v2-mxfp8-te-thd-cp-nccl-mooncake-urma-phase179.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase179-benchmark-cards.html`](docs/reports/ai-infra-phase179-benchmark-cards.html)

- **Phase 180 (`priority: 189`，2026-11-02 · 555 题与 550 实验基线 · 14 仓 31 项突破）**：
  - ![Phase 180 Visual Summary](assets/svg/ai-infra-phase180-visual-summary.svg)
  - [`docs/components/gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.md`](docs/components/gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.md)（英文版：[`.en.md`](docs/components/gpu-software-stack-flashinfer-mla-fp8-gdn-sm107a-mhc-fa-gb300-triton-tilelang-phase180.en.md)）
  - [`docs/components/inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.md`](docs/components/inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.md)（英文版：[`.en.md`](docs/components/inference-vllm-grpc-readiness-sglang-dsv41-tp-trtllm-dep4-dynamo-sticky-phase180.en.md)）
  - [`docs/components/distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.md`](docs/components/distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.md)（英文版：[`.en.md`](docs/components/distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase180-benchmark-cards.html`](docs/reports/ai-infra-phase180-benchmark-cards.html)
  - 每日归档：[`docs/daily/2026-11-02.md`](docs/daily/2026-11-02.md)

- **Phase 181 (`priority: 190`，2026-11-03 · 558 题与 553 实验基线 · 15 仓 32 项突破）**：
  - ![Phase 181 Visual Summary](assets/svg/ai-infra-phase181-visual-summary.svg)
  - [`docs/components/gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.md`](docs/components/gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.md)（英文版：[`.en.md`](docs/components/gpu-software-stack-flashinfer-12row-rmsnorm-quadroute-moe-allreduce-k1-bound-args-mhc-r200-fa4-autotune-triton-aiter-hd256-mxfp4-phase181.en.md)）
  - [`docs/components/inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.md`](docs/components/inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.md)（英文版：[`.en.md`](docs/components/inference-vllm-dsv4-packed-vision-weight-cache-sglang-warp-moe-router-trtllm-qkv-tiling-gb300-host-cache-dynamo-handoff-phase181.en.md)）
  - [`docs/components/distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.md`](docs/components/distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.md)（英文版：[`.en.md`](docs/components/distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase181-benchmark-cards.html`](docs/reports/ai-infra-phase181-benchmark-cards.html)
  - 每日归档：[`docs/daily/2026-11-03.md`](docs/daily/2026-11-03.md)

- **Phase 182 (`priority: 191`，2026-11-04 · 561 题与 556 实验基线 · 16 仓 32 项突破）**：
  - ![Phase 182 Visual Summary](assets/svg/ai-infra-phase182-visual-summary.svg)
  - [`docs/components/gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.md`](docs/components/gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.md)（英文版：[`.en.md`](docs/components/gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.en.md)）
  - [`docs/components/inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.md`](docs/components/inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.md)（英文版：[`.en.md`](docs/components/inference-trtllm-kimi-k3-pdl-attn-res-gpu-vision-norm-vllm-sw-divisor-sglang-strided-q-r3-skip-dynamo-relay-phase182.en.md)）
  - [`docs/components/distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.md`](docs/components/distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.md)（英文版：[`.en.md`](docs/components/distributed-training-megatron-moe-rl-parity-te-compiled-router-inductor-broadcast-rccl-gfx1250-dda-mooncake-hip-ipc-phase182.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase182-benchmark-cards.html`](docs/reports/ai-infra-phase182-benchmark-cards.html)
  - 每日归档：[`docs/daily/2026-11-04.md`](docs/daily/2026-11-04.md)

- **Phase 183 (`priority: 192`，2026-11-05 · 564 题与 560 实验基线 · 16 仓 32 项突破）**：
  - ![Phase 183 Visual Summary](assets/svg/ai-infra-phase183-visual-summary.svg)
  - [`docs/components/gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.md`](docs/components/gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.md)（英文版：[`.en.md`](docs/components/gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.en.md)）
  - [`docs/components/inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.md`](docs/components/inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.md)（英文版：[`.en.md`](docs/components/inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.en.md)）
  - [`docs/components/distributed-training-megatron-batch-inv-router-dcp-v2-te-nvfp4-4over6-pytorch-philox-rccl-tdm-mooncake-uring-nixl-phase183.md`](docs/components/distributed-training-megatron-batch-inv-router-dcp-v2-te-nvfp4-4over6-pytorch-philox-rccl-tdm-mooncake-uring-nixl-phase183.md)（英文版：[`.en.md`](docs/components/distributed-training-megatron-batch-inv-router-dcp-v2-te-nvfp4-4over6-pytorch-philox-rccl-tdm-mooncake-uring-nixl-phase183.en.md)）
  - 7-Card 报告：[`docs/reports/ai-infra-phase183-benchmark-cards.html`](docs/reports/ai-infra-phase183-benchmark-cards.html)
  - 每日归档：[`docs/daily/2026-11-05.md`](docs/daily/2026-11-05.md)

- **云边协同专题（2026-11-05 · Azure AMD MI300X 超大 MoE `BF16` vs `MXFP4` 部署验证 & AIPC 端侧决策模型 `Kev-0.8B/4B` 反事实 SFT / `Q8_0` GGUF 量化）**：
  - 双语工程白皮书：[`docs/components/mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.md`](docs/components/mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.md)（英文版：[`.en.md`](docs/components/mi300x-large-moe-bf16-mxfp4-and-aipc-decision-sft.en.md)）
  - 7-Card 4:5 视觉评测报告：[`docs/reports/mi300x-large-moe-bf16-mxfp4-cards.html`](docs/reports/mi300x-large-moe-bf16-mxfp4-cards.html) · [`docs/reports/aipc-kev-sft-q8-gguf-cards.html`](docs/reports/aipc-kev-sft-q8-gguf-cards.html)
  - 确定性校验实验：[`experiments/mi300x_large_moe_aipc_kev_sft/run.py`](experiments/mi300x_large_moe_aipc_kev_sft/run.py)（结果：[`results.json`](experiments/mi300x_large_moe_aipc_kev_sft/results.json)）

- **Phase 184 · 架构师选型 FAQ 专题（`priority: 193`，2026-11-08 · LLM 推理引擎选型 `vLLM` vs `SGLang` vs `TensorRT-LLM` 与 `Dynamo`+`Mooncake` PD 分离架构指南）**：
  - ![LLM Inference Engine & PD Disaggregation Selection FAQ Visual Summary](assets/svg/llm-inference-engine-pd-selection-faq-summary.svg)
  - 独立短主题双语 FAQ：[`docs/components/llm-inference-engine-selection-faq.md`](docs/components/llm-inference-engine-selection-faq.md)（[`.en.md`](docs/components/llm-inference-engine-selection-faq.en.md)）· [`docs/components/pd-disaggregation-kv-offload-selection-faq.md`](docs/components/pd-disaggregation-kv-offload-selection-faq.md)（[`.en.md`](docs/components/pd-disaggregation-kv-offload-selection-faq.en.md)）· 合集入口：[`docs/components/llm-inference-engine-pd-disaggregation-selection-faq.md`](docs/components/llm-inference-engine-pd-disaggregation-selection-faq.md)（[`.en.md`](docs/components/llm-inference-engine-pd-disaggregation-selection-faq.en.md)）
  - 7-Card 4:5 视觉评测报告：[`docs/reports/llm-inference-engine-pd-selection-cards.html`](docs/reports/llm-inference-engine-pd-selection-cards.html)
  - 交互式选型与定容工作台：[`docs/reports/ai-infra-architect-workbench.html`](docs/reports/ai-infra-architect-workbench.html)
  - 确定性仿真算例：[`experiments/llm_inference_engine_pd_selection/run.py`](experiments/llm_inference_engine_pd_selection/run.py)（结果：[`results.json`](experiments/llm_inference_engine_pd_selection/results.json)）
  - 每日归档：[`docs/daily/2026-11-08.md`](docs/daily/2026-11-08.md)

- **Phase 185 · 架构师选型 FAQ 专题（`priority: 194`，2026-11-08 Round 2 · 超大模型推理量化 `llm-quantization-selection-faq` 与 MoE 分布式并行 `moe-parallelism-selection-faq` 独立双语指南）**：
  - ![Large Model & MoE Quantization and Parallelism Selection FAQ Visual Summary](assets/svg/moe-quantization-parallelism-selection-faq-summary.svg)
  - 独立短主题双语 FAQ：[`docs/components/llm-quantization-selection-faq.md`](docs/components/llm-quantization-selection-faq.md)（[`.en.md`](docs/components/llm-quantization-selection-faq.en.md)）· [`docs/components/moe-parallelism-selection-faq.md`](docs/components/moe-parallelism-selection-faq.md)（[`.en.md`](docs/components/moe-parallelism-selection-faq.en.md)）· 合集入口：[`docs/components/moe-quantization-parallelism-selection-faq.md`](docs/components/moe-quantization-parallelism-selection-faq.md)（[`.en.md`](docs/components/moe-quantization-parallelism-selection-faq.en.md)）
  - 7-Card 4:5 视觉评测报告：[`docs/reports/moe-quantization-parallelism-selection-cards.html`](docs/reports/moe-quantization-parallelism-selection-cards.html)
  - 交互式选型与定容工作台（Section 2C）：[`docs/reports/ai-infra-architect-workbench.html`](docs/reports/ai-infra-architect-workbench.html)
  - 确定性仿真算例：[`experiments/moe_quantization_parallelism_selection/run.py`](experiments/moe_quantization_parallelism_selection/run.py)（结果：[`results.json`](experiments/moe_quantization_parallelism_selection/results.json)）

- **Phase 186 · 架构师选型 FAQ 专题（`priority: 195`，2026-11-08 Round 3 & Round 4 · 分布式训练 Checkpoint 架构 `training-checkpoint-architecture-selection-faq` 与长上下文推理 KV Cache 分层卸载 `kv-cache-tiered-offload-selection-faq` 独立双语指南）**：
  - ![Training Checkpoint & Long-Context KV Cache Tiered Offload Selection FAQ Visual Summary](assets/svg/ai-storage-checkpoint-kv-offload-selection-faq-summary.svg)
  - 独立短主题双语 FAQ：[`docs/components/training-checkpoint-architecture-selection-faq.md`](docs/components/training-checkpoint-architecture-selection-faq.md)（[`.en.md`](docs/components/training-checkpoint-architecture-selection-faq.en.md)）· [`docs/components/kv-cache-tiered-offload-selection-faq.md`](docs/components/kv-cache-tiered-offload-selection-faq.md)（[`.en.md`](docs/components/kv-cache-tiered-offload-selection-faq.en.md)）· 合集入口：[`docs/components/ai-storage-checkpoint-kv-offload-selection-faq.md`](docs/components/ai-storage-checkpoint-kv-offload-selection-faq.md)（[`.en.md`](docs/components/ai-storage-checkpoint-kv-offload-selection-faq.en.md)）
  - 7-Card 4:5 视觉评测报告：[`docs/reports/ai-storage-checkpoint-kv-offload-selection-cards.html`](docs/reports/ai-storage-checkpoint-kv-offload-selection-cards.html)
  - 交互式选型与定容工作台（Section 2D）：[`docs/reports/ai-infra-architect-workbench.html`](docs/reports/ai-infra-architect-workbench.html)
  - 确定性仿真算例：[`experiments/ai_storage_checkpoint_kv_offload/run.py`](experiments/ai_storage_checkpoint_kv_offload/run.py)（结果：[`results.json`](experiments/ai_storage_checkpoint_kv_offload/results.json)）



- 🆕 **[超长上下文（128K～1M Token）注意力机制与序列并行架构选型 FAQ：全量 FlashAttention-3 vs. DeepSpeed-Ulysses vs. 2D 混合序列并行 (`USP`) vs. 稀疏/线性混合注意力 (`MLA + NSA / 3:1 GDN`) (`docs/components/long-context-attention-parallelism-selection-faq.md`)](docs/components/long-context-attention-parallelism-selection-faq.md)** | **[English Version (`long-context-attention-parallelism-selection-faq.en.md`)](docs/components/long-context-attention-parallelism-selection-faq.en.md)** · **[7-Card 视觉评测报告 (`long-context-attention-parallelism-selection-cards.html`)](docs/reports/long-context-attention-parallelism-selection-cards.html)** · **[一图流总结 SVG](assets/svg/long-context-attention-parallelism-selection-faq-summary.svg)** · **[封存算例 (`experiments/long_context_attention_parallelism_selection/`)](experiments/long_context_attention_parallelism_selection/results.json)**
- 🆕 **[千卡 AI 训练集群故障容错与 NCCL 假死自愈架构选型 FAQ：全量重启 vs. 暖备节点替换 vs. 进程组热重构 (`ncclCommShrink`) vs. 内存副本流式恢复 (`TorchFT`) (`docs/components/gpu-cluster-fault-recovery-selection-faq.md`)](docs/components/gpu-cluster-fault-recovery-selection-faq.md)** | **[English Version (`gpu-cluster-fault-recovery-selection-faq.en.md`)](docs/components/gpu-cluster-fault-recovery-selection-faq.en.md)** · **[7-Card 视觉评测报告 (`gpu-cluster-fault-recovery-selection-cards.html`)](docs/reports/gpu-cluster-fault-recovery-selection-cards.html)** · **[一图流总结 SVG](assets/svg/gpu-cluster-fault-recovery-selection-faq-summary.svg)** · **[封存算例 (`experiments/gpu_cluster_fault_recovery_selection/`)](experiments/gpu_cluster_fault_recovery_selection/results.json)**
- 🆕 **[AI 集群以太网 RoCEv2 负载均衡与拥塞控制选型 FAQ：静态 ECMP + PFC/DCQCN vs. 多 QP 拆流 vs. 交换机 Flowlet DLB vs. 逐包喷洒 + 网卡硬件乱序重排 (`docs/components/rocev2-load-balancing-congestion-selection-faq.md`)](docs/components/rocev2-load-balancing-congestion-selection-faq.md)** | **[English Version (`rocev2-load-balancing-congestion-selection-faq.en.md`)](docs/components/rocev2-load-balancing-congestion-selection-faq.en.md)** · **[7-Card 视觉评测报告 (`rocev2-load-balancing-congestion-selection-cards.html`)](docs/reports/rocev2-load-balancing-congestion-selection-cards.html)** · **[一图流总结 SVG](assets/svg/rocev2-load-balancing-congestion-selection-faq-summary.svg)** · **[封存算例 (`experiments/rocev2_load_balancing_selection/`)](experiments/rocev2_load_balancing_selection/results.json)**
