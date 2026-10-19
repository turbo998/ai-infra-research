# AI Infra Research

**从实现机制到架构决策，而不是 GPU 规格表或每日新闻汇总。**

面向已有云/解决方案架构背景、希望深入 AI Infrastructure 的学习者。中文分析保留 English terminology，研究覆盖 GPU、HBM、互连/网络、存储、分布式训练、推理平台、调度运维以及容量/成本。优先深入 GPU/HBM/Network，再把组件连接到端到端工作负载。

## 阅读入口

| 入口 | 内容 |
|---|---|
| [研究框架](docs/framework.md) | 分层方法、组件深挖契约、来源和结论边界 |
| [全栈地图](docs/stack-map.md) | 11 个层面、技术栈、架构问题与研究进度 |
| [架构师学习路线](docs/learning/roadmap.md) | 云架构 → 单 GPU → 多 GPU → 多节点 → 平台架构 |
| [能力矩阵](docs/learning/competency-matrix.md) · [练习](docs/learning/exercises.md) | 可检查的知识/设计能力；不以文章数量推断掌握 |
| [GPU/HBM 深入研究](docs/components/gpu-hbm.md) | 执行模型、内存层级、容量账本、Roofline 与工作负载 |
| [互连与网络分层](docs/components/interconnect-network.md) | NVLink/NVSwitch、IB/RoCE/RDMA、NCCL 及固定源码分析 |
| [RDMA 接口生命周期](docs/components/rdma-lifecycle.md) | PD/MR/QP/CQ、WR/WC、缓冲重用、部分提交与错误边界 |
| [NCCL topology/transport](docs/components/nccl-topology-transport.md) | 固定选型、P2P/SHM/NET、GDR/proxy、IB 完成与错误路径 |
| [NCCL 图搜索与 Channel 映射](docs/components/nccl-graph-channel-mapping.md) | `ncclTopoCompute` 回溯搜索、`ringIntra/inter`、Rail Plane、PXN 与多通道反证 |
| [NCCL IB Port-Failover 与异步恢复不变量](docs/components/nccl-port-recovery-invariants.md) | `RDMA_READ` 完成位图探测、`activeQps[]` 选择性重传、有符号完成计数与 `IBV_QPT_UC` 端口恢复 |
| [张量 Layout/Dtype 与 SM100 Kernel 分派链路](docs/components/kernel-dispatch-layout.md) | ATen 步长置换、CuTe AOT C-ABI、TMA Swizzle、Tile-Ordered Partials 与 SM100 `tcgen05`/`TMEM` 寄存器重分配 |
| [数据加载与分布式 Checkpoint/恢复契约](docs/components/storage-checkpoint.md) | NVIDIA GDS (`cuFile`) DMA 路径、PyTorch DCP 两阶段协调、跨 Rank 去重、`BlockingAsyncStager` 与 Young-Daly 预算 |
| [推理调度、KV Cache 生命周期与 SLO/Goodput 边界](docs/components/inference-slo.md) | vLLM V1 统一 `token_budget` 调度、100% Prefix Cache 末块重算、`FreeKVCacheBlockQueue` 与 Chunked Prefill / P/D 分离 |
| [6D 分布式训练状态与通信联合账本](docs/components/training-parallelism.md) | TorchTitan `ParallelDims`、PyTorch FSDP2 双缓冲预取、`Shard(dim!=0)` 惩罚、TP+SP、CP Ring Attention 与 DeepEP v2 |
| [AI 集群拓扑感知调度、设备隔离与弹性恢复契约](docs/components/cluster-orchestration.md) | Kubelet Topology Manager、Kueue TAS、Slurm `topology/block & tree`、NVIDIA Device Plugin XID/MIG 与 `torchelastic` |
| [容量、成本与吞吐经济学](docs/components/capacity-economics.md) | TorchTitan 稠密/MoE 参数与 MFU 护栏、vLLM Gamma 突发排队与 P/D 分离 Goodput 定容、Slurm 梯形能耗积分與创新公司 I/O 边界 |
| [机架功率、液冷热力学与机柜级故障域](docs/components/rack-power-cooling.md) | 126kW 冷板液冷 vs 40kW 风冷 CDU/CFM 边界、Slurm `power_save.c` 令牌桶浪涌控制、RAPL/XCC 能耗积分与 DCGM `v4.6.1` MNNVL/IMEX 故障域 |
| [端到端 AI Infra 综合架构案例](docs/architectures/end-to-end-ai-infra-case.md) | 含工作负载数据流、Rail-Optimized 多轨与 126kW 液冷机柜拓扑、跨层故障自愈状态机三视图及全栈预算/反例/验收清单的综合架构方案 |
| [UEC 1.1 与 RoCEv2 多轨拥塞控制/喷洒路由对比](docs/components/uec-transport-congestion.md) | Linux `bnxt_re` 硬件门铃退避、NCCL 多 QP 喷洒、ECMP+DCQCN PFC 风暴反证与 UEC 风格选择性重传/信用控制模型 |
| [Prefill/Decode 分离与分层 KV 缓存路由契约](docs/components/disaggregated-kv-routing.md) | Mooncake 多轨 RDMA Write-Imm 切片与 `DeadlineMlu`、NVIDIA NIXL 非连续描述符、SGLang `DecodePrefixMatch` 与 HiCache 门控 |
| [大规模 MoE 专家并行与 DeepEP 双层去重/EPLB](docs/components/moe-expert-parallel-alltoall.md) | DeepEP `TokenLayout` 64B 对齐、`ptx::deduplicate` NVLink+RDMA 二级去重、`warpgroup_reg_realloc` 与在线 EPLB TMA 权重搬运 |
| [超长上下文混合注意力：FlashMLA、Ring/Ulysses CP 与 Mamba 检查点池](docs/components/long-context-hybrid-attention.md) | DeepSeek FlashMLA V3.2 (`656B`) / V4.1 (`528B` FP8, `288B` FP4)、Hopper 2-CTA DSM Crossover、Ring vs Ulysses CP 与 SGLang `Int8CheckpointStore` |
| [大规模推理强化学习引擎（RLVR / GRPO）：verl 权重重分片与 CuMem 休眠](docs/components/rl-post-training-engine.md) | `verl` `BucketedWeightSender` (`512 MiB` IPC/SHM)、`NCCLCheckpointEngine` `multi_sender`、Count-Only `DeltaShardedCheckpointEngine` 与 vLLM `CuMemAllocator` |
| [机密计算 GPU（CC-On / PPCIE）：SPDM/RIM 远程证明与 Bounce Buffer 边界](docs/components/confidential-gpu-tee.md) | `NVIDIA/nvtrust` Driver/VBIOS RIM 索引互斥与 `MSR 35` 豁免、8-GPU+4-NVSwitch 双向 PDI 拓扑门禁与 Bounce Buffer vs PPCIE 吞吐 |
| [共封装光学（CPO）、线性直驱（LPO）与光路交换（OCS）十万卡功耗/故障模型](docs/components/optical-interconnect-cpo-ocs.md) | SONiC `CpoManagerTask` (`OE`/`ELSFP` 兄弟端口级联关停)、NCCL RAS 28028 带外监控与十万卡 800G DSP/LPO/CPO/OCS 功耗与 Goodput |
| [DeepSeek 3FS (`hf3fs_usrbio`)、GPUDirect Storage (`cuFile`) 与多级 KV/Checkpoint 管线](docs/components/storage-gds-3fs-kv-tiering.md) | 3FS `/dev/shm` + `3fs-virt/iovs` 符号链接注册、CRAQ+BIBD 链式复制、LMCache `Hf3fsConnector` `memcpy` 瓶颈与 `nvidia-fs` BAR1 P2P / PCIe ACS 惩罚 |
| [Agentic RL 万级 MicroVM 沙箱（Firecracker UFFD）与 GPU KV Cache 协同调度](docs/components/agentic-sandbox-coscheduling.md) | Firecracker UFFD `-EAGAIN` 排空 `UFFD_EVENT_REMOVE`、`verl` `ToolAgentLoop` Token-in-Token-out、`2.56s`/`8.00s` Break-Even 与 35% 宿主机 I/O 令牌桶 |
| [FP8 细粒度分块缩放与 Blackwell NVFP4 预训练数值稳定性机制](docs/components/fp8-nvfp4-pretraining-numerics.md) | DeepSeek `1x128`/`128x128` + WGMMA 128 步晋升、`TransformerEngine` NVFP4 (`1x16 E2M1` + `16x16 RHT` + `cvt.rs` 随机舍入) 与离群通道 SQNR 账本 |
| [万卡同步训练容错与静默数据损坏（SDC）防御：NVRx 进程内热重启与 ABFT](docs/components/fault-tolerant-training-resilience.md) | `NVIDIA/nvidia-resiliency-ext` `inprocess.Wrapper` 热备切换 (`19.8s`)、`CliqueReplicationStrategy` 内存副本、`64K` 卡 `90.30%` ETTR 与正交 ABFT 单步拦截 |
| [132kW–600kW 液冷机柜毫秒级功率阶跃（`dP/dt`）与 OCP ORv3 超级电容削峰](docs/components/rack-power-transient-smoothing.md) | `DCGM` Field 1425-1438 固件功率平滑、`3,014.7–13,926.4 MW/s` 同步冲击、ORv3 `87.1 F` (`43.9 kJ`) 超级电容选型与 `264.91x` 斜率压缩 |
| [网内计算 SHARP 集合通信、NVLS 多播硬件归约与位级漂移边界](docs/components/in-network-collectives-sharp.md) | `NVIDIA/nccl` (`3e46c089`) `nvls.cc`/`collnet.cc` 代价模型、`16K` 卡 `NVLS + SHARP` `27.91x` 加速、`64` 组硬件树溢出回退与 `FP32` 累加消除 `BF16` 非结合律位级漂移 |
| [Hopper 与 Blackwell 跨代异构集群非对称流水线编排与跨代 KV 转码](docs/components/heterogeneous-cluster-scheduling.md) | `NVIDIA/Megatron-LM` (`b607c1b7`) `PipelineParallelLayerLayout` DSL、异构 `H100+B200` `1:3` 非对称 VPP 切分（气泡从 `43.87%` 压至 `0.28%`）与 `SM100 FP4 -> SM90 FP8` 紧凑 KV RDMA 转码 (`3.86x`) |
| [10 万卡 AI Factory 全栈 CapEx / OpEx / ETTR / Token 经济学帕累托模型](docs/components/ai-factory-tco-pareto.md) | 联立光互连、SHARP 集合通信、FP8/NVFP4、NVRx/3FS 与 ORv3 超级电容的十万卡 3 年 TCO 帕累托模型（有效算力产出提升 `7.36x`、每百万 Token 成本降低 `12.65x` 至 `$1.4307`） |
| [vLLM v1 分段/可中断 CUDA Graph 编译池与 SGLang Graph Runner 边界](docs/components/cuda-graph-piecewise-compilation.md) | `vLLM` `PiecewiseBackend` / `Breakable CUDAGraph`（`@eager_break_during_capture` 外置不变式、`global_graph_pool` + `weak_ref_tensors` `73.63x` 显存压缩）与非对齐 Batch (`BS=33/257`) 反超 Full Graph 机制 |
| [多模态视觉/视频大模型 Encode-Prefill-Decode (EPD) 三阶段解耦与 EncoderCacheManager](docs/components/multimodal-encoder-disaggregation.md) | `vLLM` 引用计数 `EncoderCacheManager`（多轮 `mm_hash` 锁存与 FIFO `freeable` 回收）、`SGLang` 分块 Prefill 切片与 `1E/3P/4D` EPD 降低 P99 TPOT `3.54x` (`58.64 -> 16.58 ms`) |
| [EAGLE-3 / DeepSeek MTP 树状掩码投机解码验证与高并发屋脊线坍塌边界](docs/components/eagle3-speculative-tree-verification.md) | `SGLang` `TreeMaskMode` 非因果树验证、PD 首轮 `one-hot q(X)=1` 验证与 `AdaptiveSpeculativeParams` 分桶 EMA 控制器（解决 `BS=32` `0.889x` 坍塌并恢复至 `2.422x`） |
| [SM90 Hopper 与 SM100 Blackwell Warp-Specialized Ping-Pong 算子与 TMEM 工程](docs/components/warp-specialized-kernel-engineering.md) | `CUTLASS` SM90 `setmaxnreg` (`24/240 regs`, `2.303x` 提速)、`FA3` FP8 Ping-Pong (`1.468x`) 与 SM100 `256 KB` `TMEM` + `2-CTA` (`-50%` L2 流量、`94.20%` SOL) |
| [Zero-Bubble (ZB-H1/ZB-H2/ZB-V) 与 DeepSeek-V3 双向 DualPipe/DualPipeV 流水线编排](docs/components/zero-bubble-dualpipe-scheduling.md) | `B/W` 梯度拆分、`ZB-V` (`1.26%` 气泡、`1x` 参数)、`DualPipe`/`DualPipeV` 异向 `F&B` 重叠（暴露通信压至 `0.45%–0.53%`、`1.447x` 提速）与 `rollback_adamw` (`6.94e-18`) |
| [千级多租户 Serverless Multi-LoRA 服务架构：S-LoRA 统一分页与 Punica SGMV](docs/components/multi-lora-serving-architecture.md) | `vLLM` `LoRAKernelMeta` 稳定排序、`Triton SGMV` (`_BI_SPLIT_K=8`, 64 LoRA `36.50x` 提速)、`S-LoRA` 统一分页（碎片率 `31.4%->1.6%`）与秩拐点 `r*=64` |
| [PyTorch FSDP2 (`fully_shard` 逐参数 `DTensor`) 与 `SymmetricMemory` `Async-TP` 5D 正交并行架构](docs/components/fsdp2-dtensor-async-tp-composability.md) | `pytorch/pytorch` (`1eba9b3a`) `fully_shard` 逐参数 `DTensor`（分配器碎片 `<1.5%`、省 `6.72–9.65 GiB/GPU`）与 `SymmetricMemory` 双流 `Async-TP=8`（隐藏 `91.27%–97.09%` 通信气泡、5D 单步 `1.322x` 提速） |
| [NCCL 零拷贝用户缓冲区注册 (`ncclCommWindowRegister`)、`CUMEM` NVLS 多播与免 SM 抢占通信](docs/components/nccl-zero-copy-user-buffer-registration.md) | `NVIDIA/nccl` (`3e46c089`) `register.cc`/`coll_reg.cc`/`allocator.cc` 零拷贝注册、2-CTA 消除并发 GEMM `+16.46%–47.18%` 尾波膨胀、省 `576 MiB–2.305 GiB/GPU` FIFO HBM 与 P99 延迟 `2.07x–3.74x` 提速 |
| [GPU HBM3e ECC 行重映射 (`Row Remapping`)、`Xid 63/64/94/95` 故障演化与预测性无损排空](docs/components/gpu-ecc-row-remapping-predictive-failure.md) | `NVIDIA/DCGM` (`64df9f89`) `BANK_REMAP_AVAIL_LOW/NONE` (`388/389`) 与 `m_devastatingXids` 源码、5 态马尔可夫风险模型与 10K-GPU 预测性微检查点排空（非计划中断削减 `89.04%`、ETTR `99.93%`） |
| [1M–4M 超长序列训练的 2D 混合上下文并行 (`USP = Ulysses × ZigZag Ring-Attention`)](docs/components/context-parallel-usp-zigzag-ring.md) | `feifeibear/long-context-attention` (`b92a1f4c`) 与 `NVIDIA/Megatron-LM` (`ce852bfb`) 源码、消除 GQA `8x` KV 复制与 `49.2%` 因果气泡、实现 `CP=64` 下 `2.82x–2.87x` 吞吐提速 |
| [零开销结构化受限解码：`XGrammar` 自适应位掩码缓存与 vLLM V1 异步重叠](docs/components/structured-decoding-xgrammar-bitmask.md) | `mlc-ai/xgrammar` (`7e6b29a4`) `AdaptiveTokenMask`（`97.4%` 词表预分类、`32.0x` `int32` 位集压缩）与 `vLLM V1` 异步 CPU 流水线（`BS=64` 暴露开销 `56.4 us`、吞吐 `3.67x`） |
| [算子级选择性激活重计算 (`SAC` 背包价值比) 与 NUMA 绑定 PCIe Gen5 零气泡卸载](docs/components/selective-activation-checkpoint-offload.md) | `pytorch/torchtitan` (`7c4b8e91`) `SAC` + `NUMA AffinityMode.NODE` 与 `Megatron-LM` 残差卸载规划器、70B `16K` 激活显存削减 `86.94%` (`467.81->61.09 GiB`) 且步耗时仅 `+2.53%` |
| [动态 Token 预算分块预填充 (`Chunked Prefill`) 与 Decode P99 TPOT 干扰控制](docs/components/chunked-prefill-tpot-interference-scheduler.md) | `vLLM V1` (`local_prefill_interval` / `long_prefill_token_threshold`) 与 `SGLang` (`dynamic_chunk_sizer` / `prefill_tile_block_m=64`) 源码、闭环求解 $C^*(48, 7.5\text{GiB})=640$ 将 P99 TPOT 压降 `10.27x` (`26.02 ms`) 并提升 SLA Goodput `4.49x` |
| [DiT 视频生成混合并行 (`xDiT`: `CFG × Ulysses × PipeFusion`) 与 `TeaCache` 残差跳步](docs/components/dit-video-xdit-pipefusion-serving.md) | `xdit-project/xDiT` (`f7c16aa9`) 4D 进程组、`_xdit_kv_cache` 空间微块流水与 `use_usp=True` `all_reduce` 防死锁 `TeaCache`、50 步 `Wan2.1-14B` 720p-81f 延迟降低 `4.65x` (`350.78->75.40 s`, `-0.31 dB` PSNR) |
| [`torch.compile` (`TorchInductor` / `FxGraphCache` / `MEGA_AOT`) 冷启动消除与融合](docs/components/torch-compile-inductor-cache-serving.md) | `vLLM` `TorchCompileWithNoGuardsWrapper` (`skip_all_guards_unsafe`) + `StandaloneCompiledArtifacts` 双层 SHA256 去重 (`80L->2`)、冷启动从 `29.44 min` 压至 `18.0 s / 11.8 s` (`98x–149x`) 并减少 `56.2%` 逐元素 HBM 流量 |
| [Prefill-Decode 解耦架构下 EAGLE-3 / MTP 投机解码零停顿种子交接](docs/components/disaggregated-speculative-decoding-pd.md) | `sgl-project/sglang` (`da414add`) `_draft_extend_for_prefill` + `CaptureHiddenMode.LAST` + `req_to_token` DSA 物理槽位重映射、仅增 `+1.25%` Draft KV + `8.62 KiB/rank` 种子将 `4K–128K` 首步投机锁定在 `13.435 ms`（`128K` 下较 Decode 重算快 `18.45x`） |
| [万卡同步训练灰故障 (`Grey Failure / Slow Straggler`) 双端遥测根因定位与热备替换](docs/components/grey-failure-straggler-detection.md) | `NVIDIA/Megatron-LM` (`ce852bfb`) `StragglerDetector` (`Rtt/Clk`) 与 `NVIDIA/nccl` (`3e46c089`) `ncclProfileKernelCh` `pTimer` 差分矩阵、消除单慢卡 `32.94%–63.00%` MFU 坍塌并在 `3.70 s` `100%` 归因 + `4.20 s` 热备切换（30 天 Goodput 从 `71.35%` 升至 `98.83%`） |
| [四级 KV Cache 缓存层次 (`HBM3e -> DDR5 -> CXL/Mooncake -> NVMe GDS`) 与重计算盈亏平衡](docs/components/tiered-kv-cache-offload-hierarchy.md) | `LMCache/LMCache` (`663757e3`) `StorageManager` 级联回写 + `GdsBackend` `cuFile` 与 `vLLM V1` `MooncakeStoreConnector`、推导 $L_{\text{breakeven}} = 25 / 46 / 123\text{ tokens}$（对比 POSIX Disk `6,940`）并将 1,000 并发 Agent P99 TTFT 降低 `18.91x`（`4,387.91->231.99 ms`）、Token 成本降低 `66.6%` |
| [批大小不变性确定性推理算子 (`Batch-Invariant Deterministic Kernels`) 与 0-ULP RL/合规复现](docs/components/batch-invariant-deterministic-inference.md) | `Dao-AILab/flash-attention` (`965c9212`) `num_splits_heuristic` 与 `flash_fwd_splitkv_combine_kernel` 动态归约树分析、消除 `BS=33..128` 下 `8–12 ULPs` 80 层漂移与 `96.55%–99.36%` 1,000 步 CoT 分叉率，以 `+4.21%–8.57%` TPOT 代价实现跨任意批大小的 `0 ULP` 比特级完全一致 |
| [万卡静默集合通信死锁诊断：PyTorch `FlightRecorder`、NCCL `RAS` 与 `< 650 ms` 零号病人定位](docs/components/nccl-flight-recorder-hang-diagnosis.md) | `pytorch/pytorch` (`faf5b996`) `c10d::FlightRecorder` 序列状态机与 `NVIDIA/nccl` (`3e46c089`) `src/ras/client_support.cc` (`RAS_ACE_MISMATCH` `collOpCounts` & Wait-For DAG)、在 `16,384` 卡下以 `389.4–584.2 ms` `100%` 锁定零号病人 Rank 且 `0` 误杀，恢复提速 `46.11x`（`20.384 s` vs `940.0 s`），30 天 Goodput 达 `99.93%` |
| [下一代超节点架构演进：Hopper (`NVL8`)、Blackwell (`GB200 NVL72`) 到 Vera Rubin (`VR200 NVL144` / `HBM4` / `NVLink-6`)](docs/components/vera-rubin-nvl144-hbm4-architecture.md) | 三代超节点 Roofline 与 `1T MoE (64B active, EP=64)` Token 经济学：`HBM4 2048-bit (13.5 TB/s)` + `NVLink-6 (3.6 TB/s)` + `1.6T CPO` 将 FP4 拐点推至 `1,185.2 FLOP/B`，80 层 EP 通信从 `31.20 ms` 压至 `1.48 ms`，Decode TPOT 降至 `2.28 ms`（提速 `16.52x`），1M 上下文 PD 配比收敛至 `1P:1D`，成本降至 `$1.2272/1M`（下降 `10.04x`） |
| [GPU 直发 RDMA (`IBGDA` / `NCCL_GIN_TYPE_GDAKI`)、CTA 独占 QP 与 `DeepEP V2.5` 门铃聚合](docs/components/ibgda-gpu-initiated-rdma.md) | `deepseek-ai/DeepEP` (`b5a33eef`) `NCCL_GIN_TYPE_GDAKI`、`get_qp_mode` (`SHARING_CTA` vs `SHARING_GPU`) 与 `ptx::fence_acq_rel_sys()` 源码：`512 B` 延迟降 `7.90x` (`6.561->0.831 us`)、带宽提 `16.93x` (`18.637 GB/s`)，60 层 `EP=64` MoE Decode 通信降 `5.99x` (`33.99->5.676 ms`，TPOT 提速 `2.99x` 至 `14.196 ms`) |
| [Wide-EP 解耦 `Attention-DP × Expert-EP` 调度、FP8 `SUM_LEN AllGatherv` 与双批次重叠 (`TBO`)](docs/components/attention-dp-moe-ep-serving.md) | `sgl-project/sglang` (`62cee1b1`) `dp_attention.py` 源码：`DP_attn=64` 释放 `9.34x` MLA KV 容量 (`4,147.2 GiB`)，`SUM_LEN` Group-128 FP8 `AllGatherv` + `topk_ids=-1` 掩码 + `dp_tbo_comm` 双批重叠将偏斜批耗时压降 `7.12x` (`132.54->18.62 ms`) 并通过 `_TBO_PERSIST_BUF` 省 `257.6 GiB` 显存 |
| [晶圆级/片上 SRAM (`Cerebras WSE-3` / `Groq LPU`) vs GPU HBM3e 异构投机级联经济学](docs/components/wafer-sram-gpu-cascaded-serving.md) | `vllm-project/vllm` (`973dbf80`) `SuffixDecodingProposer` (`4.484 ms`) 与 `DFlashProposer` (`3.499 ms`) 源码 + 异构级联：针对 `64K` 单会话 `10 GiB` KV 导致的片上 SRAM 容量墙，用 `SRAM 8B Draft (8K 滑窗 = 8 GiB 100% 驻留 SRAM, 0.64 ms)` + `GPU HBM3e 70B Verifier (64K 全量)` 实现 `2.563 ms` TPOT (`4.45x` 提速) 与 `$0.8566/1M` (`-73%` 成本) |
| [Blackwell SM100 块缩放微格式 (`NVFP4` vs `MXFP4`)、512B TMEM 原子与融合 Epilogue 工程](docs/components/sm100-block-scaled-nvfp4-kernels.md) | `NVIDIA/cutlass` (`78e46228`) `Sm1xxBlockScaledBasicChunk` (`128x4` `512 B` TMEM 原子) 与 `MMA_N=192 -> MMA_N_SFB=256` (`+33.33%` UTCCP 填充)、`NVFP4 (SFVecSize=16, UE4M3)` 降低离群通道 MSE `4.25x` (`+6.29 dB` SQNR)、B200 融合 `LinCombBlockScaleFactor` 提升 `M=64` Decode `+32.1%` (`1,553.4 TFLOPS`) 并使 `8x B200` `128K` KV 并发翻倍 (`74` vs `36`, `2.06x`) |
| [UEC 1.0 数据包裁剪 (`Packet Trimming`)、SONiC 动态 Alpha 缓冲池与 800G Incast 尾延迟](docs/components/uec-packet-trimming-switch-buffering.md) | `sonic-net/sonic-buildimage` (`3e670e64`) `33.0 MB` `ingress_lossless_pool` 下 32 口 `800G` RoCEv2 `XOFF` 锁死 `38.13%` (`12.58 MB`) 缓冲；UEC 免 PFC (`0%` XOFF、`80.0%` 吸收、`0.5 RTT = 6.0 us` `64 B` CP 裁剪重传) 在 `32:1 / 64:1` Incast 下缩短 P99 CCT `4.69x–4.87x` (`207.2 / 413.2 us`) 并降低无辜受害流 P99 延迟 `24.69x–43.72x` (`18.3–22.2 us`) |
| [有界陈旧度异步强化学习 (`Async RL`)、Partial Rollout 跨版本续写与逐 Token TIS 帕累托边界](docs/components/async-rl-staleness-weight-sync.md) | `THUDM/slime` (`a4202e05`) `AsyncRolloutWorker` 双门控背压与 `--partial-rollout` 源码：消除同步 GRPO `76.8%` 长尾气泡，有界异步 $\tau_{\max}=4$ + 逐 Token 版本对齐 `TIS` 实现 `140.73 eff_steps/hr` (`2.91x` 有效收敛加速，$D_{\text{KL}}=0.01657$)，超越前缀掩码 (`106.13`) 与过度陈旧 $\tau_{\max}=8$ (`131.74`) |
| [FlashInfer JIT 多级级联分页注意力 (`Cascade Attention`)、Base-2 LSE 状态归约与 PDL](docs/components/flashinfer-cascade-paged-kv-attention.md) | `flashinfer-ai/flashinfer` (`f64b4ba8`) `MultiLevelCascadeAttentionWrapper` 与 `MergeStateKernel`（Base-2 LSE `math::ptx_exp2`/`log2` 误差 `3.64e-17`、`d_sum > 0.f` `-inf` 守卫）：在 Llama-3.1-70B `25.6K` 3 级共享前缀（`B=128`）下消减 `94.44%` HBM KV 读取（`250.08 -> 13.91 GiB`），算术强度升至 `143.81 FLOPs/Byte`，提速 `16.46x`（`B=256` 达 `19.72x`） |
| [NCCL 并行聚合二项式树 (`PAT` Algorithm) 集合通信、对数跳数与 Multi-RPN `NVLS` 混合架构](docs/components/nccl-pat-binomial-allgather-reducescatter.md) | `NVIDIA/nccl` (`3e46c089`) `PatRSAlgorithm`/`PatAGAlgorithm` 与 `all_gather.h`/`reduce_scatter.h` 源码：将 `1,024` 卡顺序跳数从 Ring 的 `1,023` 压至 `7` 跳（`aggFactor=64` 削减 `141.6x` 跨节点启动），`4 MiB` 延迟降 `19.21x`（`2,227.17 -> 115.96 us`、`1 MiB` 达 `47.64x`），`16 MiB` 总线带宽提 `6.25x`（`41.80 GB/s`） |
| [Megatron-Core MoE 免辅助损失路由 (`expert_bias`)、拟牛顿对偶更新 (`qb_dual_update`) 与容量因子丢弃权衡](docs/components/megatron-moe-auxfree-routing-token-drop.md) | `NVIDIA/Megatron-LM` (`e8a302ac`) `TopKRouter` FP32 `expert_bias` 整数符号更新（`torch.sign(total_tokens - tokens_per_expert * num_experts)`）与 `qb_dual_update` 源码：在 Zipf $\alpha=0.85$ 偏斜下将最大负载从 `7.828x` 压至 `1.312x` / `1.062x`，`CF=1.25` 丢弃率从 `39.40%` 压至 `0.27%`（消除 `pad_to_capacity` `51.52%` 填充算力浪费） |
| [交互式 AI Infra 全栈架构师工作台 (H5)](docs/reports/ai-infra-architect-workbench.html) · [Phase 3](docs/reports/ai-infra-phase3-benchmark-cards.html) · [Phase 4](docs/reports/ai-infra-phase4-benchmark-cards.html) · [Phase 5](docs/reports/ai-infra-phase5-benchmark-cards.html) · [Phase 6](docs/reports/ai-infra-phase6-benchmark-cards.html) · [Phase 7](docs/reports/ai-infra-phase7-benchmark-cards.html) · [Phase 8](docs/reports/ai-infra-phase8-benchmark-cards.html) · [Phase 9](docs/reports/ai-infra-phase9-benchmark-cards.html) · [Phase 10](docs/reports/ai-infra-phase10-benchmark-cards.html) · [Phase 11](docs/reports/ai-infra-phase11-benchmark-cards.html) · [Phase 12](docs/reports/ai-infra-phase12-benchmark-cards.html) · [Phase 13](docs/reports/ai-infra-phase13-benchmark-cards.html) · [Phase 14](docs/reports/ai-infra-phase14-benchmark-cards.html) · [Phase 15](docs/reports/ai-infra-phase15-benchmark-cards.html) · [Phase 16 7-Card 报告](docs/reports/ai-infra-phase16-benchmark-cards.html) | 零外部依赖交互式 TCO/KV/SHARP 计算器与自包含 `Content-Security-Policy: default-src 'none'` 4:5 视觉卡片组，汇总全仓 63 个研究问题与 58 个可复现实验坐标 |
| [AI Infra 创新公司深挖](docs/components/ai-infra-innovators.md) · [v2 帕累托深挖](docs/components/ai-infra-innovators-v2.md) | 推理加速（Fireworks / Together / SGLang 自适应 EAGLE-3 / NVFP4 Swizzle）与新型 AI 芯片（Groq TSP / Cerebras WSE）帕累托前沿 |
| [技术趋势](docs/trends/index.md) | 证据支持的变化、成熟度与采用条件 |
| [每日研究](docs/daily/2026-10-19.md) | 最新有界研究与限制；按北京时间日期增量归档 |
| [原创离线实验](experiments/capacity_roofline/README.md) | 可重算的显存 payload 与 Roofline 理论界限 |
| [Ring cut 离线算例](experiments/ring_cut/README.md) | rank 顺序、跨节点 payload 与共享 NIC 理论下界 |
| [Rail channel cut 算例](experiments/rail_channel_cut/README.md) | 多 Channel、Multi-Rail 对齐、跨 Rail 偏斜与机内 `PATH_PHB` 瓶颈反证 |
| [IB failover invariants 算例](experiments/ib_failover_invariants/README.md) | NCCL 端口故障转移的 CQ/RQ 容量放大、选择性重传与有符号事件守恒验证 |
| [Innovator arch tradeoffs 算例](experiments/innovator_arch_tradeoffs/README.md) | Fireworks KV-Outer 字节账本、Together Sequoia 投机树 Roofline 与 Groq/Cerebras SRAM 边界 |
| [Checkpoint recovery 算例](experiments/checkpoint_recovery/README.md) | PyTorch DCP 跨 Rank 去重均衡、GDS vs POSIX 路径、异步 Stager 与 Young-Daly Goodput 仿真 |
| [Inference SLO scheduler 算例](experiments/inference_slo_scheduler/README.md) | vLLM V1 `token_budget` 调度、Prefix Cache 末块重算、Chunked Prefill 与 P/D 分离 TTFT/TPOT 权衡 |
| [Training parallelism 算例](experiments/training_parallelism/README.md) | 6D 并行显存/通信账本、FSDP2 `Shard(dim!=0)` 瞬态峰值、CP 因果倾斜与 DeepEP v2 去重 |
| [Cluster orchestration 算例](experiments/cluster_orchestration/README.md) | Kubelet NUMA/PCIe/RDMA 掩码准入、Slurm `topology/block` 反碎片与 Young-Daly/热备弹性训练 Goodput |
| [Capacity economics 算例](experiments/capacity_economics/README.md) | TorchTitan 稠密/MoE 训练成本、vLLM Gamma 突发排队与 P/D 分离 Goodput、Slurm 焦耳积分及四家创新公司吞吐/I/O 模型 |
| [Rack power & cooling 算例](experiments/rack_power_cooling/README.md) | 126kW NVL72 液冷 vs 40kW 风冷 vs PG25 退化热力学、Slurm 令牌桶浪涌限流与 RAPL 回绕、DCGM MNNVL/IMEX 故障状态机及 Groq/Cerebras 机柜画像 |
| [UEC fabric congestion 算例](experiments/uec_fabric_congestion/README.md) | ECMP+DCQCN PFC 队头阻塞、4-QP Packet Spraying 乱序窗口、RTT 梯度与 UEC 风格选择性重传尾延迟对比 |
| [Disaggregated KV routing 算例](experiments/disaggregated_kv_routing/README.md) | 多轨 RDMA Write-Imm KV 切片、小页门铃开销、慢轨动态摘除与 DRAM/NVMe HiCache 晋升排队模型 |
| [MoE EP All-to-All 算例](experiments/moe_ep_alltoall/README.md) | DeepEP `TokenLayout` 字节对齐、NVLink+RDMA 二级去重、Zipf 专家倾斜/冗余副本与 EPLB 在线权重搬运重叠 |
| [Innovator Pareto v2 算例](experiments/innovator_pareto_v2/README.md) | SGLang 自适应 EAGLE-3 步长控制、NVFP4 `128x4` vs `8x4` 缩放对齐/V-Scale 重排与 2026 四大推理流派帕累托前沿 |
| [Long-context hybrid attention 算例](experiments/long_context_attention/README.md) | `1M` 上下文 FlashMLA V4.1 (`528B`/`288B`) + DSA、Hopper 2-CTA DSM 反量化、Ring vs Ulysses CP 与 SGLang `Int8CheckpointStore` |
| [RL rollout resharding 算例](experiments/rl_rollout_resharding/README.md) | `verl` 5 种权重重分片通道（CUDA IPC / SHM / 单发与多发 NCCL / 稀疏 Delta）、vLLM `CuMemAllocator` 休眠级别与 Long-CoT 气泡消除 |
| [Confidential GPU overhead 算例](experiments/confidential_gpu_overhead/README.md) | `NVIDIA/nvtrust` RIM 索引互斥/`MSR 35` 豁免、PPCIE 8-GPU+4-NVSwitch 双向 PDI 门禁与 Bounce Buffer vs PPCIE 延迟模型 |
| [Optical CPO/OCS fabric 算例](experiments/optical_cpo_ocs_fabric/README.md) | 十万卡 800G 集群 DSP/LPO/CPO/OCS 功耗（`7.92 MW` vs `0.72 MW`）、`OE`/`ELSFP` 共享爆炸半径与同步训练 Goodput 模型 |
| [Storage GDS & 3FS KV/Checkpoint 算例](experiments/storage_gds_kv_checkpoint/README.md) | POSIX Page Cache vs `O_DIRECT` vs GDS BAR1 P2P vs 3FS `USRBIO` 在 `100 GiB` KV 恢复与 `1 TiB` Checkpoint 下的吞吐与 PCIe RC 争用 |
| [Agentic sandbox co-scheduler 算例](experiments/agentic_sandbox_coscheduler/README.md) | 万级 Firecracker UFFD 缺页风暴与 `verl` `ToolAgentLoop` GPU KV 驻留/重算/GDS 卸载（`2.56s`/`8.00s` 阈值与 35% I/O 令牌桶）协同调度 |
| [FP8 & NVFP4 numerics 算例](experiments/fp8_nvfp4_numerics/README.md) | `100x` 离群通道下 Delayed Scaling vs DeepSeek `1x128` vs NVFP4 `1x16 + RHT + SR` 的 SQNR、小梯度归零偏差与 WGMMA 晋升误差 |
| [Megascale fault tolerance 算例](experiments/megascale_fault_tolerance/README.md) | `4K–64K` GPU 规模下 Slurm+PFS vs `NVRx` 进程内热备 + 对等内存恢复的 ETTR（`18.58%` vs `90.30%`）与 BF16 比特翻转 ABFT 校验 |
| [Rack power transients 算例](experiments/rack_power_transients/README.md) | `132kW` NVL72 至 `600kW` Rubin Ultra 机柜毫秒级功率阶跃 `dP/dt`、ORv3 超级电容（`87.1 F` / `2675.3 F`）与固件平滑协同算例 |
| [SHARP in-network collectives 算例](experiments/sharp_in_network_collectives/README.md) | `64–16,384` 卡 `Ring` / `Tree` / `NVLS_TREE` / `NVLS + SHARP` AllReduce 带宽（`27.91x` 加速）、`64` 组硬件树饱和回退与 `BF16`/`FP32` 位级漂移 |
| [Heterogeneous cluster scheduler 算例](experiments/heterogeneous_cluster_scheduler/README.md) | `H100 + B200` 跨代异构 `PP=16, VPP=2` 非对称切分（`1.78x` 加速、`0.28%` 气泡）与 `128K` 跨代 `SM100 FP4 -> SM90 FP8` KV RDMA 转码 (`3.86x`) |
| [AI Factory TCO simulator 算例](experiments/ai_factory_tco_simulator/README.md) | 10 万卡 AI Factory 四代全栈配置（Gen-1 Legacy / Gen-2 Hopper / Gen-3 Het / Gen-3 Full Blackwell）3 年 TCO、有效算力天与 `$/1M Tokens` 帕累托仿真 |
| [CUDA Graph piecewise capture 算例](experiments/cuda_graph_piecewise_capture/README.md) | `488` 算子 CPU 发射墙、`32` 桶 `global_graph_pool` `73.63x` 显存压缩、B200 `BS=1` `1.92x` 提速与非对齐 `BS=33/257` Piecewise 反超 Full Graph 仿真 |
| [Multimodal encoder disaggregation 算例](experiments/multimodal_encoder_disaggregation/README.md) | `Qwen2.5-VL-72B` 共置 ViT 干扰（P99 TPOT `58.64 ms`）vs `1E/3P/4D` 三阶段 EPD + `EncoderCacheManager`（P99 TPOT `16.58 ms`、`2.70x` RPS）仿真 |
| [EAGLE-3 speculative tree 算例](experiments/eagle3_speculative_tree/README.md) | H100 FP8 `Q_ridge=229.6` 下树状投机解码在 `BS=1` (`2.605x`) 与 `BS=32` (`0.889x` 坍塌) 的屋脊线相变及自适应收缩恢复 (`2.422x`) 仿真 |
| [Warp-specialized kernels 算例](experiments/warp_specialized_kernels/README.md) | SM90 `setmaxnreg` (`24/240 regs`, `2.303x` 提速)、FA3 FP8 Ping-Pong (`1.468x`)、SM100 `256 KB` TMEM + `2-CTA` (`-50%` L2 流量、`94.20%` SOL) 与 CLC 尾波仿真 |
| [Zero-Bubble & DualPipe 算例](experiments/zero_bubble_dualpipe/README.md) | `PP=16, M=64` 下 `1F1B` / `Interleaved` / `ZB-H1/H2/V` (`1.26%` 气泡) / `DualPipe` & `DualPipeV` (`0.45%–0.53%` 暴露通信、`1.447x` 提速) 与 `rollback_adamw` (`6.94e-18`) 仿真 |
| [Multi-LoRA serving 算例](experiments/multi_lora_serving/README.md) | `Llama-3-70B` 64 并发 LoRA 下 Punica `SGMV` (`36.50x` 提速)、`S-LoRA` 统一分页（碎片 `31.4%->1.6%`、KV 容量 `+154.7%`）、Zipf 预取与秩拐点 `r*=64` 仿真 |
| [FSDP2 DTensor & Async-TP 算例](experiments/fsdp2_async_tp_composability/README.md) | `FSDP2` 逐参数 `DTensor`（碎片率 `1.15%–1.42%`、省 `6.72–9.65 GiB/GPU`）与 `SymmetricMemory` 2-Stream `Async-TP=8`（隐藏 `91.27%–97.09%` 气泡、5D 单步 `1.322x` 提速）仿真 |
| [NCCL zero-copy registration 算例](experiments/nccl_zero_copy_registration/README.md) | `132-SM` H100 上 `16–32` 通信 CTA 造成的 `+16.46%–47.18%` GEMM 尾波劣化 vs `ncclCommWindowRegister` + `CUMEM` NVLS 2-CTA 零拷贝（`0.00%` 劣化、省 `576 MiB–2.305 GiB/GPU`、P99 `2.07x–3.74x`）仿真 |
| [GPU ECC predictive drain 算例](experiments/gpu_ecc_predictive_drain/README.md) | HBM3e 5 态马尔可夫退化（`BANK_REMAP_AVAIL_LOW/NONE` 72h 故障风险 `68.4%/91.2%`）与 10K-GPU 30 天预测性微检查点排空（非计划中断削减 `89.04%`、ETTR `99.93%`）仿真 |
| [2D USP & ZigZag Ring 算例](experiments/context_parallel_usp_zigzag/README.md) | `1M–4M` 序列 `CP=64` 下 `Ulysses=8 × ZigZag-Ring=8` 对称配对 `[r, 2R-1-r]`（因果失衡 `0.16%`、无 GQA KV 冗余复制、较朴素 Ring 提速 `2.82x–2.87x`）仿真 |
| [Structured decoding XGrammar 算例](experiments/structured_decoding_xgrammar/README.md) | `128,256` 词表下 `AdaptiveTokenMask`（`97.4%` 预分类、`3,335` 动态 Token、`32.0x` `int32` 位掩码压缩）与异步重叠（`BS=64` 暴露 `56.4 us`、吞吐 `3.67x`）仿真 |
| [Selective activation offload 算例](experiments/selective_activation_offload/README.md) | 算子级 `SAC` 背包价值比排序 + NUMA 绑定 Pinned PCIe Gen5 (`56.0 GB/s`) 双缓冲卸载（削减 `86.94%` 激活显存至 `61.09 GiB`、步耗时仅 `+2.53%`）仿真 |
| [Chunked prefill interference 算例](experiments/chunked_prefill_interference/README.md) | `8x H100` `Llama-3-70B` 混批 L2/尾波干扰模型、闭环求解 $C^*(48, 7.5\text{GiB})=640$（P99 TPOT 从 `267.18 ms` 压至 `26.02 ms`、SLA Goodput `4.49x`）与 `D=96, 16K` 收缩至 `192` 仿真 |
| [DiT video PipeFusion serving 算例](experiments/dit_video_pipefusion_serving/README.md) | `Wan2.1-14B-720p-81f` (`131K` 时空 Token) 下 `TP=8` (`11.18%` 暴露通信) vs `xDiT (CFG=2 × Ulysses=2 × PipeFusion=2)` (`0.98%`) + USP 同步 `TeaCache` (`75.40 s`, `4.65x`, `-0.31 dB` PSNR) 仿真 |
| [TorchInductor compile cache 算例](experiments/torch_compile_inductor_cache/README.md) | `Llama-3-70B` 32 桶冷启动消除（`TorchCompileWithNoGuardsWrapper` + SHA256 `80L->2` 去重 + `FxGraphCache`/`MEGA_AOT` 提速 `98x–149x`）与逐元素融合（`-56.2%` HBM 流量、`+35.73%` 吞吐）仿真 |
| [Disaggregated speculative PD 算例](experiments/disaggregated_speculative_pd/README.md) | `8x H100 TP=8` PD 解耦下 Scheme A (`232.0 ms` Decode 重算气泡) vs Scheme B (`+30.0%` RDMA 膨胀) vs Scheme C SGLang 零停顿交接（`+1.25%` Draft KV + `8.62 KiB/rank` 种子、首步投机恒定 `13.435 ms`、`18.45x` 提速）仿真 |
| [Grey failure straggler detector 算例](experiments/grey_failure_straggler_detector/README.md) | `16,384x H100` 下单慢卡（`900MHz` `1.5x` GEMM 或 `3.0x` 光模块 FEC 重传）拖垮 `32.94%–63.00%` MFU vs 双端遥测 `3.70 s` `100%` 归因 + `4.20 s` 热备切换（30 天 Goodput 从 `71.35%` 升至 `98.83%`）仿真 |
| [Tiered KV cache hierarchy 算例](experiments/tiered_kv_cache_hierarchy/README.md) | `HBM3e -> DDR5 -> CXL/Mooncake -> NVMe GDS` 四级 KV 换入 vs 二次方重计算盈亏平衡（$L_{\text{be}} = 25 / 46 / 123$ vs `6,940` POSIX Disk）与 1,000 并发 Agent 会话 P99 TTFT `18.91x` 降幅仿真 |
| [Batch-invariant deterministic inference 算例](experiments/batch_invariant_deterministic_inference/README.md) | `BS=1..128` 下动态 Split-KV (`num_splits=64->1`) 与 Split-K 引发的 `8–12 ULPs` 80 层漂移及 `96.55%–99.36%` 1000 步 CoT 分叉 vs 批大小不变算子 `0 ULP`（`+4.21%–8.57%` TPOT）仿真 |
| [NCCL FlightRecorder hang triage 算例](experiments/nccl_flight_recorder_hang_triage/README.md) | `16,384` 卡静默集合通信死锁下 `FlightRecorder` + `NCCL RAS` (`RAS_ACE_MISMATCH` `collOpCounts` & Wait-For DAG) `389.4–584.2 ms` `100%` 锁定零号病人 Rank（`46.11x` 恢复提速、`99.93%` Goodput）仿真 |
| [Vera Rubin NVL144 & HBM4 roofline 算例](experiments/vera_rubin_hbm4_nvl144_roofline/README.md) | `H100 NVL8` vs `GB200 NVL72` vs `VR200 NVL144` (`HBM4 2048-bit 13.5 TB/s`, `NVLink-6 3.6 TB/s`) 在 `1T MoE (EP=64)` 下的 `2.28 ms` TPOT（`16.52x`）与 `$1.2272/1M` 成本（`10.04x` 下降）仿真 |
| [IBGDA GPU-initiated RDMA 算例](experiments/ibgda_gpu_initiated_rdma/README.md) | `DeepEP V2.5` `NCCL_GIN_TYPE_GDAKI`、`SHARING_CTA` 独占 QP (`2.27x` 消除自旋锁)、Doorbell 聚合 (`512 B` 延迟 `7.90x`、带宽 `16.93x`) 与 60 层 MoE Decode (`5.99x` 通信加速、TPOT `14.196 ms`) 仿真 |
| [Attention-DP × MoE-EP schedule 算例](experiments/attention_dp_moe_ep_schedule/README.md) | `64x H100` MLA KV 容量 `9.34x` 扩增 (`4,147.2 GiB`)、`SGLang` `SUM_LEN` Group-128 FP8 `AllGatherv` + `topk_ids=-1` 掩码 + 双批次重叠 (`TBO`, 吞吐 `7.12x`) 与 `_TBO_PERSIST_BUF` 省 `257.6 GiB` 仿真 |
| [Wafer-scale SRAM vs GPU HBM3e cascade 算例](experiments/wafer_sram_gpu_cascaded_serving/README.md) | `64K` 单会话 `10 GiB` KV 片上 SRAM 容量墙、`vLLM V1` `SuffixDecoding`/`DFlash` 与异构 `SRAM 8B Draft (8K 滑窗) + GPU HBM3e 70B Verifier (64K 全量)` (`2.563 ms` TPOT、`4.45x` 提速、`3.70x` 降本) 仿真 |
| [SM100 block-scaled NVFP4 GEMM 算例](experiments/sm100_block_scaled_nvfp4_gemm/README.md) | `CUTLASS` `Sm1xxBlockScaledBasicChunk` `512 B` TMEM 原子、`MMA_N=192 -> 256` (`+33.33%` SFB 填充)、`NVFP4` vs `MXFP4` (`4.25x` 更低 MSE、`+6.29 dB` SQNR)、B200 融合 Epilogue (`+32.1%` Decode) 与 `128K` KV `2.06x` 并发仿真 |
| [UEC packet trimming & switch buffer 算例](experiments/uec_packet_trimming_buffer/README.md) | `33.0 MB` SONiC 入向池下 32 口 `800G` RoCEv2 `38.13%` `XOFF` 税 vs UEC 免 PFC (`80.0%` 池吸收、`0.5 RTT = 6.0 us` `64 B` 裁剪重传、P99 CCT `4.69x–4.87x`、受害流 P99 `24.69x–43.72x` 改善) 仿真 |
| [Async RL staleness & TIS Pareto 算例](experiments/async_rl_staleness_pareto/README.md) | 128-GPU GRPO 下同步 Rollout `76.8%` 气泡、`--partial-rollout` (`2.42x`) 与有界异步 $\tau_{\max}=4$ + 逐 Token 版本对齐 `TIS` (`140.73 eff_steps/hr`、`2.91x` 有效收敛加速、$D_{\text{KL}}=0.01657$) 仿真 |
| [FlashInfer cascade paged KV 算例](experiments/flashinfer_cascade_paged_kv/README.md) | `flashinfer` Base-2 LSE 多级状态归约（误差 `3.64e-17`、空段 `-inf` 守卫）与 Llama-3.1-70B `25.6K` 3 级共享前缀（`B=128` 削减 `94.44%` HBM 读取、算术强度 `143.81 FLOP/B`、提速 `16.46x`）仿真 |
| [NCCL PAT binomial AllGather/RS 算例](experiments/nccl_pat_binomial_allgather_rs/README.md) | `NVIDIA/nccl` `PatRSAlgorithm`/`PatAGAlgorithm` + Multi-RPN NVLS（`1,024` 卡跳数 `1,023 -> 7`、`aggFactor=64` 削减 `141.6x` 启动、`4 MiB` 延迟 `19.21x`、`16 MiB` 带宽 `6.25x`）仿真 |
| [Megatron MoE aux-free & token drop 算例](experiments/megatron_moe_auxfree_token_drop/README.md) | Zipf $\alpha=0.85$ 专家偏斜下无均衡 `7.828x` 过载（`CF=1.25` 丢弃 `39.40%`）vs FP32 `expert_bias` 整数符号更新（`1.312x`、丢弃 `0.27%`）与 `qb_dual_update`（`1.062x`）仿真 |
| [合成容量架构练习](docs/architectures/synthetic-inference-budget.md) | 将显存账本、数据流和失败边界连接到方案设计 |
| [研究执行契约](docs/research-operations.md) | 研究 queue、跨运行续接、分支/PR、防重叠与失败恢复 |

## 证据状态

`announcement`、`specification`、`implementation`、`author-benchmark`、`own-offline`、`own-hardware`、`unverified` 分开登记。原文读过不等于完整规范审阅，源码静态分析不等于运行，离线计算不等于 GPU benchmark。

来源、主张、研究问题和进度分别保存在 [sources](research/sources.json)、[evidence](research/evidence.json)、[queue](research/queue.json)、[progress](research/progress.json)。用户学习掌握单独评估，当前未评估。

## 运行

Python 3.12 标准库；不需要 GPU 或第三方 Python 包。在仓库根目录运行：

```powershell
python -m unittest discover -s tests
python scripts\validate_research.py
python experiments\capacity_roofline\run.py --check
python experiments\ring_cut\run.py --check
python experiments\rail_channel_cut\run.py --check
python experiments\ib_failover_invariants\run.py --check
python experiments\innovator_arch_tradeoffs\run.py --check
python experiments\checkpoint_recovery\run.py --check
python experiments\inference_slo_scheduler\run.py --check
python experiments\training_parallelism\run.py --check
python experiments\cluster_orchestration\run.py --check
python experiments\capacity_economics\run.py --check
python experiments\rack_power_cooling\run.py --check
python experiments\uec_fabric_congestion\run.py --check
python experiments\disaggregated_kv_routing\run.py --check
python experiments\moe_ep_alltoall\run.py --check
python experiments\innovator_pareto_v2\run.py --check
python experiments\long_context_attention\run.py --check
python experiments\rl_rollout_resharding\run.py --check
python experiments\confidential_gpu_overhead\run.py --check
python experiments\optical_cpo_ocs_fabric\run.py --check
python experiments\storage_gds_kv_checkpoint\run.py --check
python experiments\agentic_sandbox_coscheduler\run.py --check
python experiments\fp8_nvfp4_numerics\run.py --check
python experiments\megascale_fault_tolerance\run.py --check
python experiments\rack_power_transients\run.py --check
python experiments\sharp_in_network_collectives\run.py --check
python experiments\heterogeneous_cluster_scheduler\run.py --check
python experiments\ai_factory_tco_simulator\run.py --check
python experiments\cuda_graph_piecewise_capture\run.py --check
python experiments\multimodal_encoder_disaggregation\run.py --check
python experiments\eagle3_speculative_tree\run.py --check
python experiments\warp_specialized_kernels\run.py --check
python experiments\zero_bubble_dualpipe\run.py --check
python experiments\multi_lora_serving\run.py --check
python experiments\fsdp2_async_tp_composability\run.py --check
python experiments\nccl_zero_copy_registration\run.py --check
python experiments\gpu_ecc_predictive_drain\run.py --check
python experiments\context_parallel_usp_zigzag\run.py --check
python experiments\structured_decoding_xgrammar\run.py --check
python experiments\selective_activation_offload\run.py --check
python experiments\chunked_prefill_interference\run.py --check
python experiments\dit_video_pipefusion_serving\run.py --check
python experiments\torch_compile_inductor_cache\run.py --check
python experiments\disaggregated_speculative_pd\run.py --check
python experiments\grey_failure_straggler_detector\run.py --check
python experiments\tiered_kv_cache_hierarchy\run.py --check
python experiments\batch_invariant_deterministic_inference\run.py --check
python experiments\nccl_flight_recorder_hang_triage\run.py --check
python experiments\vera_rubin_hbm4_nvl144_roofline\run.py --check
python experiments\ibgda_gpu_initiated_rdma\run.py --check
python experiments\attention_dp_moe_ep_schedule\run.py --check
python experiments\wafer_sram_gpu_cascaded_serving\run.py --check
python experiments\sm100_block_scaled_nvfp4_gemm\run.py --check
python experiments\uec_packet_trimming_buffer\run.py --check
python experiments\async_rl_staleness_pareto\run.py --check
python experiments\flashinfer_cascade_paged_kv\run.py --check
python experiments\nccl_pat_binomial_allgather_rs\run.py --check
python experiments\megatron_moe_auxfree_token_drop\run.py --check
python scripts\source_monitor.py
```

来源监控联网，只取得 allowlist 中的公开页面并保留哈希/状态，不归档第三方全文。失败返回非零；网页哈希变化不是新技术发布。结构验证也不能代替人工事实、版权与隐私审阅。

## 公开与实验边界

这是干净历史的公开研究仓库，只保存原创分析、公开来源链接和合成实验。研究通过分支/PR 更新，**不自动合并**；PR 在合并前也已经公开。首轮实质成果位于研究 PR，不应把初始化的 `main` 当成全部内容。

默认只做静态阅读和 CPU/offline 实验。GPU、付费云、模型/API 调用需单独批准；不自动安装驱动、不占用现有集群、不执行上游安装器/训练脚本。

每日研究沿用北京时间 22:30 的 Local Copilot 自动化。运行需要本机、App 与网络可用；不能保证关机、休眠或退出后继续。每次从远端有效研究分支恢复，不每天重建基线。

第三方材料保留原作者和原有权利，见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。未声明覆盖全仓库的开源许可证；公开可读不等于取得第三方再分发权。
Evidence-driven AI infrastructure research: GPU, HBM, interconnects, distributed systems, and solution architecture
