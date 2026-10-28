# Megatron-LM 单 DP 组 Refit 负载均衡、NCCL 2.32.3-1 峰值 BusBW 与 CFT Counted 窗口、TransformerEngine 128x128 MXFP8 重量化、FlashAttention-4 SM100 Sparse-MLA 64-Head 尾界、Mooncake、LMCache、Dynamo 与 NIXL 深度解析（Phase 122）

> **English Version**: [megatron-refit-dp-nccl-2323-cft-te-mxfp8-fa4-mla-mooncake-lmcache-dynamo-nixl-phase122.en.md](megatron-refit-dp-nccl-2323-cft-te-mxfp8-fa4-mla-mooncake-lmcache-dynamo-nixl-phase122.en.md)
> **所属阶段**: Phase 122 (`priority: 131`) · Track C 分布式训练、集合通信、KV 分层与多进程遥测
> **实验脚本**: [`experiments/megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122/run.py`](../../experiments/megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122/run.py)

---

## 1. 一手来源与证据分类（Primary Evidence）

| 证据 ID / 来源 | 仓库与 PR/Commit | 合入日期 | 证据分级 | 核心贡献 |
|---|---|---|---|---|
| `src-megatron-lm-resharding-single-dp-group-spread-replicas-pr8502-2026` | [`NVIDIA/Megatron-LM#8502`](https://github.com/NVIDIA/Megatron-LM/pull/8502) (`4a55e6fe`) | 2026-10-28 | `implementation` | 移除 `_round_robin_dp()` 中 `if len(grouped_by_dp) == 1: return src_meta_list[0]` 的错误快路径，使 `TP=1, PP=1` 下所有目标 rank 通过 `src_meta_list[dst_rank % len(src_meta_list)]` 均匀从各 DP 副本拉取权重（`DP=4` 时单源网卡出站流量降低 **4.0x**） |
| `src-nccl-v2323-gb300-b200-h100-peak-busbw-53a84c7e-2026` | [`NVIDIA/nccl@53a84c7e`](https://github.com/NVIDIA/nccl/commit/53a84c7ec23abe56e152affce86fae36e47c5ef9) | 2026-09-30 | `author-benchmark` | 发布 NCCL 2.32.3-1 在 GB300 (`NVL4..NVL64`)、B200 (`NVL8`) 与 H100 (`NVL8`) 上 Legacy (`8GB`)、Symmetric (`2GB`) 与 CE (`8GB`) 的峰值 BusBW：GB300 `NVL64` Symmetric AllReduce 达 **992 GB/s**（较 Legacy `878 GB/s` 提升 **+12.98%**），GB300 `NVL8` CE AllGather/Alltoall 达 **751 GB/s** |
| `src-nccl-cft-counted-operations-symmetric-window-3a87143f-2026` | [`NVIDIA/nccl@3a87143f`](https://github.com/NVIDIA/nccl/commit/3a87143fee5d132217e3e1dd8b04c54dfd08b984) | 2026-09-28 | `specification` + `implementation` | 文档化 `NCCL_WIN_CFT_COUNTED` 与 `putCounted` / `putMultimemCounted` / `redCounted`，用落盘后递增的对称内存字节计数器取代 `data + fence + flag` 三步同步（`256B` 计数器对齐、`16B` 传输对齐、每 GPU $\le 32$ 活动计数器、$\le 256\text{ GB}$ 窗口上限） |
| `src-te-grouped-mxfp8-requantize-128x128-persistent-tiles-pr3656-2026` | [`NVIDIA/TransformerEngine#3656`](https://github.com/NVIDIA/TransformerEngine/pull/3656) (`15dcc848`) | 2026-10-09 | `implementation` | 新增 `128x128` 宽 tile 持久化分组 MXFP8 重量化内核（寄存器驻留 `128B` 合并行段、`512B` 连续 swizzled 缩放块），在触达 `live_rows` 后的首个 tile 立即终止，消除容量填充缓冲区上的空跑 CTA（`4096/32768` 活跃行下 tile 工作量缩减 **32.0x**） |
| `src-fa4-sm100-sparse-mla-bwd-preprocess-lse-log2-tail-bound-2865a891-2026` | [`Dao-AILab/flash-attention@2865a891`](https://github.com/Dao-AILab/flash-attention/commit/2865a891a90c01f21d9d1390c063b4fbf2331fd1) | 2026-09-23 | `implementation` | 修复 SM100 Sparse-MLA 反向预处理在 `use_padded_offsets=False` 且 `64` 个 Q head、奇数 token 序列下将 `lse_log2 = +inf` 尾部写入下一条序列首 token（`64` 行）导致 `P = exp2(S - inf) = 0`、梯度清零的竞态破坏 |
| `src-mooncake-conductor-configurable-max-blocks-per-context-pr4907-2026` | [`kvcache-ai/Mooncake#4907`](https://github.com/kvcache-ai/Mooncake/pull/4907) (`179ffb21`) | 2026-10-28 | `implementation` | Mooncake Conductor 新增 `max_blocks_per_context` 配置项（默认 `200,000`，超限时批量淘汰至 `90%` 即 `180,000`；`0` 禁用容量淘汰；负数/非整数启动即报错退出） |
| `src-mooncake-ascend-hccs-2mib-register-alignment-check-pr4903-2026` | [`kvcache-ai/Mooncake#4903`](https://github.com/kvcache-ai/Mooncake/pull/4903) (`e70c0998`) | 2026-10-28 | `implementation` | 在 Ascend Direct `HCCS` 模式下，于 `registerLocalMemory` 阶段立即拦截未对齐 `2 MiB` (`2,097,152 B`) 的 `MEM_DEVICE` 指针（如 `torch_npu` 缓存分配器 `+512 B` 子块）并返回 `ERR_INVALID_ARGUMENT` |
| `src-mooncake-pg-apply-view-update-atomic-active-ranks-pr5008-2026` | [`kvcache-ai/Mooncake#5008`](https://github.com/kvcache-ai/Mooncake/pull/5008) (`5ef59b0c`) | 2026-10-28 | `implementation` | `MooncakeCommunicator::applyViewUpdate` 先在栈上构建 `next_active_ranks[64]` 再单趟写回 `meta_->activeRanks`，消除先清零再回填期间并发 worker 误读 `activeRanks == false` 的竞态窗口 |
| `src-lmcache-mp-grpc-aio-stream-read-ahead-semaphore-pr5858-2026` | [`LMCache/LMCache#5858`](https://github.com/LMCache/LMCache/pull/5858) (`d181fafa`) | 2026-10-28 | `implementation` + `author-benchmark` | LMCache MP 在 `grpc.aio` 服务端通过 `_STREAM_READ_AHEAD = 1` 信号量限制流式事件积压，慢订阅者下在途批次从 `998` 降至 `158`、服务端 RSS 增量从 `128 MB` 降至 `4 MB`（**32.0x** 降幅），单记录批次吞吐从 `794.5` 提升至 `2163.5 batches/s`（**2.723x**） |
| `src-dynamo-sglang-shared-build-postprocessor-streaming-parity-pr17440-2026` | [`ai-dynamo/dynamo#17440`](https://github.com/ai-dynamo/dynamo/pull/17440) (`b7bf2cc4`) | 2026-10-28 | `implementation` | 抽出 `_build_postprocessor` 统一 SGLang `--dyn-preprocess-workers 0` 内联路径与多进程 worker 池路径的 `15` 项流式后处理配置（`skip_special_tokens`、reasoning/tool 解析器等） |
| `src-nixl-telemetry-prometheus-mp-collision-only-pid-labels-pr2387-2026` | [`ai-dynamo/nixl#2387`](https://github.com/ai-dynamo/nixl/pull/2387) (`250581c1`) | 2026-10-12 | `implementation` | `prometheus_mp` 仅在同一 `(hostname, agent_name, local_rank)` 下存在 $\ge 2$ 个存活写入进程时才附加 `pid` 与 `agent_instance` 标签，使同名进程重启后无缝延续原 Prometheus 时间序列 |

---

## 2. 核心架构机制与公式推导

### 2.1 Megatron-LM 单 DP 组 Refit 均衡 (`#8502`) 与 NCCL 2.32.3-1 峰值带宽 (`53a84c7e`, `3a87143f`)

- **单 DP 组 Refit 源选择公式 (`#8502`)**：在 `TP=1, PP=1` 训练向推理集群同步权重（Refit）时，所有 $D$ 个源副本共享同一个 `data_parallel_group_ranks` 元组（`len(grouped_by_dp) == 1`）。旧快路径直接返回 `src_meta_list[0]`，导致所有目标 rank 全部从 rank 0 拉取权重；移除快路径后简化为：
  $$\text{src\_replica}(\text{dst\_rank}) = \text{src\_meta\_list}\big[\text{dst\_rank} \bmod D\big]$$
  在 $D=4$ 时将源 rank 0 的峰值出站请求数从 $4$ 降至 $1$（**4.0x** 减负）。
- **NCCL 2.32.3-1 GB300 / B200 / H100 峰值 BusBW (`53a84c7e`)**：
  - 在 GB300 上，Symmetric (`2GB`) AllReduce 从 `NVL4` 的 `673 GB/s` 扩展至 `NVL64` 的 **992 GB/s**（较 Legacy `8GB` 的 `878 GB/s` 高出 **+12.98%**）；
  - Copy Engine (`CE`, `8GB`) AllGather 与 Alltoall 在 GB300 `NVL8` 上达到 **751 GB/s**（较 Legacy `710 / 703 GB/s` 提升 `+5.8% / +6.8%`）。

### 2.2 TE `128x128` 持久化 MXFP8 重量化 (`#3656`) 与 FA4 Sparse-MLA 64-Head 尾界 (`2865a891`)

- **TE `128x128` 持久化网格 (`#3656`)**：将每个 CTA 的重量化块从 `32x128` 扩大至 `128x128`，2 个方向的 swizzled E8M0 缩放因子各自形成连续的 $128 \times 128 / 32 = 512\text{ B}$ 内存块，且持久化网格在超过 `live_rows` 的首个行块处直接停止，避免为容量填充区启动空跑 CTA。
- **FA4 Sparse-MLA 64-Head `lse_log2` 越界覆写 (`2865a891`)**：预处理 tile 固定为 `128` 个打包 `(token, head)` 行。当 `Q heads = 64` 且 `use_padded_offsets=False` 时，1 个 tile 跨 2 个 token；奇数长度序列（如 `3` tokens = `192` 行）的最后一个 tile 仅半满，若按 `seqlen_q_rounded = 256` 写出，尾部 `64` 行 `lse_log2 = +inf` 会直接覆写下一条序列的第 1 个 token，使其重计算的注意力概率 $P = 2^{S - \infty} = 0$，梯度完全消失。改为在非填充模式下以 `seqlen_q` 为界即可彻底消除该跨序列污染。
