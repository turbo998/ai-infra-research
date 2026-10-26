# Phase 91 Track C：FlashAttention-4 SM100 稀疏 MLA 残差越界防护与 ROCm/CK Split-KV 启发式修复、Megatron-LM 序列打包中间流水级防护、DeepEP 跨节点低延迟 Combine 代理栅栏、verl 蒸馏 Teacher CP 对齐、Mooncake SSD 预取与权重生命周期、NIXL 双向 NVTX 请求关联与 Dynamo KV-Router 标量遍历

> **English Version**: [fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91.en.md](fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91.en.md)  
> **阶段状态**：Phase 91（`priority: 100 — 100-Priority Milestone!`）· 关联问题 `q-fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91` · 关联论断 `claim-fa4-sm100-mla-res-megatron-seqpack-deepep-verl-cp-mooncake-ssd-nixl-dynamo-phase91`  
> **可复现实验**：[`experiments/fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91/`](../../experiments/fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91/)

---

## 1. 一手源码证据矩阵（10 项上游提交）

| # | 仓库与提交 | 日期 | 证据级别 | 核心架构机制 |
|---|---|---|---|---|
| 1 | `Dao-AILab/flash-attention` [`2cb0a8e471`](https://github.com/Dao-AILab/flash-attention/commit/2cb0a8e471b837338275b56fc8dc792cc0bf6070) | 2026-09-23 | `implementation` | SM100 CuTe 稀疏 MLA 前向在 `real_q_heads < 128` 时对寄存器到全局显存的 `o_lo` 残差写出增加 `is_valid_qhead_row` 谓词保护，防止覆盖下一 token 或触发越界访问 |
| 2 | `Dao-AILab/flash-attention` [`0b4cb4e64b`](https://github.com/Dao-AILab/flash-attention/commit/0b4cb4e64b0dd53a6042a596844f692aa5b473ab) (`#2901`) | 2026-09-24 | `implementation` + `author-benchmark` | 修复 ROCm/CK `override_num_splits_if_necessary` 恒返回 `1` 的两处缺陷（传入真实 `seqlen_k` 与 `nhead_k`），并将 `num_splits` 上限收敛至 `<= 8`（`batch*nhead_k=64` 时加速 `3.21x`） |
| 3 | `NVIDIA/Megatron-LM` [`ede2c586cf`](https://github.com/NVIDIA/Megatron-LM/commit/ede2c586cf0f8e1a26c8ca7fc9fa7d7dfea83c7a) (`#8378`) | 2026-10-21 | `implementation` | 当启用 `sequence_packing_scheduler` 时，`is_dataset_built_on_rank` 在中间流水线级（middle PP stages）返回 `False`，仅在首尾 PP 级构建数据集 |
| 4 | `deepseek-ai/DeepEP` [`d4f41e4e93`](https://github.com/deepseek-ai/DeepEP/commit/d4f41e4e937d99fd262ca53f5e4221dd17fd5086) (`#642`) | 2026-05-26 | `implementation` | `internode_ll.cu` 低延迟 combine 接收路径在 `mbarrier_arrive(empty_barriers[stage_idx])` 前插入 `fence_view_async_shared()`（`fence.proxy.async.shared::cta`） |
| 5 | `volcengine/verl` [`67bf1df3f2`](https://github.com/volcengine/verl/commit/67bf1df3f205ab06fd73915049f180b183dc8771) (`#8332`) | 2026-10-22 | `implementation` | `compute_forward_kl_topk` 将模型的 `cp_layout`（如 `dsv4_hybrid` 的连续切分）与 `local_cp_size` 透传给 teacher top-k 打包，消除 THD 下 CP=2 最高 100% 行的静默错位 |
| 6 | `kvcache-ai/Mooncake` [`f751cff2b6`](https://github.com/kvcache-ai/Mooncake/commit/f751cff2b698a9df2c9b3efbfdc36ca765656df9) (`#4272`) | 2026-10-26 | `implementation` | Mooncake Store 新增无租约只读查询 `QueryReadOnly` 与 `is_exist(prefetch_to_memory=true)` 异步 SSD→DRAM 预取提升 |
| 7 | `kvcache-ai/Mooncake` [`2e533c97fe`](https://github.com/kvcache-ai/Mooncake/commit/2e533c97fe93ee916967a7e29a9f5590c82e7e7a) (`#4313`) | 2026-10-26 | `implementation` | Mooncake Store 新增受管权重组版本生命周期（`BeginWeightRevision` 等 RPC）与 HA 原子复制，阻止普通缓存淘汰误删受管权重 |
| 8 | `ai-dynamo/nixl` [`a95227d202`](https://github.com/ai-dynamo/nixl/commit/a95227d20229df99b0363042d37c3de75ffdf378) (`#2495`) | 2026-10-26 | `implementation` | NIXL 将发送端 `postXferReq` / `nixl::xfer.complete` 与接收端 `nixl::notif.received` 的 NVTX `uint64` payload 统一设为 `nixl.xfer_request_id` |
| 9 | `ai-dynamo/dynamo` [`75d0686fb7`](https://github.com/ai-dynamo/dynamo/commit/75d0686fb7f07a5fb9665e33d8723cb0cb77a78e) (`#16765`) | 2026-10-23 | `implementation` | Dynamo SGLang `native_http.rs` 通过 `Map::new()` 直接移动 `response` 进入 `engine_data`，消除 64 KiB `routed_experts` 等大字段的额外主机内存拷贝 |
| 10 | `ai-dynamo/dynamo` [`30e972003a`](https://github.com/ai-dynamo/dynamo/commit/30e972003a0e4fa3577cf899848b715bd4c97797) (`#17149`) | 2026-10-23 | `implementation` | Dynamo `kv-router` 针对“单 worker + 单 hint 源”常见拓扑引入标量下层驻留遍历快路径（`OwnerView::exact_id`），实现零堆分配查询 |

---

## 2. 关键实现路径与架构深挖

### 2.1 FlashAttention-4 SM100 稀疏 MLA 越界防护与 ROCm/CK Split-KV 启发式（`2cb0a8e471`, `#2901`）
- **SM100 稀疏 MLA `o_lo` 残差保护**：当真实查询头数 `real_q_heads < 128`（如 24 或 64 头）时，内核将 Q 头填充至 128 行 tile。主输出通过 TMA 写出会自动丢弃越界行，但高精度残差 `o_lo` 采用普通寄存器到全局显存拷贝；若不加 `is_valid_qhead_row` 谓词，填充行会覆写下一 token 的 `o_lo` 并在末 token 越界触发非法显存访问。
- **ROCm/CK Split-KV 修复（`#2901`）**：原 `override_num_splits_if_necessary` 误用 `ceildiv(hdim_v, hdim_v) == 1` 作为 KV 块数 `num_n_blocks`，且用 `nhead` 代替 `nhead_k`（高估并行度达 GQA 倍数），导致 `num_splits` 永远为 `1`。修复后在 `batch*nhead_k = 64` 时取得 `3.21x` 加速，并将最大切分数限制为 `<= 8`（规避 `gfx950` 上 CK combine 在 `num_splits >= 9` 时相对误差跃升至 `0.28..2.0` 的数值发散）。

### 2.2 verl 蒸馏 CP 布局对齐与 Mooncake / NIXL / Dynamo 分布式数据面（`#8332`, `#4272`, `#2495`, `#17149`）
- **verl 蒸馏 Teacher-Student CP 对齐（`#8332`）**：在 `dsv4_hybrid` 下，student 采用连续（contiguous）CP 切分，而 `compute_forward_kl_topk` 原先对 teacher top-k 默认使用 `zigzag` 切分；在 `CP=2`、MXFP8、16 条 17-token 序列下，rank 0 的 136 个有效 token 中有 128 个错配，rank 1 的 136 个有效 token 100% 错配。透传 `cp_layout` 与 `local_cp_size` 后彻底消除错位。
- **Mooncake `is_exist` SSD→DRAM 预取（`#4272`）与 Dynamo 标量遍历（`#17149`）**：Mooncake 通过零租约 `QueryReadOnly` 探测仅驻留 `LOCAL_DISK` 的 KV 对象并在后台异步提升至 DRAM；Dynamo `kv-router` 则将单 worker + 单 hint 源的下层驻留查询降级为标量 ID 比较，消除逐块集合分配开销。
