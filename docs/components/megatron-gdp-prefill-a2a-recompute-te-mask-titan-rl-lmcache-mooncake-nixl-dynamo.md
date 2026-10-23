# Megatron-LM 融合 GDP 预填充准备与 EP A2A 重叠分段全激活重计算、TE Triton 掩码容量丢弃与编译态 FP8 权重缓存、TorchTitan Muse Spark GB300 RL 128 卡布局及 LMCache/Mooncake/NIXL/Dynamo 传输面演进

> **Phase 56 (`priority: 65`) · 一手源码与 Commit 级深挖**
> - `NVIDIA/Megatron-LM@fe0e608ca417488cfdeac47b471574c45163360d` (`#6477`)
> - `NVIDIA/Megatron-LM@da2c0ec614d9bb7bdfc16c833ee6568f64f73af9` (`#6429`)
> - `NVIDIA/TransformerEngine@c9c92255465b4518380561b1bbb0aebb177c0b6d` (`#3347`)
> - `NVIDIA/TransformerEngine@9ea05cf2e8dc275d2fd3c88d39062ba20947b77d` (`#3505`)
> - `pytorch/torchtitan@43d1bb98014e0dab19a83e5220ec7ea1ab06d49e` (`#5581`)
> - `pytorch/torchtitan@094c884685c2583323b76be244b76526b85fb520` (`#5152`)
> - `LMCache/LMCache@ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be` (`#5960`), `a81b4e5aea` (`#5922`), `a0ff470128` (`#5976`)
> - `kvcache-ai/Mooncake@dd1217b6c461491e7c73ff29b4ffad215e0a9ac0` (`#4793`), `dae22d472e` (`#4434`), `b66bb2b5f6` (`#4873`)
> - `ai-dynamo/nixl@f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4` (`#2487`), `70badd6aad` (`#2478`)
> - `ai-dynamo/dynamo@3acf24807325febd0462bdff6056e74f17659e71` (`#17143`), `30e972003a` (`#17149`), `c178cbae9b` (`#16501`), `a246c3f227` (`#16351`)

---

## 1. Megatron-LM 融合 GDP Prefill Prepare 与 EP A2A 重叠分段全重计算 (`#6477`, `#6429`)

- **Gated Delta Product (`GDP`) 融合 `gdp_prefill_prepare` (`#6477`)**：在短卷积与分块递推算子之间，Eager 预填充路径原先需执行约 **14 个独立 PyTorch 算子**（含 3 次跨步张量拷贝：切分 `value/key/query`、Householder 副本折叠至序列维、GQA 复制扩展及 `beta = sigmoid(b)` 与 `g = -exp(A_log) * softplus(a + dt_bias)`）。`gdp_prefill_prepare` 将其融合为 **1 次 Triton 内核发射**，每个线程块负责一个 `(token, copy)` 的全部注意力头，并统一调用 `triton.language.extra.libdevice`（`log1p`, `exp`, `div_rn`）保证与 PyTorch Eager 路径逐位一致（`torch.equal`）。
- **EP A2A 重叠下的分段全激活重计算 (`RecomputeSegment`, `#6429`)**：
  - 在 `--overlap-moe-expert-parallel-comm` 1F1B 调度计划中支持 `--recompute-granularity full`（兼容 `uniform` 与 `block`）。
  - `RecomputeSegment` 在初始前向（`forward_no_grad = True`）开始前快照 `ModelChunkState` 的 **5 个原地可变字段** `_MUTABLE_CHUNK_STATE_FIELDS = ("input_ids", "position_ids", "padding_mask", "mtp_input_mask", "mtp_hidden_states")`，防止 MTP `_get_embeddings` 在反向重放时对已移位的 `mtp_input_mask` 再次循环移位导致梯度静默错位。
  - 在 `ScheduleNode._forward` 中对 `forward_no_grad` 的输出显式执行 `.detach()`，防止 `TransformerLayer` 内部在 `torch.enable_grad()` 下运行的 `bias_dropout_add` 将前向计算图钉死在段间边界张量上。

---

## 2. TransformerEngine 容量截断与编译态缓存 & TorchTitan GB300 RL 128 卡布局 (`#3347`, `#3505`, `#5581`, `#5152`)

- **TE Triton 掩码路由容量上限截断 (`#3347`) 与 `torch.compile` FP8 权重缓存 (`#3505`)**：
  - `_row_id_map_pass_2_kernel` 新增 `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)`，当路由映射分配的目标行数超出 `num_out_tokens` 容量上限时将其标记为 `-1` 丢弃，对齐 CUDA 索引映射路径并消除写穿输出缓冲区的越界风险（JAX 路径传入 `2**31 - 1` 哨兵保持无截断）。
  - `te.Linear` 在 `torch.compile` 下遇到 `is_first_microbatch=True` 时不再向函数式自定义算子传入旧缓存（`not (use_compiled_op and is_first_microbatch)`），从而在不破坏函数式不可变契约的前提下创建新 FP8 权重缓存并在后续微批次复用。
- **TorchTitan Muse Spark 816B (A42B) GB300 DAPO-Math RL 训练拓扑 (`#5581`) 与 Varlen FlashAttention (`#5152`)**：
  - 将训练器并行策略从 **256 卡 (`FSDP=64 × TP=4 × EP=4`)** 重构为 **128 卡 (`FSDP=8 × TP=4 × CP=4 × EP=16`)**：通过 `ContextParallelTransform` 将全部注意力层（含滑动窗口层）映射为 `UlyssesVarlenInnerAttention`（CP=4 切分序列长度），同时将 EP 扩大至 16（4 台 GB300 主机共处同一 NVLink 域，`scratch_capacity_factor = 16.0`），使专家权重的跨机架 FSDP All-Gather 组从 **64 Rank 缩减 8 倍至 8 Rank**。在 GB300 131K 上下文 RL 实测中，单卡峰值显存为 `186 GiB`，训练 MFU 从 `15%` 提升至 **`28%`**。
  - `VarlenInnerAttention` (`#5152`) 通过 `sdpa_kernel(self.sdpa_backends, set_priority=True)` 显式绑定后端优先级，避免回退至高显存开销的路径。

---

## 3. LMCache、Mooncake、NIXL 与 Dynamo 分布式 KV 传输面演进

- **LMCache 回调式 `LookupState` (`#5960`)、L1/L2 `MemoryTags` 过滤 (`#5922`) 与 SWA 前缀回放块对齐 (`#5976`)**：`LookupState` 通过 `MessagingFuture.add_done_callback` 在单服务器 `LOOKUP` 确认瞬间立即发送 `QUERY_PREFETCH_STATUS`（消除 1 个调度步延迟），并在全服 `RESOLVED` 后对超出 `min(hits)` 的服务器自动发送 `free_lookup_locks([min, hits))`；`#5976` 在启用 SWA 有界前缀回放（`_prefix_replay_tokens > 0`）且全提示命中（`ret == len(all_token_ids)`）时，将检索长度向下对齐至整块 `aligned = (ret - 1) // chunk * chunk` 并释放尾部锁 `(aligned, ret)`，防止检索超出 vLLM 向下取整分配的块数。
- **Mooncake 单次时钟 RDMA 投递链 (`#4793`)、TENT 通知槽位守卫 (`#4434`) 与 NVLink IPC 诊断 (`#4873`)**：`RdmaEndPoint::submitPostSend` 在每条 QP 链（最多 128 个 WR）构建前仅读取一次 `getCurrentTimeInNano()`，将 256-WR 调用的锁持有与构建耗时从 **`6.3 us` 降至 `1.9 us`**（`25.0 -> 7.5 ns/WR`，**3.32x 加速**）；`sendNotification` 仅在 `ibv_post_send` 成功后递增 `notify_send_wr_id_`，避免在 255 个未完成发送下因第 256 次投递失败而回绕覆盖在途槽位 `0`；`IntraNodeNvlinkTransport` 新增 `cuMemRetainAllocationHandle` VMM 检测与容器跨 PID 命名空间冲突诊断。
- **NIXL UCX 16B `treq` 接收端关联 (`#2487`) 与重投递遥测 (`#2478`)**：UCX 在 26B `tctx` 标签后追加 16B 小端 `treq` 标签（`requestId: u64` + `execution: u64`，合计 42B），使接收端 `nixl::notif.received` 与发送端阶段可通过 `(xfer_request_id, xfer_execution)` 精确关联；`postXferReq` 仅在进入后端 `postXfer` 前更新计时器，修复被拒绝的重投递重置在途传输计时的问题。
- **Dynamo KV 路由器 33B 低层边槽位 (`#17143`)、标量单路径遍历 (`#17149`)、多层会话驻留 (`#16501`) 与 SGLang 按需流读取 (`#16351`)**：`OwnerTable` 利用无锁 `ArcSwap<OwnerSnapshot>` 将 24B `IndexedResidencyOwner` 驻留为 4B `OwnerId(u32)`，将低层索引每条边的槽位体积从 **113 B 压缩至 33 B**（**3.42x 缩减**）；当剩余至多 1 个 Worker 与 1 个 Hint 源时切入无集合开销的 `advance_scalar`；SGLang 取消监控改为单读者任务按需读取流，消除每个流式 chunk 创建与销毁临时协程的开销。

---

## 4. 确定性复现实验验证

运行 `python experiments/megatron-gdp-prefill-a2a-recompute-te-mask-titan-rl-lmcache-mooncake-nixl-dynamo/run.py --check` 全部不变量通过（详见 `summary.json`）。
