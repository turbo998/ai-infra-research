# SGLang FlashAttention 统一内核派发、DeepSeek-V4.1 Engram 环境恢复、vLLM PCP/MLA-DCP 投机解码修复、KVCR 卸载减负与 TensorRT-LLM Kimi-K3 Helix-A2A（Phase 118 / Priority 127）

> **证据分类**：Upstream `implementation`（10 个上游公开提交，2026-10-27 至 2026-10-28）+ 本仓库 `own-offline` 确定性调度/显存建模（`experiments/sglang_fa_dispatch_dsv41_vllm_pcp_mla_dcp_kvcr_trtllm_helix_a2a_phase118/run.py`）。本文不包含未标注的真实 GPU 硬件重测声明。

## 1. 核心架构问题与背景

在超大规模 MoE 与多模态实时生成推理栈（SGLang、vLLM V1、TensorRT-LLM PyTorch Backend）中，近期修复并演化了五项关键运行时机制：

1. **SGLang FlashAttention 静态导入绕过内核选择器与 Engram 异常环境污染**：`flashattention_backend.py` 此前在模块顶层直接静态导入 `flash_attn_varlen_func` 与 `flash_attn_with_kvcache`，导致部分调用分支绕过了运行时选定的架构专属内核（如 `SM120` FA4）；同时 `environ.py` 的 `override()` 上下文管理器未使用 `try ... finally`，一旦内部抛出异常便会永久污染进程环境变量。
2. **SGLang 实时视频/扩散模型初始化槽位泄漏与冷启动 Hub API 阻塞**：`realtime_video_api.py` 在会话初始化完成前过早预留生成槽位，初始化异常时槽位无法释放；`weights/source.py` 在本地已缓存命名权重时仍发起 HuggingFace Hub API 请求。
3. **vLLM V1 投机解码在 PCP 与 MLA DCP 下的分片错位**：
   - 当开启 Prefill Context Parallelism (`pcp_size > 1`) 时，`speculator.py` 若启用 `batch_sharded_sampling`，按 `req_state_idx // draft_logits_num_shards` 索引的 Draft Logits 行会与 PCP 跨 Rank 请求切分冲突；
   - 在 Decode Context Parallelism (`DCP`) 下，当投机解码 Batch 含有非均匀 Query 长度（如 `query_lens = (3, 1)`）时，`flashinfer_mla.py` 未按每个拆分行（Split Row）传入后续因果 Query 行数 `causal_rows_after`，导致非均匀投机验证掩码失真。
4. **vLLM KVCR 分层卸载容器分配开销、ROCm Kimi-K3/Qwen3.8-Flash-Next 适配与 TensorRT-LLM Kimi-K3 `helix_a2a`**：
   - `kvcr/manager.py` 在每次 `create_store_job` 热路径上构造多余的 `tuple(int(c) for c in job.chunk_ids)` 与 `set` 比较；
   - TensorRT-LLM 在 Kimi-K3 开启 Helix Context Parallelism + Attention DP 时，需要通过 `helix_a2a` 对 MoE 输入 Token 进行跨 Rank 切分，避免每个 CP Rank 重复计算全量 Token 的专家前传，并在 TP16 下妥善处理 12 头视觉塔的整除性失配。

---

## 2. 上游一手实现剖析（10 个零重叠 Commit）

| # | 仓库与提交 | 日期 | 核心实现路径与调度机制 |
|---|---|---|---|
| 1 | `sgl-project/sglang@586355a83b` (`#46695`) | 2026-10-28 | `python/sglang/srt/layers/attention/flashattention_backend.py` & `flash_attention_v4_sm120.py`：移除模块级 `flash_attn_varlen_func`/`flash_attn_with_kvcache` 静态导入，所有注意力调用统一路由至选定内核，为 SM120 FA4 补齐 `mask_mod` 参数，并新增 AST 静态检查单测防止回退。 |
| 2 | `sgl-project/sglang@655059819c` (`#46896`) | 2026-10-27 | `python/sglang/srt/environ.py` & `docs/docs/advanced_features/engram.mdx`：将 `override()` 恢复逻辑包裹在 `try ... finally` 中确保异常下环境变量零泄漏，并将 DeepSeek-V4.1-Flash Engram `peer_read` 文档参数更新为 `--attn-dp-size 8`。 |
| 3 | `sgl-project/sglang@9694a99d55` (`#46252`) | 2026-10-27 | `python/sglang/multimodal_gen/runtime/entrypoints/openai/realtime/realtime_video_api.py`：将生成槽位预留推迟至会话初始化成功之后，并加固 `_cleanup_realtime_session` 的异常释放路径。 |
| 4 | `sgl-project/sglang@5884e4fec6` (`#46679`) | 2026-10-27 | `python/sglang/multimodal_gen/runtime/weights/source.py`：优先通过 `snapshot_download(..., local_files_only=True)` 解析本地已缓存的命名权重，仅在捕获 `LocalEntryNotFoundError` 时才回退至远端下载，消除热缓存启动时的 Hub API 网络依赖。 |
| 5 | `vllm-project/vllm@820be160ce` (`#64151`) | 2026-10-28 | `vllm/v1/worker/gpu/spec_decode/speculator.py`：在 `_init_draft_logits` 中检查 `parallel_config.pcp_size == 1`，在开启 PCP 时显式禁用 Batch-Sharded Draft Logits。 |
| 6 | `vllm-project/vllm@8697bbae4c` (`#59392`) | 2026-10-27 | `vllm/v1/attention/backends/mla/flashinfer_mla.py`：为 FlashInfer MLA 原生 DCP 解码计算并传入每拆分行的后续因果 Query 行数 `causal_rows_after`，修复非均匀投机解码批次（如 `query_lens=(3, 1)`）下的精度错误。 |
| 7 | `vllm-project/vllm@36f5ff3172` (`#63889`) | 2026-10-27 | `vllm/v1/kv_offload/tiering/kvcr/manager.py`：移除 `_resolve_pin` 热路径中的 `tuple(int(...) ...)` 生成器遍历与双 `set()` 构造开销，同时保持 `MemoryRef.element_index` 为原生 `int`。 |
| 8 | `vllm-project/vllm@bb7ea479c6` (`#63409`) | 2026-10-27 | `vllm/models/kimi_k3/amd/mla.py`：在 ROCm Kimi-K3 融合 MLA 解码 `q-prep` 内核调用中显式传入全零位置张量（Zero Positions），修复无位置偏移解码步下的指针/未初始化访问。 |
| 9 | `vllm-project/vllm@5314c58281` (`#63332`) | 2026-10-27 | `recipes/models/Qwen/Qwen3.8-Flash-Next.yaml`：新增 AMD Instinct MI355X 上的 Qwen3.8-Flash-Next（125B 主模型 + 51B N-gram 嵌入，每 Token 激活 6B 参数）Quark MXFP4（TP1/TP2）部署配方。 |
| 10 | `NVIDIA/TensorRT-LLM@8786940e8b` (`#20597`) | 2026-10-28 | `tensorrt_llm/_torch/models/modeling_kimi_linear.py` & `modeling_kimi_k3_vl.py`：实现 Kimi-K3 在 Helix CP + Attention DP 下的 `helix_a2a` Token 切分 MoE（借助 `_helix_cp_pad` 对齐并通过 All-to-All 将每 Rank 专家计算量降低 $1/\text{CP}$），并在视觉塔头数（12）无法被 TP（如 16）整除时自动以 `tp=1` 副本模式运行。 |

---

## 3. 关键公式、数据布局与离线验证结果

### 3.1 TensorRT-LLM `helix_a2a` Token-Sliced MoE 与 MLA DCP Ragged 因果偏移
1. **Helix CP 下 Token-Sliced MoE 计算复杂度**：设总 Token 数为 $T$，Helix Context Parallelism 度为 $C_{\text{helix}}$。未切分时每个 CP Rank 需对全部 $T$ 个 Token 执行 MoE 路由与专家 GEMM；启用 `helix_a2a` 后，先按 `_helix_cp_pad` 补齐并通过 All-to-All 切分，单 Rank 仅处理：
   $$T_{\text{rank}} = \left\lceil \frac{T}{C_{\text{helix}}} \right\rceil$$
   当 $T = 4096, C_{\text{helix}} = 8$ 时，$T_{\text{rank}} = 512$，MoE 算力开销缩减 **$8\times$**（`helix_a2a_moe_compute_reduction_factor = 8`）。
2. **MLA DCP Ragged Spec-Decode `causal_rows_after`**：对于请求 $i$ 的查询长度 $q_i$，其第 $j \in [0, q_i - 1]$ 个查询行之后同属该请求的查询行数为：
   $$\text{causal\_rows\_after}(i, j) = q_i - 1 - j$$
   对于 `query_lens = (3, 1)`，拼接后的向量严格为 `[2, 1, 0, 0]`，确保 DCP 各 KV 分片在合并部分注意力状态时精确裁剪未来位置的 KV 贡献。

### 3.2 离线模型验证汇总（`run.py --check`）

- `fa_bypassed_calls`: `0`（AST 静态校验确保 100% FlashAttention 调用经由选定内核派发）
- `env_override_leaks_on_exception`: `0`（`try/finally` 确保异常中断后环境变量 100% 恢复）
- `leaked_realtime_slots_on_init_failure`: `0`（初始化后再预留生成槽位）
- `batch_sharded_draft_logits_under_pcp`: `false`（`pcp_size > 1` 时自动关闭分片 Draft Logits）
- `mla_dcp_ragged_causal_rows_after`: `[2, 1, 0, 0]`（精确匹配 `query_lens=(3, 1)` 的每行因果尾长）
- `kvcr_containers_saved_per_job`: `3`（每次 KVCR 卸载任务省去 3 个临时元组/集合容器分配）
- `qwen38_flash_next_mxfp4_main_gib`: `65.48`（125B 主模型在 Quark MXFP4 下仅占 `65.48 GiB`，相比 BF16 压缩 `3.5557x`，单卡 288 GiB MI355X TP1 即可轻松容纳 125B 主权重 + 51B N-gram 嵌入）
- `helix_a2a_moe_compute_reduction_factor`: `8`（`CP=8` 下每 Rank 仅处理 `512` tokens）
- `kimi_k3_vision_effective_tp_at_tp16`: `1`（12 头视觉塔在 TP16 下自动回退为 `tp=1` 副本模式，避免非整除崩溃）

---

## 4. 面向云解决方案架构师的练习与可证伪问题

1. **容量计算练习**：Qwen3.8-Flash-Next 拥有 125B 主模型参数与 51B N-gram 嵌入参数（每 Token 仅激活 6B 参数）。若主模型采用 Quark MXFP4（每 32 个 FP4 权重共享 1 个 E8M0 8-bit 缩放因子，平均每参数 `0.5625 B`），51B N-gram 嵌入采用 FP8（`1.0 B`/参数），计算在单张 AMD MI355X（288 GiB HBM3e）上采用 TP1 部署时的静态权重+嵌入总显存占用，以及剩余可用于 KV Cache 的显存空间。（答案：主权重 $125 \times 10^9 \times 0.5625 / 2^{30} \approx 65.48\text{ GiB}$，N-gram 嵌入 $51 \times 10^9 \times 1.0 / 2^{30} \approx 47.50\text{ GiB}$，合计约 $112.98\text{ GiB}$，单张 MI355X 仍余约 $175.02\text{ GiB}$ 供长上下文 KV Cache 与运行时缓冲区使用）。
2. **可证伪假设**：在 Kimi-K3 TP16 部署下，将 12 头视觉编码器复制为 `tp=1` 是否会造成严重的显存或吞吐瓶颈？不会：视觉塔参数量通常不足主语言模型的 2%，复制开销可忽略，且在 Attention-DP 模式下不同 Rank 本就并行处理不同图像的视觉 Token，`tp=1` 反而消除了跨 16 卡的视觉层 All-Reduce 通信。
