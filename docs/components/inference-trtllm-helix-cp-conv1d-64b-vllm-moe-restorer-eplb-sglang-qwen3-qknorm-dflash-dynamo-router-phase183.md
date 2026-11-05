# 推理服务与路由系统：TensorRT-LLM 单 Rank CP 旁路与 64 位因果卷积偏移、vLLM EPLB 感知专家权重过滤与零重解析批处理输出、SGLang Qwen3 融合 QK-Norm+RoPE 与 DFlash 批量解码准备、ROCm AITER 3D Split-KV 掩码及 Dynamo 预备路由行保留（Phase 183）

> 🌐 **Language / 语言切换**: **[中文](inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.md)** | **[English](inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 183（`priority: 192`）审计了 TensorRT-LLM、vLLM、SGLang、ROCm AITER 与 NVIDIA Dynamo 的 **10 个上游主干提交**：

1. **ROCm AITER 3D Split-KV 尾行统计掩码与 TensorRT-LLM 单 Rank CP 旁路 / 64 位因果卷积偏移**：
   - `656e98e68b49` (#7472) 修复 AITER Triton `unified_attention` 3D Split-KV 解码内核在 `BLOCK_Q * num_queries_per_kv < BLOCK_M`（如 `GQA=5, BLOCK_Q=3, BLOCK_M=16` 时的第 16 行尾行）未对 `segm_max` 与 `segm_expsum` 施加 `query_mask_0` 的竞态缺陷：消除了有效分段写 `(M=0, L=0)` 与越界分段写 `(M=-inf, L=1)` 并发覆盖引发的 `~1e-3` 跨轮次非确定性漂移（修复后跨轮漂移为 **`0.0`**）。
   - `1399e996099f` (#21192) 在 TensorRT-LLM PyTorch 执行器中当 `context_parallel_size == 1` 时忽略 `cp_config["cp_type"]`（避免默认 `ULYSSES` 或 `RING` 触发 `NotImplementedError` 崩溃），并在 `cp_size > 1` 时强制要求 `HELIX`；`eaa88a0fed55` (#21684) 将 `causal_conv1d` 前向与更新内核的 `conv_state` 槽位偏移计算提升为 64 位整数（`int64_t` / `size_t`），消除 $> 2^{32}$ 元素混合 Mamba/卷积状态池上的 32 位有符号溢出。
2. **vLLM EPLB 感知专家并行权重过滤、共享路由专家行恢复器与零重解析 `run-batch` JSON 写出**：
   - `2f8c840032c6` (#64187) 使 `--enable-ep-weight-filter` 与专家并行负载均衡器（`--enable-eplb`）协同工作：按 EPLB 初始物理槽位映射（含冗余专家副本）仅读取本 Rank 所需的专家权重，在 DeepSeek-R1-0528-NVFP4-v2（`DEP=8` + `32` 冗余专家）上将每 Rank 权重读取量从 **`384.9 GiB` 降至 `105.2 GiB`（缩减 `3.659x` / `-72.67%`）**，热页缓存加载耗时从 `43.6 s` 降至 `39.8 s`，GSM8K 精度保持一致（`0.9636` vs `0.9621`，97.4% 逐题一致）。
   - `463e23b1dee0` (#64689) 在 `routed_experts_capturer.py` 中允许 MoE 后端绑定共享的 `restore_routed_experts_rows(topk_ids, ctx)` 方法，使 vLLM-Ascend 等后端无需替换 `capture()` 即可恢复微批次（microbatch）与 DeepSeek-V4.1 重放批次的专家路由行布局。
   - `14d147a7a05d` (#65867) 消除 `vllm run-batch` 对 `/v1/embeddings`、`/score`、`/rerank` 缓冲 JSON 响应的二次反序列化与重序列化，将单行输出事件循环耗时在 128 维向量上从 `85.3 us` 降至 **`17.9 us`（`4.765x`）**、在 512 维向量上从 `188.1 us` 降至 **`29.4 us`（`6.398x`）**，在单张 H100 处理 `864,891` 条短文本时将端到端挂钟时间缩短 **`12.5%`**（128 维）与 **`18.0%`**（Qwen3-Embedding-0.6B，512 维）。
3. **SGLang Qwen3 连续 Q 融合预填充 QK-Norm+RoPE、DFlash 批量解码准备与 AITER `page_size=1` 图捕获修复**：
   - `c07ec416dabf` (#47785) 为 Qwen3 预填充融合 QK RMSNorm 与 RoPE 并直接写出连续布局 Q 张量；`e32f88502e8c` (#47678) 将 `DFlashDraftInputV2.prepare_for_decode` 中的逐请求 Python `__setitem__` 循环替换为向量化批量张量填充，在 DeepSeek-V4.1-Flash GB300 分离式 `DEP=16`（每 Rank `bs=40`）上削减约 **`0.4 ms`**（占 `1.0–1.4 ms` 预协商宿主路径的 30%+）每步解码准备开销；`89b264138157` (#46928) 修复 AMD AITER `unified_attention` 在默认 `page_size=1` 与 CUDA/HIP 图捕获下未拷贝页表导致的乱码输出（GSM8K 从 `0.00` 恢复至正常基线）。
4. **NVIDIA Dynamo 阈值饱和下保留预备路由行与混合 Conductor LPX Sidecar 强制门控**：
   - `4f7153790d4e` (#18009) 优化 Dynamo 调度器 Actor：当部分 Rank 超过繁忙拒绝阈值（`active_decode_blocks_threshold` / `active_prefill_tokens_threshold`）时，不再丢弃调用方预构建的候选行并全量重扫，而是保留预备行并仅剔除所有 Rank 均满载的 Worker 行。在 **`2,048` 个 Worker** 的 `agentx` 负载下，饱和吞吐从 **`7,172 req/s` 提升至 `10,013 req/s`（`1.396x` / `+39.6%`）**，每次调度 Actor CPU 周期从 `437` 降至 **`314 kcycles`（`-28.15%`）**，在固定 `4.6k req/s` 下的准入 p99 延迟从 **`2,764 ms` 骤降至 `2.1 ms`（提速 `1,316.2x`）**；`7e9dc291b12a` (#18069) 在 Operator 调谐阶段强制校验混合 LPX Conductor 必须配置 `runtime` LPX Sidecar。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+--------------------------------------------------------------------------------------------------------+
| Phase 183 LLM Inference & Serving Stack: TRT-LLM, vLLM, SGLang, ROCm AITER & NVIDIA Dynamo (2,048 Wkr) |
+--------------------------------------------------------------------------------------------------------+
  [AITER 3D Split-KV Mask + TensorRT-LLM CP Gate & 64-Bit Conv1d (#7472, #21192, #21684)]
    |-- AITER 3D Split-KV (#7472)    --> Mask Leftover Rows (3*5=15 < 16) in segm_max/expsum (0.0 Drift)
    |-- TRT-LLM CP Gate (#21192)     --> Bypass cp_type at cp_size=1 + Enforce HELIX for cp_size>1
    +-- TRT-LLM Conv1d 64b (#21684)  --> 64-Bit int64_t Slot Offsets for >2^32 Element Hybrid State Pools
                                      |
                                      v
  [vLLM EPLB Weight Filter, Shared MoE Restorer & Zero-Reparse run-batch (#64187, #64689, #65867)]
    |-- EPLB Weight Filter (#64187)  --> DeepSeek-R1 DEP8+32 Redundant: 384.9 -> 105.2 GiB/Rank (-72.67%)
    |-- Shared MoE Restorer (#64689) --> Unified restore_routed_experts_rows Across Microbatch/Replay
    +-- Direct run-batch JSON (#65867)--> 6.398x Event-Loop Speedup (512d) & -18.0% H100 Batch Wall Time
                                      |
                                      v
  [SGLang Qwen3 QK-Norm+RoPE, Bulk DFlash Prep & Dynamo 2,048-Worker Router (#47785, #47678, #46928, #18009)]
    |-- SGLang Qwen3 & DFlash (#47678)--> Fused Contiguous QK-Norm+RoPE + -0.4 ms/step GB300 DEP16 Host Prep
    |-- SGLang AITER Graph (#46928)  --> Copy cuda_graph_page_table at page_size=1 (GSM8K 0.00 -> Full)
    +-- Dynamo Retained Rows (#18009)--> 10,013 vs 7,172 req/s (+39.6%) & Admission p99 2,764 -> 2.1 ms (1316x)
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/trtllm_helix_cp_conv1d_64b_vllm_moe_restorer_eplb_sglang_qwen3_qknorm_dflash_dynamo_router_phase183/run.py`](../../experiments/trtllm_helix_cp_conv1d_64b_vllm_moe_restorer_eplb_sglang_qwen3_qknorm_dflash_dynamo_router_phase183/run.py) `--check` 100% 验证通过：
- `aiter_split_kv_valid_rows = 15`、`aiter_split_kv_block_m = 16`、`aiter_split_kv_masked_drift = 0.0`
- `trtllm_cp_single_rank_bypass_size = 1`、`trtllm_conv1d_offset_bits = 64`
- `vllm_eplb_weight_load_reduction_ratio = 3.659`（`384.9 -> 105.2 GiB/rank`，节省 **`72.67%`**）
- `vllm_run_batch_128d_speedup = 4.765`、`vllm_run_batch_512d_speedup = 6.398`、`vllm_run_batch_qwen3_emb_walltime_saving_pct = 18.0`
- `sglang_dflash_prep_saved_ms = 0.4`（`DEP=16, bs=40`）、`sglang_aiter_cudagraph_fixed_page_size = 1`
- `dynamo_router_workers = 2048`、`dynamo_router_throughput_speedup = 1.396`（`7,172 -> 10,013 req/s`）、`dynamo_router_kcycles_reduction_pct = 28.15`、`dynamo_router_p99_speedup = 1316.2`（`2,764.0 -> 2.1 ms`）

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **千节点级（2,048 Workers）Agentic 在线推理集群调度** | 在启用 `active_decode_blocks_threshold` / `active_prefill_tokens_threshold` 的 Dynamo 生产集群上启用预备候选行保留机制（`#18009`），将饱和吞吐提升 `39.6%` 并将准入 p99 从 `2.76 s` 压至 `2.1 ms`（`1,316x`）。 | 在任意单 Rank 触及繁忙阈值时即丢弃调用方预构建候选行并在 Actor 上重扫全集群（会在高并发下造成严重的调度队列阻塞）。 |
| **DeepSeek-R1 / V4.1 大规模 EP + EPLB 部署与 GB300 DFlash 推测解码** | 在 vLLM `DEP=8` + EPLB 部署中同时开启 `--enable-ep-weight-filter`（`#64187`，每 Rank 读取从 `384.9 GiB` 降至 `105.2 GiB`），并在 SGLang GB300 `DEP=16` DFlash 解码中使用批量长度填充（`#47678`，省 `0.4 ms/step`）。 | 在开启 EPLB 时关闭 EP 权重过滤（导致每 Rank 冗余加载全部 256+ 专家权重，拉长冷启动并挤占主机页缓存）。 |
| **高吞吐 Embedding 批处理与 AMD AITER 统一注意力服务** | 在 vLLM `run-batch` 向量批处理中启用零重解析 JSON 写出（`#65867`，Qwen3-Embedding-0.6B 端到端提速 `18.0%`），在 AITER `unified_attention` 中启用 3D Split-KV 尾行掩码（`#7472`）与 `page_size=1` 图页表拷贝（`#46928`）。 | 在 `page_size=1` 且开启 HIP/CUDA Graph 时跳过 `cuda_graph_page_table` 同步（会导致解码阶段读取全零页表并输出乱码）。 |
