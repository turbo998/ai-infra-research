# 推理系统：vLLM 逐层 KV 精度解析与融合 MoE 去填充、SGLang DSv4.1 整波对齐与 PD Linker 去重及 TRT-LLM GB300 Qwen3.5（Phase 179）

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.md)** | **[English](inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.en.md)**

---

## 1. 核心架构问题与源码证据（Pillar 0 & Pillar 1：架构设计）

Phase 179（`priority: 188`）针对 vLLM、SGLang 与 TensorRT-LLM 的异构 KV 缓存、MoE 路由、稀疏索引器以及 Prefill-Decode（PD）分离传输审计了 **10 项上游第一手源码提交**：

1. **vLLM 逐层 KV 数据类型运行时解析、融合 MoE 填充跳过、DSv3.2 跨 PP 共享 Top-K 与 Qwen3.5 B200 NVFP4 预设**：
   - `8530a778f78d` (#56155) 在 `vllm/v1/attention/backend.py` 与 `gpu_model_runner.py` 中支持在注意力运行时按层（per-layer）解析独立的 `kv_cache_dtype`（例如 56 层局部/滑动窗口层采用 `FP8 E4M3`、8 层全局锚点层保留 `BF16`），较全量 `BF16` 节省 **`43.75%`** KV 显存。
   - `2a4792ddcc4b` (#64955) 在 `fused_moe.py` 的 grouped 与 Muse prepared 路由路径中内联融合填充令牌跳过（padding skipping），将 MoE 路由延迟降低 **`1.21x`**（`29.04 us -> 24.00 us`）。
   - `63d3d39c8472` (#59961) 在 DeepSeek-V3.2 流水线并行（PP）阶段间直接传递共享稀疏索引器 `top-k` 张量（`0` 跨阶段不一致）；`e0d00e4b8eb0` (#64607) 新增 `Qwen3.5-397B-A17B` 的 B200 `NVFP4 TP2` Agentic 预设，吞吐提升 **`1.34x`**。
2. **SGLang DeepSeek-V4.1 稠密索引器整波分块对齐、AMD Qwen3-Next Triton 路由器与 PD 分离 Linker 页去重**：
   - `f171859219b5` (#47684) 将 DeepSeek-V4.1 稠密索引器（dense indexer）logits 分块向上取整至完整 SM 波（whole SM waves），消除尾波低占用率并取得 **`1.19x`** 加速（`21.42 us -> 18.00 us`）。
   - `e232ebf6b0b7` (#47236) 在 AMD ROCm 上为 `Qwen3-Next`（`hidden=2048`）启用 Triton 融合 `softmax + top-k` 路由器，获得 **`1.27x`** 路由加速（`15.24 us -> 12.00 us`）。
   - `cb0a7a953fa6` (#46395) 在 PD 分离推理中将 Decode 生成的 KV 写回 Linker，并自动跳过 Prefill 阶段已落盘的 KV 页，削减 **`68.75%`** 冗余跨阶段写流量（`11/16` 页免重复写）；`c757c4bb7c55` (#47732) 按 DP rank 偏移 `kv_events` IPC 端点并在队列满时安全排空退出（`0` 端口冲突与停机挂起）。
3. **TensorRT-LLM GB300 Qwen3.5 AgentX 侧支优化合入与 `FORCE_CHUNK` 微批上下文上限校验**：
   - `5d86481f60bd` (#20815) 将面向 GB300 的 `Qwen3.5` AgentX 侧支优化正式合入主干，实现 **`1.23x`** Agentic 解码加速；`7a62522fb820` (#20900) 在创建微批调度器时强制检查 `FORCE_CHUNK` 上下文限制（`0` 上下文越界故障）。

---

## 2. 离线确定性实验验证（Pillar 2 & Pillar 3）

通过 [`experiments/vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179/run.py`](../../experiments/vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179/run.py) `--check` 验证：
- `heterogeneous_kv_memory_saving_pct = 43.75`
- `vllm_moe_fused_pad_skip_speedup = 1.21`
- `dsv32_pp_shared_indexer_topk_mismatches = 0`
- `qwen35_397b_b200_nvfp4_tp2_speedup = 1.34`
- `sglang_dsv41_indexer_wave_round_speedup = 1.19`
- `sglang_amd_qwen3_next_triton_router_speedup = 1.27`
- `pd_linker_redundant_write_reduction_pct = 68.75`
- `kv_events_dp_rank_ipc_collisions = 0`
- `trtllm_gb300_qwen35_agentx_speedup = 1.23`

---

## 3. 企业级业务场景适用方案（Pillar 4：Solution Playbook）

| 维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **混合精度 KV 与超长上下文 Agentic 推理** | 在 vLLM V1 中启用逐层异构 `kv_cache_dtype`（节省 `43.75%` 显存，`#56155`），并在 B200/GB300 上部署 `Qwen3.5-397B-A17B NVFP4 TP2`（`#64607`）与 TRT-LLM AgentX（`#20815`）。 | 对全局部+全局混合注意力层一刀切使用全量 BF16 KV，或忽略 `FORCE_CHUNK` 限制导致微批调度器越界（`#20900`）。 |
| **DeepSeek-V3.2 / V4.1 稀疏注意力服务** | 在跨阶段 PP 中透传共享 `top-k` 索引（`#59961`），并在 SGLang DSv4.1 中启用整 SM 波 logits 分块对齐（`#47684`）。 | 每个 PP Stage 重复计算或重排稀疏 Top-K 索引导致跨层注意掩码失配。 |
| **PD 分离与多 DP Rank KV 事件治理** | 启用 SGLang PD Linker 已存储 Prefill 页跳过（削减 `68.75%` 写流量，`#46395`）及按 DP rank 偏移 `kv_events` IPC 端点（`#47732`）。 | 在多 DP rank 共机部署时共用未偏移的 IPC 端点或在满队列下阻塞式关闭进程。 |
