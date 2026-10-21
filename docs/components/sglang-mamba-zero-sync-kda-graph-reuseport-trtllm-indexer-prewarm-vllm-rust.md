# SGLang Mamba 零同步规划 / KDA Prefill 图捕获 / ReusePort、TensorRT-LLM DeepSeek-V4 Indexer 预热与 vLLM Cosmos3-Edge / Rust 前端对齐

> **关联上游提交（一手源码）**：
> - [`sgl-project/sglang@740d5f9`](https://github.com/sgl-project/sglang/commit/740d5f9f9e33aeebc6b5bcb1ba73f25bf6db062b) (`#45472`, 2026-10-21) — *perf: eliminate host-device syncs in hybrid Mamba extend batch planning*
> - [`sgl-project/sglang@f06d1bb`](https://github.com/sgl-project/sglang/commit/f06d1bb1f45df617aa455c103ce5ede2bc6c6a47) (`#45473`, 2026-10-21) — *feat: support FlashInfer KDA prefill CUDA graph*
> - [`sgl-project/sglang@a3d00ef`](https://github.com/sgl-project/sglang/commit/a3d00eff58b6c22ea7539f882cc10c537634200b) (`#45487`, 2026-10-21) — *perf: compute `contains_last_prefill_chunk` on host and preserve in TBO `filter_batch`*
> - [`sgl-project/sglang@7e5256d`](https://github.com/sgl-project/sglang/commit/7e5256d1ba3aaa628bb8b756fafb904a76a1123a) (`#45482`, 2026-10-21) — *perf: add `--enable-http-worker-reuseport` for multi-worker HTTP server*
> - [`NVIDIA/TensorRT-LLM@370fe4c`](https://github.com/NVIDIA/TensorRT-LLM/commit/370fe4c5096cff7ffb457092d36a208da34006d7) (`#20734`, 2026-10-21) — *fix(deepseek_v4): fix `_prewarm_cute_dsl_indexer_q` dead import and cover all fallback tactics*
> - [`vllm-project/vllm@4a3ee9a`](https://github.com/vllm-project/vllm/commit/4a3ee9a0f6a2958f2712ceb3c9f60fe48cd764b3) (`#57393`, 2026-10-21) — *[Model] Support Cosmos3-Edge ModelOpt quantized checkpoints*
> - [`vllm-project/vllm@2916b4b`](https://github.com/vllm-project/vllm/commit/2916b4b4a6b7e9ae552e661b365fec65117134ac) (`#62822`), [`bc4794e`](https://github.com/vllm-project/vllm/commit/bc4794ef9a9fea5f8a003ef4dc5a49bbe8145241) (`#62825`), [`8a21d0a`](https://github.com/vllm-project/vllm/commit/8a21d0a328bb3cbb9c73ca0ae65d83465768cca4) (`#62416`), [`e0ca7d9`](https://github.com/vllm-project/vllm/commit/e0ca7d9cc11da2011389c725e5d31c54c3b47259) (`#62824`, 2026-10-21) — *[Rust Frontend] Render `return_token_ids`, env vars, CLI flags, and `PromptFillsContext` parity*

---

## 1. SGLang 混合 Mamba 零同步规划与 KDA 图捕获 / ReusePort

### 1.1 消除 Hybrid Mamba Extend 批次中的隐式 Host-Device 同步（`#45472`）
- **Host 侧 `mamba_ping_pong_empty: list[bool]` 镜像**：此前验证未分配槽位时调用 `(ping_pong_mapping[req_indices] == -1).any().item()` 会触发 CUDA 流同步；现通过 Host 布尔数组校验，并通过 Device 端二维索引 `ping_pong_mapping[req_pool_indices, track_positions]` 直接收集槽位，`clear_slots` 改用 `index_fill_(1, idx64, 0)`。
- **消除 `torch.isin` 在多模态掩码中的 `unique()` 同步**：PyTorch `torch.isin(elements, test_elements)` 在满足 $P \ge 10 \cdot N^{0.145}$ 时（例如 $N=4096$ 时 $P \ge 34$）会走排序 + `torch.unique()` 分支，因输出形状依赖数据而强制触发 Device-to-Host 同步。`#45472` 将 `_get_multimodal_mask` 替换为排序后 `torch.searchsorted(sorted_p, input_ids)` + `clamp_(max=P-1)` + 逐元素相等比较，彻底实现零同步。

### 1.2 KDA Prefill CUDA Graph、`contains_last_prefill_chunk` 与 ReusePort（`#45473` / `#45487` / `#45482`）
- **`FlashInferKDAPrefillGraph`（`#45473`）**：在 `KDAPrefillGraphMetadata` 中封装 SM100/SM103 `RecurrentKDAPrefillWrapper`，使混合线性注意力的 Prefill 阶段可完整纳入分段 CUDA Graph 捕获与重放。
- **Host 侧 `contains_last_prefill_chunk` 与 TBO 传递（`#45487`）**：在 `ForwardBatch` 初始化时通过 `any(req.extend_end == len(req.full_untruncated_fill_ids) for req in extend_reqs)` 直接在 CPU 计算，并在 Two-Batch Overlap (TBO) `filter_batch` 切分微批次时按 `keep_indices` 重新计算。
- **`_ReusePortMultiprocess`（`#45482`）**：新增 `--enable-http-worker-reuseport`（要求 `uvicorn>=0.51`），每个 HTTP Worker 独立绑定 `SO_REUSEPORT` 套接字由 Linux 内核负载均衡 `accept()`，同时父进程保留第 1 个监听套接字以防止外部第二个服务进程静默抢占同一端口组。

---

## 2. TensorRT-LLM DeepSeek-V4 CuTe DSL Indexer Q 三分支预热修复（`#20734`）

`TensorRT-LLM@370fe4c` 修复了 DeepSeek-V4 稀疏注意力索引器预热函数 `_prewarm_cute_dsl_indexer_q` 的三项缺陷：
1. **死导入修复**：将失效的导入路径修正为 `from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4Indexer`，此前 `ImportError` 被 `try/except` 静默吞掉导致预热从未执行；
2. **PP 阶段过滤**：仅筛选当前 Pipeline Parallel 阶段持有 `wq_b.weight is not None` 且启用 `_is_fused_project_mxfp4_enabled(torch.bfloat16)` 的本地层；
3. **全覆盖 `_fallback_tactic()` 三分支**：将预热 Token 数设为 `(4, 8, 16)`，分别命中 `m <= 4`、`4 < m <= 8` 与 `m > 8` 三个 CuTe DSL JIT 编译分支。

---

## 3. vLLM Cosmos3-Edge ModelOpt 权重清洗与 Rust 前端对齐

- **Cosmos3-Edge ModelOpt 伪量化器剥离（`#57393`）**：在加载包含 ModelOpt 校准状态（`.input_quantizer.`、`.weight_quantizer.`、`.output_quantizer.`）的 Checkpoint 时直接丢弃伪量化缓冲区，并将 `.weight_scale_inv` 与 `.act_quant_scale` 分别重映射为 `.weight_scale` 与 `.input_scale`。
- **Rust 前端精确对齐（`#62822` / `#62825` / `#62416` / `#62824`）**：
  - 使用 `prompt_len.cmp(&max_model_len)` 区分 `Ordering::Equal`（`PromptFillsContext`，当 `max_tokens >= 1` 时拒绝并返回 `param="input_tokens"`）与 `Ordering::Greater`（`PromptTooLong`）；
  - `/v1/chat/completions/render` 支持 `return_token_ids`；对齐 `VLLM_LOG_STATS_INTERVAL`、`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` 及 `--tokenizer` / `--tokenizer-revision` / `--hf-config-path`。

---

## 4. 确定性复现实验与验证

```bash
python experiments/sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust/run.py --check
```
