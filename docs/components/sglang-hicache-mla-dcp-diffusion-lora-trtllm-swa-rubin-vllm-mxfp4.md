# SGLang HiCache MLA DCP L3 去重写与稀疏尾页位图投票、Diffusion 动态 LoRA 守卫、TensorRT-LLM SWA 分支快照与 Rubin SM107、vLLM Humming MXFP4 激活覆盖与 Rust 协议对齐

> **Phase 56 (`priority: 65`) · 一手源码与 Commit 级深挖**
> - `sgl-project/sglang@c677171c1df08ef20da92c8ba6be12507c92a248` (`#41187`)
> - `sgl-project/sglang@02b983603dc743d2d37eb34410d7e3cb0b238a8c` (`#45943`)
> - `sgl-project/sglang@0e3496288488ff4c70aa22d0b44643f5d1ca2b86` (`#45966`)
> - `NVIDIA/TensorRT-LLM@ae1aa43964947a272ef170933b3b9f5194a5955d` (`#19877`)
> - `NVIDIA/TensorRT-LLM@b0af321b0c9372fa6e80501c51dc9d5b6da380f2` (`#20334`)
> - `NVIDIA/TensorRT-LLM@854ccee8a944a7597149a96421d92f9dd5496579` (`#20167`)
> - `NVIDIA/TensorRT-LLM@448321fb98261098a3eb5bd319f896959dc7b676` (`#20605`)
> - `NVIDIA/TensorRT-LLM@d7ec5162ac8d935b21bb5eac719e5eac99664cec` (`#20242`)
> - `vllm-project/vllm@97b6a67d94f7078221dcc9839141a0e2b5ff9a20` (`#62751`)
> - `vllm-project/vllm@7b1d07440dde8426a8b3d85e5f5815c06913eb82` (`#63295`)
> - `vllm-project/vllm@4a5ee8fc77519a5fc1c68700809ac972277d8f6a` (`#63316`)
> - `vllm-project/vllm@933f7962aeb81aa581eadb24dedd746fab20b2dd` (`#63317`)

---

## 1. SGLang HiCache MLA + Decode Context Parallelism (`dcp_size > 1`) L3 存储与 Diffusion 动态 LoRA (`#41187`, `#45943`, `#45966`)

- **HiCache MLA DCP L3 页映射与写入去重 (`#41187`)**：
  - 在 MLA 模型启用 Decode Context Parallelism (`dcp_size > 1`) 时，每个请求的第 `k` 个逻辑页跨越 `logical_page_size = page_size * dcp_size` 个 token，但在每个 DCP 分片上仅占用 1 个物理存储页，因此主机与 L3 (`file` / `mooncake`) 索引按 `_storage_page_index(index) = index // dcp_size` 映射。
  - **跨 TP 写入去重**：由于 MLA 的 KV 潜变量在相同 `dcp_rank` 的不同 TP Rank 上完全复制，`HiCacheController` 定义 `is_storage_writer = (not is_mla_model) or (tp_rank == dcp_rank)`。例如在 `TP=4, DCP=2` 下，仅 `tp_rank ∈ {0, 1}` 向 L3 写出 `dcp0` 与 `dcp1` 对象，而 `tp_rank ∈ {2, 3}` 将 `backup_skip` 置为 `True` 并直接从 L3 读取，将 L3 写放大降低 **`tp_size / dcp_size = 2x`**。
  - **稀疏尾页检查点位图投票 (`PoolHitPolicy.TRAILING_PAGES`)**：混合模型（如 DeepSeek + Mamba/KDA 状态池）仅在离散页边界保存状态检查点（如页 512 与 1024 存在检查点，页 768 无检查点）。若像普通 KV 池一样对各 Rank 的标量命中页数做 `ReduceOp.MIN`，结果可能落在无检查点的中间页（如 768）。`_sync_prefetch_hit_query` 改为构造长度为 `num_pages + 1` 的命中位图向量 `hits[0..num_pages]` 并执行逐位 `ReduceOp.MIN`，精确求出所有 Rank 共同命中的最大有效检查点页。
- **Diffusion 融合 `FUSED_LINEAR_GELU` 动态 LoRA 守卫与 LTX-2 去同步 (`#45943`, `#45966`)**：
  - `can_use_linear_gelu` 在静态检查 `_static_reject_reason` 中解包 `BaseLayerWithLoRA` 以便挂载融合算子，但在每次前向调用时检查 `isinstance(linear, BaseLayerWithLoRA) and linear.lora_active`：一旦激活 LoRA 适配器即自动切回包含 LoRA 分支的 `linear(x)`，防止 cublasLt 基座权重 epilogue 静默丢弃 LoRA 增量。
  - LTX-2 去噪循环预生成 `sigmas_host` (`cpu`) 与 `sigmas_device` (`cuda`) 并将 `encoder_attention_mask` 移至块循环外，消除每步每层的 `.item()` / `.to(device)` 流同步。

---

## 2. TensorRT-LLM SWA 分支快照、Low-M GEMM 图捕获安全、Rubin SM107 与 MLA 独立 V 步长 (`#19877`, `#20334`, `#20167`, `#20605`, `#20242`)

- **SWA 树状前缀缓存分支快照 (`#19877`)**：滑动窗口注意力（SWA）在滑出窗口后会驱逐旧块，导致同批次或跨轮次的共享前缀请求无法复用中间前缀。`_swa_branch_snapshot_point` 计算分支点 `point = (divergence // tokens_per_block) * tokens_per_block`（要求 `divergence < num_lookup_tokens` 且 `reused < point < prompt_len`），并将调度器切为 `(ContextChunkingPolicy.FORCE_CHUNK, chunk_unit_size)`，使预填充精确停在块对齐的分支点以固化 Radix 快照。
- **Low-M GEMM CUDA Graph 捕获防覆盖 (`#20334`)**：`low_m_gemm.py` 在 `torch.cuda.is_current_stream_capturing()` 期间不再复用按 `(module_key, m_bucket, n)` 索引的缓存输出张量，而是每次调用分配独立 `torch.empty((m, n))`，消除同一图内多次调用（如 Eagle3 多步草稿 logits）相互覆盖输出的内存别名缺陷。
- **Rubin (`SM107`) MiniMax-M3 使能 (`#20167`) 与多模态 `DecoderLoraOwner` (`#20605`)**：`FlashInferTrtllmGenFmha` 将 `SM107` 加入 `EXTRA_PAGE_SIZES_ONLY_SMS = frozenset({107})`，使 128-token 页（Eagle3 草稿层）走 TRTLLM-Gen FMHA、32-token 页走回退 FMHA；`MiniMaxM3KVCacheManagerV2` 在初始化时清零 KV/scale/index 缓冲区以消除掩码尾槽位 `0 * NaN = NaN`；`DecoderLoraOwner` 将多模态 LoRA 严格限制在解码器子树并拒绝视觉塔权重。
- **Context MLA 独立 V 张量物理步长 (`#20242`)**：在 `AttentionOp`、`FusedMHARunnerV2`、`quantizeCopyInputToFp8Kernel` 与 `TllmGenFmhaKernel` 中透传 `v_stride_in_bytes` / `v_token_stride_elements`（强制校验 16B TMA 对齐），修复独立连续 V 张量被硬编码按打包步长 `numKvHeads * (headDimQkNope + headDimV)` 跨行读取的越界/错位问题。

---

## 3. vLLM Humming MXFP4 激活覆盖与 Rust 前端协议对齐 (`#62751`, `#63295`, `#63316`, `#63317`)

- **Humming MXFP4 MoE 激活覆盖 (`#62751`)**：新增 `_humming_input_schema(model_activation_key)`，统一调用 `_resolve_activation_key` 将 `--quantization-config.moe.activation mxfp8` 应用于 `gpt_oss_mxfp4` 与 `mxfp4` 权重转换。
- **Rust 前端 `LoRARequest` 6 字段元组对齐与 `MooncakeTransferStats` (`#63295`, `#63316`, `#63317`)**：移除 Rust 端多余的 `tensorizer_config_dict` 字段以对齐 Python `msgspec.Struct(array_like=True)` 的 6 元组布局；引入 `LoadedLora` 使缓存生成的推理请求始终令 `load_inplace = false`；在 `#[serde(untagged)] enum KvConnectorStats` 中将 `MooncakeTransfer` 置于 `Multi` 之前，导出完整的 6 项 Mooncake Prometheus 传输指标。

---

## 4. 确定性复现实验验证

运行 `python experiments/sglang-hicache-mla-dcp-diffusion-lora-trtllm-swa-rubin-vllm-mxfp4/run.py --check` 全部不变量通过（详见 `summary.json`）。
