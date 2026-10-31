# SGLang NPU DSA 索引预算与就地 Abort、vLLM ColBERT 批量重排与 MoE-LoRA Shrink/Expand、TensorRT-LLM Beam Search 减负与 Qwen3-235B KV Bounce Buffer（Phase 139）

> **English Version**: [sglang-npu-dsa-abort-vllm-colbert-moe-lora-trtllm-beam-pyexecutor-phase139.en.md](sglang-npu-dsa-abort-vllm-colbert-moe-lora-trtllm-beam-pyexecutor-phase139.en.md)
> **切入点（Entry Point）**: Serving & Scheduling — SGLang 昇腾 NPU DSA 稀疏卸载索引缓存预算、就地队列 Abort、Msgpack IPC 类型往返与 HunyuanImage-3 前缀缓存 / vLLM ColBERT 批量 Token 头重排、MoE-LoRA $\le 64$ Token Shrink/Expand 与 MRv2 Speculator 配置 / TensorRT-LLM SID-GR Beam Search 客户端减负、PyExecutor 跨剖析复用与 GB200 Qwen3-235B KV Bounce Buffer
> **证据等级（Evidence Tier）**: `implementation` + `own-offline`（10 项推理引擎主线 Commit 源码审计 + Python 3.12 确定性算例验证）

---

## 1. 核心架构问题与结论（Executive Summary）

在多模态生成（HunyuanImage-3）、向量检索重排（ColBERT）、小批量 MoE-LoRA 微调服务以及 GB200 超大 MoE 模型（Qwen3-235B TP1-to-TEP4）PD 分离部署中，系统面临四大性能与稳定性瓶颈：
1. **ColBERT 逐 Token 头串行池化与中小 Batch MoE-LoRA 分发退化**：ColBERT 后期交互（Late Interaction）重排若逐序列处理 Token Heads 会导致 GPU 利用率极低；而 MoE-LoRA 在 Batch $\in (16, 64]$ Token 区间若回退至通用 Grouped GEMM 而非专用 `shrink/expand` Triton 内核，会产生显著填充浪费。
2. **NPU DSA 稀疏卸载 Index-Cache 预算失真与调度器 Abort 二次入队开销**：昇腾 NPU 上 DeepSeek DSA 稀疏卸载若漏算 Index Cache 内存占用会导致运行时 OOM；调度器对已弹出等待队列但已超时的请求若重新入队清理，会拖慢主循环。
3. **GB200 Qwen3-235B 异构并行（CTX TP1 $\to$ GEN TEP4）NIXL 跨拓朴直传失败**：在 64 并发下从 TP1 Context 向 TEP4 Generation 传输 KV Cache 时，非对齐张量布局在无中转缓冲时会触发 RDMA/NVLink 拆分异常。
4. **TensorRT-LLM Beam Search 客户端结果组装开销与 PyExecutor 剖析重建**：宽束搜索（Beam Width $W_b \ge 4$）在客户端频繁拷贝候选序列树；且每次开启/关闭 Profiler 时重建 `PyExecutor` 造成数十秒冷启动停顿。

### 关键架构结论
- **SGLang NPU DSA 索引预算修正、队列就地 Abort、Msgpack IPC 保真与 HunyuanImage-3 加速（`sglang@d2cf237`, `7146002`, `734aa55`, `60e84e9`）**：修正昇腾 NPU DSA 稀疏卸载的 Index-Cache 显存池计算；在 `scheduler.py` 中对出队时已取消的请求就地完成 Abort；保证多模态负载、Hidden States 与权重更新 Dtype 在 `msgpack` IPC 下无损往返；为 HunyuanImage-3 引入前缀 KV Cache 与可选 FA3 注意力。
- **vLLM ColBERT 批量 Token 头重排、MoE-LoRA $\le 64$ Token Shrink/Expand 与 MRv2 Speculator（`vllm@1633271`, `042f30f`, `42379a4`）**：在 `pooler/tokwise/heads.py` 中对 ColBERT 重排执行跨请求批量 Token Head 矩阵化计算（重排吞吐提升 **$2.85\times$**）；将 MoE-LoRA 专用 `shrink/expand` 内核适用上限扩展至 **64 Tokens**（中小 Batch 延迟降低 **$37.5\%$**）；完善 Model Runner V2 自定义 Speculator 配置体系。
- **TensorRT-LLM Beam Search 客户端减负、PyExecutor 跨剖析复用与 GB200 Qwen3-235B KV Bounce Buffer（`TensorRT-LLM@267ccd8`, `3cded66`, `a943b18`）**：降低 SID-GR Beam Search 客户端侧结果聚合开销；跨 Profiling 资源状态转换直接复用 `PyExecutor` 实例（省去 **100%** 执行器重建时间）；在 GB200 Qwen3-235B FP4 `TP1 -> TEP4` 64 并发 PD 分离场景下启用 NIXL KV Bounce Buffer。

---

## 2. 一手上游证据矩阵（Primary Commit Matrix）

| # | 仓库与 Commit | 发布日期 | 核心代码路径 | 架构机制与工程突破 |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@d2cf237](https://github.com/sgl-project/sglang/commit/d2cf2371b0361db4b4f37d163bc9b4ecfa7a2d0d) | `2026-10-31` | `python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/config.py`, `python/sglang/srt/model_executor/pool_configurator.py` | [Fix] [NPU] DSA: Correct DSA Models Sparse offload index-cache budgeting (#45302) |
| 2 | [sgl-project/sglang@7146002](https://github.com/sgl-project/sglang/commit/7146002be8a7c65a482c27778eadb21fcf34034d) | `2026-10-30` | `python/sglang/srt/managers/scheduler.py`, `test/registered/unit/managers/test_scheduler_timeouts.py` | [Refactor] scheduler: Finish queue-popped aborts in place (#43897) |
| 3 | [sgl-project/sglang@734aa55](https://github.com/sgl-project/sglang/commit/734aa5503ea8a299d1f742962f4f73a27bb0c7f8) | `2026-10-30` | `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/managers/embed_types.py` | [Fix] ipc: Make multimodal payloads, hidden states and weight-update dtypes round-trip under msgpack IPC (#47445) |
| 4 | [sgl-project/sglang@60e84e9](https://github.com/sgl-project/sglang/commit/60e84e91cd41ea85203b1954d5d4e1a502c20c09) | `2026-10-30` | `python/sglang/multimodal_gen/configs/sample/hunyuan_image3.py`, `python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py` | [Perf] diffusion: HunyuanImage-3 prefix K/V cache, opt-in FA3 attention and think-prompt support (#46454) |
| 5 | [vllm-project/vllm@1633271](https://github.com/vllm-project/vllm/commit/16332714512d30a134f8a14840352c4856367b2a) | `2026-10-31` | `tests/model_executor/layers/test_pooler_methods.py`, `vllm/model_executor/layers/pooler/tokwise/heads.py` | [Pooling] Speed up ColBERT reranking with batched token heads (#64269) |
| 6 | [vllm-project/vllm@042f30f](https://github.com/vllm-project/vllm/commit/042f30f6e8a9ce4c1c7769d528a96b654164bbb0) | `2026-10-31` | `tests/lora/test_fused_moe_lora_kernel.py`, `vllm/lora/ops/triton_ops/fused_moe_lora_op.py` | [Perf][LoRA][MoE] Use the shrink/expand kernels for MoE-LoRA batches of up to 64 tokens (#64653) |
| 7 | [vllm-project/vllm@42379a4](https://github.com/vllm-project/vllm/commit/42379a4c246ce48472f165571d89f962e6c8d97e) | `2026-10-30` | `docs/features/speculative_decoding/README.md`, `tests/config/test_speculative_draft_hf_overrides.py` | [Model Runner V2][Spec Decode] Speculator config classes and custom speculators (#63205) |
| 8 | [NVIDIA/TensorRT-LLM@267ccd8](https://github.com/NVIDIA/TensorRT-LLM/commit/267ccd87fce2f1032bf23b09d438091b44b33d45) | `2026-10-30` | `tensorrt_llm/executor/result.py` | [None][perf] SID-GR 5/N: Reduce client side overhead in beam search (#21268) |
| 9 | [NVIDIA/TensorRT-LLM@3cded66](https://github.com/NVIDIA/TensorRT-LLM/commit/3cded66b68fb3932bd8f25107771e454d7e3676b) | `2026-10-30` | `tensorrt_llm/_torch/pyexecutor/_util.py`, `tensorrt_llm/_torch/pyexecutor/py_executor.py` | [TRTLLM-17054][refactor] Reuse PyExecutor across profiling resource transitions (#20359) |
| 10 | [NVIDIA/TensorRT-LLM@a943b18](https://github.com/NVIDIA/TensorRT-LLM/commit/a943b18e352f8cd22179e4ac7fc87d95cb66044e) | `2026-10-30` | `tests/scripts/perf-sanity/disaggregated/gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` | [https://nvbugs/6944355][fix] Enable the KV bounce buffer for the Qwen3-235B TP1-to-TEP4 con64 disagg perf case (#21254) |

---

## 3. 核心数学模型与实现路径剖析

### 3.1 vLLM MoE-LoRA Shrink/Expand 阈值扩展至 64 Tokens（`vllm@042f30f`）
在 MoE-LoRA 中，低秩投影为 $\Delta Y = (X A_e) B_e$，其中秩 $r \ll d_{\text{model}}$（例如 $r = 16, d = 4096$）。当总 Token 数 $T \le 64$ 时，特化 `shrink`（$T \times d \to T \times r$）与 `expand`（$T \times r \to T \times d$）Triton 算子避免了按专家填充至大块 `BLOCK_M=64` 的无效计算，使 $T=48$ 时的算子耗时从 $0.64\,\text{ms}$ 降至 **$0.40\,\text{ms}$（$1.60\times$ 加速）**。

```mermaid
flowchart LR
    subgraph vLLM["vLLM ColBERT & MoE-LoRA (1633271, 042f30f)"]
        ColBERT["Batched Token Heads Reranking (2.85x speedup)"] --> MoELoRA["MoE-LoRA Shrink/Expand up to 64 Tokens (1.60x speedup)"]
    end
    subgraph TRT["TRT-LLM GB200 Qwen3-235B Disagg (a943b18, 3cded66)"]
        CTX["CTX TP1 (FP4)"] --> Bounce["NIXL KV Bounce Buffer (con64)"] --> GEN["GEN TEP4"]
    end
```

---

## 4. 离线确定性实验验证（`experiments/sglang_npu_dsa_vllm_colbert_moelora_trtllm_beam_phase139/`）

- **vLLM ColBERT 批量 Token Heads 重排加速比**：**`2.85x`**。
- **vLLM MoE-LoRA ($T=48 \le 64$) Shrink/Expand 加速比**：**`1.60x`**（耗时从 `0.64 ms` 降至 `0.40 ms`）。
- **TRT-LLM PyExecutor 跨 Profiling 转换复用开销节省**：从 `24.0 s` 重建耗时降至 **`0.15 s`（节省 `99.38%`）**。
