# SGLang Mamba DCP Chunk Grid & Kimi-K3 SM107 Split-K, vLLM RL NCCL Rendezvous & TensorRT-LLM Packed Ulysses Unscatter (Phase 87)

> **Evidence Classification**: First-hand Implementation Audit (`implementation` + `own-offline` deterministic Python 3.12 stdlib model `experiments/sglang_vllm_trtllm_phase87/run.py`).

## 1. Key Architectural Findings & First-Hand Source Paths

| Upstream Commit / PR | Source Paths | Architectural Change & Quantitative Metrics |
|---|---|---|
| `sgl-project/sglang#41701` (`8252e3af63`, 2026-10-25) & `#46442` (`94d2b61e38`, 2026-10-25) | `python/sglang/srt/managers/{schedule_policy,scheduler}.py`, `python/sglang/srt/mem_cache/kv_cache_configurator.py` | `#41701` keeps chunked prefill cuts on the Mamba kernel chunk grid (`mamba_cache_chunk_size`, e.g., `64`) when DCP (`dcp_enabled`) and `enable_mamba_extra_buffer` are active, combining with existing alignments via `math.lcm(24, 64) = 192` and flooring intermediate cuts via `PrefillAdder.floor_chunked_tokens()`. Prevents an unaligned cut from forfeiting every subsequent checkpoint donation (**4 vs 1** donations on a 1024-token prefill with budget 260) while allowing cuts `< grid` or final chunks (`trunc_len == remaining`) to make forward progress without deadlock. `#46442` names `--kv-cache-dtype-draft` explicitly in the unified-memory fused-region mismatch error. |
| `sgl-project/sglang#46464` (`1b112fd62a`, 2026-10-25) & `#45733` (`933853a3f2`, 2026-10-25) | `python/sglang/kernels/ops/gemm/{cutedsl_bf16_gemm,dense_bf16_gemm_warp_splitk}.py`, `python/sglang/multimodal_gen/*` | `#46464` adds an FP32-output warp split-K BF16 GEMM kernel (`4` compute warps, `2` A-loader warps, `cp.async` ring stages `{8: (8, 11), 16: (16, 9), 32: (32, 6)}`) on `SM107` for Kimi-K3's merged MoE front projection (`N=6016, K=7168`), preserving full FP32 router-logit precision across columns `[1536:2432]` (**896 experts**, top-16 sigmoid + `correction_bias` selection). `#45733` generalizes exact GPU image preprocessing (Lanczos + validation) across Flux, Qwen-Image, and Cosmos3 pipelines. |
| `vllm-project/vllm#63460` (`c9dbc92630`, 2026-10-25), `#63496` (`8e11a8a49e`, 2026-10-25), `#60149` (`75fc05a53a`, 2026-10-25), `#63538` (`b70de52965`, 2026-10-25) | `vllm/distributed/weight_transfer/{nccl_common,nccl_engine,sparse_nccl_engine}.py`, `vllm/config/lora.py`, `vllm/model_executor/models/dory.py` | `#63460` unifies dense and sparse RL weight-transfer initialization via `build_worker_init_payload()` (`rank_offset=1`) and `run_trainer_rendezvous()`, running the worker RPC in a background thread concurrently with `open_trainer_endpoint` on Rank 0; `#63496` moves MegaMoE + LoRA rejection (`MEGA_MOE_BACKENDS` and `"aiter_mega_moe"`) into `LoRAConfig.verify_with_kernel_config()` during config validation; `#60149` adds Dory model support; `#63538` borrows Gemma4 `_get_max_soft_tokens` in `EmbeddingGemma2` and `DiffusionGemma`. |
| `NVIDIA/TensorRT-LLM#17147` (`1586d85375`, 2026-10-23), `#20270` (`87342f4474`, 2026-10-24), `#20307` (`39a44fab74`, 2026-10-23) | `cpp/tensorrt_llm/kernels/ulyssesPostUnscatterKernel.cu`, `cpp/tensorrt_llm/thop/ulyssesPostUnscatterOp.cpp`, `tensorrt_llm/_torch/pyexecutor/*` | `#17147` introduces `ulyssesPackedQkvPostUnscatterKernel` (`VEC = 8` via `uint4` 16-byte loads/stores) to unscatter packed all-to-all buffers `[P, B, Sp, 3, H, D]` directly into NHD-contiguous `Q, K, V` `[B, P*Sp, H, D]` in **1 fused launch** instead of 3, guarded by host-side `checkInt32Dim` / `checkInt32Product` / `checkInt32SumProduct` checks. `#20270` narrows the PyExecutor runner surface and makes the executor interface explicit; `#20307` adds Cosmos3-Edge Reasoner support. |

## 2. Mathematical Formulation & Architect Sizing Rules

### 2.1 Mamba + DCP Truncation Alignment via LCM
When combining an existing truncation alignment $A_{\text{dsa}} = 24$ with the Mamba DCP chunk grid $G_{\text{mamba}} = 64$, the scheduler enforces:
$$A_{\text{total}} = \operatorname{lcm}(A_{\text{dsa}}, G_{\text{mamba}}) = 192$$
and floors intermediate prefill chunks via:
$$\operatorname{floor\_grid}(L_{\text{trunc}}, L_{\text{rem}}, G) = \begin{cases} L_{\text{trunc}}, & \text{if } G \le 1 \lor L_{\text{trunc}} \ge L_{\text{rem}} \lor L_{\text{trunc}} < G \\ \lfloor L_{\text{trunc}} / G \rfloor \cdot G, & \text{otherwise} \end{cases}$$

### 2.2 Packed QKV Post-Ulysses Unscatter Grid Mapping
For Ulysses degree $P$, batch $B$, sub-sequence length $S_p$, heads $H$, and head dim $D$:
$$\text{gridDim} = (3 \cdot P \cdot S_p, B, 1), \quad \text{blockDim} = (H \cdot (D / 8), 1, 1)$$
At $P=2, B=1, S_p=2048, H=24, D=128$, this launches $12,288 \times 1$ blocks of $384$ threads using 16-byte `uint4` transfers.

## 3. Reproducible Verification (`experiments/sglang_vllm_trtllm_phase87`)

Running `python experiments/sglang_vllm_trtllm_phase87/run.py --check` validates all deterministic metrics in `summary.json`.
