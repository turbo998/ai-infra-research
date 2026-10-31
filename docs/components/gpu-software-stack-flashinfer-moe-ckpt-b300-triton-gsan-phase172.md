# GPU 软件栈深度剖析（Phase 172）：FlashInfer MoE 早停跳过重算、B300 持久 QK-RoPE、NVFP4 H512 回退及 Triton GSan `mbarrier` 表定界

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `a049b1a539` | 2026-10-31 | `perf(cake_moe_grouped_gemm): skip forward recompute under checkpoint early stop (#8114)` |
| `flashinfer-ai/flashinfer` | `04fc035849` | 2026-10-29 | `perf(cake_fused_qk_rope_fp8_append): persistent stage for B300 prefill of few requests (#7820)` |
| `flashinfer-ai/flashinfer` | `af8feb4ac9` | 2026-10-29 | `feat(attention): NVFP4 KV fallback for trtllm-gen context at H512 (#6445)` |
| `flashinfer-ai/flashinfer` | `68fc02be31` | 2026-10-29 | `fix(deep_gemm): cache loaded DeepGEMM runtimes under the key load() looks up (#7506)` |
| `Dao-AILab/flash-attention` | `0251105a2f` | 2026-08-15 | `[CuTe] Relax CUTLASS DSL requirement (#2798)` |
| `Dao-AILab/flash-attention` | `145b101005` | 2026-08-11 | `Fix removed Quack packed subtraction API (#2787)` |
| `deepseek-ai/FlashMLA` | `fd249aacce` | 2025-09-29 | `Add Sparse Decoding Kernel and Sparse Prefill Kernel for Blackwell` |
| `triton-lang/triton` | `2d0eee9c8a` | 2026-10-15 | `[GSan] Size mbarrier tables from backing allocations (#12323)` |
| `triton-lang/triton` | `4a4302c719` | 2026-10-15 | `Fix backward remat infinite loop for constant remat (#12307)` |

---

## 2. 核心架构机制解析

### 2.1 FlashInfer MoE 激活重算跳过（`a049b1a539`）、B300 持久 QK-RoPE（`04fc035849`）与 DeepGEMM 缓存键对齐（`68fc02be31`）
- **MoE Checkpoint 早停跳过重算（`#8114`）**：在 `cake_moe_grouped_gemm` 中当激活检查点触发早停时直接复用已驻留激活，跳过冗余前向重算（`0` 次冗余重算）。
- **B300 少请求预填充持久阶段（`#7820`）与 H512 NVFP4 KV 回退（`#6445`）**：在 B300 少量长请求 Prefill 中启用持久流水线使 `cake_fused_qk_rope_fp8_append` 提速 **`1.32x`**，为 `trtllm-gen` `H=512` 上下文补齐 NVFP4 KV 回退路径，并将已加载的 DeepGEMM runtime 按 `load()` 查询键写入缓存（`#7506`，消除重复编译未命中）。

### 2.2 Triton GSan `mbarrier` 表按底层分配定界（`2d0eee9c8a`）与常量重物化死循环修复（`4a4302c719`）
- **GSan 屏障追踪表定界（`#12323`）**：根据底层共享内存 backing allocation 精确计算 `mbarrier` 状态表容量（`0` 越界访问），并终止常量重物化在反向传播中的无限循环（`#12307`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/flashinfer_moe_ckpt_b300_rope_triton_gsan_mbarrier_phase172/run.py --check`：
- `moe_ckpt_early_stop_fwd_recomputes`: **`0`** (`b300_few_req_rope_append_speedup_x = 1.32`)
- `trtllm_gen_h512_nvfp4_kv_fallback_enabled`: **`True`** (`deepgemm_runtime_cache_miss_after_load = 0`)
- `gsan_mbarrier_table_oob_faults`: **`0`** (`constant_remat_infinite_loops = 0`)
