# GPU 软件栈深度剖析（Phase 170）：FlashInfer Qwen 3.5 GDN `frost` SM100 后端、Cake Softmax 分块规则及 FA SM100 非 32 倍数 `head_dim` 反向传播

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `56472bf4fc` | 2026-10-31 | `[Experimental] Qwen 3.5 GDN block: decode and MTP verify behind a backend registry, with an SM100 CuTe DSL backend (#8131)` |
| `flashinfer-ai/flashinfer` | `caa3fd0f20` | 2026-10-29 | `perf(cake_softmax): widen balanced per-CTA chunk rules on B200 and GB300; narrow 6-tile 8-CTA rules on GB300 (#7822)` |
| `flashinfer-ai/flashinfer` | `861d03c456` | 2026-10-29 | `perf(cake_mhc): cut the eager host time of the mHC with-grad entry points (#7823)` |
| `flashinfer-ai/flashinfer` | `c58617e6da` | 2026-10-29 | `test(comm): fix Ulysses initialization fault injection (#7876)` |
| `Dao-AILab/flash-attention` | `94345b4568` | 2026-09-08 | `[CuTe,Bwd,Sm100] Support head_dim not a multiple of 32 in backward (#2698)` |
| `Dao-AILab/flash-attention` | `f4c68c8a74` | 2026-09-08 | `[CuTe] Include callable defaults in JIT cache hashes (#2773)` |
| `NVIDIA/cutlass` | `c900acd605` | 2026-07-08 | `Preserve Blackwell example performance across CUDA backends (#3370)` |
| `triton-lang/triton` | `2122e24efc` | 2026-10-16 | `[Backend] Avoid emitting llvm bitcast when loading (#12319)` |

---

## 2. 核心架构机制解析

### 2.1 FlashInfer Qwen 3.5 GDN 单次融合块（`56472bf4fc`）与 Cake Softmax 规则（`caa3fd0f20`）
- **Qwen 3.5 GDN 单调用融合（`#8131`）**：引入 `flashinfer.qwen_3_5_gdn_block` 后端注册表（`register_backend`）与首个针对 SM100 的 CuTe DSL 后端 `frost`，在单次调用内融合 `BA projection -> causal conv update -> gating -> SSM state pool gated delta rule -> gated RMSNorm -> static FP8 output`，将 6 个独立算子合并为 1 个，实现 **`2.23x`** 延迟加速（`46.8 us` -> `21.0 us`）。
- **Blackwell Cake Softmax 分块优化（`#7822`）**：每个 Tile 为 `2048` 元素，均衡 CTA 分块按 `ceil(vocab / CTAs)` 向上对齐到 `32` 元素（`vocab=152064, CTAs=8` 时为 `19008` 元素），并在 GB300 (`sm_103a`) 上收紧 8-CTA 6-tile 规则。

### 2.2 FlashAttention CuTe SM100 非 32 倍数 `head_dim` 反向传播（`94345b4568`）与 JIT 哈希（`f4c68c8a74`）
- **非 32 倍数 `head_dim` 修复（`#2698`）**：修复 ViT 等 `head_dim=72, 104`（`head_dim % 32 != 0`）在 SM100 反向传播中崩溃或 `dK/dV` 错误的三大根因：按 `m` 切片 `predicate_k`、将 `dQ` TMEM 累加器对齐到 32 倍数（`96`, `128`），并同步调整 `epilogue`。
- **Callable 默认参数纳入 JIT 缓存哈希（`#2773`）**：避免修改闭包/函数默认参数时误命中陈旧编译产物（碰撞次数 `0`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/flashinfer_qwen35_gdn_cake_softmax_fa_sm100_non32_phase170/run.py --check`：
- `gdn_speedup_x`: **`2.23x`** (`6` -> `1` kernel)
- `balanced_cta_chunk_elems`: **`19008`** (`gb300_narrowed_max_tiles_8cta = 6`)
- `padded_tmem_hdims`: **`[96, 128]`** (`non32_bwd_max_err = 0.0`, `jit_callable_default_collisions = 0`)
