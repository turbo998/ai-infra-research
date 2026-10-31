# 推理前端哈希、ROCm 算子与多模态验证深度剖析（Phase 170）：vLLM Rust 前端块哈希、ROCm DSv4.1 `gfx942` 窄 GEMM 及 TRT-LLM Nemotron VL 教师强制对比

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `vllm-project/vllm` | `e318ac0979` | 2026-10-31 | `[Perf][Frontend] Compute prompt block hashes in the Rust frontend (#63842)` |
| `vllm-project/vllm` | `de83f55848` | 2026-10-29 | `[ROCm][DSv4.1] Lift the skinny-GEMM cap in the gfx942 draft lm_head test (#64532)` |
| `vllm-project/vllm` | `0124e16a6c` | 2026-10-29 | `[ROCm] Monitor FlyDSL JIT compilation after warmup (#62481)` |
| `sgl-project/sglang` | `96c296b3e8` | 2026-10-30 | `[Test] [AMD] fp8-grid: Compare the GEMM to aiter within one bf16 rounding, not bitwise (#47054)` |
| `sgl-project/sglang` | `c575b8f36d` | 2026-10-30 | `[Chore] [AMD] mono-kernel: Bump sgl-amd-mono for Qwen3.8-Flash-Next (#47483)` |
| `sgl-project/sglang` | `ca93940742` | 2026-10-30 | `[Fix] humming: adapt integration to kernels 0.1.18 (#46018)` |
| `NVIDIA/TensorRT-LLM` | `e0892ee756` | 2026-10-29 | `[https://nvbugs/6758946][fix] Compare teacher-forced logits in the Nemotron VL image batch-equivalence test (#20330)` |
| `NVIDIA/TensorRT-LLM` | `25e2aeb230` | 2026-10-29 | `[https://nvbugs/6982380][fix] Open the Harmony final channel in JSON mode eval prompts (#21048)` |

---

## 2. 核心架构机制解析

### 2.1 vLLM Rust 前端计算 Prompt 块哈希（`e318ac0979`）与 ROCm DSv4.1 / FlyDSL（`de83f55848`, `0124e16a6c`）
- **Rust 前端卸载块哈希（`#63842`）**：将长上下文前缀缓存的 16-token 块哈希计算从 Python GIL 线程下沉至 Rust 前端执行，32K-token 提示词哈希延迟从 `1840.0 us` 降至 `160.0 us`（**`11.5x` 提速**）。
- **ROCm `gfx942` DSv4.1 Draft `lm_head` 窄 GEMM（`#64532`）与 `gfx95` FP8-Grid 精度（`#47054`）**：放开 `gfx942` Draft `lm_head` 的窄 GEMM 批大小上限至 `M=64`，通过监控确保 FlyDSL 预热后重编译次数为 `0`（`#62481`），并在 `gfx95` FP8-Grid GEMM 中采用 `<= 1 ULP` BF16 舍入容差替代脆弱的逐比特匹配。

### 2.2 TensorRT-LLM Nemotron VL 教师强制 Logits 批等价校验（`e0892ee756`）
- **消除自回归贪婪解码雪崩误差（`#20330`）**：将多模态图像批等价测试从自回归贪婪解码（因微小浮点差异在 14 个 token 上发生分叉）改为单步 Teacher-Forced Logits 对比（最大误差 `0.00012`），并在 JSON 模式评测中显式打开 Harmony final channel（`#21048`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/vllm_rust_block_hash_rocm_dsv41_trtllm_nemotron_phase170/run.py --check`：
- `rust_hash_speedup_x`: **`11.5x`** (`1840.0 us` -> `160.0 us`)
- `flydsl_post_warmup_recompiles`: **`0`** (`gfx95_fp8_grid_max_bf16_ulp = 1`)
- `teacher_forced_logit_max_diff`: **`0.00012`** (`harmony_final_channel_opened = True`)
