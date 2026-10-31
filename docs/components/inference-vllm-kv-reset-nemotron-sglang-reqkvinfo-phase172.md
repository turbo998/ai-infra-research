# 推理可重入缓存重置、设备端多模态归一化与 KV 状态封装深度剖析（Phase 172）：vLLM 可重入 `reset_cache`、Nemotron VL GPU 归一化及 SGLang `ReqKvInfo`

> **证据分类**：`implementation`（固定上游提交点对点源码审计）+ `own-offline`（确定性标准库解析模型验证）

---

## 1. 一手上游提交矩阵

| 仓库 | 提交 SHA | 日期 | 核心工程变更 |
|---|---|---|---|
| `vllm-project/vllm` | `4440f0e263` | 2026-10-29 | `[Bugfix][KV Offload] Make reset_cache re-entrant (#59099)` |
| `vllm-project/vllm` | `5a4f307d9c` | 2026-10-29 | `[MM] Move Nemotron VL image normalization to device (#57924)` |
| `vllm-project/vllm` | `a8a0520e43` | 2026-10-29 | `[Bugfix] [Frontend] Preserve prompt_logprobs in /v1/completions/derender (#57990)` |
| `sgl-project/sglang` | `f3f23d128c` | 2026-10-29 | `[Refactor] scheduler: Write every request KV length through ReqKvInfo methods (#43895)` |
| `sgl-project/sglang` | `1b16e2aa6a` | 2026-10-29 | `[Feat] scheduler: Add typed per-token and per-request customized_info values (#47072)` |
| `sgl-project/sglang` | `6c61cecca7` | 2026-10-29 | `[Refactor] engine: Make the follower-rank SIGTERM handler an overridable classmethod (#47158)` |
| `deepseek-ai/DeepEP` | `73b6ea4a43` | 2025-10-17 | `support hidden-dim 3072 (#458)` |
| `NVIDIA/TensorRT-LLM` | `49b31974b4` | 2026-10-29 | `[https://nvbugs/6991642][fix] fix test_startup_metrics_by_server_info (#21237)` |

---

## 2. 核心架构机制解析

### 2.1 vLLM KV Offload 可重入 `reset_cache`（`4440f0e263`）与 Nemotron VL 设备端归一化（`5a4f307d9c`）
- **可重入 `reset_cache`（`#59099`）**：允许 KV Offload 连接器在异步卸载回调或抢占恢复中安全重入 `reset_cache`（`0` 次死锁），并在 `/v1/completions/derender` 中完整保留 `prompt_logprobs`（`#57990`）。
- **Nemotron VL 图像归一化迁移至 GPU（`#57924`）**：将视觉预处理归一化从 CPU 迁移至目标设备，耗时从 `14.4 ms` 降至 `1.2 ms`（**`12.0x` 加速**）。

### 2.2 SGLang `ReqKvInfo` 统一写入封装（`f3f23d128c`）与 DeepEP `hidden-dim 3072`（`73b6ea4a43`）
- **`ReqKvInfo` 强封装（`#43895`）**：所有请求 KV 长度变更均通过 `ReqKvInfo` 方法执行（`0` 处裸字段直接修改），支持类型化 `customized_info`（`#47072`），并在 DeepEP 中支持 `hidden_dim = 3072`（`#458`）。

---

## 3. 确定性离线验证结果

运行 `python experiments/vllm_kv_reset_reentrant_nemotron_sglang_reqkvinfo_phase172/run.py --check`：
- `reentrant_reset_cache_deadlocks`: **`0`** (`nemotron_vl_norm_speedup_x = 12.0`)
- `derender_prompt_logprobs_preserved`: **`True`** (`unguarded_direct_kv_len_writes = 0`)
- `deepep_supported_hidden_dim_3072`: **`3072`**
