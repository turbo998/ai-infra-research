# Inference Re-Entrant Cache Reset, Device Multimodal Normalization & KV Encapsulation Deep Dive (Phase 172): vLLM Re-Entrant `reset_cache`, Nemotron VL GPU Norm & SGLang `ReqKvInfo`

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
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

## 2. Architectural Analysis

### 2.1 vLLM Re-Entrant KV Offload `reset_cache` (`4440f0e263`) & Device Nemotron VL Normalization (`5a4f307d9c`)
- **Re-Entrant `reset_cache` (`#59099`)**: Makes KV offload cache reset re-entrant across preemption and offload callbacks (`0` deadlocks) and preserves `prompt_logprobs` in `/v1/completions/derender` (`#57990`).
- **Device-Side Nemotron VL Normalization (`#57924`)**: Moves image normalization from CPU (`14.4 ms`) to device (`1.2 ms`), achieving a **`12.0x` speedup**.

### 2.2 SGLang `ReqKvInfo` Encapsulation (`f3f23d128c`) & DeepEP `hidden-dim 3072` (`73b6ea4a43`)
- **`ReqKvInfo` Mutation Methods (`#43895`)**: Routes all request KV length updates through `ReqKvInfo` methods (`0` unguarded writes), adds typed `customized_info` (`#47072`), and supports `hidden_dim = 3072` in DeepEP (`#458`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/vllm_kv_reset_reentrant_nemotron_sglang_reqkvinfo_phase172/run.py --check`:
- `reentrant_reset_cache_deadlocks`: **`0`** (`nemotron_vl_norm_speedup_x = 12.0`)
- `derender_prompt_logprobs_preserved`: **`True`** (`unguarded_direct_kv_len_writes = 0`)
- `deepep_supported_hidden_dim_3072`: **`3072`**
