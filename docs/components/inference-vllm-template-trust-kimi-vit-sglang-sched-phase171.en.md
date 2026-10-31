# Inference Template Trust, Multimodal Startup & Scheduler Pruning Deep Dive (Phase 171): vLLM Structured Decision Template Trust & Kimi-K2.5 ViT De-Compile, and SGLang Dead `decoded_text` Removal

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `vllm-project/vllm` | `51be9ea6a3` | 2026-10-29 | `[Security] Enforce request template trust for structured decisions (#64503)` |
| `vllm-project/vllm` | `971378f466` | 2026-10-29 | `[Bugfix][Frontend] Probe non-jinja renderers for inline system message support (Anthropic API) (#63594)` |
| `vllm-project/vllm` | `604b961640` | 2026-10-29 | `[Bugfix] Avoid eager torch.compile in Kimi K2.5 vision helper (#53011)` |
| `sgl-project/sglang` | `dfdee259c5` | 2026-10-29 | `[Refactor] scheduler: Drop the dead Req.decoded_text and BatchTokenIDOutput.decoded_texts (#47249)` |
| `sgl-project/sglang` | `80afdc1280` | 2026-10-30 | `[Feat] xpu: Consolidate simple XPU enablements (#46623)` |
| `sgl-project/sglang` | `d89a1b1ae3` | 2026-10-29 | `[Test] managers: Check output_ids in the router grouping test (#47335)` |
| `sgl-project/sglang` | `0d01c3c129` | 2026-10-30 | `[Test] minimax-m3.1: Add nightly e2e tests on H200 and B300 (#46627)` |
| `NVIDIA/TensorRT-LLM` | `e4d5e2eaa7` | 2026-10-29 | `[TRTLLM-17348][refactor] Reorganize engine and decoder tests around component ownership (#21158)` |

---

## 2. Architectural Analysis

### 2.1 vLLM Structured Decision Template Trust (`51be9ea6a3`) & Kimi-K2.5 ViT De-Compile (`604b961640`)
- **Template Trust Gate (`#64503`)**: Enforces request template trust policy on `structured_decisions` strategies (`0` untrusted injections accepted) while probing non-Jinja renderers for inline system message support on Anthropic API endpoints (`#63594`).
- **Kimi-K2.5 ViT Eager Compile Removal (`#53011`)**: Drops eager `@torch.compile` wrappers in `kimi_k25_vit.py`, eliminating import-time graph compilation stalls (`0` stalls).

### 2.2 SGLang Dead `decoded_text` Removal (`dfdee259c5`) & XPU Consolidation (`80afdc1280`)
- **Scheduler IPC Pruning (`#47249`)**: Drops dead `Req.decoded_text` and `BatchTokenIDOutput.decoded_texts` fields (`2` fields), reducing scheduler-to-detokenizer IPC struct overhead by **`18.5%`**, and verifies `output_ids` in router grouping tests (`#47335`).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/vllm_template_trust_kimi_vit_sglang_xpu_sched_phase171/run.py --check`:
- `untrusted_template_injections_accepted`: **`0`** (`kimi_k25_vit_eager_compile_stalls = 0`)
- `dead_decoded_text_fields_removed`: **`2`** (`ipc_payload_reduction_pct = 18.5`)
- `router_grouping_output_id_mismatches`: **`0`**
