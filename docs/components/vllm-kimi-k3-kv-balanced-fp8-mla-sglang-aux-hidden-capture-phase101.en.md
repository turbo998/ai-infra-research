# vLLM Kimi-K3 KV-Balanced FP8 MLA Decode, SGLang Draft-Free Auxiliary Hidden State Capture, `persistent_topk` Barrier & TRT-LLM EagerMoeWorkspace Analysis (Phase 101)

- **Date / Phase**: 2026-10-27 · Phase 101 (`priority: 110`)
- **Component**: `inference`
- **Question ID**: `vllm-kimi-k3-kv-balanced-fp8-mla-sglang-aux-hidden-capture-phase101`
- **Evidence ID**: `claim-vllm-kimi-k3-kv-balanced-mla-sglang-aux-hidden-phase101`
- **Reproducible Experiment Suite**: [`experiments/vllm_kv_balanced_mla_sglang_aux_phase101/`](../../experiments/vllm_kv_balanced_mla_sglang_aux_phase101/run.py)

---

## 1. Architectural Context & Core Problem

Modern agentic inference and speculative/distillation pipelines face four runtime bottlenecks:

1. **Split-KV Load Imbalance Under Skewed Agentic Context Lengths**: When serving Kimi-K3 MLA (`512 + 64 = 576` FP8 values/token) on agentic batches that mix 128K tool histories with 1K interactive turns, static split counts either starve SM clusters or over-split short sequences, while host-side planning breaks CUDA Graph capture.
2. **Dormant Draft-Model Memory Overhead During Auxiliary Hidden State Extraction**: Capturing multi-layer target hidden states using EAGLE3, DFLASH, or DSPARK conventions previously required keeping `--speculative-algorithm` active, loading the draft model weights and reserving draft KV cache even when no draft generation was needed.
3. **Cross-Warp Race Before Radix Barrier Arrival in `persistent_topk`**: In `csrc/persistent_topk.cuh`, threads cleared `next_hist[i] = 0` in parallel and `tx == 0` immediately issued `red_release(&state->arrival_counter, 1)` without a CTA-wide `__syncthreads()`, allowing warps 1..7 to race with the barrier release.
4. **MoE Workspace Reclamation Under Allocator Quota Pressure**: TensorRT-LLM's `EagerMoeWorkspace` and PEFT cache manager required deterministic scratch reclamation under native allocator quota limits and explicit `py_lora_ckpt_source` propagation.

---

## 2. Primary Source Empirical Analysis (12 Zero-Overlap Commits)

### 2.1 vLLM `#63638` (`6597ceafbc`): Kimi-K3 Opt-In KV-Balanced FP8 MLA Decode (`KIMI_K3_KV_BALANCED_MLA`)

`vllm/models/kimi_k3/nvidia/kv_balanced_mla/{backend.py,core.py,decode.py,helpers.py}` introduces an SM100/SM103/SM107 FP8 MLA decode backend built around a device-planned split-KV work list:

- **Kernel Parameters & Register / FP8 Prescale Tuning**:
  - Cache stores `LATENT = 512` + `ROPE = 64` = **576 FP8 values/token** in pages of 32 or 64; queries pack into `ROWS = 128` tiles (2-CTA M128 clusters), `KV_TILE = 128`, `CAPACITY = 8` max splits/request, `MIN_SPLIT_TOKENS = 1024` (`min_tiles = 8`), and `OVERHEAD_TILES = 6`.
  - Rebalances warpgroup registers from `(208, 256, 40)` to **`REG_SPLIT = (200, 240, 64)`** to eliminate MMA warpgroup register spills that throttled Tensor Core descriptor arithmetic.
  - Uses **`P_SCALE = (2.0, 6.0)`** (`skip-correction threshold = 2.0`, `P prescale = 6.0` in $\log_2$), keeping max scaled $P = 2^{2+6} = 256 \le 448$ (E4M3 max) while preserving probabilities down to $2^{-(9+6-2)} = 2^{-13}$ of the running row max.
- **Device-Side `[W=1..8, B]` Binary-Search Work-List Planner (`_plan_work_list_kernel`)**:
  - Searches candidate wave budgets $W \in \{1, \dots, 8\}$ (`budget = W * clusters`) via binary search to find the smallest chunk $c \ge 8$ tiles whose total clusters $\sum_b n_b \cdot q_{\text{tiles}}$ fit within `budget`, minimizing:
    $$\text{estimate}(W) = \left\lceil \frac{\sum_b n_b \cdot q_{\text{tiles}}}{\text{clusters}} \right\rceil \times \left(\max_b \text{per}_b + \text{OVERHEAD\_TILES}\right)$$
  - Encodes work-list entries **split-major** as `((q_tile * num_reqs + req_idx) << 8) | split` so adjacent query tiles of a request share the split's KV pages through L2.
  - Plans once on the first MLA layer per forward (`_PLANNED`) and reuses the device buffer across all subsequent layers inside a captured CUDA Graph.

### 2.2 SGLang `#43449` (`e732f5d41e`), `#46340` (`eb8c2a40ef`), `#46254`, `#46570`, `#39774`

- **Draft-Free Target Auxiliary Hidden State Capture (`#43449`)**: Adds `--enable-return-aux-hidden-states`. `--speculative-algorithm` (`EAGLE3`, `DFLASH`, or `DSPARK`) then only names the layer capture convention (`aux_hidden_state_capture`) before `handle_aux_hidden_state_capture` resets `speculative_algorithm = None`, avoiding loading any draft model or reserving draft KV cache.
- **AMD MiniMax-M3 Top-K Index Frequency Default (`#46340`)**: Changes `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ` default on ROCm from `2` to `1`, preventing skip layers from reusing stale sparse KV block selections.

### 2.3 vLLM `#63475` (`3090266d88`), `#63828` (`70d67f19dc`), `#45188`, `#55699` & TensorRT-LLM `#20947` / `#20666`

- **`persistent_topk` Radix Barrier CTA Sync (`#63475`)**: Inserts `__syncthreads()` after clearing `next_hist[i] = 0` and before `tx == 0` calls `red_release(&state->arrival_counter, 1)` in `csrc/persistent_topk.cuh`.
- **ROCm Checkpoint Page-Cache Release (`#63828`), `MAX_JOBS` Build Fix (`#45188`), SSL Watcher Shutdown (`#55699`), and TensorRT-LLM `EagerMoeWorkspace` Reclamation & LoRA Ownership (`#20947`, `#20666`)**.

---

## 3. Deterministic Verification Summary

Running `python experiments/vllm_kv_balanced_mla_sglang_aux_phase101/run.py --check` produces:

| Dimension | Metric | Verified Result |
|---|---|---:|
| Kimi-K3 KV-Balanced MLA Planner | 16-request skewed agentic batch (`1K–128K`) tile cycles vs fixed 4-split | **`134` vs `524` tile cycles (`3.9104x` speedup)** |
| Kimi-K3 KV-Balanced MLA Capacity | Max addressable requests at $H=128, q=2$ (`q_tiles=2`) | **`1,024` concurrent requests** |
| SGLang `--enable-return-aux-hidden-states` | Draft weights + KV VRAM saved (EAGLE3 / DFLASH) | **`3,920.19 MiB` / `5,481.23 MiB`** |
| vLLM `persistent_topk` Barrier Fix | Unsynchronized warp exposure before / after `__syncthreads()` | **`87.5%` (7/8 warps) $\to$ `0.0%`** |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Int32 Partial-Buffer Addressing Limit**: Given `MAX_TILES = (1 << 31) // (128 * 8 * 512 * 2) = 2048`, compute `query_tiles(128, 4)` and `max_planned_requests(128, 4)` when verifying a 4-token DSpark speculative draft block.
2. **FP8 E4M3 Prescale Saturation Boundary**: Why must $2^{\text{threshold} + \text{prescale}} \le 448$ in `P_SCALE = (2.0, 6.0)`, and what happens to attention weights if `prescale` is raised to `7.0` when the running row max lags by $2^2$?
