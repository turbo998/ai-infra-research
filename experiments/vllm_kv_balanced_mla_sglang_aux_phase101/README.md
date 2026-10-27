# vLLM Kimi-K3 KV-Balanced FP8 MLA Decode, SGLang Draft-Free Auxiliary Hidden Capture & Persistent Top-K Barrier (`Phase 101`)

Deterministic Python 3.12 standard-library model reproducing:
1. **vLLM Kimi-K3 KV-Balanced FP8 MLA Decode Planner** (`6597ceafbc`, PR `#63638`): exact reference implementation of `_plan_work_list_kernel` from `vllm/models/kimi_k3/nvidia/kv_balanced_mla/core.py` (`LATENT=512`, `ROPE=64`, `KV_TILE=128`, `ROWS=128`, `CAPACITY=8`, `MAX_WAVES=8`, `MIN_SPLIT_TOKENS=1024`, `OVERHEAD_TILES=6`, `REG_SPLIT=(200,240,64)`, `P_SCALE=(2.0,6.0)`), comparing `[W, B]` binary-search split-KV planning against fixed 4-split allocation on skewed agentic contexts (`1K..128K`).
2. **SGLang Draft-Free Target Auxiliary Hidden State Capture** (`e732f5d41e`, PR `#43449`): models `--enable-return-aux-hidden-states` across EAGLE3, DFLASH, and DSPARK conventions while resetting `speculative_algorithm = None` to avoid loading dormant draft model weights and KV cache.
3. **vLLM `persistent_topk` Radix Barrier CTA Synchronization** (`3090266d88`, PR `#63475`): models the `__syncthreads()` barrier inserted between parallel `next_hist[i] = 0` clearing and `tx == 0` `red_release(&state->arrival_counter, 1)`.

## Usage
```bash
python experiments/vllm_kv_balanced_mla_sglang_aux_phase101/run.py --check
```
