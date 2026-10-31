# SGLang Joint-LRU Unified KV Eviction & Rust Attention-DP Abort Fan-Out, vLLM NIXL 1.5.2 & TRT-LLM Owning Fragment Deep Dive (Phase 161)

> **Evidence Classification**: Primary upstream commit analysis (`implementation`: `a65c9c596905`, `60b183d5c3ce`, `2410a09c6e1a`, `78161c28da3b`, `886b845ef14c`, `06f36c122297`, `2d1a7d170eb8`, `3302a317466d`, `f6aa38fe06cd`, `93eb6eb23812`) + deterministic Python 3.12 standard-library model (`own-offline`: `experiments/sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161/run.py`).

## 1. Core Architectural Findings & Implementation Paths

1. **SGLang Unified-Memory Joint-LRU Shared Eviction (`a65c9c596905`) & DeepSeek V4 Buffer Info Pool Descriptors (`2410a09c6e1a`)**:
   - Unifies Full-Attention and Sliding-Window-Attention (SWA) KV eviction into a single `joint_lru` ordering (`74.5% -> 86.2%` prefix hit rate, `+11.70 pts` gain in `own-offline` modeling) and formalizes DeepSeek V4 device pool buffer metadata.
2. **SGLang Rust-Server Attention-DP Abort Broadcast & Listener Map (`60b183d5c3ce`)**:
   - Publishes the attention-DP listener map and fans out client aborts across all `16` DP ranks (`15 -> 0` orphaned DP workers, `1.85 ms` fan-out latency).
3. **vLLM CohereASR FP16 Bias Fix (`78161c28da3b`), Canary-1B-v2 (`886b845ef14c`), NIXL 1.5.2 (`06f36c122297`), TRT-LLM Owning Fragment (`2d1a7d170eb8`) & DeepEP HTTPS Submodule (`f6aa38fe06cd`)**:
   - Fixes FP16 attention bias dtype mismatches in CohereASR (`0` errors) and returns an owning fragment from TRT-LLM `util_convert_type` (`0` dangling views).

## 2. Deterministic Verification

```bash
python experiments/sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161/run.py --check
```
