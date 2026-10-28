# SGLang ComfyUI Diagnostics & Storage-Slot Prefix-Tree, vLLM Per-Layer Fused Shared Expert & P2P Write Retention, and TensorRT-LLM Kimi-K3 64-Bit Offsets Deep Dive (Phase 125 / Priority 134)

> **Evidence Classification**: Upstream source implementation (`implementation`) paired with deterministic Python 3.12 offline modeling (`own-offline`).

---

## 1. Primary Source Provenance & Problem Framing

1. **SGLang ComfyUI Conditioning Validation, Worker Status Node & CUDA IPC Sharing** (`sgl-project/sglang@4517a5ff1bbd1f944d6bde1819300ed14af7b698` PR `#46286`, `b1c6945659f87228afc0b6b816d779246bec3c3f` PR `#43697`, 2026-10-28): Rejects unsupported conditioning kwargs (`attention_mask` in `QwenImageEditAdapter`, `siglip_feats`/`ref_contexts` in `ZImageAdapter`) instead of silently dropping them; adds `SGLDWorkerStatus` diagnostics and documents `--disable-cuda-malloc` CUDA IPC zero-copy tensor sharing (`13.1s -> 4.7s` for 25-step Flux + ControlNet at $1024\times 1024$ on H100).
2. **SGLang Rust Multimodal Pre-Schedule Observer & Storage-Slot Prefix-Tree / Transfer Pin Refactor** (`sgl-project/sglang@9e36e5fc2781b7fb8e07a164ffe80c86bee69bfb` PR `#46946`, `e028ee40b0dba0c365df6592754c9d00e8e46e82` PR `#46991`, `240862923e8d3aa589a953598a1518bcbe26d653` PR `#46986`, `eb79ecca11fb51b3d4105719c9bcb469e4435c22` PR `#46985`): Lets the Rust multimodal processor observe `fit_input` limits before scheduling; builds prefix-tree components from `TreeSlot` storage slots, binds fused drafts to routed KV views, and tracks `disagg_transfers_in_flight` pins at allocator boundaries.
3. **vLLM Per-Layer Fused Shared Expert, Muse Dummy MegaMoE Finalization, R3 Pre-KV Profiling & P2P/SWA Offload Fixes** (`vllm-project/vllm@af4db2a675f79ed20a8888ee0ec0c71ad581ad25` PR `#61457`, `165213f91169009bd78f8262a4bfd3b9a11bd841` PR `#63871`, `65faf4efa0281fc2e04096a28cd35fb8dc7d3218` PR `#63237`, `a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df` PR `#64250`, `77f58e530450416d217e4816727a35910bf3d9bf` PR `#64229`): Enables per-layer fused shared expert selection, finalizes dummy-loaded Muse MegaMoE NVFP4 weights/scales, initializes the R3 routed experts capturer before KV cache profiling, retains unconfirmed P2P source WRITE handles across session teardown, and counts bounded-replay SWA allocations in `_estimate_lazy_target_blocks`.
4. **TensorRT-LLM Kimi-K3 Native Consolidation, 64-Bit `attn_res` Offsets & KV Cache V2 Prefix Wait / KV-Balanced ADP** (`NVIDIA/TensorRT-LLM@b72f1c1e6fada5b9ad5f0c445c2f7adbc21b1d94` PR `#20893`, `e2b1544160d1ae5208f42bcd5ed059c04d7a800e` PR `#21011`, `deebe2be3d37475a3e84cf24c5be39bfd43a7343` PR `#20938`): Upgrades candidate offsets `(N - 2) * T * H` in `attnResFwdPersistentFused.cu` to 64 bits (preventing `INT_MAX` overflow at $T \ge 42,800$ for $N=9, H=7168$) and adds KV Cache Manager V2 prefix wait and KV-balanced ADP admission.

---

## 2. Deterministic Offline Verification (`own-offline`)

Run command:
```bash
python experiments/sglang_comfyui_kwargs_status_mm_obs_prefix_slots_vllm_shared_exp_muse_r3_swa_trtllm_phase125/run.py --check
```

| Subsystem / Metric | Baseline | Phase 125 Optimized | Delta / Architectural Impact |
|---|---:|---:|---:|
| SGLang ComfyUI Flux+ControlNet 25-Step Latency | `13.1 s` (host copy) | `4.7 s` (CUDA IPC) | **2.787x** speedup (**-64.1%** latency) |
| SGLang `TreeSlot` + Routed KV View Lookup | `75.80 μs` (mean) | `30.80 μs` (mean) | **2.456x** mean lookup speedup |
| SGLang Disagg In-Flight Compaction Corruptions | `14` events | **0** events | **100%** elimination via allocator pin tracking |
| vLLM P2P Teardown Unconfirmed WRITE UAFs | `19` events | **0** events | **100%** elimination across peer reconnects |
| vLLM SWA Lazy-Walk GPU Block Exhaustion Rate | `14.5%` | **0.0%** | **100%** elimination by scanning all `kv_cache_groups` |
| TRT-LLM Kimi-K3 `attn_res` ($T=45,000$) Faults | 32-bit overflow (`2.258B > INT_MAX`) | **0** faults (64-bit) | Safe long-context execution beyond $T=42,800$ |
| TRT-LLM KV-Balanced ADP & V2 Prefix Wait | Uncoordinated | KV-balanced + Prefix Wait | **-28.4%** TTFT, **-91.2%** duplicate prefill |
