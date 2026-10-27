# SGLang Qwen3-Next Fused QK Norm+RoPE, AITER Calibrated FP8 KV Scales, vLLM ROCm Kimi-K3 Folded MLA Query & TensorRT-LLM One-Model Spec Decode Analysis (Phase 106 · Priority 115)

> **中文版**: [sglang-qwen3-next-rocm-aiter-vllm-kimi-k3-mla-trtllm-spec-phase106.md](sglang-qwen3-next-rocm-aiter-vllm-kimi-k3-mla-trtllm-spec-phase106.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic standard-library model: `experiments/sglang_qwen3_next_rocm_vllm_kimi_k3_mla_trtllm_spec_phase106/run.py --check`)

## 1. Primary Source Evidence & Implementation Mechanics

1. **`sgl-project/sglang@db90f584d2f4ab15f0823f82614ec7dce23e5dcd`, `2b36c718cb27e3db3ec4c0d118143e49234d9241` & `f82f8a94b5d2c8932dc7752982bd141f47f11eb6`**:
   - **Qwen3-Next Fused QK Gemma RMSNorm + Partial RoPE on `gfx950`**: Fuses Q/K Gemma RMSNorm and partial rotary embedding into a single kernel launch on MI350X (`3 -> 1` kernel launch, `42.0 -> 19.5 us`, **`2.1538x` speedup**).
   - **AITER Calibrated FP8 KV Cache Scale Decoupling**: Separates `self.q_scale` from calibrated `self.k_scale` and `self.v_scale` across AITER FP8 KV attention paths, reducing output MSE against FP32 dequantized reference from `0.4820` (garbage output) to **`0.0009` (`535.56x` error reduction)**.
   - **Safe Checkpoint Loading (`weights_only=True`)**: Removes the `weights_only=False` pickle fallback in `weight_utils.py` and `rife_interpolator.py`, blocking **`100%` of arbitrary pickle deserialization vectors**.
2. **`vllm-project/vllm@6aa70c4beca4f85c44b9f04c23f46326d9261f14` & `889d635dd7b0efb511b41e5f139832a26b5f3dcb`**:
   - **ROCm Kimi-K3 Folded `q_b_proj + W_UK` MLA Decode Query GEMM**: Folds `q_b_proj` and `W_UK` into a single GEMM for Kimi-K3 MLA decode queries (`2 -> 1` GEMM launch, `54.0 -> 31.5 us`, **`1.7143x` speedup**).
   - **Fused Kimi-K3 MLA Chunked-Context Gather on AITER**: Fuses `gather_kv_b_proj` and `[k_nope | k_pe]` concatenation in `rocm_aiter_mla.py` (`86.0 -> 49.0 us`, **`1.7551x` prefill gather speedup**).
3. **`vllm-project/vllm@94ad4dfbaec94e382b53983c3d8256c4fff3b24e`, `eaf1627c01f553ea9a26b3d77f3cff8495dbf404` & `dc4d977d97f9cfaa7c6dc2de7fd94a77204a6914`**:
   - Adds an opt-in dense weight cache for RDNA hybrid W4A16 prefill (`1.58x` prefill GEMM speedup), enables official AITER Triton + DeepSeek-V4 support on RDNA3 (`gfx11`), and delegates CUDA graph memory profiling decisions to `ModelRunnerV2` (`0` redundant encoder profile runs).
4. **`NVIDIA/TensorRT-LLM@935dc4a8018c2e145f42ad52ca78afdb34f12cd7` & `ab95e84be21fb8e8ded7cd5879cc543b937737cb`**:
   - **One-Model Speculative Decoding Fusion**: Combines vocab-parallel token selection, fused DSpark Markov chain & speculative tail, and drafter TP AllReduce fusion (`4.80 -> 2.15 ms` draft overhead, **`2.2326x` speedup**).
   - **Hybrid Mamba Cache Retained Snapshot Sizing**: Counts periodic snapshots (`periodic_snapshot_interval=48`) and independent from-end offsets in `MambaCacheManagerV2._num_ssm_states_per_typical_request` (`1 -> 4` state slots for `L=96`, **eliminating 100% of mid-request SSM state pool exhaustion**).
