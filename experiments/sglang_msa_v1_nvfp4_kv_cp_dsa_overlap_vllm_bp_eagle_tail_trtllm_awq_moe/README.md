# SGLang MSA-v1 NVFP4 KV Pool & CP DSA Indexer Overlap, vLLM Secondary-Tier Backpressure & EAGLE/MTP Partial-Tail Offload, and TensorRT-LLM MegaMoE NVFP4_AWQ Pre-Quant Scale

Deterministic Python 3.12 standard-library model auditing six first-hand inference engine commits (`sgl-project/sglang` `#46141`, `#46139`, `#46140`, `#46179`, `vllm-project/vllm` `#61780`, `#61019`, and `NVIDIA/TensorRT-LLM` `#20823`):

1. **SGLang MSA-v1 Sparse Attention, NVFP4 KV Pool & MiniMax-M3.1 NVFP4 MoE (`#46141`, `#46139`, `#46140`)**: Models NVFP4 KV storage (`kNvfp4BlockSize = 16`, 8 bytes E2M1 payload + 1 byte E4M3 scale = `9 B` per 16 elements vs `32 B` BF16 -> `3.56x` KV footprint compression), `rcp.approx.ftz.f32` tie-exact reciprocal quantization (`0.1%` nibble drift avoided vs IEEE divide), and 2-CTA `page_row_copy` (`kPageRowLanes = 64`, `kPageRowBlockQuota = 2`, leaving `146/148` B200 SMs for compute).
2. **SGLang DSA Context-Parallel KV Gather Overlap with Indexer (`#46179`)**: Models `run_indexer_with_kv_gather` in `interleave_bcg.py`, hiding CP KV all-gather latency (`185 us` at `CP=4`, `8192` tokens) behind `dsa_indexer` top-k computation (`240 us`), cutting combined step latency from `425 us` to `248 us` (`1.71x` speedup).
3. **vLLM Tiered KV Offload Failure-Induced Backpressure (`#61780`)**: Models `EMABackpressureDetector.on_store_failed()` (`DEFAULT_FAILURE_THRESHOLD = 3`, `_FAILURE_EMA_FACTOR = 4.0 * HIGH_WATER`) coupled with `ThrottledDropPolicy(ramp_factor=2.0)`, halting `100%` of futile `ENOSPC`/`EDQUOT` write storms after 3 consecutive failures until idle decay probes recovery.
4. **vLLM `OffloadingConnector` Partial-Tail Serving & Storing with EAGLE/MTP (`#61019`)**: Models `supports_partial_tail` with `eagle_proof_margin`, recovering unaligned tail KV blocks (`cache_hit_alignment_tokens = 4` inside `block_size = 64`) for EAGLE/MTP hybrid models (`Qwen3-Next MTP`), lifting prefix cache hit rate on unaligned multi-turn prompts from `75.0%` to `96.88%`.
5. **TensorRT-LLM MegaMoE CuteDSL `NVFP4_AWQ` Pre-Quant Scale (`#20823`)**: Models `x = x * fc31_act_scale` before FP4 activation quantization or `w4a16` dequantized execution (`W / s * (x * s) = W * x`), eliminating `14.8x` relative L2 quantization error caused by unscaled outlier channels.

## Usage

```bash
python experiments/sglang_msa_v1_nvfp4_kv_cp_dsa_overlap_vllm_bp_eagle_tail_trtllm_awq_moe/run.py --check
```
