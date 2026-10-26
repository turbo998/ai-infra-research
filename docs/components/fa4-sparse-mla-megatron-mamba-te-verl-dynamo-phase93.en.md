# Phase 93 Track 3: FlashAttention-4 SM100 Sparse MLA Training Accuracy, Megatron-LM Batch-Invariant Mamba2, TE MXFP8/NVFP4, verl FSDP2 Offload & Dynamo KV-Router Slot Compression

> **中文版 / Chinese Version**: [`fa4-sparse-mla-megatron-mamba-te-verl-dynamo-phase93.md`](fa4-sparse-mla-megatron-mamba-te-verl-dynamo-phase93.md)  
> **Reproducible Experiment**: [`experiments/fa4_sparse_mla_megatron_mamba_te_verl_dynamo_phase93/`](../../experiments/fa4_sparse_mla_megatron_mamba_te_verl_dynamo_phase93/)  
> **Upstream Commits Covered (`2026-09-23` – `2026-10-26`)**:
> - `Dao-AILab/flash-attention@71dbd9372d`: `[CuTe, SM100] Sparse MLA training: dpsum from an O residual`
> - `Dao-AILab/flash-attention@93c9acac10`: `[CuTe, SM100] Sparse MLA training fwd: exact running softmax max`
> - `NVIDIA/Megatron-LM@aa46a57229` (`#8221`): `fix(ssm): batch-invariant Mamba decode without a z replay buffer`
> - `NVIDIA/TransformerEngine@2e959f352c` (`#3662`): `[PyTorch] Fix NVFP4 weight views of the single-group fused grouped MLP`
> - `NVIDIA/TransformerEngine@33e7ff3160` (`#3658`): `[Common] Swizzle MXFP8 scaling factors with one CTA per scale tile`
> - `pytorch/torchtitan@094c884685` (`#5152`): `Varlen attention backend falls back to CuDNN even when FA is installed.`
> - `volcengine/verl@edd2c63b6a` (`#8187`): `[fsdp] fix: prevent parameter corruption during FSDP2 CPU offload`
> - `LMCache/LMCache@a0ff470128` (`#5976`): `[Fix][MP] Keep a full-prompt hit inside vLLM's allocation under SWA prefix replay`
> - `ai-dynamo/dynamo@4b487d890d` (`#16303`): `fix(llm): deliver OpenAI logit_bias to the vLLM backend instead of silently dropping it`
> - `ai-dynamo/dynamo@3acf248073` (`#17143`): `perf(kv-router): shrink lower-tier edge slots from 113 to 33 bytes`

---

## 1. Architectural Overview

Across large-scale sparse-attention training, hybrid SSM decoding, RLHF parameter offloading, and distributed KV routing, Phase 93 Track 3 isolates 10 critical upstream commits addressing numerical cancellation, asynchronous DMA races, and index memory footprints:

1. **FlashAttention-4 SM100 Sparse MLA Training Accuracy (`71dbd9372d`, `93c9acac10`)**:
   - **Saving the `O` Rounding Residual (`o_lo = fp32(O) - bf16(O)`)**: In backward preprocess, $dS = P \odot (dP - dP_{\text{sum}})$ where $dP_{\text{sum}} = \text{rowsum}(dO \odot O)$. On peaked attention rows ($dP \approx dP_{\text{sum}}$ on the dominant key), coherent BF16 rounding of $O$ survives cancellation and causes `0.89%` (`80%` self weight, with `~1%` of rows at `~100%` error) to `3.80%` (`97%` self weight) `dQ/dK` rel-L2 error vs FP64. Writing `o_lo` during forward and computing `dpsum = rowsum(dO * (out + o_lo))` drops `dQ/dK` rel-L2 error to **`0.24%`** (**15.83x reduction** at 97% self weight).
   - **Exact Running Softmax Max (`rescale_threshold = 0.0`)**: Because the SM100 sparse MLA kernel walks gathered index blocks from last to first, top-k slot 0 (the strongest key) arrives last; lazy `rescale_threshold = 8.0` leaves a non-integer $\delta$ in $p = \exp_2(\delta)$ whose BF16 rounding injects a coherent $\sim 2^{-9}$ row gain error. Using `rescale_threshold = 0.0` for sparse training forwards reduces coherent row gain error by **8.79x** (`1.23e-3 -> 1.40e-4`) and removes index-order dependence.
2. **Megatron-LM Batch-Invariant Mamba2 Decode Without a `z` Replay Buffer (`#8221`, `aa46a57229`)**:
   - Introduces `_chunk_scan_row_kernel` and row-level state kernels in `batch_invariant_decode_rows.py` that compute only the target 16-row Tensor Core MMA tile per step (**16x fewer rows** than a 256-row chunk tile), commit boundary states only for full chunks ($1/256$), and broadcast the target row's `z` gate with row stride 0—eliminating the **128.0 MiB** `z` replay buffer (`B=64, C=256, H=64, D=64`).
3. **TransformerEngine Single-Group NVFP4 Weight Views (`#3662`, `2e959f352c`) & Per-Tile MXFP8 Scale Swizzle (`#3658`, `33e7ff3160`)**:
   - Fixes single-group fused grouped MLP forward FC1 scale view (`ceil_div(fc1_weight_k, k_sf_divisor)` using packed bytes instead of logical K) and backward FC2 dgrad transposed columnwise shape `(1, in_features, out_features // 2)`.
   - Adds `NVTE_MXFP8_PER_TILE_SWIZZLE=1` kernels that assign one CTA per 128-row (row-wise) or `128x32` (column-wise) scale tile instead of packing 32 tiles into a single CTA (**32x CTA parallelism** on a 4096-row activation).
4. **TorchTitan Varlen Flash Priority (`#5152`, `094c884685`) & verl FSDP2 Blocking CPU Offload (`#8187`, `edd2c63b6a`)**:
   - Enforces `sdpa_kernel(self.sdpa_backends, set_priority=True)` in TorchTitan `VarlenInnerAttention` to prevent PyTorch's default `[CUDNN, FLASH]` order in `varlen_attn` from triggering cuDNN backward OOM on packed batches.
   - Replaces `non_blocking=True` with blocking D2H transfers in verl's `offload_fsdp2_model_to_cpu` and `fsdp2_load_full_state_dict`, preventing `FSDPParam.reset_sharded_param` repadding on uneven shards from racing with in-flight DMA and silently corrupting parameters.
5. **LMCache SWA Prefix-Replay Alignment (`#5976`, `a0ff470128`) & Dynamo `logit_bias` (`#16303`, `4b487d890d`) + KV-Router Edge Slot Compression (`#17143`, `3acf248073`)**:
   - Aligns LMCache full-prompt external hits to vLLM's block-floored allocation under sliding-window `prefix_replay_tokens`.
   - Forwards OpenAI `logit_bias` to vLLM while raising `InvalidArgument` on TensorRT-LLM in Dynamo.
   - Compresses Dynamo KV-Router lower-tier edge slots from **113 bytes to 33 bytes (-70.80%)** by storing compact `OwnerId` handles and XOR fingerprints instead of full `IndexedResidencyOwner` structs (`1,077.65 MiB -> 314.71 MiB` across 10M edges).
