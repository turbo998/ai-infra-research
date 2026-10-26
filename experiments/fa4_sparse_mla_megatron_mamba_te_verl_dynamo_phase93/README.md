# Phase 93 Track 3: FlashAttention-4 SM100 Sparse MLA Training, Megatron Mamba2, TE MXFP8/NVFP4, verl FSDP2 & Dynamo

Deterministic Python 3.12 standard-library model reproducing the training, compiler, and KV-routing architectures in Phase 93 Track 3:

1. **FlashAttention-4 SM100 Sparse MLA Training (`71dbd9372d`, `93c9acac10`)**:
   - Saves the BF16 output rounding residual `o_lo = fp32(O) - bf16(O)` so backward preprocess computes `dPsum = rowsum(dO * (out + o_lo))`, eliminating catastrophic cancellation on peaked rows and reducing `dQ/dK` rel-L2 error at 97% self-attention weight by `15.83x` (`3.80% -> 0.24%`).
   - Uses `rescale_threshold = 0.0` (exact running softmax max) for sparse training forwards instead of `8.0`, eliminating the coherent `~2^-9` row gain error (`1.23e-3 -> 1.40e-4`, `8.79x` reduction) and index-order sensitivity.
2. **Megatron-LM Batch-Invariant Mamba2 Row Kernels (`#8221`, `aa46a57229`)**: Eliminates the `z` replay buffer (`128 MiB` saved at `B=64, C=256, H=64, D=64`) and computes only the target 16-row MMA tile per decode step (`16x` chunk-scan row reduction).
3. **TransformerEngine Single-Group NVFP4 Views (`#3662`, `2e959f352c`) & Per-Tile MXFP8 Scale Swizzle (`#3658`, `33e7ff3160`)**: Expands MXFP8 scale swizzle for a 4096-row activation from 1 CTA (`32` tiles/CTA) to `32` parallel CTAs (`1` CTA per 128-row scale tile).
4. **TorchTitan Varlen Flash Priority (`#5152`, `094c884685`), verl FSDP2 Blocking CPU Offload (`#8187`, `edd2c63b6a`), LMCache SWA Prefix Replay (`#5976`, `a0ff470128`), and Dynamo vLLM `logit_bias` (`#16303`, `4b487d890d`) & KV-Router Lower-Tier Edge Slot Compression (`#17143`, `3acf248073`, `113B -> 33B`, `-70.80%`)**.
