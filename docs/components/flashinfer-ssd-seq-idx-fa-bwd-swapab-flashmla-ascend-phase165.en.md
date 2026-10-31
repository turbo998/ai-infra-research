# FlashInfer SM107a Mamba2 SSD Single-Sequence Varlen `seq_idx` Check, FlashAttention CuTe SM90 Swapped Bwd MMA Fix & FlashMLA Ascend 950 Token-Interleaved KV (Phase 165)

> **中文版**: [flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.md](flashinfer-ssd-seq-idx-fa-bwd-swapab-flashmla-ascend-phase165.md)

## 1. Primary Source Audit Scope (Track 1 · 11 Zero-Overlap Commits)

- **FlashInfer**: `#8132` (`6ccf040752`, 2026-10-31), `#8025` (`e7ee9232eb`, 2026-10-30), `#7993` (`ed68b15bef`, 2026-10-30), `#7970` (`a111133e0b`, 2026-10-30)
- **FlashAttention**: `#2888` (`3c2c6b31e7`, 2026-09-14), `#2867` (`8d3a3b80d4`, 2026-09-13)
- **CUTLASS**: `be1fcc750f` (2026-08-05)
- **Triton**: `#12356` (`d1e7340cd5`, 2026-10-27), `#12471` (`5c36ec57d1`, 2026-10-23)
- **FlashMLA**: `#229` (`2e5429fc56`, 2026-09-30), `063a9f05aa` (2026-09-14)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **FlashInfer `cake_ssd_combined` SM107a Single-Sequence Varlen `seq_idx` Check & KDA Lazy-Upload Plan Parity (`#8132`, `#7970`)**:
   - Adds an `sm_107a` `seq_idx` check program across `cake_mamba_ssd_combined`'s 56 generated prefill programs to bypass per-chunk sequence boundary scans when a varlen invocation contains a single sequence, cutting latency from `20.40 us` to `18.25 us` (**-10.54%**), while fixing GDN/KDA KCP plan-parity verification after lazy KDA table upload (`#7970`) and staged MoE FFI signature checks (`#8025`).
2. **FlashAttention CuTe SM90 Backward `dKV_swapAB` WGMMA M-Mode Fix & 2-Stage `hdim256` (`#2888`)**:
   - Fixes `FlashAttentionBackwardPostprocess` reversing the entire `(tile / atom_layout)` tiler under `swap_ab` (which requested `M = 192` instead of the required WGMMA atom `M = 64`), restoring `M = 64` with a 2-stage `hdim256` schedule and reducing GQA/MQA postprocess compile failures from `4` to **0**.
3. **FlashMLA Ascend 950 (`dav-3510`) & SM100a/SM103a Token-Interleaved FP8/FP4 KV Cache (`#229`, `063a9f05aa`)**:
   - Releases DeepSeek-V4.1 kernels targeting `sm_100a`, `sm_103a`, and Huawei Ascend 950 (`dav-3510`) with token-interleaved FP8/FP4 KV cache layout (storing each token's quantized payload immediately followed by its scales), consolidating per-token memory streams from `2` to **1** and resolving CUDA 13.0 compilation errors.

## 3. Deterministic Reproduction

```bash
python experiments/flashinfer_ssd_seq_idx_fa_bwd_swapab_flashmla_ascend_phase165/run.py --check
```
