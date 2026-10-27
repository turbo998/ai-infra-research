# Phase 111 Track A Experiment: FlashInfer Frost W8A16 & Hopper MSA Top-K, FA4 SM100 Sparse MLA Recompute-P, and TileLang Extern Write Analysis

Deterministic Python standard-library model covering:
1. **FlashInfer (`d42c276bd872`, `3d01e3fec63f`, `47e64b0e96c4`, `73c83aac3125`, `2b5a5ffff99a`, `9c4aacf1bfd6`, `0c3d03c7656a`)**:
   - Frost W8A16 grouped MoE Tensor Core schedules across SM90 (`K256`), SM103 (`serial_gather=True`), and SM120 (`group8/stages2/K128`).
   - Hopper `cake_hopper_msa` `msa_topk_select` single-pass exact schedules and `MoEEpCommunication.dispatch` `runtime_max_tokens_per_rank`.
   - SM120 MiniMax-H3 `quant_out_proj` NVFP4 (`quant=1`) routing on 170-SM RTX 5090 (`PROGRAM_1_SMS`), GB300 `cake_dsa_indexer` `sample_shift_rule = [4096, 375]`, SM12x W4A16 `> 2 GiB` static CuTe layout `from_dlpack`, and `selective_state_update` non-multiple-of-64 `head_dim` (`80/96/288`).
2. **FlashAttention-4 (`294cfb3593ed`, `38ddafeb35e5`)**:
   - SM100 Sparse MLA backward in-kernel `recompute-P` (`32 GiB -> 0 GiB` saved `p` at `64K` tokens) + token-chunked `dS` transient (`32 GiB -> 2 GiB` at `chunk=4096`), and SM100 `HD256` varlen grid bounding.
3. **TileLang (`f3ffeaa6b8ee`)**:
   - `AnnotateReadOnlyParams` conservative bare-data-var write tracking for `call_extern`.

## Run
```bash
python experiments/flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111/run.py --check
```
