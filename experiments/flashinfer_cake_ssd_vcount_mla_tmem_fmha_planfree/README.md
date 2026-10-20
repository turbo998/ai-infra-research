# FlashInfer Cake SSD Look-Back VCount Cap, Blackwell MLA TMEM Layout-B Score Addressing & FMHA Plan-Free Static One-Wave Decode

## Overview / 概述

- **English**: Deterministic Python 3.12 standard-library verification model (`own-offline`) for three upstream FlashInfer commits landed on 2026-10-20:
  1. `flashinfer-ai/flashinfer@4af20723d2` (`#6884`): Capping `_chunk_parallel_vcount(capability, sm_count)` at `min(table_vcount, sm_count)` so 208-SM Rubin R200 (`sm_107a`) boards bind `V = 208` instead of `212`, avoiding a second wave of ranges on 4 CTAs (`1.345x–1.641x` warm and `1.512x–1.639x` cold speedups).
  2. `flashinfer-ai/flashinfer@bc45bb9878` (`#6590`): Fixing Blackwell `cta_group::2` M128 N128 TMEM Layout-B score addressing in 9 Cake MLA decode kernels (`s_base = taddr + s_off + (((warp % 4) * 32) << 16)`), eliminating the erroneous `n_half * 64` column offset that routed softmax warps 2 & 3 (`keys 64..127`) into the neighboring phase slot (`max abs error` reduced from `3.2e-2` / `12.2%` mismatch to `3.1e-4` on GB300 at `input_scale=0.5`).
  3. `flashinfer-ai/flashinfer@fda1694904` (`#6119`): Adding Program 2 (`static_one_wave`, plan-free static ticket assignment `ticket = CTA index`) to `cake_fmha` E4M3 `head_dim=128` balanced DCP decode (`dcp_spec_bf16_fp8_balanced`), cutting CUDA-graph latency on `prod_b1_s8192_q4_cp4_graph` from `12.58 us` to `10.85 us` (`1.159x`) on B200 and `11.87 us` (`1.059x`) on GB300.
- **中文**：针对 2026-10-20 合入的 3 项 FlashInfer 一手提交的 Python 3.12 标准库确定性建模与校验（`own-offline`）：涵盖 Mamba2 Cake SSD Look-Back 虚拟 CTA 范围上限 `_chunk_parallel_vcount`、Blackwell `cta_group::2` MLA TMEM Layout-B 1/4 通道分数寻址修复，以及 `cake_fmha` E4M3 `head_dim=128` DCP 投机解码免规划静态单波次（Program 2）选择器。

## Run / 运行方式

```bash
python experiments/flashinfer_cake_ssd_vcount_mla_tmem_fmha_planfree/run.py --check
```
