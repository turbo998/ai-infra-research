# FlashInfer Hopper Cake VSA SM90 Sage-FP8 40-Stage Sparse Attention, KDA KCP `wave_fill1_p1tie` Wave Planner & Lean Arena, 64-Head DSA On-Device `-1` Hole Compaction, and Causal Conv1d+SiLU 6-Route Training

> **Phase 56 (`priority: 65`) · Primary Source & Commit-Level Deep Dive**
> - `flashinfer-ai/flashinfer@8ec9464ce841bb75521303f827f41838bb28044e` (`#7092`)
> - `flashinfer-ai/flashinfer@baddd518c5433b39f21f9937e5888eb6212ac710` (`#7097`)
> - `flashinfer-ai/flashinfer@fb23d9702ce9e6a32506ec5329346894127a7a3e` (`#7084`)
> - `flashinfer-ai/flashinfer@462a48a4cd0fd91a98bc7dfdc218824c6da9f05e` (`#7098`)
> - `flashinfer-ai/flashinfer@69124ac26bd658fdba5e581ef51d0b5a55521303` (`#7089`)
> - `flashinfer-ai/flashinfer@8e8342f909352ec8cfdd351eb4759b5c8b5e351e` (`#6466`)
> - `flashinfer-ai/flashinfer@562917330df30b7a6ff79bbbd6a2d0baae749d80` (`#7086`)

---

## 1. Hopper (`sm_90a`) Cake VSA SM90 Sage-FP8 40-Stage Variable Block-Sparse Attention (`#7092`)

On Hopper H100 (`sm_90a`), `cake_vsa_sm90` adds 19 FP8 twin stages (`vsa_fwd_k64_d2_fp8_sm90`, `vsa_fwd_k64_d3_fp8_sm90`, `vsa_fwd_k128_fp8_sm90`, `vsa_bwd_dq_fp8_sm90`, `vsa_bwd_dkv_*_fp8_sm90`, etc.) and 2 quantizer kernels (`vsa_q8_sm90`, `vsa_kv8_sm90`), expanding `MODULES` in `cake_jit.py` to **40 total stages**.

- **Sage-FP8 E4M3 Quantization Contract (`_FP8_MAX = 448.0`, `_K_GROUP = 16`)**:
  - `q_scale` has shape `(1, H, Sq)` (`amax(dim=-1) / 448.0` per query token);
  - `k_scale` has shape `(1, H, ceil(Sk / 16))` (`amax(dim=(-1, -2)) / 448.0` per 16-token key group, folded with `sm_scale` on load);
  - `v_scale` has shape `(H, 128)` (`amax(dim=-2) / 448.0` per channel, applied once at epilogue store);
  - Both $Q K^T$ and $P V$ execute on Hopper WGMMA E4M3, shifting online softmax row maxima so $P \in [0, 1]$ maps into $[0, 448]$ before E4M3 rounding.
- **Forward Dispatch & SMEM Footprint**: Dispatches `k128_fp8` (`163,840 B` SMEM) when `kv_tile == 128`, 2-stage `k64_d2_fp8` (`114,688 B` SMEM) when `kv_tile == 64`, `active <= 18`, and `q_blocks >= 32`, and 3-stage `k64_d3_fp8` (`147,456 B` SMEM) otherwise.
- **Author H100 80GB HBM3 Benchmark (`author-benchmark`, cold L2, 137 shapes)**: Achieves geomean speedups of `1.0868x` (`b20`), `1.1537x` (`ext33`), `1.2238x` (`real39`), and `1.1542x` (`long45`), with overall geomean **1.1632x** (procedure A, 136/137 wins) and **1.1602x** (procedure B, 137/137 wins), peaking at **1.6698x** on `h32-m512-n8192-k16` (`0.0680 ms` vs `0.1135 ms`).

---

## 2. Cake Packed KDA Context-Parallel (`KCP`) `wave_fill1_p1tie` Planner & `KdaArena` Lean Workspace (`#7097`, `#7084`)

- **8-Stage Direct Launch & `wave_fill1_p1tie` Segment Rule (`#7097`)**:
  - Expands `PROGRAMS` from 6 to **8 launches** by adding `kcp_fwd_o` (`kcp_sm_bounded`) and `kcp_dav` (`kcp_pairs`: `(num_chunks_pad / 2, num_heads, 1)`).
  - Default segment rule `P_RULE_DEFAULT = "wave_fill1_p1tie"` (`cake_kcp_plan.py`):
    1. When `n_frag <= 3` and `NC_max_frag >= 257` (`P_chain >= 2`), minimizes the equal-segment wave cost $\lceil n_{\text{frag}} P H / \text{SM}\rceil \lceil \text{NC}_{\max}/P\rceil t_{\text{walk}} + (P-1)t_{\text{compose}}$.
    2. When `n_frag in {2, 3}`, `NC_max_frag < 257`, and `n_frag * H <= SM`, evaluates Longest-Processing-Time (LPT) makespan across $\lfloor \text{SM}/H \rfloor$ slots and picks **`P = 1`** on a tie (`ms_p1 <= ms_pb`), eliminating `kcp_affine_merge` and backward state re-walk.
  - **Memory & GB300 Benchmark (`author-benchmark`)**: `kcp_saved_bytes` drops `256 * T_r * H` bytes (`256 MiB` per rank at `T_r=16,384, H=64`). Across all 144 GB300 (`sm_103a`) CP2/4/8/16 configurations, Cake KCP achieves **1.4640x–2.7942x** speedup over FLA KCP.
- **`KdaArena(lean=True, tail_passes=4)` (`#7084`)**: Batches the 3 chunk-parallel backward tail stages (`fused_grad`, `glm_intra`, `gate_epilogue`) into `B = 4` passes and aliases `v_new`, lowering the backward workspace floor from **4,488 B to 2,820 B per internal token-head** (**37.17% reduction**) with bitwise-identical gradients.

---

## 3. Cake 64-Head DSA On-Device `-1` Hole Compaction & Causal Conv1d+SiLU 6-Route Training (`#7098`, `#7089`, `#6466`, `#7086`)

- **Cake 64-Head DSA Training (`#7098`)**: Compacts valid indices in `gather_kv_indices` to `0 .. k_count-1` with `-1` tail fill on device (`index_prep` with `compact=1, use_pdl=True`) and consumes head-major strided `dout` (`[512, Tq * 512, 1]`) in `bwd_delta` without copy. On GB300 (`sm_103a`), `nope_docs264x992_rank0_fixedtail` (75.8% hole rows) drops from `14.109 ms` to `6.280 ms` (**2.247x**); on GB200 (`sm_100a`), from `13.964 ms` to `6.352 ms` (**2.198x**).
- **Causal Conv1d + SiLU 6-Route (`#7089`) & SM120 FP8 Prefill (`#6466`)**: `cake_causal_conv1d_train` splits into 6 routes (`fwd_dt`, `fwd_bias`, `bwd_dt`, `bwd_bias`, `bwd_fp32g`, `bwd_bias_fp32g`) with channel tiles for `C in {1280, 1536, 2560}`, achieving **2.959x–6.649x** over FLA on GB300; `fmha_v2_prefill_deepseek` on SM120 adds `D=256/256` and `D=128/128` FP8 E4M3-to-BF16 kernels (`1.53x–1.61x` over BF16).

---

## 4. Deterministic Verification

Run `python experiments/flashinfer-vsa-sm90-sage-fp8-kda-kcp-dsa-holefree-conv1d-gdp/run.py --check` to verify all invariants (`summary.json`).
