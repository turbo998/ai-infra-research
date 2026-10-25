# FlashInfer SM100 MegaMoE Flag-in-Data (`<=32` Rows), SM120 QSA NVFP4 Decode, PCIe CE Ring All-Reduce, Kimi-K3 LatentMoE Evict-First & DeepGEMM Mega-mHC (`h=4096`)

> **中文版本**: [flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096.md](flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096.md)  
> **Evidence ID**: `ev-flashinfer-mega-moe-ll32-qsa-nvfp4-sm120-pcie-ce-ring-kimi-k3-mhc-h4096-2026`  
> **Experiment**: [`experiments/flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096/run.py`](../../experiments/flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096/run.py)

---

## 1. First-Hand Source Audit Scope

This document audits five first-hand upstream commits landed in `flashinfer-ai/flashinfer` on 2026-10-25 (`#7348`, `#7313`, `#7301`, `#7312`, `#7321`), covering expert-parallel MoE communication, sparse NVFP4 GQA decode on SM120, PCIe Copy-Engine ring collectives, L2 eviction policies for LatentMoE, and persistent hyper-connection (mHC) fusion on Blackwell (`sm_100a`/`sm_103a`/`sm_120a`):

1. **SM100 BF16 Rank-Major EP8 MegaMoE Flag-in-Data (`<=32` Rows) & Pushed Route Codes (`#7348`, `d4d4bd06d1`)**: Eliminates inter-rank staging barriers for small-batch decode (`T <= 32`) and drops the standalone route kernel for `32 < T <= 128`.
2. **SM120 Cake QSA NVFP4 Sparse Paged GQA Decode (`#7313`, `09a77503f0`)**: Delivers `mma.sync` plain, cooperative fused, and split-KV merge kernels for `sm_120a` (RTX PRO 6000 Blackwell, 188 SMs, which lacks `tcgen05` and TMEM).
3. **SM120 Pipelined Topology-Weighted PCIe CE Ring All-Reduce (`#7301`, `68e3e09ad3`)**: Introduces `IpcVariant.COPY_ENGINE_RING_PIPED` with stream memory operations and `ce_slow_hops` dual-socket topology weighting for chunks `>= 4 MiB/rank`.
4. **SM100/SM103 Kimi-K3 Stable LatentMoE Evict-First Decode & Resident TP1 Tail GEMM (`#7312`, `1c0d2be434`)**: Applies `a_evict_first` L2 policy to streamed weights for `T <= 128` so activations remain L2-resident, and adds a claimed-item resident `256x512` fused-RMSNorm tail GEMM for `T >= 1024`.
5. **SM100/SM103 DeepGEMM Mega-mHC (`h=4096`, 4-Route, 20 Sinkhorn Iters) (`#7321`, `847172573a`)**: Fuses GLM-5.3-Flash post-mixing, BF16x3 split-K projection, 20 Sinkhorn iterations, route collapse, and input RMSNorm into a single persistent kernel launch.

---

## 2. Kernel Implementation Paths & Data Movement

### 2.1 SM100 MegaMoE BF16 Rank-Major EP8 Flag-in-Data (`_LL_MAX_T = 32`) (`#7348`)

In `flashinfer/moe_ep/kernel_src/blackwell_bf16_rank_major/session.py` and `src/flashinfer_blackwell_moe_ep_layer_sm100.cu` (`H=7168, I=2048, E=256`, `top_k=8`, `EP=8`, `1 <= T <= 128`):
- **Low-Latency Flag-in-Data Path (`T <= _LL_MAX_T = 32`)**: Senders write 8-byte words (`4-byte payload + 4-byte epoch`) directly into the owner rank's receive bank without staging barriers. Receivers poll these 8-byte words across two alternating parity banks (`_LL_BANKS = 2`) ordered by a system-scope release/acquire pair. The full forward pass launches only **6 kernels**, and `stage_inputs` immediately dispatches rows once weights are bound.
- **Pushed Route Codes (`32 < T <= 128`)**: The `input-arrive` kernel pushes per-owner route codes and the dispatch kernel derives routing in one extra CTA, eliminating the separate route kernel (**8 kernels** total). The `unpermute` kernel prefetches `top-k` weights and inverse permutation indices into L2 before its grid-dependency wait and reads them only after the wait.
- **Measured Performance (8x B200 `sm_100a`)**: At `T=1` (`all-ranks`), eager latency drops from `164.6 us` to `143.4 us` (**1.15x**) and CUDA Graph latency drops from `155.5 us` to `141.2 us` (**1.10x**) with bit-identical outputs across all 8 ranks.

### 2.2 SM120 Cake QSA NVFP4 Sparse Paged GQA Decode (`#7313`)

In `flashinfer/experimental/cake_qsa_nvfp4/` (`Hq=24, Hkv=2, D=256`, `2051` selected tokens per row in `64-token` tiles, page size `16`):
- Because `sm_120a` has no `tcgen05` or TMEM, Cake provides `mma.sync` K-first decode, cooperative single-launch decode+merge, and split-KV merge programs.
- **Measured Performance (RTX PRO 6000 Blackwell `sm_120a`)**: Achieves **1.27x – 1.89x** eager speedup (`320.93 us -> 169.62 us` at `T=128` HND) and **1.21x – 1.90x** CUDA Graph speedup (`321.59 us -> 169.10 us` at `T=128` HND) over FlashInfer's reference `qsa_nvfp4` backend.

### 2.3 SM120 Pipelined Topology-Weighted PCIe CE Ring All-Reduce (`#7301`)

In `csrc/pcie_ipc_all_reduce.cu` and `flashinfer/comm/pcie_ipc_tuning.py`, `IpcVariant.COPY_ENGINE_RING_PIPED` pipelines 2–4 pieces per rank chunk (`>= 4 MiB/rank`) over the DMA Copy Engine using stream memory operations and `set_ce_slow_hops` cross-socket awareness. On 8x SM120 PCIe Gen5 GPUs across two CPU sockets, TP=8 BF16 `96 MiB/rank` latency improves from `4.4753 ms` to `4.0287 ms` (**1.111x**).

### 2.4 Kimi-K3 Stable LatentMoE Evict-First & Resident TP1 Tail GEMM (`#7312`)

In `flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py`:
- For `T <= 128`, streaming weights with `a_evict_first = DECODE_A_EVICT_FIRST` preserves activations in L2, yielding up to **1.201x** front projection speedup (`53.07 us -> 44.18 us` at `T=64` on GB300) and **1.191x** TP1 tail speedup (`34.98 us -> 29.36 us` on GB300).
- For `T >= 1024`, CTA pairs claim `256x512` work items from a device counter in a resident GEMM with fused RMSNorm, yielding **1.121x** speedup (`120.87 us -> 107.81 us` at `T=1024` on GB200).

### 2.5 SM100/SM103 DeepGEMM Mega-mHC (`h=4096`) (`#7321`)

In `flashinfer/experimental/deepgemm_mega_mhc/mega_mhc.py`, one persistent kernel executes GLM-5.3-Flash's 4-route mHC block (`h=4096`, 20 Sinkhorn iterations), outperforming vLLM's multi-kernel `MHCFusedPostPreOp.forward_cuda` by **1.426x** on B200 (`39.095 us -> 27.424 us`) and **1.387x** on GB300 (`35.510 us -> 25.608 us`) at `T=1024`.

---

## 3. Summary Table

| Kernel / Path | Commit | Hardware | Baseline Latency | Optimized Latency | Speedup |
|---|---|---|---|---|---|
| MegaMoE BF16 EP8 Flag-in-Data (`T=1`, Eager) | `#7348` (`d4d4bd06d1`) | 8x B200 (`sm_100a`) | $164.6\ \mu\text{s}$ | $143.4\ \mu\text{s}$ | **1.15x** |
| Cake QSA NVFP4 Decode (`T=128`, HND, Graph) | `#7313` (`09a77503f0`) | RTX PRO 6000 (`sm_120a`) | $321.59\ \mu\text{s}$ | $169.10\ \mu\text{s}$ | **1.90x** |
| PCIe CE Ring Piped (`TP=8`, BF16, `96 MiB/rank`) | `#7301` (`68e3e09ad3`) | 8x SM120 PCIe Gen5 | $4.4753\ \text{ms}$ | $4.0287\ \text{ms}$ | **1.111x** |
| Kimi-K3 LatentMoE Front (`T=64`, `evict_first`) | `#7312` (`1c0d2be434`) | B300 / GB300 (`sm_103a`) | $53.07\ \mu\text{s}$ | $44.18\ \mu\text{s}$ | **1.201x** |
| DeepGEMM Mega-mHC (`h=4096`, `T=1024`) | `#7321` (`847172573a`) | B200 (`sm_100a`) | $39.095\ \mu\text{s}$ | $27.424\ \mu\text{s}$ | **1.426x** |
