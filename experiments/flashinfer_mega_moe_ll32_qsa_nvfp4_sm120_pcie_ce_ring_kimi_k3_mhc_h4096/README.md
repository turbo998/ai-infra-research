# FlashInfer SM100 MegaMoE Flag-in-Data (`<=32` Rows), SM120 QSA NVFP4 Decode, PCIe CE Ring All-Reduce, Kimi-K3 LatentMoE Evict-First & DeepGEMM Mega-mHC (`h=4096`)

Deterministic Python 3.12 standard-library model auditing five first-hand `flashinfer-ai/flashinfer` commits (`#7348`, `#7313`, `#7301`, `#7312`, `#7321`):

1. **SM100 BF16 Rank-Major EP8 MegaMoE Flag-in-Data (`_LL_MAX_T = 32`) & Pushed Route Codes (`#7348`)**: Quantifies 8-byte `(payload_u32, epoch_u32)` dual-bank polling for `T <= 32` (`6` launches vs `9` baseline) and pushed route codes for `32 < T <= 128` (`8` launches), achieving `1.15x` eager (`164.6 us -> 143.4 us`) and `1.10x` CUDA Graph (`155.5 us -> 141.2 us`) speedups at `T=1` on 8x B200.
2. **SM120 Cake QSA NVFP4 Sparse Paged GQA Decode (`#7313`)**: Models `Hq=24, Hkv=2, D=256`, `2051` selected tokens (`33` tiles of `64` tokens, page `16`) using `mma.sync` cooperative and split-KV decode on RTX PRO 6000 Blackwell (`sm_120a`), reaching `1.89x` eager (`320.93 us -> 169.62 us`) and `1.90x` CUDA Graph (`321.59 us -> 169.10 us`) speedups at `T=128` HND.
3. **SM120 Topology-Weighted PCIe Copy-Engine Pipelined Ring All-Reduce (`#7301`)**: Models `IpcVariant.COPY_ENGINE_RING_PIPED` with stream memory operations and `ce_slow_hops` NUMA/socket awareness (`>= 4 MiB/rank`), achieving `1.111x` (`4.4753 ms -> 4.0287 ms`, 3 pieces/chunk) at TP=8 BF16 `96 MiB/rank`.
4. **SM100/SM103 Kimi-K3 Stable LatentMoE Evict-First Decode & Resident TP1 Tail GEMM (`#7312`)**: Models `a_evict_first` L2 weight streaming for `T <= 128` (`1.201x` front at `T=64` on B300/GB300; `1.191x` TP1 tail on GB300) and claimed-item resident `256x512` fused RMSNorm tail GEMM for `T >= 1024` (`1.121x` at `T=1024` on GB200).
5. **SM100/SM103 DeepGEMM Mega-mHC (`h=4096`, 4-Route, 20 Sinkhorn Iters) (`#7321`)**: Models persistent single-launch post-mixing + BF16x3 split-K projection + 20 Sinkhorn iterations + collapse + input RMSNorm vs vLLM `MHCFusedPostPreOp`, reaching `1.426x` on B200 (`39.095 us -> 27.424 us`) and `1.387x` on GB300 (`35.510 us -> 25.608 us`) at `T=1024`.

## Usage

```bash
python experiments/flashinfer_mega_moe_ll32_qsa_nvfp4_sm120_pcie_ce_ring_kimi_k3_mhc_h4096/run.py --check
```
