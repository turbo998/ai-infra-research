# FlashInfer `cake_causal_conv1d` Dynamic Counter, Kimi-K3 AttnRes SM103 384-Thread Pair, Grouped/Batch FP8 GEMM & CUTLASS SM110a / TileLang Bounds (Phase 104 · Priority 113)

Deterministic Python 3.12 stdlib experiment verifying:
- `flashinfer-ai/flashinfer@c289936e6b`: `cake_causal_conv1d_train` dynamic `int32` chunk work-stealing (`2.2505x` packed-doc imbalance speedup, `1.052x–1.926x` author B200 speedup) & zero-left workspace reuse (`0` extra memsets).
- `flashinfer-ai/flashinfer@53c4cd2341`: `cake_kimi_k3_attn_res` SM103 384-thread 3-warpgroup (`232/40` regs, `64,512 <= 65,536` regs/CTA) eliminating `28` spilled registers/lane at `K=4` (`1.0575x` B300 / `1.0553x` GB300).
- `flashinfer-ai/flashinfer@216d52e164`, `7e2ee76104` & `4a439a7e64`: `bs_ue8m0_swap_rp32` 32-row-block swap-AB (`7.82 -> 6.86 us`, `1.1399x` B200) and masked `BLOCK_N` FP8 GEMM (`4.35 -> 3.63 us`, `1.1983x` B200).
- `NVIDIA/cutlass@78e46228a1` & `f614dc40e1` + `tile-ai/tilelang@323b02ac2a`, `ee4151896d` & `060e1ff31b`: SM110a `setmaxnreg` activation, `T.fill` end-of-slice bound (`4` OOB writes prevented), and `T.ws(-1)` all-thread execution (`0 -> 256` active threads).

```powershell
python experiments\flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104\run.py --check
```
