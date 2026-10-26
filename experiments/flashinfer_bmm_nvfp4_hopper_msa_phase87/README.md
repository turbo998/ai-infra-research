# FlashInfer Decode BMM Swap-AB, Cake NVFP4 Attention, Hopper MSA Consolidation & TileLang Barrier Isolation (Phase 87)

Deterministic Python 3.12 stdlib model auditing:
- `flashinfer-ai/flashinfer#7351` (`563a9021ad`): `tcgen05` swap-AB decode-row `bmm_bf16` routing on SM100/SM103 (`1.01x-1.20x` on B200, `1.03x-1.22x` on GB300).
- `flashinfer-ai/flashinfer#7362` (`f83439d2a6`): Cake NVFP4 attention on SM100, SM103, and SM120 (`1.03x-1.08x` vs FA4 BF16 on GB300 for `S >= 32760`; `1.41x-1.84x` vs cuDNN SDPA on RTX PRO 6000 Blackwell).
- `flashinfer-ai/flashinfer#7332` (`01b18dc98d`): Hopper MSA translation-unit consolidation (`179 -> 68` files, `-77,342` net lines while preserving SASS/symbol parity across all 89 programs).
- `flashinfer-ai/flashinfer#7331` (`85fcf53577`), `#7317` (`506fac6203`), `#7320` (`a02c8c5e6b`), `#7354` (`66a780899a`), and `tile-ai/tilelang#3643` (`efa1ffc1e3`).

## Run
```bash
python experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py
python experiments/flashinfer_bmm_nvfp4_hopper_msa_phase87/run.py --check
```
