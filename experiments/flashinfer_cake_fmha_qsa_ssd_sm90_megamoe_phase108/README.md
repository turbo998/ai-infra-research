# FlashInfer Cake FMHA NVFP4 Paged Context, QSA Workspace Reuse, cuDNN Ragged Offsets & CUTLASS/TileLang Guards (Phase 108)

Reproducible stdlib-only model validating:
- `flashinfer-ai/flashinfer@07cc0d60a9` (`#7512`): 20 paged context attention programs (`BF16`, `FP8`, `NVFP4` KV) on `sm_100a` / `sm_103a`.
- `flashinfer-ai/flashinfer@fa8fb88e0f` (`#7514`): Cake QSA NVFP4 checked workspace carve reuse across decode steps.
- `flashinfer-ai/flashinfer@a3452cd405` (`#7438`): cuDNN prefill explicit `seq_lens_q` separation from physical ragged `q_offsets`.
- `NVIDIA/cutlass@1d7cde107f` (`#3278`) & `tile-ai/tilelang@876a59e6eb` (`#2818`): `PredicatedTileIterator` `ScatterD` group pointer guard and TileLang MMA K-atom padding enforcement.

```bash
python experiments/flashinfer_cake_fmha_qsa_ssd_sm90_megamoe_phase108/run.py --check
```
