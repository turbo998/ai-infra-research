# Track A Experiment (Phase 109): FlashInfer cuDNN Plan Reuse, Cake Ulysses Direct NVLink, GEMM Host Dispatch Cut, CUTLASS SM89 Scale & TileLang Reduce/L0 Pitch

Deterministic Python 3.12 standard-library model of Phase 109 (`priority: 118`) Track A operator commits:

- `flashinfer-ai/flashinfer` `5a06916e7b` (`#7052`), `35591b5d6a` (`#7513`), `946eabe5a5` (`#7485`), `ecc3b2f8e3` (`#7492`), `4dd1d9b964` (`#4674`), `4d1799541e` (`#7487`)
- `NVIDIA/cutlass` `564d267e4c` (`#3461`), `b50d8fd719` (`#3394`)
- `tile-ai/tilelang` `c362c962d0` (`#2815`), `0bd33f953c` (`#3511`)

## Run

```bash
python experiments/flashinfer_cudnn_plan_ulysses_gemm_host_minimax_phase109/run.py
python experiments/flashinfer_cudnn_plan_ulysses_gemm_host_minimax_phase109/run.py --check
```
