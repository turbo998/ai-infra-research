# DiT Video Generation Serving (`xDiT`: `CFG × Ulysses × PipeFusion` + `TeaCache`)

Deterministic offline model for 720p 81-frame video generation (`Wan2.1-14B` / `HunyuanVideo` class 3D DiT: `40` blocks, `H=5120`, `N=131,072` spatiotemporal tokens, `50` denoising steps, `2` CFG passes) on `8x H100-SXM-80GB`.

## Primary Code References

- **`xdit-project/xDiT` (`commit f7c16aa958379a9f0a0302350fe0e1bfcec5f23c`, `2026-10-15`)**:
  - `xfuser/core/distributed/parallel_state.py` (`blob e927fd4ed11a`) & `runtime_state.py` (`blob c26d8cfd5f00`): 4D hybrid process groups (`_CFG`, `_SP` Ulysses/Ring, `_PP` PipeFusion, `_TP`) and spatial patch micro-pipelining (`num_pipeline_patch`, `pp_patches_token_start_end_idx_global`).
  - `xfuser/core/cache_manager/cache_manager.py` (`blob 52f66633d229`): `_xdit_kv_cache` fresh/stale KV reuse across spatial patches inside PipeFusion.
  - `xfuser/model_executor/cache/utils.py` (`blob 98f3465973bc`): `TeaCachedTransformerBlocks` polynomial rescaled relative L1 distance (`VectorizedPoly1D`), sequence-parallel `self.all_reduce(diff)` / `self.all_reduce(norm)` synchronization (`L134-138`), and boundary step protection `reset_mask = (cnt == 0) | (cnt == num_steps - 1)` (`L315`).

## Key Results (`results.json`)

- **`TP=8` AllReduce Bottleneck**: Because 3D full bidirectional attention has no causal mask (`4 N^2 H` FLOPs) and hidden size `H=5120` is relatively narrow, `TP=8` executes `160` large `[131072, 5120]` AllReduces per step, exposing **`784.60 ms/step`** (`11.18%` communication overhead, `350.78 s` for 50 steps).
- **`xDiT (CFG=2 × Ulysses=2 × PipeFusion=2, M=4 patches)`**: Splits conditional/unconditional passes across 2 subgroups and pipelines 4 spatial micro-patches, cutting exposed communication + bubble overhead to **`0.98%`** (`58.30 ms/step`, **`298.30 s`** for 50 steps, `1.18x` parallel speedup over `TP=8`).
- **`xDiT + TeaCache (rel_l1_thresh=0.12)`**: Skips `39 / 50` middle denoising steps (`78.0%` skip ratio) with USP-synchronized L1 distance checks while forcing full compute on steps `0` and `49`, reducing 50-step video generation latency from `350.78 s` to **`75.40 s`** (**`4.65x`** end-to-end speedup over `TP=8`) with only **`0.31 dB`** PSNR drop (`34.88 dB`).

## Reproduction

```bash
python experiments/dit_video_pipefusion_serving/run.py
python experiments/dit_video_pipefusion_serving/run.py --check
```
