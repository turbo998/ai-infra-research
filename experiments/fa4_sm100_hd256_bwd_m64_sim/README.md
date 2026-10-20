# FlashAttention-4 Blackwell SM100 `hdim=256` 2-CTA Backward (`M=64` TMEM & `AccumTileOrder`) Simulator / 模拟器

## 中文说明

本确定性标准库模拟器（`run.py`）基于 `Dao-AILab/flash-attention` 提交 `2b00d76f222c842b245fd98847223c898bc476db` 与 `965c9212e6be33c821ee9f2b461564c5c91a2503`（`flash_bwd_sm100.py`、`bwd_accum_tile.py`、`flash_bwd_postprocess.py`）验证：
1. **SM100 `512` 列 TMEM 预算与 `M=64` 2-CTA KV Tile (`tile_n=64, acc_m64=True`)**：对比 `hdim=256` 在 `M=128` 1-CTA (`768` 列溢出) 与 `M=128` 2-CTA Serial (`640` 列溢出) 下的 TMEM 分配失败，验证 `M=64` 下 `lane_rows=2` 使每个累加器宽度减半（`dV=128, dK=128, S=64, dP=64, dQ=128`，尾端重叠后 `tmem_end = 512`），且 `P^T` / `dS^T` 驻留 SMEM（`tmem_cols_P = tmem_cols_dS = 0`）。
2. **零额外缓冲区 DSMEM `dS` 交换 (`has_sdS_xchg=False`) 与 `sdQaccum` ステージ预算**：`sdS_dK` 在 SMEM 中按 `(q, kv)` M-major 排布使对端 CTA (`cta_rank ^ 1`) 的半块天然连续，省去 `sdS_xchg` 缓冲区；同时在 `32 KiB` 独立 `sKt` 约束下采用 `dQ_reduce_ncol=16, sdQaccum_stage=2`。
3. **`AccumTileOrder` 三种存储序双射与 1-D Epilogue 写出校验**：验证 `ROW_MAJOR`、`TMEM_FRAG_1CTA`、`TMEM_FRAG_2CTA` 在 `(128, 256)` 下的严格双射及与 SM100 backward reduce 写出偏移的 100% 匹配。

## English Summary

This deterministic Python standard-library simulator (`run.py`) models `Dao-AILab/flash-attention` commits `2b00d76f22` and `965c9212e6`:
1. **SM100 `512`-Column TMEM Map with `M=64` 2-CTA KV Tiles (`tile_n=64, acc_m64=True`)**: Demonstrates why `hdim=256` overflows TMEM at `tile_n=128` (`768` cols in 1-CTA, `640` cols in 2-CTA serial) and proves how `M=64` (`lane_rows=2`) halves every accumulator's TMEM column footprint (`dV=128, dK=128, S=64, dP=64, dQ=128`, fitting exactly in `512` columns with tail overlap).
2. **Zero-Extra-Buffer DSMEM `dS` Exchange (`has_sdS_xchg=False`)**: Verifies `sdS_dK` contiguous sub-tile reuse and `dQ_reduce_ncol=16, sdQaccum_stage=2` alongside the `32 KiB` `sKt` buffer.
3. **`AccumTileOrder` Bijection & Writer Equivalence**: Verifies closed-form index bijections across `ROW_MAJOR`, `TMEM_FRAG_1CTA`, and `TMEM_FRAG_2CTA`, plus 512 B contiguous warp global loads in `FlashAttentionBackwardPostprocess`.

```bash
python experiments/fa4_sm100_hd256_bwd_m64_sim/run.py --check
```
