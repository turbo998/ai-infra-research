# FlashInfer `cake_warp_decode` K256 FC2, Rubin MLA Pairpipe, `cake_mok` Load Bins & `cake_selective_state_update`

Reproducible Python 3.12 standard-library model auditing five first-hand `flashinfer-ai/flashinfer` commits (`53f6bc4d3f47`, `5c314c510256`, `1e47a692ef2a`, `882d4a6d65e2`, `5e5835ed42f0`):

1. **`cake_warp_decode` Linear-Grid `K256 FC2`** (`53f6bc4d3f47876542206e5c8e430dcee0102c2f`, `#7339`): 1-D linear-grid unthrottled workfeed `K256 FC2` (`kernel_cake_warp_decode_4f408b642a6c28d2940e`, `179,456 B` SMEM) for `H=4096, E=512, K=10` (`38` rows on `sm_100a`, `47` rows on `sm_103a`, up to `1.05x` faster).
2. **Rubin (`sm_107a`) MLA NVFP4 Paged Decode Dual Epilogue Tiles & `decode_full_pairpipe`** (`5c314c510256271da2e84dad887b5acc9b6a9b2b`, `#7329`): Two `4 KB` epilogue staging tiles per correction warp (`323,584 -> 332,800 B` SMEM) and converter software pipeline (`H96-Q8-B64-KV64K`), reaching up to `1.047x` (`H448-Q1-B1-KV64K` `26.22 -> 25.04 us`).
3. **`cake_mok` Owner-Rank Load Bins & Early Backward Scratch Release** (`1e47a692ef2a37e5c1decd7d43944d6e73623b34`, `#7326`): Owner-rank load binning inside the 9 scheduler count kernels, precomputed `MoKSchedule.layout` (`int32[3 * local_experts + schedule_capacity / 256]`), and early backward scratch release (`glm52_ep8_mxfp8_full_ragged_defaults` `15.975 -> 15.596 ms`, `1.024x` on B200).
4. **`cake_selective_state_update` for Mamba2 Decode** (`882d4a6d65e29c71c73ad58224acfe9c8f8d3116`, `#7319`): Faster generated programs across all `122` Mamba2 decode shapes (`1.015x–1.253x` on B200 `sm_100a`, `1.018x–1.248x` on GB300 `sm_103a`).
5. **`cake_mamba_ssd_combined` Unified `0x7FFF` SM107 Descriptor Mask** (`5e5835ed42f037838c1f0e11ba9dfa67825a134d`, `#7349`): Folds `21` mask-only `sm_107a` renders (`93 -> 72` device files, `_SM107_MODULES` `45 -> 19` entries) via `CAKE_SSD_SMEM_DESC_ADDR_MASK = 0x7FFF` (`15`-bit `tcgen05` SMEM descriptor start-address field on SM107 vs `14`-bit `0x3FFF` on SM100/SM103).

## Usage

```bash
python experiments/flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update/run.py --check
```
