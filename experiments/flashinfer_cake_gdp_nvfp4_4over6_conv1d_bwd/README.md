# FlashInfer Cake GDP Decode/MTP, NVFP4 4/6 Slot-Append & Causal Conv1d Backward Experiment

Deterministic Python 3.12 standard-library model reproducing:
1. **`flashinfer-ai/flashinfer` PR `#6883` (`fc1790f4ec`)**: Cake Gated DeltaProduct (GDP) decode & MTP verification (`cake_gated_delta_product_mtp`) multi-Householder state recurrence and `splitv8_tile8` vs `lanerow` (`l8r4` / `l8r2` / `l4r4`) routing on `sm_100a` / `sm_103a`.
2. **`flashinfer-ai/flashinfer` PR `#6013` (`52fbbd090c`) & PR `#6204` (`3821440929`)**: `nvfp4_4over6` MSE block scale selection (`sf_6 = amax / 6` vs `sf_4 = 1.5 * sf_6`) in `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping` and removal of `.cpu()` host sync during eager prefill (`499.4 -> 451.7 ms`, `-9.55%`).
3. **`flashinfer-ai/flashinfer` PR `#6930` (`7c866e85d3`)**: `cake_causal_conv1d_train` backward channel-threshold routing (`fold_overlap`, `fast_roll`, `roll_prefetch`) on `sm_103a` (GB300) and `sm_107a` (Rubin R200).

## Run

```powershell
python experiments\flashinfer_cake_gdp_nvfp4_4over6_conv1d_bwd\run.py --check
```
