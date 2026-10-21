# FlashInfer TRTLLM-Gen `SlidingWindowCustom` / `VariableWindow` FMHA & Cake MiniMax-H3 / Small / Dense GEMM Consolidation Experiment

Deterministic Python 3.12 standard-library verification for Phase 50 Topic 1 (`flashinfer-trtllm-gen-sliding-varwindow-fmha-cake-minimax-h3-small-dense-gemm`):

- **`flashinfer#4905` (`d4383335ba`)**: Models `TllmGenFmhaMaskType` (`SlidingWindowCustom=4`, `VariableWindow=5`), `fixedWindowBinds` causal fast-path retention (`4.2624 us` at `KV=128, W=127` vs `4.4736 us` at `KV=4096` on B300), `+4.17%` (B300) / `+3.69%` (B200) prefill overhead, `mChunkedAttentionSizeLog2` trailing `= 0` fix, and non-causal sliding-window `ValueError` guard.
- **`flashinfer#6947` (`c78528451f`), `#6946` (`a509a27181`), `#6938` (`2852797429`)**: Verifies the `-39,298` net line reduction (`+6,238 / -45,536` across `263` files) and `84`-entry (`44 + 54 - 14`) deduplicated `EXPORTED_TEMPLATES` union.

```powershell
python experiments\flashinfer-trtllm-gen-sliding-varwindow-fmha-cake-minimax-h3-small-dense-gemm\run.py --check
```
