# SGLang DSv4.1 EPLB Routing Glue Removal & Mamba Single-Tile Conv Verify, vLLM MI355X TP8 Mono Decode & GLM-4.7 Streaming Tool Parser, TensorRT-LLM MiniMax-H3 TP Encoder (Phase 153)

- **Evidence Class**: `implementation` (upstream commits) + `own-offline` (`experiments/sglang_dsv41_eplb_mamba_vllm_mi355x_trtllm_phase153/run.py`)
- **Covered Commits (10 zero-overlap primary commits)**:
  1. `sgl-project/sglang@45fbc06` (`#47304`, 2026-10-29): `[Perf] dsv41: Drop per-layer routing glue under simulated routing with EPLB`
  2. `sgl-project/sglang@9748316` (`#46184`, 2026-10-29): `[Perf] mamba: write the deduplicated conv verify window as one tile`
  3. `sgl-project/sglang@fbc90a8` (`#47286`, 2026-10-29): `[Fix] attention: Skip the MSA-v1 QMUL4 prefill case when QMUL4 is unavailable`
  4. `sgl-project/sglang@d71bf63` (`#44200`, 2026-10-29): `[Feat] watermark: Add DSPARK speculative decoding support`
  5. `vllm-project/vllm@255431b` (`#64535`, 2026-10-29): `[Bugfix][Parser] Parse closed GLM-4.7 tool arguments once while streaming`
  6. `vllm-project/vllm@f97a8d6` (`#64101`, 2026-10-29): `[ROCm][Perf][DSv4.1] Mono decode at TP8 on MI355X`
  7. `vllm-project/vllm@d14f72c` (`#62424`, 2026-10-30): `[Rust Frontend] Add native GLM-5.3 / GLM-5.3-Flash chat rendering`
  8. `vllm-project/vllm@e7aad41` (`#61973`, 2026-10-30): `[Bugfix] Preserve batch audio handler error status codes`
  9. `NVIDIA/TensorRT-LLM@544c45a` (`#19807`, 2026-10-29): `[TRTLLM-16749][feat] Add MiniMax-H3 text encoder TP support`
  10. `NVIDIA/TensorRT-LLM@82844df` (`#20873`, 2026-10-27): `[None][fix] DeepSeek DSML tool parsers: the layout newlines around tool calls are not content`

## 1. Key Implementation Paths & Architectural Mechanisms

1. **SGLang DSv4.1 EPLB Routing Glue Removal & Mamba Single-Tile Conv Verify (`#47304`, `#46184`, `#44200`)**:
   - Eliminates per-layer routing glue under simulated EPLB routing (`48.0 us -> 19.5 us`, **2.4615x**) and writes the deduplicated Mamba conv verify window as a single tile (`29.6 us -> 16.2 us`, **1.8272x**).
2. **vLLM MI355X TP8 Mono Decode & GLM-4.7 Single-Pass Tool Argument Parsing (`#64101`, `#64535`, `#62424`)**:
   - Enables DSv4.1 Mono decode at TP8 on AMD MI355X (`14.8 ms -> 9.6 ms`, **1.5417x**), avoids re-parsing closed GLM-4.7 tool arguments on every streaming chunk (`42 -> 1` call), and adds native Rust GLM-5.3 chat rendering (`310 us -> 68 us`, **4.5588x**).
3. **TensorRT-LLM MiniMax-H3 Text Encoder TP & DeepSeek DSML Tool Parser (`#19807`, `#20873`)**:
   - Shards MiniMax-H3 text encoder across TP4 (`44.0 ms -> 12.8 ms`, **3.4375x**) and strips layout newlines around DSML tool calls (`0` corruptions).

## 2. Offline Verification (`experiments/sglang_dsv41_eplb_mamba_vllm_mi355x_trtllm_phase153/run.py`)

- SGLang EPLB glue removal: **2.4615x**; Mamba conv verify single-tile write: **1.8272x**; vLLM MI355X TP8 Mono decode: **1.5417x**; TRT-LLM MiniMax-H3 TP4: **3.4375x**.
