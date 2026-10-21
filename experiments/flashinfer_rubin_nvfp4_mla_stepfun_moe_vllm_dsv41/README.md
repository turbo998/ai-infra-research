# FlashInfer Rubin (`sm_107a`) Cake NVFP4 MLA Decode, StepFun Pair-Raster MoE & vLLM DeepSeek-V4.1 Replay Fix

Deterministic Python 3.12 standard-library model reproducing:
1. **FlashInfer PR #6931 (`adca7b87d2`)**: Cake dense NVFP4 MLA paged decode on Rubin (`sm_107a` / R200 208 SMs), `352 B/token` KV cache (`256 B` packed E2M1 `ckv` + `32 B` E4M3 `ckv_sf` + `64 B` E4M3 `kpe`), `356 B/row` quantized query buffer, 256-byte aligned 4-slot workspace (`partial_o`, `partial_lse`, `lse_scratch`, `plan_block`), mixed 4-CTA / 2-CTA clusters (`50` fours + `4` pairs on 208 SMs), and 16-config R200 benchmarks (`1.044x` overall geomean, `1.104x` Q4 geomean, `1.248x` peak at `H96-Q4-B64-KV128K`).
2. **FlashInfer PR #6934 (`fd4de5acce`)**: StepFun SwiGLU fused MoE (`H=4096, I=1536, E=64, top_k=8`) kernel regeneration (`111` kernels/arch on `sm_100a` / `sm_103a`), pair-rasterized BF16 FC1 tile-128 (`2.0x` L2 weight tile reuse across M-tile pairs), and two-CTA `nvfp4_bf16tok` FC2 tile-256 (`1.032x` on B200 and `1.0245x` on GB300 at `T=1024`).
3. **vLLM PR #62652 (`85752a43c3`)**: `DeepseekV41ModelState.add_request` sets `_keeps_rows_np[req_index] = True` when `prompt_logprob_token_ids is not None`, preserving full `300`-token and `150+150`-token chunked prefill rows during `DecoderReplayLayers` replay while still window-trimming (`64` rows) non-scoring requests.

## Run

```bash
python experiments/flashinfer_rubin_nvfp4_mla_stepfun_moe_vllm_dsv41/run.py --check
```
