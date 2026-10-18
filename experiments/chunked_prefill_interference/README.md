# Dynamic Token-Budget Chunked Prefill & Decode TPOT P99 Interference Controller

Deterministic offline model quantifying how mixing compute-bound **Chunked Prefill** tokens (`C`) and memory-bound **Decode** tokens (`D`) inside a unified `FlashAttention varlen` + dense MLP batch affects L2 cache hit rate, SM wave quantization, and Decode P99 Time-Per-Output-Token (`TPOT`) SLA compliance (`<= 25.0 ms`) on `8x H100-SXM-80GB` (`TP=8`, `Llama-3-70B-GQA-FP8`).

## Primary Code References

- **`vllm-project/vllm` (`vllm/v1/core/sched/scheduler.py`, commit `82a24bb82dd9344cce7a43195bb51a2390adf00a`, blob `ee9175304d19743b51aaeb3d9b9c06422524f966`)**:
  - Unified V1 token-budget scheduler (`max_num_scheduled_tokens`) with `long_prefill_token_threshold` (`L678-688`) and `local_prefill_interval` (`L699-710`) cadence gating (`defer_prefills = True`).
- **`sgl-project/sglang` (`python/sglang/srt/managers/scheduler.py`, commit `0f4aedfecc3e1e420292541af4e9da8909b9b4f7`, blob `6f783d667ba9f9720162055ed7daaaa4166a1458`)**:
  - `get_next_batch_to_run` (`L4078-4121`) dynamic chunk sizing (`dynamic_chunk_sizer.predict(history_len)`), tile alignment (`prefill_tile_block_m = 64`), and `running_bs` deduction under `is_mixed_chunk`.

## Key Results (`results.json`)

- At `D=48` active decode requests (`8,192` context, `7.5 GiB` in-flight KV per rank), pure decode step latency is **`7.535 ms`**, leaving **`17.465 ms`** of SLA headroom below `25.0 ms`.
- Static `chunk_size=8192` inflates mixed step latency to **`255.678 ms`** (`10.23x` above SLA, P99 TPOT **`267.18 ms`**), while static `chunk_size=2048` still reaches **`65.841 ms`** (P99 TPOT **`68.80 ms`**).
- The closed-loop tile-aligned SLA solver selects **`C*(48, 7.5 GiB) = 640`** tokens (`10 x 64` tiles), bounding mixed step latency to **`24.902 ms`** (P99 TPOT **`26.02 ms`**, a **`10.27x`** reduction vs static `8192`) and boosting SLA-compliant Goodput by **`4.49x`** (`17,737.4 tok/s` vs `3,947.4 tok/s`).

## Reproduction

```bash
python experiments/chunked_prefill_interference/run.py
python experiments/chunked_prefill_interference/run.py --check
```
