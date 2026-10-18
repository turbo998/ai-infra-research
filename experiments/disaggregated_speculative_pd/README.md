# Disaggregated Speculative Decoding (`EAGLE-3` / `MTP` under PD Disaggregation)

Deterministic `own-offline` simulator for **Prefill-Decode (PD) Disaggregated Speculative Decoding**, grounded in `sgl-project/sglang` (`python/sglang/srt/speculative/eagle_disaggregation.py`, `eagle_worker_v2.py`, `eagle_info.py`) and `deepseek-ai/DeepSeek-V3` Multi-Token Prediction (MTP).

## Scenarios Compared
- **Scheme A (`kv_only_decode_reprefill`)**: Transfers only target model FP8 KV cache (`20,480 B/tok/rank`); the Decode node must run full draft-extend prefill over $L$ tokens under active batch contention (`58.74 ms` warmup stall at `32K`, `232.00 ms` at `128K`).
- **Scheme B (`full_bf16_hidden_rdma`)**: Transfers target KV plus full-sequence 3-layer fused BF16 hidden states (`+30.0%` RDMA volume inflation) and still runs draft-extend on the Decode node.
- **Scheme C (`zero_stall_seed_handover`)**: Prefill runs `_draft_extend_for_prefill` overlapped with target KV RDMA DMA and transfers Draft KV (`+1.25%` RDMA overhead) plus `CaptureHiddenMode.LAST` seed states (`8.62 KiB/rank`: last token FP8 hidden state, `topk_p`, `topk_index`, and `dsa_topk_indices` remapped via `req_to_token` on Decode), cutting first speculative step latency to `13.435 ms` (`3.21x` over autoregressive decode and `18.45x` faster than Scheme A at `128K`).

## Verification
```powershell
python experiments\disaggregated_speculative_pd\run.py --check
```
