# Megatron-LM 512-Dim Absorbed MLA / Mixture-of-Kittens / Shared-Expert TP, veRL vLLM Submission-Gate Barrier, Mooncake EFA Security / `IOUringStagingPool` / HCCS Alignment, LMCache Atomic Rollback & Dynamo / NIXL UCCL Notification Guard (Phase 123)

Deterministic Python 3.12 standard-library model covering:
- Megatron-LM `fused_sparse_mla_absorbed` supporting 512-dim NoPE queries/keys (`+64` zero-padded channels to match 576-dim absorbed layout while preserving caller scale `0.044194`) and unaligned `topk=100 -> 128` sentinel `-1` padding (`#7617`), Mixture-of-Kittens (`mok`) MoE megakernel backend (`7 -> 1` kernel launch per layer on `SM100`/`SM103`, `#8210`), and fused shared-expert `GroupedLinear` `FC1` local TP shard `out_features` fix (`4x` weight inflation eliminated at `TP=4`, `#8145`).
- veRL vLLM rollout submission-gate barrier (`_generate_admitted`) releasing `_admitting` immediately after `AsyncLLM.add_request()` returns rather than waiting for the first yielded token (`11.2 s -> 0.012 s` pause stall, `933.33x` faster when `batch=256 > max_num_seqs=32`, `#8397`).
- Mooncake EFA `remote_accessible=false` security enforcement (`FI_READ | FI_WRITE` only + `kLocalOnlyKey = 1`, `#4976`), TENT `IOUringStagingPool` (`32 x 4 MiB = 128 MiB` pinned DMA bounce pool, `#5012`), `hccs:device` 2 MiB alignment gating (`#5049`), Ascend failed-lookup `slice->markFailed()` (`#4888`), and shared `libtent_metrics.so` HTTP 500 self-observability (`#4938`).
- LMCache Coordinator atomic metadata rollback (`capture`/`restore` + HTTP `503` on `MetadataWriteError`, `#6077`), lazy Mooncake SDK import (`#6134`), Dynamo OTLP 4-byte UTF-8 token packing (`#6090`), Dynamo SGLang KVCR HiCache & caller-spoofed `kv_hints` stripping (`#16117`), EPP `FULL_DUPLEX_STREAMED` validation (`#17125`), `WorkerType::Prefill` discovery filtering (`#17325`), and NIXL UCCL `std::optional<nixl_blob_t> notif_msg` eliminating 6 spurious empty notifications (`10 -> 4` notifications, `6 -> 0` empty, `#2429`).

## Run & Verify

```powershell
python experiments\megatron_mok_mla_verl_gate_mooncake_efa_iouring_lmcache_dynamo_nixl_phase123\run.py --check
```
