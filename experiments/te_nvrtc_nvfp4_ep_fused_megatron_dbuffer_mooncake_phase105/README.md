# TransformerEngine NVRTC NVFP4 & Fused EP Dispatch, Megatron-LM `DBuffer` P2P & Batch-Invariant Mamba Prefix Cache, Mooncake Borrowed-Buffer COO Planner & NIXL GPU-to-CPU Proxy Ring (Phase 105)

Deterministic standard-library verification for Phase 105 (`priority: 114`) Track C:
- **TransformerEngine NVRTC NVFP4 `4over6` (`ec39dcf368`) & Fused EP Prepare+Dispatch (`f7273d7d80`)**: Compiles NVFP4 `4over6` via NVRTC (`24 -> 0` static AOT instantiations) and fuses EP `prepare+dispatch` with device-resident `recv` counts (`1 -> 0` host syncs/layer, `46.0 -> 21.5 us`, `2.1395x` speedup).
- **Megatron-LM MFSDP v2 `DBuffer` Direct P2P (`a666bd60ef`) & Batch-Invariant Mamba Prefix Caching (`3888b34882`)**: Eliminates `256.0 MiB` staging buffers (`4.80 -> 3.10 ms`, `1.5484x`) and enables `block_size_tokens % ssm_chunk_alignment == 0` Mamba prefix caching (`4,096 -> 1,024` prefill tokens, `4.0x` reduction with `0.0` drift).
- **Mooncake Borrowed-Buffer Byte Ranges & COO Sparse Planner (`81b3694b1b`), LMCache Shared `KVEventSubscriber` (`a0d918a77c`), Dynamo LPX Sidecar (`9ea06e0cd3`) & NIXL GPU-to-CPU Proxy Ring (`f1de8fc286`)**: Cuts 12.5%-dense sparse weight transfer from `64.0 -> 8.25 MiB` (`7.7576x`) with `0` intermediate allocations and reduces NIXL warp proxy ring contention by `32.0x` via 64B leader-thread commands.

```bash
python experiments/te_nvrtc_nvfp4_ep_fused_megatron_dbuffer_mooncake_phase105/run.py --check
```
