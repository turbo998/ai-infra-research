# FA4 SM100 Empty-Tile TMEM Zero-Fill & Sparse MLA 64-Head Bwd, veRL/Megatron/TE Training Fixes, and Mooncake/NIXL/LMCache Transport Hardening (Phase 88)

Deterministic Python 3.12 standard-library verifier covering:
- **FlashAttention-4 `#2906` (`86dcb5d60c`), `2865a891a9`, `6a3ecb9be2`, `2cb0a8e471`, `#3007` (`e9d8944c85`)**: SM100 `constexpr zero_fill` in `correction_epilogue` (eliminating `0.0 * Inf = NaN` from uninitialized `tcgen05.alloc` TMEM), Sparse MLA 64-head `recompute_p`, `lse_log2` unpadded sequence-end stop, `o_lo` residual row guard, and zero-host-sync CUDA graph capture/replay tests.
- **veRL `#8361` (`4f8f090af3`), `#8235` (`0bd668c1aa`), `#8234` (`3dd97a0e1b`), Megatron-LM `#7780` (`ed9c2c8c1c`), TransformerEngine `#3397` (`e32e1238b0`)**: Preserving 20+ Megatron-Bridge layer specs under QAT, onloading HDO CPU-offloaded optimizer shards before checkpoint load, omitting `data_parallel_start_rank` on vLLM >= 0.28 DP leader, resetting GTP `symmetric_wgrad_pool` reuse fences before full-iteration CUDA Graph capture, and typing distributed-weight grouped `wgrad` in FP32 (`65,536x` finer ULP resolution).
- **Mooncake `#4887`/`#4814`/`#3751`/`#4799`/`#4821`/`#4923`, NIXL `#2222` (`9e7800fc16`)/`#2494`/`#2442`, LMCache `#6031` (`a873f152f8`)**: Per-tenant `NamespaceTable`, HA SIGTERM leader lease release, tiled COO sparse reshard, target-grouped multi-buffer TE, SSE4.2 CRC-32C, GPUNETIO independent QP completion progress (`5.58x`–`7.36x` small-request p50 speedup under 2 MiB head-of-line load), EFA unsolicited write recv disable, `nixl_ep` barrier put completion wait, and LMCache `pygrpc-native` Rust-tokio gRPC server mode.

## Run
```bash
python experiments/fa4_sparse_mla_verl_mcore_mooncake_nixl_phase88/run.py
python experiments/fa4_sparse_mla_verl_mcore_mooncake_nixl_phase88/run.py --check
```
