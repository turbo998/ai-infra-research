# SGLang Confirmed-First Router & HiCache L3, vLLM RL ParamMeta & KV Offload Hardening, and TensorRT-LLM Mamba Replay & Beam Histories (Phase 88)

Deterministic Python 3.12 standard-library verifier covering:
- **SGLang `#46505` (`eabb11a4c1`) & `#40467` (`64fc5f0bdb`)**: `sgl-router` confirmed-first pending prefix stakes (`--cache-pending-prefix-consult-below`) with follower cap (`--cache-pending-prefix-max-followers`, `8.0x` lower cold-burst worker overload) and `/get_hicache_l3_cache_layout` Mooncake L3 key layout exposure.
- **SGLang `#33250` (`ef11eb3b76`), `#45713` (`6a28c96a49`), `#46308` (`9841183827`)**: DeepSeek-V4 non-EP TBO for `attn_tp > 1`, `gfx950` MI350X/MI355X HIP fused KDA decode (`5.0x` fewer launches), and pipelined Ulysses own-row copy hiding.
- **vLLM `#63256` (`50a184f435`), `#61050` (`e113018f50`), `#63708` (`51d2a85594`), `#62929` (`14aefcd4f9`), `#63628` (`8a4b018f1c`), `#63655` (`8c6e770a1c`), `#61636` (`14d593115d`)**: RL `ParamMeta` ownership/layout, LRU running-prefix preservation (`100%` prefix retention), non-blocking ZMQ P2P control (`_SNDHWM=1000`), cache-reset promotion hold, NIXL registration rollback, OBJ `blocks_per_chunk` namespacing, and 3D MoE LoRA single-pinning (`2.0x` less pinned host memory).
- **TensorRT-LLM `#18843` (`ba6c1fc87d`), `#21027` (`e7d3441e77`), `#20915` (`732fd63463`)**: Modular Mamba/KDA replay cache managers with 128 KiB TMA backing guard (`131,072 B`) and async per-beam hypothesis reconstruction for logits processors.

## Run
```bash
python experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py
python experiments/sglang_router_hicache_vllm_kv_offload_trtllm_phase88/run.py --check
```
