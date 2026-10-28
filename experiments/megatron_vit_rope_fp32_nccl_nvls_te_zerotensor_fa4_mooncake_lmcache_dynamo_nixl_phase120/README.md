# Phase 120 Track C Experiment: Megatron ViT FP32 RoPE & Deterministic NVLS, TE ZeroTensor Wgrad & SM12x FA4, Mooncake EFA/HIXL/PeerMem, LMCache, Dynamo & NIXL

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `NVIDIA/Megatron-LM`, `NVIDIA/TransformerEngine`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`:

- `NVIDIA/Megatron-LM@3ca52d9` & `62c063f`: Vision `Pixtral2DRotaryEmbedding` FP32 frequency preservation across `Float16Module` and deterministic `NCCL_ALGO=Ring,NVLS` / `""` acceptance on non-Hopper GPUs.
- `NVIDIA/TransformerEngine@fc48751` & `6d6a500`: `ZeroTensor` dummy weight gradients (`0.44 ms` and `5308.42 MiB` HBM traffic saved on shared `[129280, 7168]` MTP/LM head) + per-device dummy cache key and SM12x deterministic backward FA4 exclusion.
- `kvcache-ai/Mooncake@c596310`, `c8fe63e`, `cb670d5`: EFA stale peer segment descriptor single-refetch per target, Ascend HIXL `MC_CUSTOM_TOPO_JSON` injection, and `/sys/module` dynamic peer-memory probing (`dma_buf` default on GB300 open kernel modules).
- `LMCache/LMCache@e45d4d5`, `ai-dynamo/dynamo@cffc4fa`, `ai-dynamo/nixl@842eedf`: Request-tag propagation in `MooncakeLookupClient`, logprob-conditional token text decoding in Dynamo Rust frontend, and `LOCK_SH | LOCK_NB` liveness probes in NIXL `prometheus_mp`.

## Usage

```bash
python experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/run.py --check
```
