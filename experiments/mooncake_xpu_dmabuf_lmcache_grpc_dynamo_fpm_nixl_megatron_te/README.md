# Mooncake XPU dma-buf RDMA, LMCache gRPC Bulk Repeated-Scalar Codec, Dynamo GIL-Free FPM & Session Control, NIXL Telemetry, and Megatron GDN / TE JAX Triton mHC

Deterministic Python 3.12 standard-library reproduction for Phase 47 Topic 3 (`q-phase47-mooncake-xpu-dmabuf-lmcache-grpc-dynamo-fpm-gil-megatron-gdn-te-jax-mhc`).

## Covered Upstream Commits
- `kvcache-ai/Mooncake@042ab8a` (`#4378`), `4137ac0` (`#4756`), `e3e9d70` (`#4643`)
- `LMCache/LMCache@caba6c3` (`#5935`)
- `ai-dynamo/dynamo@4d31cd2` (`#16742`), `ebfcce7` (`#14264`)
- `ai-dynamo/nixl@03d9b74` (`#2479`), `56c1c2f` (`#2445`)
- `NVIDIA/Megatron-LM@29e1a7b` (`#8353`), `NVIDIA/TransformerEngine@53d1a3c` (`#3680`), `b9715e5` (`#3548`), `pytorch/torchtitan@9698a73` (`#5394`)

## Run
```bash
python experiments/mooncake_xpu_dmabuf_lmcache_grpc_dynamo_fpm_nixl_megatron_te/run.py --check
```
