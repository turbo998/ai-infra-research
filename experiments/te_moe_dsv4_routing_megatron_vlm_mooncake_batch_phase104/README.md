# TE DeepSeek-V4 Custom MoE Routing, Megatron Batched VLM Prefill Encoding, TorchTitan Qwen3 RoPE & Mooncake Multi-Buffer / Dynamo DeepStack Cache (Phase 104 · Priority 113)

Deterministic Python 3.12 stdlib experiment verifying:
- `NVIDIA/TransformerEngine@83d8c4e337` & `NVIDIA/Megatron-LM@1142209a19`: TE `RoutingMapInfo` custom DSv4 MoE routing (`61.50 MiB` HBM & `15.03 GFLOPs` saved at `T=4096, H=7168, E=256`) and Megatron-LM batched waiting-request image encoding before prefill (`42.4 -> 14.5 ms`, `2.9241x` speedup).
- `pytorch/torchtitan@13806cbae0` & `3ed5aa976f`: `Qwen3-235B-A22B` published `rope_theta = 1,000,000.0` (`5.0x` wavelength distortion eliminated) and lazy `attn_gym` backend error deferral.
- `kvcache-ai/Mooncake@f7baeb138d`, `679ae25172` & `b03155412e` + `LMCache/LMCache@f5feba8054` + `ai-dynamo/dynamo@2e96c98ba7` + `ai-dynamo/nixl@42f1714f46`: Mooncake HTTP Bearer auth & store bounds, C/Rust `batch_put_from_multi_buffers` (`61 -> 1` RPC calls, `3,997,696 B` staging copy saved per block), EFA `fi_getinfo()` discovery, LMCache SM80 `bf16 -> fp32 -> fp8e4b15` upcast, Dynamo DeepStack 3-level embedding cache budgeting (`3.0x` under-accounting eliminated), and NIXL GPUNetIO early OOB IPv4 failure rejection.

```powershell
python experiments\te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104\run.py --check
```
