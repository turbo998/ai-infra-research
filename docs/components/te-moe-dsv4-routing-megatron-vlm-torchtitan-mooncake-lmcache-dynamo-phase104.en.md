# Deep Dive: TransformerEngine DeepSeek-V4 Custom MoE Routing, Megatron-LM Batched VLM Prefill Encoding, TorchTitan Qwen3 RoPE & Mooncake Multi-Buffer / Dynamo DeepStack Cache (Phase 104 · Priority 113)

> **中文版本**: [te-moe-dsv4-routing-megatron-vlm-torchtitan-mooncake-lmcache-dynamo-phase104.md](te-moe-dsv4-routing-megatron-vlm-torchtitan-mooncake-lmcache-dynamo-phase104.md)
> **Experiment**: [`experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/run.py`](../../experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/run.py) · **Summary**: [`summary.json`](../../experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104/summary.json)
> **7-Card Visual Report**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. Evidence Classification & Upstream Anchors

| Source ID | Repo & Commit | Path | Classification |
|---|---|---|---|
| `src-te-moe-dsv4-custom-routing-expert-counts-2026` | `NVIDIA/TransformerEngine@83d8c4e3370bc0117189adbbd88a50fd57440acd` | `tests/jax/test_te_ep_moe.py` | `implementation` |
| `src-megatron-inference-batch-waiting-image-encodes-2026` | `NVIDIA/Megatron-LM@1142209a19054a89dba0c868d6af2cb474152bd9` | `megatron/core/inference/engines/dynamic_engine.py` | `implementation` |
| `src-torchtitan-qwen3-235b-a22b-rope-theta-1e6-2026` | `pytorch/torchtitan@13806cbae05bb9a6ceecc0ce9aa0642d53e7ba25` | `torchtitan/models/qwen3/flavors.py` | `implementation` |
| `src-torchtitan-attn-gym-defer-lazy-backend-error-2026` | `pytorch/torchtitan@3ed5aa976fdf38e05fd50885ab192ea66ed6c5b7` | `torchtitan/models/common/attn_gym.py` | `implementation` |
| `src-mooncake-http-metadata-bearer-auth-bounds-2026` | `kvcache-ai/Mooncake@f7baeb138db8f14bb158e9fc6c84d1a1bd6178f6` | `mooncake-store/tests/e2e/run_nof_heartbeat_tcp_e2e.sh` | `implementation` |
| `src-mooncake-store-c-rust-multi-buffer-batch-io-2026` | `kvcache-ai/Mooncake@679ae2517249d38ca99f18ea640b3524cb6dd180` | `mooncake-store/include/store_c.h`, `mooncake-store/rust/src/ffi_dlopen.rs` | `implementation` |
| `src-mooncake-efa-discover-via-libfabric-2026` | `kvcache-ai/Mooncake@b03155412e7d41782b24558548b952fd7f2d6ee3` | `mooncake-transfer-engine/src/transport/efa_transport/efa_context.cpp` | `implementation` |
| `src-lmcache-sm80-bf16-fp8-key-store-upcast-2026` | `LMCache/LMCache@f5feba8054e82ce56d531661379ab1d7d87b316f` | `lmcache/v1/distributed/serde/turboquant/store_kernel.py` | `implementation` |
| `src-dynamo-vllm-deepstack-multimodal-cache-budget-2026` | `ai-dynamo/dynamo@2e96c98ba75372091b6f2bdc1f5b08e13f5a5dc5` | `components/src/dynamo/vllm/multimodal_utils/multimodal_embedding_cache_connector.py` | `implementation` |
| `src-nixl-gpunetio-reject-failed-oob-ipv4-early-2026` | `ai-dynamo/nixl@42f1714f46054e4ef9b16605a51bba03255c278a` | `src/plugins/gpunetio/gpunetio_backend.cpp` | `implementation` |

---

## 2. Implementation Paths, Data Layout & Synchronization

### 2.1 TE `RoutingMapInfo` for DeepSeek-V4 & Megatron-LM Batched VLM Prefill Encoding (`83d8c4e337`, `1142209a19`)
- **TE `RoutingMapInfo` (`83d8c4e337`)**: Splits MoE routing into `RouterComputationInfo` vs `RoutingMapInfo` and returns `expert_counts`, bypassing TE's internal gate GEMM for DeepSeek-V4 custom/hash routers and saving **`15.03 GFLOPs` and `61.50 MiB` HBM traffic per layer** at $T=4096, H=7168, E=256$.
- **Megatron-LM Batched Waiting-Request Image Encoding (`1142209a19`)**: Batches deferred vision encodes across `8` waiting multimodal requests before prefill (`42.4 -> 14.5 ms`, **`2.9241x` speedup**) with a `0.0 us` `_has_deferred_image_encodes` fast-exit on text-only queues.

### 2.2 TorchTitan `Qwen3-235B-A22B` RoPE Base Fix & `attn_gym` Lazy Import (`13806cbae0`, `3ed5aa976f`)
- Restores `rope_theta = 1,000,000.0` (instead of `5,000,000.0`), eliminating **`5.0x` wavelength distortion**, and defers `attn_gym` `ImportError` until lazy backend invocation.

### 2.3 Mooncake Multi-Buffer Batch IO, HTTP Auth, EFA `fi_getinfo()` & LMCache / Dynamo / NIXL (`f7baeb138d`, `679ae25172`, `b03155412e`, `f5feba8054`, `2e96c98ba7`, `42f1714f46`)
- Exposes `mooncake_store_batch_put_from_multi_buffers` in C ABI and Rust (**`61 -> 1` RPC calls (`61.0x`), `3,997,696 B` staging copy saved per block**), adds HTTP Bearer auth & capacity bounds, discovers EFA via `fi_getinfo()`, upcasts `bf16 -> fp32 -> fp8e4b15` on SM80 in LMCache, sizes Dynamo DeepStack 3-level VLM cache rows via `mm_encoder_output_widths` (**eliminating `3.0x` budget under-accounting**), and rejects failed NIXL GPUNetIO OOB IPv4 discovery at init.
