# Megatron-LM Vision FP32 RoPE Preservation & Deterministic NVLS AllReduce, TransformerEngine ZeroTensor Dummy Wgrad & SM12x FA4 Guard, Mooncake EFA Stale Descriptor Refetch & Dynamic PeerMem Probe, LMCache Request-Tag Keys, Dynamo Conditional Token Decode & NIXL Shared-Lock Probes (Phase 120)

> **Module Positioning**: Written for cloud Solution Architects designing distributed training (Megatron-LM, TransformerEngine) and disaggregated KV / transport fabrics (Mooncake, LMCache, Dynamo, NIXL). Based on 10 upstream commits, this document analyzes FP32 frequency preservation in `Pixtral2DRotaryEmbedding`, deterministic `Ring,NVLS` on Blackwell, zero-storage `ZeroTensor` dummy weight gradients for shared MTP/LM heads, single-refetch EFA segment descriptors, automatic `dma_buf` fallback on GB300 open kernel modules, and conflict-free `flock(LOCK_SH)` telemetry liveness probes.
> **Evidence Class**: `implementation` (verified at fixed commit SHAs) + `author-benchmark` (upstream PR measurements) + `own-offline` (reproducible Python standard-library model in this repository).

---

## 1. Architectural Problem Statement & 10 Upstream Commits Overview

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| C1 | `NVIDIA/Megatron-LM` [`3ca52d9`](https://github.com/NVIDIA/Megatron-LM/commit/3ca52d95d6f4ce8cec12103af40cd774be43ee30) | 2026-10-27 | `vit_model.py`: Keep `Pixtral2DRotaryEmbedding` `self.freqs` outside `register_buffer` so `Float16Module` preserves FP32 frequencies | `implementation` |
| C2 | `NVIDIA/Megatron-LM` [`62c063f`](https://github.com/NVIDIA/Megatron-LM/commit/62c063f2b731b57b1c2031b2b4f4c6a769271ccb) | 2026-10-18 | `determinism.py`: Accept `NCCL_ALGO=Ring,NVLS` and `""` under `--deterministic-mode` on non-Hopper GPUs (`arch != 9`) | `implementation` |
| C3 | `NVIDIA/TransformerEngine` [`fc48751`](https://github.com/NVIDIA/TransformerEngine/commit/fc48751409f2d0bd8c07260efcf9803e84eb52de) | 2026-10-13 | `module/base.py`: Allow `ZeroTensor` dummy weight gradients (`zero_tensor_dummy_wgrad`) and key cache by `(*shape, dtype, device)` | `implementation` + `author-benchmark` |
| C4 | `NVIDIA/TransformerEngine` [`6d6a500`](https://github.com/NVIDIA/TransformerEngine/commit/6d6a500860d0fb60bd4438a0df0dd4d5d0846f1b) | 2026-10-06 | `dot_product_attention`: Disable FlashAttention-4 on SM12x (`arch == 12`) when `deterministic and requires_backward` | `implementation` |
| C5 | `kvcache-ai/Mooncake` [`c596310`](https://github.com/kvcache-ai/Mooncake/commit/c596310fbb184d30e2ff557876ad41a29069089a) | 2026-10-28 | `efa_transport.cpp` / `efa_context.cpp`: Refetch stale peer segment descriptors once per target per call before failing slices | `implementation` |
| C6 | `kvcache-ai/Mooncake` [`c8fe63e`](https://github.com/kvcache-ai/Mooncake/commit/c8fe63e52cbb9e1af6c3ed760d6ce1b7262ba0d4) | 2026-10-28 | `ascend_direct_transport`: Inject `MC_CUSTOM_TOPO_JSON` into HIXL `comm_resource_config.nic_topo_path` | `implementation` |
| C7 | `kvcache-ai/Mooncake` [`cb670d5`](https://github.com/kvcache-ai/Mooncake/commit/cb670d5847b128fea7d8606327911475f48ec8ac) | 2026-10-28 | `config.cpp`: Default `with_nvidia_peermem` by probing `/sys/module/nvidia_peermem` or `nv_peer_mem` | `implementation` |
| C8 | `LMCache/LMCache` [`e45d4d5`](https://github.com/LMCache/LMCache/commit/e45d4d5a85bdb0f3b1343b561a8b3f616774b0d2) | 2026-10-22 | `mooncake_lookup_client.py`: Pass `request_configs` into `token_database.process_tokens()` during lookup | `implementation` |
| C9 | `ai-dynamo/dynamo` [`cffc4fa`](https://github.com/ai-dynamo/dynamo/commit/cffc4fa75c15017bcd32e62f7d0fe4bad468c35d) | 2026-10-27 | `lib/llm/src/backend.rs`: Skip per-token UTF-8 text decode for streaming chunks when `data.log_probs.is_none()` | `implementation` |
| C10 | `ai-dynamo/nixl` [`842eedf`](https://github.com/ai-dynamo/nixl/commit/842eedf0117fe19d864414df980576e55c939c6b) | 2026-10-12 | `prometheus_mp`: Probe stores and owner lock files with `flock(LOCK_SH \| LOCK_NB)` instead of `LOCK_EX \| LOCK_NB` | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 Megatron-LM Vision FP32 RoPE & Deterministic NVLS AllReduce (`3ca52d9`, `62c063f`)

1. **Vision `Pixtral2DRotaryEmbedding` FP32 Frequency Preservation (`3ca52d9`)**:
   - Keeping `self.freqs` outside `register_buffer` prevents `Float16Module` from casting inverse frequencies $\omega_k = 10000^{-2k/d}$ to BF16/FP16 before multiplying by large patch coordinates (e.g., $109$ on a `110×110` grid), eliminating up to `0.029879 rad` angle drift and `0.025852` cosine error.
2. **Deterministic Mode Support for `Ring,NVLS` and `""` (`62c063f`)**:
   - Expands `ACCEPTED_NCCL_ALGO_TOKENS` to include `NVLS` (requiring `Ring` alongside it for Broadcast/Reduce) and accepts `NCCL_ALGO=""` on non-Hopper architectures (`get_device_arch_version() != 9`);
   - Verifies via `ctypes.CDLL(None).getenv(b"NCCL_ALGO") == b""` after first process-group initialization that PyTorch's C++ initializer did not overwrite an explicit empty string with `^NVLS`.

### 2.2 TransformerEngine `ZeroTensor` Dummy Wgrad & SM12x FA4 Guard (`fc48751`, `6d6a500`)

1. **ZeroTensor Dummy Weight Gradients (`fc48751`)**:
   - When `fuse_wgrad_accumulation=True` is combined with a shared weight used twice in one backward pass (such as a `[129280, 7168]` BF16 LM head shared with an MTP head), PyTorch autograd sums the two returned dummy tensors ($3 \times 129280 \times 7168 \times 2\text{ B} = 5308.42\text{ MiB}$ of wasted HBM traffic, costing **0.44 ms** on VR200);
   - Setting `zero_tensor_dummy_wgrad=True` returns `torch._efficientzerotensor(shape, dtype=dtype, device=device)` with zero storage and zero-cost addition, while keying `_dummy_wgrads` by `(*shape, dtype, device)`.
2. **SM12x Deterministic Backward FA4 Exclusion (`6d6a500`)**:
   - Disables FlashAttention-4 when `deterministic and attention_params.requires_backward and device_compute_capability[0] == 12`, while allowing forward-only calls (`requires_backward=False`) to continue using FA4.

### 2.3 Mooncake EFA Stale Descriptor Refetch, HIXL NIC Topo & Dynamic PeerMem (`c596310`, `c8fe63e`, `cb670d5`)

- **EFA Stale Segment Descriptor Refetch (`c596310`)**: Tracks `refreshed_targets` / `refreshed_descs` so a batch of 32 slices targeting newly registered peer chunks triggers at most **1 metadata force-update per target per call** and succeeds with **0 failed slices** instead of failing with `TRANSFER_FAIL (-800)`.
- **Ascend HIXL `MC_CUSTOM_TOPO_JSON` Injection (`c8fe63e`)**: Injects `MC_CUSTOM_TOPO_JSON` into `comm_resource_config.nic_topo_path` with precedence over `MC_TENT_CONF`.
- **Dynamic `WITH_NVIDIA_PEERMEM` Probing (`cb670d5`)**: Checks `/sys/module/nvidia_peermem` and `/sys/module/nv_peer_mem` at startup so GB300 hosts running NVIDIA open kernel modules without `nvidia-peermem` default to `dma_buf` export instead of failing with `EFAULT (14)`.

### 2.4 LMCache Request Tags, Dynamo Conditional Token Decode & NIXL Shared-Lock Probes (`e45d4d5`, `cffc4fa`, `842eedf`)

- **LMCache Request-Tag Lookup Keys (`e45d4d5`)**: Passes `request_configs=request_configs` to `token_database.process_tokens()` inside `MooncakeLookupClient.lookup()`.
- **Dynamo Logprob-Conditional Token Text Decode (`cffc4fa`)**: Sets `needs_token_text = data.log_probs.is_some()`, eliminating 512 per-token `String` allocations across a 512-token response when logprobs are not requested.
- **NIXL `prometheus_mp` Shared-Lock Liveness Probes (`842eedf`)**: Switches `storeWriterAlive()` and `heldAddressesExcept()` from `LOCK_EX | LOCK_NB` to `LOCK_SH | LOCK_NB` so concurrent scrapers never mistake each other's probe lock for a live writer.

---

## 3. Offline Reproducible Verification (`own-offline`)

Experiment directory: `experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/`

- **Command**: `python experiments/megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120/run.py --check`
- **Verified Outputs**:
  - `megatron_vit_and_nvls.vit_rope_max_angle_rad_err_if_cast_to_bf16 = 0.029879`, `ring_nvls_on_blackwell_sm100 = true`, `ring_nvls_on_hopper_sm90 = false`;
  - `te_zerotensor_and_fa4.avoided_dummy_add_hbm_mib = 5308.42`, `vr200_saved_ms_per_backward = 0.44`;
  - `mooncake_efa_hixl_peermem.efa_new_failed_slices = 0`, `gb300_open_kmod_unset_env_path = "dma_buf_export"`;
  - `lmcache_dynamo_nixl.dynamo_token_string_allocs_without_logprobs = 0`, `nixl_probe_lock_mode = "LOCK_SH | LOCK_NB"`.
