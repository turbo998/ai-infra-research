# Deep Dive into LMCache MP Subclass Module Override, Dynamo `</think>` Tool-Call Split & KV Replay Ports, NIXL Shared `prometheus_mp` Endpoint, Megatron GDN CUDA Graph, TE JAX MoE FSDP Layout & TorchTitan RL Pipeline Parallelism

> **Language / 语言**: [English](lmcache-mp-module-dynamo-nixl-te-jax-moe-torchtitan-rl-pp.en.md) | [中文](lmcache-mp-module-dynamo-nixl-te-jax-moe-torchtitan-rl-pp.md)  
> **Phase**: Phase 49 (`priority: 58`) · **Date**: 2026-10-22  
> **Primary Upstream Sources**:
> - `LMCache/LMCache@12ade8880e30694eecb1d2c46862e0b35edb37f8` (`#5807`, 2026-10-21)
> - `ai-dynamo/dynamo@bc2b62478adc7c64055e0e99507adecb8b58bff3` (`#16965`, 2026-10-21)
> - `ai-dynamo/dynamo@39ea07ea104ee3e59b5340581a466e08db633970` (`#16750`, 2026-10-21)
> - `ai-dynamo/nixl@98a17bb1fc58b6c50b6dfcafb550c6a558addeb2` (`#2408`, 2026-10-19)
> - `ai-dynamo/nixl@56c1c2f7062c31a69a1c08a1d2b3cc444c8e834b` (`#2445`, 2026-10-21)
> - `NVIDIA/Megatron-LM@e592c042d575ad86b9a0b9b20239e4c1a1c1f060` (`#8322`, 2026-10-19)
> - `NVIDIA/TransformerEngine@eb3c860a4d3394b7eb5e21dba3021fa10782a52e` (`#3638`, 2026-10-16)
> - `pytorch/torchtitan@a4a62b349d669aea0bd413518bed6cb58a691b2b` (`#3980`, 2026-10-19)
> - `pytorch/torchtitan@a242475253433725f10f39c18f5ccbfdf188e097` (`#5390`, 2026-10-19)

---

## 1. Context & Problem Statement

Across distributed KV caching (LMCache), disaggregated serving and streaming tool-call parsing (Dynamo), RDMA/multiprocess telemetry (NIXL), and large-scale training engines (Megatron-LM, TransformerEngine JAX, TorchTitan), eight architectural improvements were landed:

1. **LMCache MP Server Module Override via Subclassing (`LMCache#5807`)**: `ModuleSpec` previously keyed dependencies by concrete Python class types, preventing subclasses from replacing built-in modules (`Engine`, `StorageManager`) without breaking downstream dependency lookup.
2. **Dynamo `</think>` Tool-Call Opener Split (`dynamo#16965`) & KV Replay Port Collisions (`dynamo#16750`)**: When `</think>` and `<tool_call>` arrived in the same token chunk, `extract_content_ids` failed to split the chunk at `reasoning_end_id`, dropping the tool-call opener; meanwhile, vLLM's KV event `replay_endpoint` lacked per-DP-rank contiguous port allocation.
3. **NIXL `prometheus_mp` Creator-Agent Teardown Gap (`nixl#2408`) & Null-Unsafe `std::getenv` (`nixl#2445`)**: `prometheus_mp` previously owned one `scrapeEndpoint` per agent instance, taking down `/metrics` whenever the creator agent exited even if sibling agents in the same process remained alive.
4. **Megatron-LM GDN CUDA Graph CPU Allocation (`Megatron-LM#8322`) & TE JAX MoE FSDP Gated-Shard Layout (`TransformerEngine#3638`)**: `torch_chunk_gated_delta_rule` allocated `torch.zeros(...).to(value)` on CPU during CUDA Graph capture; TE JAX MoE sharded packed `(gate, up)` weights across FSDP without preserving gated pair layout, corrupting SwiGLU output (`4.652113` instead of `2.193176`).
5. **TorchTitan RL Trainer Pipeline Parallelism (`torchtitan#3980`) & LoRA TP SPMD Typechecking (`torchtitan#5390`)**: The RL trainer previously rejected Pipeline Parallelism due to per-microbatch token-level loss arguments and stage-local state dicts; `_LinearFunction` rejected `Partial` inputs on `RowParallelLinear` LoRA adapters under Tensor Parallelism.

---

## 2. Architectural & Implementation Analysis

### 2.1 LMCache MP Subclass Module Resolution (`LMCache#5807`)
- Every built-in module declares `module_name: ClassVar[str]` (`"telemetry_config"`, `"gpu_context_pool"`, `"storage_manager"`, `"session_manager"`, `"l1_memory_manager"`, `"engine"`).
- `order_modules` and `ModuleBuildContext.require` resolve dependencies by `module_name`, allowing a subclass inheriting `module_name = "engine"` to replace `Engine` transparently.

### 2.2 Dynamo `</think>` Tool Split & KV Replay Ports (`dynamo#16965`, `dynamo#16750`)
- **`extract_content_ids` Split at `reasoning_end_id` (`#16965`)**: Splits incoming token chunks immediately after `reasoning_end_id` (`</think>`), forwarding the trailing tokens (`\n<tool_call>...`) to the tool-call parser on the same step.
- **Contiguous KV Replay Ports (`#16750`)**: Reserves `dp_size` contiguous ports via `allocate_contiguous_ports` and binds `DYN_VLLM_KV_REPLAY_PORT` per DP rank.

### 2.3 NIXL Process-Shared `prometheus_mp` Endpoint & `nixl_getenv` (`nixl#2408`, `nixl#2445`)
- **Process-Shared `scrapeEndpoint` (`#2408`)**: Keyed in a mutex-guarded map by `(canonicalDir(dir_), bind_address, stale_ttl) -> std::weak_ptr<scrapeEndpoint>`. The first agent in a process creates the `shared_ptr<scrapeEndpoint>` and runs `claim()`; sibling agents join it, and `/metrics` stays up until the last agent in the process is destroyed. `reclaim()` uses atomic `nextAttempt_` + `std::try_to_lock` for lock-free export hot paths.
- **`nixl_getenv` (`#2445`)**: Returns `std::optional<std::string>` (`std::nullopt` on unset or empty env vars) across all NIXL plugins and telemetry modules.

### 2.4 Megatron GDN CUDA Graph & TE JAX MoE FSDP Layout (`Megatron-LM#8322`, `TransformerEngine#3638`)
- **Megatron GDN (`#8322`)**: Replaces `torch.zeros(...).to(value)` with `value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))` so CUDA Graph capture allocates recurrent state directly on device.
- **TE JAX MoE Block Optimization (`#3638`)**: Adds `MeshResource`-only resolution, `quant_before_fsdp_ag` MXFP8 pre-all-gather quantization, SM100+ `use_cudnn_fusion` GLU/dGLU fusion, and fixes FSDP gated-shard `(gate, up)` packing so 2-rank FSDP SwiGLU matches unsharded `3 * silu(1) = 2.193176` instead of `4.652113`.

### 2.5 TorchTitan RL Pipeline Parallelism & LoRA TP SPMD Typecheck (`torchtitan#3980`, `torchtitan#5390`)
- **RL Pipeline Parallelism (`#3980`)**: Passes per-microbatch loss arguments via `step(loss_kwarg_mbs=...)`, clears per-microbatch `loss_metrics` dicts before updating (eliminating stale stage-metadata inference calls), slices Monarch trainer ranks as `(pp, dp, cp, tp)` so each DP slice spans all PP stages (`dp0=[0,1,4,5], dp1=[2,3,6,7]` for 8 GPUs), and publishes merged stage state dicts into a single TorchStore `model_state_dict` key.
- **LoRA TP SPMD Typecheck (`#5390`)**: Implements `_LinearFunction.spmd_typecheck` using `spmd.rules.einsum("...k,_nk->...n", input, weight, linear_in=(0, 1))`, allowing `lora_b` (`Replicate`) to propagate `Partial` outputs from `lora_a` (`Shard(k)`) into `RowParallelLinear`'s All-Reduce.

---

## 3. Deterministic Standard-Library Verification

Verified via `experiments/lmcache-mp-module-dynamo-nixl-te-jax-moe-torchtitan-rl-pp/run.py --check`:

| Dimension | Invariant / Metric | Reproduced Value | Status |
|---|---|---|---|
| LMCache MP Subclass Override | `module_name="engine"` Topological Resolution | `CustomVendorEngine` ordered after `storage_manager` | `PASS` |
| Dynamo `</think>` Split & KV Ports | Post-`</think>` Tool Opener / Contiguous DP Ports | `151657` preserved / `[29400, 29401, 29402, 29403]` | `PASS` |
| NIXL `prometheus_mp` Shared Endpoint | Survives Creator Exit & Releases After Last Agent | `survives_creator = True`, `released_last = True` | `PASS` |
| TE JAX MoE FSDP SwiGLU Parity | 2-Rank FSDP vs Unsharded `3 * silu(1)` | **`2.193176`** (vs buggy `4.652113`) | `PASS` |
| TorchTitan RL PP & LoRA TP SPMD | 8-GPU `(pp=2,dp=2,tp=2)` Slices & LoRA Rule | `dp0=[0,1,4,5], dp1=[2,3,6,7]`, `lora_b -> Partial` | `PASS` |
