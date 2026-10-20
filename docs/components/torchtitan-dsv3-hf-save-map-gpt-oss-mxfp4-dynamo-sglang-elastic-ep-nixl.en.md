# Deep Dive: TorchTitan DeepSeek-V3 HF Save Map & GPT-OSS Dual-Layout Adapter, Dynamo SGLang Elastic-EP NIXL First-Forward & TRT-LLM OpenMPI `ob1` Snapshot Transport, and Mooncake Accessor / NIXL Trace V2

> **Lang / 语言**: [中文](./torchtitan-dsv3-hf-save-map-gpt-oss-mxfp4-dynamo-sglang-elastic-ep-nixl.md) | **English (Current)**
>
> **Evidence Class**: `implementation` (verified against upstream commits) + `own-offline` (deterministic Python 3.12 standard-library model `experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py --check`).

---

## 1. Architectural Context & Core Problem Statement

Across large-scale MoE checkpoint interoperability (TorchTitan <-> HuggingFace Safetensors), online elastic expert-parallel scaling (Dynamo + SGLang Elastic-EP over NIXL), process snapshot/restore lifecycles (TRT-LLM), and distributed KV store control planes (Mooncake Master + NIXL Tracing), six upstream commits landed on 2026-10-20 eliminate critical production failure modes:

1. **Corrupted Safetensors Headers (`unknown variant ''`) on TorchTitan DeepSeek-V3 671B HF Exports**: `DeepSeekV3StateDictAdapter` reads `fqn_to_index_mapping` from `hf_assets_path/model.safetensors.index.json` (`91,991` entries in `deepseek-ai/DeepSeek-V3`). Because DCP consolidation emits an output header entry for every key in the mapping, any mapped name that `to_hf` never writes receives `{"shape": [], "dtype": ""}`, causing `safetensors` to reject the file with `unknown variant ''`. Three groups of entries in the released index were previously unhandled: `45,808` FP8 `X.weight_scale_inv` scales, `788` `model.layers.61.*` MTP tensors when `num_mtp_layers=0`, and shared MTP embedding/head aliases (`model.layers.61.embed_tokens.weight` and `model.layers.61.shared_head.head.weight`) when `num_mtp_layers=1`.
2. **Transposed Shape Mismatch Between BF16 and MXFP4 Expert Layouts in TorchTitan GPT-OSS**: TorchTitan stores routed experts output-dimension-first (`w13.weight [E, 2, F, D]`, `w2.weight [E, D, F]`), matching dequantized MXFP4 (`gate_up_proj_blocks`, `down_proj_blocks`). However, unquantized BF16 HuggingFace GPT-OSS checkpoints (e.g., `unsloth/gpt-oss-20b-BF16`) store expert weights input-dimension-first (`gate_up_proj [E, D, 2F] = [32, 2880, 5760]`, `down_proj [E, F, D]`) because `transformers` computes `x @ W`. Emitting `_blocks` names and output-first layouts unconditionally broke BF16 checkpoint loads and HF exports.
3. **First-Forward MoE Crash-Loops and Scale-Up Hangs in Dynamo SGLang Elastic-EP over NIXL**: Under SGLang 0.5.21/0.5.22, unquantized MoE experts routed over NIXL hit a deprecated `forward_deepgemm_masked` path unless BF16 dispatch and `deep_gemm` runner flags are set on both the primary and joining workers; additionally, NIXL's default 128-token dispatch budget aborts on 1024-token chunked prefill steps, and omitting `--moe-dense-tp-size 1` hangs the primary on the joiner's first forward after admission.
4. **Dropped MPI Connections After TRT-LLM Snapshot Restore**: OpenMPI's default `ucx` PML transport enables TCP keepalive probes between ranks. Because snapshot mode halts network traffic before freezing ranks, unanswered keepalive probes cause the OS kernel to drop connections so the restored engine fails on its first collective.
5. **ABA Metadata Races in Mooncake Master & Cross-Thread Full `TraceContext` Propagation in NIXL**: Mooncake Store requires per-key metadata accessors paired with shard-monotonic publication generations (`route::Generation`), while NIXL requires bumping its trace plugin API to V2 (`pushContext` / `popContext`) so exporting backends receive full W3C trace contexts rather than a bare 64-bit correlation ID.

---

## 2. First-Hand Source Implementation Analysis

### 2.1 TorchTitan DeepSeek-V3 HF Index Filtering & MTP Shared-Copy Cloning (`1d581795cd`)

In `pytorch/torchtitan` commit [`1d581795cd`](https://github.com/pytorch/torchtitan/commit/1d581795cd014a46e871cc9fc605325fbbceee85) (PR `#5065`, 2026-10-20), `DeepSeekV3StateDictAdapter` filters `fqn_to_index_mapping` during initialization and clones shared MTP tensors during `to_hf()`:

$$
N_{\text{saved}}(M) = 91{,}991 - 45{,}808 - (1 - M) \times 788 = \begin{cases} 45{,}395, & M = 0 \\ 46{,}183, & M = 1 \end{cases}
$$

- Drops all `.weight_scale_inv` entries (`45,808`) and any `model.layers.{L}.*` where $L \ge \text{len(layers)} + \text{len(mtp\_layers)}$.
- On `to_hf()`, clones `tok_embeddings.weight` and `lm_head.weight` into `model.layers.{61+i}.embed_tokens.weight` and `shared_head.head.weight` (~3.71 GB total for 671B BF16) so `safetensors` serializes independent tensors without alias errors; `from_hf()` skips those copies.

### 2.2 TorchTitan GPT-OSS Dual-Layout Adapter (`da54a64ede`)

In `pytorch/torchtitan` commit [`da54a64ede`](https://github.com/pytorch/torchtitan/commit/da54a64ede6983f52f5d66a1290eefc5e13e3260) (PR `#5064`, 2026-10-20), `GptOssStateDictAdapter` switches between `from_hf_map` (default BF16 input-first `gate_up_proj [E, D, 2F]`, `down_proj [E, F, D]`) and `from_hf_map_quantized` (`to_quantized_layout=True` / `is_quantized_layout=True`, output-first `gate_up_proj_blocks` and `down_proj_blocks`), while stripping `_blocks` suffixes and dropping `_scales` in `fqn_to_index_mapping`.

### 2.3 Dynamo SGLang Elastic-EP NIXL Invariants & TRT-LLM `ob1` Snapshot Transport (`02da7f4486`, `c8b3a8e0e5`)

1. **SGLang Elastic-EP over NIXL (`02da7f4486`, PR `#16865`)**: Enforces `SGLANG_DEEPEP_BF16_DISPATCH=1`, `SGLANG_NIXL_EP_BF16_DISPATCH=1`, `--moe-runner-backend deep_gemm`, `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` matching `--chunked-prefill-size 1024`, and `--moe-dense-tp-size 1` across both primary and joiner groups.
2. **TRT-LLM Snapshot Mode OpenMPI `ob1` Transport (`c8b3a8e0e5`, PR `#16372`)**: Sets `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=tcp,self` in `components/src/dynamo/trtllm/__main__.py` when `is_snapshot_enabled()` is true prior to importing `dynamo.trtllm.main`.

### 2.4 Mooncake Per-Key Metadata Accessor & NIXL Trace Plugin API V2 (`cc6b43e989`, `56c1c2f706`)

- **Mooncake Store (`cc6b43e989`, PR `#4717`)**: Associates each `ObjectRecord` in the 1024 metadata shards (`kNumShards = 1024`) with a shard-monotonic `route::Generation next_generation = 1`, tying in-flight tasks to a specific publication generation.
- **NIXL Trace Plugin API V2 (`56c1c2f706`, PR `#2445`)**: Replaces `pushCorrelationId` / `popCorrelationId` with `pushContext(const TraceContext &)` / `popContext()`, bumps `nixl_trace_plugin_api_version` to `V2`, and rejects stale `V1` plugins in `traceLoader`.

---

## 3. Deterministic Verification (`own-offline`)

Running `python experiments/torchtitan_dsv3_hf_save_map_gpt_oss_mxfp4_dynamo_elastic_ep/run.py --check` verifies:
- Exact DeepSeek-V3 index counts (`45,395` at `num_mtp_layers=0`, `46,183` at `num_mtp_layers=1`, `0` empty safetensors headers);
- Lossless roundtrip between TorchTitan `[E, 2, F, D]`, MXFP4 `[E, 2F, D]`, and HF BF16 `[E, D, 2F]` (`[32, 2880, 5760]`);
- First-forward MoE Elastic-EP liveness and post-restore collective completion under `OMPI_MCA_pml=ob1`;
- Stale generation rejection in Mooncake and V1 plugin rejection in NIXL.

---

## 4. Architect Design Guidelines & Falsifiable Follow-Ups

1. **Always Filter Quantization Scale Keys and Disabled Optional Layers from `model.safetensors.index.json` Before Unquantized DCP Exports**: Leaving unpopulated keys in `fqn_to_index_mapping` produces empty `dtype: ""` safetensors entries that break downstream readers.
2. **Disable Keepalive-Bearing MPI Transports (`ucx`) Before Initializing Engines in Checkpoint/Restore Pods**: Set `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=tcp,self` before importing MPI-initializing libraries.
3. **Falsifiable Follow-Up Question**: How much pinned RDMA buffer memory does `SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024` consume per rank as EP world size scales from 8 to 64 GPUs?
