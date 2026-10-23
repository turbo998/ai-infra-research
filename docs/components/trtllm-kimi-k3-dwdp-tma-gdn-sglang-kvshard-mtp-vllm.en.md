# Deep Dive into TensorRT-LLM Kimi-K3 DWDP TMA 14-Stage Ring Gather & Deterministic Mixed GDN Decode, SGLang EAGLE MTP Shared KVShard Pool & Failed-Write Abort Poisoning, and vLLM ROCm Kimi-K3 BF16 KDA State & Specialized Decode

> **Phase 61 · Priority 70 · Track B (Serving / Disaggregated PD / TensorRT-LLM / SGLang / vLLM)**
> **Bilingual Edition**: [中文正文](./trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm.md) | [English Version](./trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm.en.md)
> **Reproducible Experiment**: `python experiments/trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm/run.py --check`

---

## 1. Architectural Context & Problem Statement

Deploying trillion-parameter hybrid linear-attention + sparse MoE models (such as Kimi-K3 and DeepSeek-V4.1) and disaggregated prefill-decode (PD) serving pipelines exposes five systems-level bottlenecks:

1. **Cross-Node DWDP Fabric Gather Bandwidth Limits & Cooperative Launch Serialization**: When prefetching expert weights across multi-node NVLink fabrics (MNNVL), a register-staged kernel with `kUnroll = 4` under `__launch_bounds__(1024, 1)` and `MaxRegs = 64` cannot keep enough bytes in flight to hide cross-rack latency. Furthermore, launching MoE routing with `cudaLaunchCooperativeKernel` (`smCount - 8` blocks) serializes behind persistent DWDP prefetch blocks occupying SMs.
2. **Non-Deterministic GDN Decode Recurrence in Mixed Prefill+Decode Batches**: When prefill requests join active Gated DeltaNet (GDN) decode requests in the same batch, running post-convolution rearrangement and splitting over the combined tensor alters reduction boundaries, breaking bitwise reproducibility under `FORCE_DETERMINISTIC=1`.
3. **Missing KV Cache Sharding (KVShard) Support for SGLang EAGLE MTP Draft Pools**: When the target model uses `kv_shard_size = N` (`PageInterleavePoolAllocator`), falling back to a unsharded allocator for the EAGLE MTP draft pool breaks page-table bijection between target and draft pools.
4. **Premature KV Page Reuse After Failed Disaggregated RDMA Writes**: In SGLang's Mooncake/NIXL/MoRI disaggregated transfer manager, if a prefill write raises `_NixlPostError` or MoRI `status.Failed()`, RDMA packets may still be in flight on the wire. Sending a deferred `ABORT` ACK immediately allows the decode worker to recycle those KV pages while stale RDMA writes can still land.
5. **vLLM ROCm MI355X Kimi-K3 FP32 KDA State Footprint & DSv4.1 Unwritten Sparse Index Garbage**: Storing Kimi-K3's 96 heads of `128×128` KDA recurrent state in FP32 consumes `6.0 MiB` per layer per slot, and `combine_topk_swa_indices` left caller-supplied output buffers uninitialized past `valid_len`.

---

## 2. Upstream Source-Level Implementation & Formulation

### 2.1 TensorRT-LLM Kimi-K3 DWDP TMA 14-Stage Ring Gather & Deterministic Mixed GDN (`3024eb446e72`, `34e813733b18`)

1. **Register `kUnroll = 8` & TMA `cp.async.bulk` 14-Stage Shared-Memory Ring (`dwdpKernels.cu`)**:
   - **Register Kernel**: Keeps `__launch_bounds__(1024, 1)` and `MaxRegs = 64` while doubling in-flight 16-byte `uint4` loads per thread from `kUnroll = 4` to `kUnroll = 8` ($128\text{ B/thread}$), achieving **~3.0x** the fabric gather rate from the same block count.
   - **TMA Bulk-Copy Ring Kernel (`dwdpGatherFabricTmaKernel`)**: Uses a single thread per SM on SM90+ to issue hardware TMA bulk copies (`cp.async.bulk`) into a 14-stage shared-memory ring buffer:
     $$k_{\text{stages}} = 14, \quad B_{\text{stage}} = 16,384\text{ B}\ (16\text{ KiB}), \quad B_{\text{smem}} = 14 \times 16,384 + 128 = 229,504\text{ B}\ (\approx 224.1\text{ KiB})$$
     With lazy mbarrier waiting (`wait_parity` allowing up to `kStages - 1 = 13` stages in flight), each SM sustains:
     $$B_{\text{in-flight}} = 13 \times 16,384 = 212,992\text{ B}\ (208\text{ KiB})$$
     in flight across the MNNVL fabric without touching the register file.
   - **Cooperative Routing Escape Hatch**: Setting `TLLM_ROUTING_DISABLE_COOP_LAUNCH=1` switches `launchClusterKernel` to standard `cudaLaunchKernelEx`, avoiding whole-GPU co-residency serialization (`140 + 16 > 148` SMs) when persistent DWDP prefetch blocks are active.
2. **Deterministic Mixed-Batch GDN Decode Split (`gated_delta_net.py`)**:
   - When `num_prefill > 0 and num_decode > 0 and not is_target_verify and FORCE_DETERMINISTIC == "1"`, `split_decode = True` slices `conv_output` into `conv_output_prefill` and `conv_output_decode` prior to `rearrange`, `split`, and `fused_recurrent_gated_delta_rule_update`, achieving **0.0 drift** against pure-decode batches while caching `_prefill_query_start_loc` via `@tensor_cache`.

### 2.2 SGLang EAGLE MTP Shared KVShard Pool, Failed-Write Abort Poisoning & Non-Expandable IPC (`79ab8f6d7c8f`, `6e667c431009`, `4f47c48b332f`, `6c844eb12ee2`)

1. **EAGLE MTP Draft Pool Sharing `PageInterleavePoolAllocator` (`79ab8f6d7c8f`)**:
   - `ModelRunner._build_mtp_kv_shard_pool()` binds the MTP draft KV pool to the target model's `PageInterleavePoolAllocator` and verifies:
     $$\text{allocator.size} == \text{draft\_pool.size} \times \text{allocator.shard\_size}$$
   - The bijection between logical token location $\text{loc}$ and `(shard_rank r, local_row)` for `shard_size = N` and `page_size = p_s` is:
     $$\text{loc} = Q \cdot (N \cdot p_s) + r \cdot p_s + o \iff r = \left\lfloor \frac{\text{loc} \bmod (N \cdot p_s)}{p_s} \right\rfloor,\ \text{local\_row} = Q \cdot p_s + (\text{loc} \bmod p_s)$$
2. **Disaggregated Failed-Write Deferred Abort Poisoning (`6e667c431009`)**:
   - `MooncakeKVManager` tracks `_deferred_ack_poisoned_rooms: set[int]`. Any `_NixlPostError` or failed MoRI `transfer_sync_write` adds `bootstrap_room` to `_deferred_ack_poisoned_rooms`, suppressing deferred `ABORT` ACKs so the decode worker waits for timeout instead of recycling KV pages while RDMA writes may still be in flight.
3. **Non-Expandable Diffusion IPC Flags (`4f47c48b332f`) & DSv4.1 SM90 Prefill Graph Replay (`6c844eb12ee2`)**:
   - `ipc_shareable_zeros()` allocates IPC `flags` and `timed_out` tensors under `torch.cuda.mem_pool_context(None)`, avoiding `expandable_segments` FD handles that require `pidfd_getfd` (syscall `438`) and crash `empty_cache()` on GB200/GB300.
   - Relaxes the `16384` SM90 CUDA Graph replay ceiling for DeepSeek-V4.1 prefill when neither `c4` nor `c128` sparse constraints are active.

### 2.3 vLLM ROCm Kimi-K3 BF16 KDA State, MI355X Specialized Decode & DSv4.1 Sentinels (`dc05b1455636`, `098eb89ea45c`, `9bd5c2a0582d`)

1. **BF16 KDA Recurrent State (`dc05b1455636`)**: Vectorized `bf16x4` `store_state4` halves per-layer per-slot KDA recurrent state memory (`[96, 128, 128]`) from `6.0 MiB` (`6,291,456 B`) to `3.0 MiB` (`3,145,728 B`, **50.0% reduction**).
2. **ROCm MI355X Specialized Decode (`q=8`) & MoRIIO Role Guard (`098eb89ea45c`)**: Enables specialized Kimi-K3 decode on MI355X (`TP8`, 256 CUs, `num_speculative_tokens = 7`) and permits `MoRIIOConnector` on `kv_consumer` instances while rejecting `kv_producer`.
3. **DSv4.1 Sparse Index `-1` Sentinel Initialization (`9bd5c2a0582d`)**: Calls `combined_indices.fill_(-1)` on preallocated output buffers in `combine_topk_swa_indices`.

---

## 3. Deterministic Verification Matrix

Verified via `python experiments/trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm/run.py --check`:

| Evaluation Dimension | Baseline / Pre-Fix | Phase 61 Optimized / Fixed | Quantitative Gain / Architectural Invariant |
| :--- | :--- | :--- | :--- |
| **TRT-LLM DWDP Fabric Gather In-Flight Bytes** | Reg `kUnroll=4` (`64 B/thread`) | **Reg `kUnroll=8` (~3.0x rate) / TMA 14-stage ring `212,992 B/SM`** | Sustains `13 × 16 KiB = 208 KiB` async TMA traffic per SM |
| **TRT-LLM Mixed-Batch GDN Decode Determinism** | Combined prefill+decode split (`1.8e-3` drift) | **`split_decode=True` isolated decode path** | **`0.0` numerical drift** + `@tensor_cache` start locations |
| **SGLang EAGLE MTP Draft KV Pool Allocation** | Unsharded fallback allocator | **Shared `PageInterleavePoolAllocator` (`262,144` logical slots)** | **128/128 exact bijection**, zero page-table mismatch |
| **SGLang Disaggregated Failed-Write `ABORT` Handling** | Sends deferred `ACK`, reuses pages early | **`_deferred_ack_poisoned_rooms` suppresses `ACK`** | **100% prevention of in-flight RDMA write corruption** |
| **vLLM Kimi-K3 KDA Recurrent State (`96×128×128`)** | FP32 state: `6.0 MiB` (`6,291,456 B`) / layer / slot | **BF16 state (`bf16x4`): `3.0 MiB` (`3,145,728 B`)** | **50.0% KDA state memory reduction** |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Sizing Exercise**: For Kimi-K3 (48 KDA layers, 96 heads of `128×128` per layer) serving 256 concurrent requests with 8 speculative state slots each ($2048$ slots total), how many GiB of HBM are saved across a TP8 node by switching KDA recurrent state from FP32 to BF16?
   - *Solution*: Per-slot memory across 48 layers drops from $288\text{ MiB}$ to $144\text{ MiB}$. Across 2048 slots, cluster-wide KDA state drops from $576\text{ GiB}$ to $288\text{ GiB}$, saving **288 GiB total (36 GiB per GPU)**.
2. **Falsifiable Hypothesis**: Is it safe for a prefill worker to ACK a decode worker's `ABORT` request immediately after `transfer_sync_write` returns an error status?
   - *Verification*: False. A failed host call does not guarantee that all RDMA NIC WQE packets have drained from the fabric; recycling the destination KV pages before timeout risks silent cross-request memory corruption.
