# Mooncake AWS Trainium PCIe-Switch EFA Affinity & Stale-Peer Eviction, LMCache Cambricon MLU HND Normalization & Checksum Verifier, TorchTitan Muse Spark 816B GB300 RL Re-Sharding & Qwen3 MXFP8 Rollout, TE Clamp & NIXL v6 256B Telemetry Ring

> **中文版本**: [mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo.md](mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo.md)  
> **Question ID**: `mooncake-neuron-pci-efa-evict-lmcache-mlu-torchtitan-te-nixl-dynamo` (Phase 74, Priority 83)  
> **Reproducible Experiment**: [`experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py`](../../experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py)  
> **Evidence Class**: `implementation` / `author-benchmark` (upstream commits & PR benchmarks) + `own-offline` (deterministic stdlib simulator)

---

## 1. Architectural Motivation & Bottlenecks

Across heterogeneous multi-cloud accelerators (AWS Trainium, Cambricon MLU, ARM Neoverse/Grace, and NVIDIA GB300/B200), six production bottlenecks in distributed KV transport, RL post-training, and control-plane telemetry were resolved between `2026-10-22` and `2026-10-25`:

1. **Mooncake EFA PCIe-Switch Mismatch on AWS Trainium Collapsing Aggregate Bandwidth by 94%** ([`Mooncake@f88d94d1bce8`](https://github.com/kvcache-ai/Mooncake/commit/f88d94d1bce85e959d8cfb5fdd02c840c791d632)): Because Trainium lacks `libcudart.so`, Mooncake fell back to host-level topology and bound all 16 Neuron devices to a single EFA NIC across PCIe switches (`2.5 GB/s` cross-switch vs `19.2 GB/s` switch-local, `7.68x` slower), capping 16-device aggregate bandwidth at `12.31 GB/s`.
2. **Mooncake EFA Stale-Peer Handle Leak Exhausting Libfabric Address Vectors (`fi_av_insert`)** ([`Mooncake@487da6af8205`](https://github.com/kvcache-ai/Mooncake/commit/487da6af8205b131b845d535cbe513152051a6d6), [`Mooncake@44094f2e01a0`](https://github.com/kvcache-ai/Mooncake/commit/44094f2e01a05bde80a5a7440caa6d324aa02c12)): Rolling restarts on `2 × p5en.48xlarge` (120 Decode + 120 Prefill) generated new `"ip:port:ts@nic"` keys on every reconnect without removing dead entries from `peer_map_` or the Libfabric AV table, eventually causing `fi_av_insert` to return `0`.
3. **LMCache Cambricon MLU Layout Mismatch & Unbounded Device Staging** ([`LMCache@a2cd25df48b7`](https://github.com/LMCache/LMCache/commit/a2cd25df48b741cde7776971b38f9c476f48bdd3), [`LMCache@6b2b75465484`](https://github.com/LMCache/LMCache/commit/6b2b75465484f38ee7d7d7e0949e239444e1c564)): `vllm-mlu` reports `"NHD"` for physical `HND` (`[2, NB, NH, BS, HS]`) tensors, and holding full-layer staging buffers across all chunks simultaneously inflated peak MLU memory.
4. **TorchTitan Muse Spark 816B-A42B GB300 RL Cross-Rack AllGather Bottleneck & Qwen3 Rollout Wait** ([`torchtitan@43d1bb98014e`](https://github.com/pytorch/torchtitan/commit/43d1bb98014e0dab19a83e5220ec7ea1ab06d49e), [`torchtitan@df4e4dc22a33`](https://github.com/pytorch/torchtitan/commit/df4e4dc22a335031a9952cbe599b206c410f9d60)): A `256`-GPU trainer topology (`FSDP=64 × TP=4 × EP=4`) suffered from a 64-GPU cross-rack expert FSDP group at `131,072` sequence length, while BF16 rollout generation bottlenecked Qwen3-30B-A3B DAPO-Math on B200.
5. **TransformerEngine Triton `mask_map` Out-of-Bounds Write Under Capacity Factor < 1.0** ([`TransformerEngine@c9c92255465b`](https://github.com/NVIDIA/TransformerEngine/commit/c9c92255465b4518380561b1bbb0aebb177c0b6d)): `_row_id_map_pass_2_kernel` failed to clamp `row_id >= num_out_tokens` to `-1`, writing out of bounds during MoE permute/unpermute.
6. **NIXL SPSC Telemetry Ring False Sharing on ARM & Dynamo `/health` Discovery Serialization** ([`nixl@a9f1f7a76f46`](https://github.com/ai-dynamo/nixl/commit/a9f1f7a76f46f0ee78ec7448936d48ca52073fe3), [`nixl@f3673f6be3f4`](https://github.com/ai-dynamo/nixl/commit/f3673f6be3f4f80f6e1e38ff86cebcf9af5750d4), [`dynamo@a789babbe37a`](https://github.com/ai-dynamo/dynamo/commit/a789babbe37a53175b962085f8a7f89909425e9b)): Placing `write_pos` and `read_pos` in adjacent 8-byte slots caused severe cache-line bouncing on 256B-exclusivity ARM cores, while Dynamo serialized all discovered worker instances on every readiness probe.

---

## 2. Core Implementation Mechanics

### 2.1 Mooncake Trainium PCIe-Switch EFA Affinity & Stale-Peer Eviction

- **Neuron `/proc/self/maps` Discovery & PCIe-Switch Grouping** (`f88d94d1bce8`): Discovers `/dev/neuron<N>` mappings from `/proc/self/maps`, loads `libnrt.so.1` with `RTLD_GLOBAL` to query `nec_get_device_pci_bdf()`, groups Neuron devices and EFA NICs by shared PCIe switch bridge, and sorts HCA names in `Topology::resolve()`, lifting 16-device aggregate EFA bandwidth from `12.31 GB/s` to **`231.16 GB/s` (`18.78x`)** and single-thread bandwidth from `6.07 GB/s` to **`18.30 GB/s` (`3.01x`)**.
- **Stale-Peer Eviction & C API Batch Status** (`487da6af8205`, `44094f2e01a0`): Indexes peers by `stableHostNicKey("ip:port:ts@nic") -> "ip@nic"` and evicts idle superseded endpoints via `disconnectIfIdle()` + `fi_av_remove`, bounding `peer_map_` at **`1` entry across `5,000` churn iterations**, while `getBatchTransferStatus` provides $O(1)$ batch completion checks.

### 2.2 LMCache Cambricon MLU `MluDeviceSpec` & Request-Scoped Checksum Verifier

- **`MluDeviceSpec` Normalization & Bounded Staging** (`a2cd25df48b7`): Normalizes `"NHD"` to physical `"HND"`, rejects raw integer pointers, and frees temporary device buffers per chunk on the transfer stream, reducing 61-layer 16-chunk staging from `976.0 MiB` to **`62.0 MiB` (`15.74x`)**.
- **`BlockAllocator` End-to-End Checksum Verifier** (`6b2b75465484`): Allocates disjoint `GroupBlocks` for cold `STORE` vs warm `RETRIEVE` and verifies per-chunk checksums in local worker buffers.

### 2.3 TorchTitan Muse Spark 816B GB300 RL Re-Sharding, Qwen3 MXFP8 Rollout & TE Clamp

- **Muse Spark 816B-A42B 128-GPU GB300 RL Re-Sharding** (`43d1bb98014e`): Re-shards the trainer from `256` GPUs (`FSDP=64 × TP=4 × EP=4`) to **`128` GPUs (`FSDP=8 × TP=4 × CP=4 × EP=16`)** with `UlyssesVarlenInnerAttention` on both global and sliding-window layers, shrinking the expert FSDP group **`8x` (`64 -> 8`)**, halving trainer GPUs (`-50%`), and reaching `186 GiB/GPU` peak memory and `28%` MFU at `131,072` context.
- **Qwen3-30B-A3B B200 MXFP8 Rollout & TE Clamp** (`df4e4dc22a33`, `c9c92255465b`): `mxfp8_mega_fused_inference=True` speeds up generator decode **`1.14x` (`20.2 -> 17.8 s`)** and rollout wait **`1.16x` (`15.8 -> 13.6 s`)** (`1.12x` step time), while TE's `_row_id_map_pass_2_kernel` clamps `row_id >= num_out_tokens` to `-1`.

### 2.4 NIXL v6 256B-Aligned SPSC Telemetry Ring, UCX `treq` & Dynamo Lightweight `/health`

- **NIXL `TELEMETRY_VERSION = 6`** (`a9f1f7a76f46`): Separates `write_pos` and `read_pos` into distinct 256-byte blocks (`512 B` header, keeping `version` at offset `16 B`) with cached opposing indices and reader shared `flock`, achieving **`51.9M vs 11.9M events/s` (`4.36x`, `1,022 -> 555 ns` RTT `1.84x`)** same-socket and **`9.2M vs 2.5M events/s` (`3.68x`)** cross-socket on ARM.
- **UCX 16B `treq` & Dynamo `/health`** (`f3673f6be3f4`, `a789babbe37a`): Appends a 16-byte `treq` (`requestId`, `execution`) tag to UCX notifications and makes Dynamo `/health` status-only unless `?instances=true` is requested.

---

## 3. Quantitative Summary Table

Run `python experiments/mooncake_neuron_pci_efa_evict_lmcache_mlu_torchtitan_te_nixl_dynamo_sim/run.py --check`:

| Component / Commit | Architectural Mechanism | Baseline | Optimized | Gain / Impact |
|---|---|---|---|---|
| **Mooncake Trainium EFA** (`f88d94d1bce8`) | `/dev/neuron` discovery + PCIe switch grouping + sorted HCAs | `12.31 GB/s` (16-dev) | **`231.16 GB/s`** | **18.78x** (**7.68x** switch-local NIC) |
| **Mooncake EFA AV Evict** (`487da6af8205`) | `stableHostNicKey` + `disconnectIfIdle` + `fi_av_remove` | `5,000` leaked peers | **`1` active peer** | Eliminates `fi_av_insert == 0` outage |
| **LMCache MLU Backend** (`a2cd25df48b7`) | `"NHD" -> "HND"` normalization + per-chunk buffer release | `976.0 MiB` staging | **`62.0 MiB` staging** | **15.74x** staging reduction |
| **TorchTitan Muse Spark RL** (`43d1bb98014e`) | `FSDP=8 × TP=4 × CP=4 × EP=16` + Ulysses all-layer varlen | `256` GPUs (`FSDP=64`) | **`128` GPUs (`FSDP=8`)** | **8x** smaller expert FSDP / **-50% GPUs** |
| **TorchTitan Qwen3 RL** (`df4e4dc22a33`) | B200 `mxfp8_mega_fused_inference=True` | `15.8 s` rollout wait | **`13.6 s` rollout wait** | **1.16x** (**1.14x** decode) |
| **NIXL v6 SPSC Ring** (`a9f1f7a76f46`) | 256B cache-line isolation (`512 B` header) + index caching | `11.9M events/s` (`1,022 ns`) | **`51.9M events/s` (`555 ns`)** | **4.36x** throughput / **1.84x** RTT |
