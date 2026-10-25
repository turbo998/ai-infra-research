# Mooncake AWS Neuron (Trainium) Direct HBM-over-EFA, Stale-Peer AV Eviction & HCA/NUMA Store Affinity, LMCache Cambricon MLU Backend & Request-Exclusive Local Checksum Bench, TorchTitan FlashAttention Fallback & Lazy Export Deferral, Megatron-LM GDP Prefill Prepare Fusion & EP Consensus Enforcement, and Dynamo KV-Router Event Filtering, Worker GC Freeze & Batched Token Metrics

> **Language / 语言**: [English](mooncake-neuron-efa-lmcache-mlu-megatron-dynamo.en.md) | [中文](mooncake-neuron-efa-lmcache-mlu-megatron-dynamo.md)
> **Phase**: Phase 73 (`priority: 82`, Track C)
> **Question ID**: `q-phase73-mooncake-neuron-efa-lmcache-mlu-megatron-dynamo`
> **Evidence Tiers**: `implementation` + `author-benchmark` + `own-offline` (`mastery: not-assessed`)

---

## 1. Executive Summary & Architectural Insights

Between `2026-10-22` and `2026-10-25`, upstream commits across `kvcache-ai/Mooncake`, `LMCache/LMCache`, `pytorch/torchtitan`, `NVIDIA/Megatron-LM`, and `ai-dynamo/dynamo` delivered five major system-level advances across heterogeneous accelerators (AWS Trainium Neuron HBM, Cambricon MLU, Huawei Ascend MemFabric), large-scale disaggregated rolling upgrades, Gated Delta Product (GDP) hybrid state-space models, and high-concurrency serving runtimes:

1. **Mooncake AWS Neuron (Trainium `trn2.48xlarge`) Direct HBM-over-EFA, Rolling-Upgrade Stale-Peer Eviction & HCA/NUMA Store Affinity** (`Mooncake@f88d94d1bc`, `487da6af82`, `868a6d05ab`, `6a835cbf93`, `44094f2e01`):
   - **Headerless AWS Neuron HBM Direct EFA Transport**: Probes `/proc/self/maps` for `/dev/neuron<N>` mappings at runtime (overriding caller `"cpu:N"` hints to prevent `preTouchMemory()` CPU stores from corrupting device HBM), `dlopen`s `/opt/aws/neuron/lib/libnrt.so.1` with `RTLD_GLOBAL` prior to `fi_getinfo()`, sets `FI_HMEM=neuron` and `FI_EFA_ENABLE_SHM_TRANSFER=0` (preventing libfabric's SHM provider from falling back to **`0.07 GB/s`** uncached PCIe BAR `memcpy`, routing intra-node traffic over EFA loopback at **`19.23 GB/s` — a `274.7x` speedup**), and uses `nec_get_device_pci_bdf()` to pair each Neuron device with its switch-local EFA NICs. On `trn2.48xlarge`, switch-local NIC affinity boosts single-device throughput from **`2.5 GB/s` to `19.2 GB/s` (`7.68x`)** and 16-device aggregate bandwidth from **`12.31 GB/s` to `231.16 GB/s` (`18.78x`)**, while sorting `Topology::resolve()` storage keys so both peers number HCAs identically.
   - **Stale-Peer AV Eviction Under 120P+120D Rolling Upgrades**: Extracts `stableHostNicKey("ip:port:ts@nic") -> "ip@nic"` and evicts superseded volatile peer entries via `disconnectIfIdle()` (using non-blocking `tryLock()` on the CQ thread to avoid deadlocking against submitters holding the read lock while awaiting CQ credits) and auto-invalidates `prov_errno in {4, 8, 9, 14}`. Under 5,000 peer-restart churn iterations on 2×`p5en.48xlarge`, `peer_map_` final size drops from **`5000` to `1`**, eliminating `fi_av_insert == 0` silent refusals.
   - **Request-Local HCA/NUMA Affinity & O(1) C API `getBatchTransferStatus`**: Adds `MC_STORE_HCA_AFFINITY=1` and `MC_STORE_HCA_NUMA_SEGMENTS=1` for RL put/get placement and exposes O(1) `getBatchTransferStatus()` in `transfer_engine_c.h`.
2. **LMCache Cambricon MLU Engine-Driven Backend & Request-Exclusive Local Checksum Benchmark** (`LMCache@a2cd25df48`, `6b2b754654`):
   - **Cambricon MLU (`MluDeviceSpec` / `MluDeviceOps`)**: Supports rank-5 K/V-first NHD/HND and rank-3 MLA transfers (normalizing legacy vLLM MLU NHD reports to physical HND) and forces synchronous D2H copies into strided host slices (`skip_prefix_n_blocks > 0`) to work around a `torch_mlu 1.26.1 + torch 2.6.0` `non_blocking=True` data-corruption bug while keeping contiguous D2H copies asynchronous.
   - **`lmcache bench server` Request-Exclusive `BlockAllocator`**: Reserves exclusive block IDs (`sequential` or `random`), zeroes destination blocks before warm `RETRIEVE`, and verifies MD5 digests directly in local worker buffers without HTTP server dependencies.
3. **TorchTitan FlashAttention Safe Fallback & `attn_gym` Lazy Export Error Deferral** (`torchtitan@89e4339c96`, `3ed5aa976f`):
   - Checks `importlib.util.find_spec` for `flash_attn_interface` (Hopper FA3) and `flash_attn.cute` (Blackwell FA4), falling back to PyTorch FA2 with zero numerical drift against explicit FA2 on L4 (`atol=0, rtol=0`, loss `0.19502778`, grad L2 norm `0.01268001`) while failing fast at construction for unequal Q/K and V MLA dimensions.
   - Captures `ModuleNotFoundError` raised inside `attn_gym`'s lazy `__getattr__` exports (`causal_conv1d` -> `cutlass`) into `_MissingKernel` placeholders, restoring 18 CPU test modules on macOS (`1371 -> 1640` tests collected).
4. **Megatron-LM Gated Delta Product (GDP) Prefill Prepare Fusion & Removal of `disable_ep_consensus`** (`Megatron-LM@b291235082`, `7d3f0fb3b5`):
   - Fuses ~**14 per-layer PyTorch elementwise/strided-copy ops** between short-conv and varlen chunk recurrence into a single Triton launch (`gdp_prefill_prepare`, reducing 40-layer prefill prep launches from **`560` to `40`**) using `triton.language.extra.libdevice` for bitwise agreement (`torch.equal`) with eager PyTorch.
   - Removes `--inference-disable-ep-consensus` in favor of periodic `ep_consensus_interval = 20` coordination.
5. **Dynamo 1.7.0 KV-Router Non-Main Group Pre-Filtering, Lightweight `/health`, Worker GC Freeze & Batched Token Counter** (`dynamo@8bf042187a`, `a789babbe3`, `45cc9e8a1a`, `906e89d3a3`):
   - Filters `NonMainAttentionKind` / `NonMainAttentionGroup` before validating `block_size` on `KvEventOwnership::Kvcr` events; makes `/health` status-only by default (opt-in `?instances=true`); adds `--freeze-gc-after-init` to freeze ~**1,000,000** static Python objects in the Dynamo worker process; and batches per-token `output_tokens` atomic updates every **`LOCAL_FLUSH_TOKENS = 64`** tokens (**`64x`** reduction in cross-core cache-line bouncing).

---

## 2. Reproducible Verification (`own-offline` vs. `author-benchmark`)

```powershell
python experiments\mooncake_neuron_efa_lmcache_mlu_megatron_dynamo_sim\run.py --check
```

| Subsystem | Hardware / Scenario | Baseline | Optimized | Improvement / Outcome | Evidence Tier |
|---|---|---|---|---|---|
| Mooncake EFA Neuron HBM & Switch Affinity | AWS `trn2.48xlarge` (16× Trainium2 + EFA) | Cross-switch `2.5 GB/s` (1 dev) / `12.31 GB/s` (16 dev); SHM `0.07 GB/s` | Switch-local `19.2 GB/s` (1 dev) / **`231.16 GB/s` (16 dev)**; EFA loopback **`19.23 GB/s`** | **`7.68x`** (1 dev), **`18.78x`** (16 dev), **`274.7x`** (loopback) | `author-benchmark` + `own-offline` |
| Mooncake EFA Stale-Peer Eviction | 2× `p5en.48xlarge` (5,000 peer restart churn) | `peer_map_` grows to `5000`, `fi_av_insert == 0` silent refuse | `stableHostNicKey` + `disconnectIfIdle` evicts stale AV slots | `peer_map_` size **`5000 -> 1`** | `author-benchmark` + `own-offline` |
| LMCache Cambricon MLU Backend | `torch_mlu 1.26.1` + `torch 2.6.0` | Strided host D2H (`skip_prefix_n_blocks > 0`) corrupts data under `non_blocking=True` | Forces blocking copy on strided host slices; keeps contiguous async | **Eliminates silent prefix-skip D2H corruption** | `implementation` + `own-offline` |
| TorchTitan FA Fallback & Lazy `attn_gym` | NVIDIA L4 (CUDA 13.4) / macOS 15 CPU | Crashes at import/build without FA3/FA4 wheel or `cutlass` (23 collection errors) | Safe FA2 fallback (`atol=0, rtol=0`); lazy export errors deferred to build | **`1371 -> 1640`** CPU tests collected (+18 modules) | `author-benchmark` + `own-offline` |
| Megatron-LM GDP Prefill Prepare Fusion | 40-layer Gated Delta Product model | ~14 PyTorch kernels/layer (`560` launches total) | Single `gdp_prefill_prepare` Triton kernel (`libdevice` bit-exact) | **`560 -> 40` launches (`14x` reduction)** | `implementation` + `own-offline` |
| Dynamo 1.7.0 Runtime Optimizations | High-concurrency vLLM worker & metrics | Gen2 GC walks ~1M static objects; 1 shared atomic op per generated token | `--freeze-gc-after-init` + `LOCAL_FLUSH_TOKENS = 64` + status-only `/health` | **`64x` fewer shared atomic ops** | `implementation` + `own-offline` |

---

## 3. Architect Exercises & Falsifiable Questions

1. **Troubleshooting Exercise (Mooncake EFA CQ-Poller Lock Ordering)**: Why must `EfaEndPoint::disconnectIfIdle()` use `lock_.tryLock()` rather than a blocking write lock on the CQ-polling thread?
   - *Answer Guide*: Submitter threads hold a read lock on `lock_` inside `submitPostSend()` while spinning for CQ credits that only the CQ-polling thread can release. A blocking write lock on the CQ-polling thread would wait for the reader to exit while preventing the CQ poller from draining completions, causing a deterministic deadlock.
2. **Falsifiable Hypothesis (Dynamo KV-Router Hybrid Model Quarantine)**: Validating `block_size == kv_block_size` on `KvEventOwnership::Kvcr` events before filtering out `NonMainAttentionGroup` events will falsely quarantine `CacheOwner` on the very first Mamba/sliding-window block-store event in hybrid models.
