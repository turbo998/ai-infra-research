# Distributed Training, Collectives & KV Storage Fabric: Megatron-LM Batch-Invariant Triton Sigmoid MoE Routing & DCP V2 Hybrid Propagation, TransformerEngine NVFP4 4-over-6 Factorized-Reciprocal Kernel (2.50x B200), PyTorch Raw-Philox Exponential Tails & Deterministic Bilinear2d Gather Backward (85.4x), RCCL `gfx1250` TDM/LL128 AllGather, Mooncake Shared `io_uring` Cloned Buffers (4.77x), LMCache Single-Copy TP KV Dedup (8.0x) & NIXL Vectorized Descriptor Merge (Phase 183)

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-batch-inv-router-dcp-v2-te-nvfp4-4over6-pytorch-philox-rccl-tdm-mooncake-uring-nixl-phase183.md)** | **[English](distributed-training-megatron-batch-inv-router-dcp-v2-te-nvfp4-4over6-pytorch-philox-rccl-tdm-mooncake-uring-nixl-phase183.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 183 (`priority: 192`) audits **11 upstream mainline commits** across Megatron-LM, TransformerEngine, PyTorch, RCCL (`rocm-systems`), Mooncake, LMCache, and NIXL:

1. **Megatron-LM Batch-Invariant Triton Sigmoid Top-K MoE Routing & DCP V2 Runtime Hybrid Layer Propagation**:
   - `611b430d5c02` (#8602) routes sigmoid MoE layers in `batch_invariant_mode` through the fused Triton sigmoid top-k kernel in both the training router and `InferenceTopKRouter` (replacing uncompiled `sigmoid + sorted aten.topk + gather + normalize`), maintaining **`0` ULP bitwise-identical routing** between trainer and engine while accelerating batch-invariant decode by **`0.57–0.62 ms/step`** on Nemotron-3-Nano.
   - `8817e0fd0a81` (#7048) completes **DCP V2 (4/4)** by propagating each packed microbatch's runtime Dynamic Context Parallelism topology through `HybridStack` layout conversion into Mamba, GDP, GDN/GDN2, MTP, MLA/absorbed MLA, and DSA/indexer layers, allowing models built at `CP=1` to consume successive `CP1 / CP2 / CP4` microbatches without mutating build-time geometry.
2. **TransformerEngine NVFP4 4-over-6 Factorized-Reciprocal CUDA Kernel (`2.50x` on B200) & In-Place Eager Softmax Backward**:
   - `f93ad4db05d7` (#3643) overhauls the Blackwell B200 NVFP4 4-over-6 quantization kernel: factorizing strict E2M1 candidate errors via $2^k \times \{1, 1.5\}$ cuts slow IEEE divisions from 1 per element (16 per candidate) to 2 reciprocals per candidate using `rcp.approx.ftz`; mapping consecutive threads to 16-element row groups eliminates `shfl_xor_sync`, reduces SMEM bank conflicts from 8-way to **2-way (`4x` reduction)**, and coalesces output writes into a single 32B sector; and hoisting per-tensor `rcp_amax` removes per-block divisions. On B200 (`8192 x 8192` BF16), this delivers **`2.50x` speedup on rowwise+columnwise strict** (`0.432 -> 0.173 ms`), **`2.34x` on per-tensor strict** (`0.145 -> 0.062 ms`), **`2.27x` on per-token strict** (`0.161 -> 0.071 ms`), **`2.17x` on 2D block strict** (`0.369 -> 0.170 ms`), and **`1.28x` on fast mode** (`0.109 -> 0.085 ms`) with `0` ULP bitwise identity.
   - `eb10c80d9f3b` (#3815) restores in-place gradient reuse via `overwrite_output_grads` in PyTorch eager `ScaledMaskedSoftmax`, `ScaledUpperTriangMaskedSoftmax`, and `ScaledAlignedCausalMaskedSoftmax` backward while bumping tensor version counters.
3. **PyTorch Raw-Philox 32-Bit Exponential Tails (Fixing $p = 10^{-8}$ Multinomial Bias) & Zero-Atomic Deterministic Bilinear2d Gather Backward (`85.44x`)**:
   - `e946fc999750` (#202464) fixes severe rare-event quantization in CUDA `exponential_()` (where converting 32-bit Philox ints to $(0, 1]$ FP32 floored small values to multiples of $2^{-24} \approx 5.96 \times 10^{-8}$ and broke Gumbel-max `torch.multinomial` below $p \approx 10^{-7}$): constructing $u \in [0, 1)$ directly from raw Philox bits with cubic Taylor $u + u^2/2 + u^3/3$ for $u < 2^{-8}$ and bitwise `~x` + `lg2.approx.ftz` elsewhere restores GB300 `torch.multinomial` rare-token sampling at $p = 10^{-8}$ from **`0.05x` (`4/81` hits) to `0.99x` (`80/81` hits)** across $8.1 \times 10^9$ draws with zero runtime penalty.
   - `735a2ec9b17c` (#197714) replaces the slow deterministic `upsample_bilinear2d` backward with an input-centric gather kernel (each thread gathers at most $4\times 4 = 16$ output gradients with 16-byte `aligned_vector` NHWC loads/stores and **`0` atomics**), accelerating `(48, 384, 256, 256) -> (512, 512)` BF16 NCHW from `1,022.66 ms` to **`11.97 ms` (`85.44x` vs old deterministic, and `14.19x` faster than the `169.81 ms` non-deterministic atomic baseline)** and NHWC BF16 from `1,076.62 ms` to **`9.98 ms` (`107.88x`)**.
4. **RCCL `gfx1250` SPX TDM / DPX DDA LL128 AllGather, Mooncake Shared `io_uring` Cloned Buffers (`4.77x`), LMCache Single-Copy TP KV Dedup (`8.0x`) & NIXL Vectorized Descriptor Merge (`2.51x`)**:
   - `0faf966d5a23` (#14339) enables the `gfx1250` SPX symmetric TDM AllGather kernel for `[64 MiB, 128 MiB]` eager and up to `4 GiB` intra-tray / `512 MiB` cross-UALink-tray under HIP graph capture while relaxing tile-store waits from every `2` peers to every **`4` peers**; `478db7d54462` (#14710) optimizes `gfx1250` DPX `1–32 MiB` DDA LL128 AllGather with batched-load polling, cooperative atomic barrier removal, and a $\le 75$-block (256-thread) cap.
   - Mooncake `ed0986045ba4` (#4788) splits $> 1\text{ GiB}$ staging buffers into $\le 1\text{ GiB}$ iovecs registered once on a source ring and cloned via `io_uring_clone_buffers()` across **`385` worker rings** (`32 GiB` pool), eliminating `errno=14` (`EFAULT`) registration failures and lifting cold SSD KV reads from `4.4 GB/s` (`pread` fallback) to **`15–21 GB/s` (`4.773x` speedup)**; `19ded3ff83cd` (#5226) drains in-flight MemFabric batches before teardown.
   - LMCache `dc8c116decb8` (#6125) threads `(tp_replicas, dcp_shards)` into IPC-to-object key generation so `LMCacheMPConnector` stores TP-replicated KV groups (such as Kimi K3 `TP=8` MLA compressed latent KV alongside sharded KDA state) once per TP group, cutting 256-token FP8 chunk writes from `28.0 MB` to **`3.5 MB` (`8.0x` reduction)** and 1536-token chunks from `170 MB` to `21 MB`; `298140c41690` (#6266) adds fd-based VMM memory sharing.
   - NIXL `732612574c2c` (#2434) vectorizes contiguous descriptor range merging prior to `makeXferReq`, achieving **`2.512x` speedup on x86_64** (`19.34 -> 7.70 us` at `20,000 -> 128`) and **`2.256x` on ARM64** (`13.90 -> 6.16 us`), cutting DeepSeek-V4-Flash `ISL=100k` `makeXferReq` latency at concurrency 16 from `69.7 us` to **`44.4 us` (`-36.30%`)**.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 183 Distributed Training, Collectives & KV Fabric: Megatron, TE, PyTorch, RCCL, Mooncake & NIXL  |
+--------------------------------------------------------------------------------------------------------+
  [Megatron-LM Batch-Invariant Sigmoid Router & DCP V2 + TE NVFP4 4-over-6 (#8602, #7048, #3643, #3815)]
    |-- Megatron BI Router (#8602)   --> Shared Triton Sigmoid Top-K in Train & Infer (0 ULP, -0.62 ms/step)
    |-- Megatron DCP V2 (#7048)      --> Runtime CP1/CP2/CP4 Propagation Across Mamba/GDN/MTP/MLA/DSA
    |-- TE NVFP4 4-over-6 (#3643)    --> Factorized 2^k*{1,1.5} + 2-Way SMEM (2.50x Row+Col Strict on B200)
    +-- TE In-Place Softmax (#3815)  --> Eager Attention Backward overwrite_output_grads Buffer Reuse
                                      |
                                      v
  [PyTorch Raw-Philox Exponential Tails & Deterministic Bilinear2d Gather (#202464, #197714)]
    |-- Raw-Philox Exp Tail (#202464)--> Taylor u<2^-8 + ~x lg2.approx (p=1e-8 Multinomial 0.05x -> 0.99x)
    +-- Det Bilinear2d Bwd (#197714) --> Input-Centric 16B NHWC Gather, 0 Atomics (85.44x vs Det, 14.19x vs Atomic)
                                      |
                                      v
  [RCCL gfx1250 TDM/LL128 + Mooncake io_uring Clone + LMCache TP Dedup + NIXL (#14339, #4788, #6125, #2434)]
    |-- RCCL gfx1250 AG (#14339/#14710)-> SPX Symmetric TDM AG (4-Peer Wait, 4 GiB) + DPX DDA LL128 (<=75 Blks)
    |-- Mooncake io_uring (#4788)    --> <=1 GiB Slices Cloned Across 385 Rings (4.4 -> 21.0 GB/s, 4.773x)
    |-- LMCache TP Dedup (#6125)     --> Store TP-Replicated MLA KV Once per TP8 Group (28 -> 3.5 MB, 8.0x)
    +-- NIXL Vec Desc Merge (#2434)  --> 2.512x x86 / 2.256x ARM64 Merge & -36.30% DSv4 ISL=100k makeXferReq
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/megatron_batch_inv_router_dcp_v2_te_nvfp4_4over6_pytorch_philox_rccl_tdm_mooncake_uring_nixl_phase183/run.py`](../../experiments/megatron_batch_inv_router_dcp_v2_te_nvfp4_4over6_pytorch_philox_rccl_tdm_mooncake_uring_nixl_phase183/run.py) `--check`:
- `megatron_bi_sigmoid_decode_saved_max_ms = 0.62`, `megatron_bi_train_infer_ulp_drift = 0`
- `te_nvfp4_4o6_rowcol_speedup = 2.50`, `te_nvfp4_4o6_pertensor_speedup = 2.34`, `te_nvfp4_4o6_smem_bank_conflict_reduction = 4`
- `pytorch_philox_p1e8_prev_rate_ratio = 0.05`, `pytorch_philox_p1e8_new_rate_ratio = 0.99`
- `pytorch_bilinear2d_bwd_vs_old_det_speedup = 85.44`, `pytorch_bilinear2d_bwd_vs_atomic_speedup = 14.19`
- `rccl_gfx1250_tdm_ag_tile_wait_peers = 4`, `rccl_gfx1250_tdm_ag_intratray_max_mib = 4096`, `rccl_gfx1250_dda_ll128_max_blocks = 75`
- `mooncake_uring_cold_read_speedup = 4.773` (`4.4 -> 21.0 GB/s`, `385` rings), `lmcache_tp8_storage_reduction_ratio = 8.0`, `nixl_desc_merge_x86_speedup = 2.512`, `nixl_dsv4_isl100k_latency_saving_pct = 36.30`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Blackwell B200/GB300 NVFP4 Pretraining & RL Rare-Token Exploration** | Enable TE's factorized-reciprocal NVFP4 4-over-6 kernel (`#3643`, `2.50x` row+col strict), Megatron batch-invariant Triton sigmoid routing (`#8602`), and PyTorch raw-Philox exponential tails (`#202464`, restoring $p = 10^{-8}$ multinomial accuracy from `0.05x` to `0.99x`). | Sampling rare tail tokens ($p \le 10^{-7}$) in RL/post-training using legacy FP32-cast `exponential_()` (quantizes Gumbel noise to $2^{-24}$ steps and loses 95% of $p=10^{-8}$ hits). |
| **Deterministic Multimodal Vision Training & Dynamic Context Parallelism (DCP V2)** | Use PyTorch's input-centric gather `bilinear2d` backward under `torch.use_deterministic_algorithms(True)` (`#197714`, `85.44x–107.88x` speedup) and Megatron DCP V2 runtime `CP1/CP2/CP4` propagation across hybrid Mamba/MLA/DSA models (`#7048`). | Running legacy output-centric deterministic bilinear backward (`>1.0 s` per call) or rebuilding static model geometry for mixed sequence-length microbatches. |
| **32 GiB+ NVMe/RDMA Disaggregated KV Pools & Hybrid MLA+KDA Multi-Process Caching** | Use Mooncake's $\le 1\text{ GiB}$ sliced `io_uring_clone_buffers()` across 385 rings (`#4788`, `21 GB/s` cold SSD read), LMCache `(tp_replicas, dcp_shards)` single-copy TP deduplication on Kimi K3 `TP=8` (`#6125`, `8.0x` savings), and NIXL vectorized descriptor merging (`#2434`). | Calling `io_uring_register_buffers` on single $>1\text{ GiB}$ iovecs or repeating registration per worker ring (fails with `EFAULT`/`ENOMEM` and degrades to `4.4 GB/s` `pread`). |
