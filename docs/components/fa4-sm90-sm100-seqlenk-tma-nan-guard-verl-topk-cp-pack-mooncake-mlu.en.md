# FlashAttention-4 SM90/SM100 Load-Safe 5D Ragged TMA `seqlen_k` Zero-Fill (`0 * NaN` Guard), `verl` Megatron Top-k Distillation `quant_align_size` & `cp_layout` Packing, and Mooncake TENT Async Cambricon MLU Host-Device Copies

> **Document Status**: Phase 77 (`priority: 86`) First-Hand Source Audit + CPU/Offline Reproducible Verification  
> **Target Audience**: Solution Architects working on Hopper/Blackwell TMA descriptor wraparound, Megatron-LM Context Parallel (CP) + MXFP8 online distillation, and heterogeneous AI accelerator (Cambricon MLU) transfer engines  
> **中文伴侣文档**: [fa4-sm90-sm100-seqlenk-tma-nan-guard-verl-topk-cp-pack-mooncake-mlu.md](fa4-sm90-sm100-seqlenk-tma-nan-guard-verl-topk-cp-pack-mooncake-mlu.md)

---

## 1. Architectural Context & Root Causes

1. **FlashAttention-4 `0 * NaN = NaN` Poisoning from Unused K/V Rows Past `seqlen_k` (`#2983`, `#3012`, `#3013`)**: When a preallocated KV cache or packed varlen tensor contains uninitialized `NaN` rows beyond `seqlen_k` (`cu_seqlens_k` or `seqused_k`), online softmax masks out-of-bounds attention weights to $P_{ij} = 0$ (and $dS_{ij} = 0$ in backward). However, IEEE-754 FMA evaluates $0 \times \text{NaN} = \text{NaN}$ during $O = P V$, $dP = V dO^T$, and $dQ = dS K$ if those out-of-bounds K/V rows are loaded by TMA into shared memory. Moreover, `quack`'s store-only `ptr_shift=True` 3D descriptor shifts the base pointer outside mapped memory (causing MMU faults on TMA loads) and only supports rank $\le 3$.
2. **`verl` Megatron Top-k Distillation MXFP8 Assertion Crash (`#8325`) & Silent Token Misalignment Under Mixed `cp_layout` (`#8332`)**:
   - Passing a boolean `use_fp8_padding` failed under MXFP8 (`quant_align_size = 32` vs `16` for blockwise FP8).
   - Worse (`#8332`), when a `dsv4_hybrid` student used `cp_layout="contiguous"` while the teacher defaulted to `"zigzag"` at `CP=2` and `MXFP8` (`quant_align_size=32`), 16 sequences of 17 valid tokens produced the exact same packed shape (`256` rows/rank) on both sides—raising no assertion error while silently pairing **128 of 136 valid student tokens on rank 0 (`94.1%`) and 136 of 136 on rank 1 (`100%`)** with the wrong teacher token or padding row!
3. **Synchronous Copy Bottleneck & Unregister Rollback Races in Mooncake TENT on Cambricon MLU (`#4720`)**: Previously, MLU host-device copies in TENT were strictly synchronous, preventing overlap between device DMA and TCP/HP-TCP network transfers.

---

## 2. First-Hand Source Audit & Mathematical Mechanics

### 2.1 FlashAttention-4 Load-Safe 5D Ragged TMA Descriptor (`#2983`, `#3012`, `#3013`)

In `flash_attn/cute/copy_utils.py`, `flash_fwd_sm100.py`, `flash_bwd_sm100.py`, `flash_fwd_sm90.py`, and `flash_bwd_sm90.py`, `create_ragged_tensor_for_tma_load` and `offset_ragged_tensor_for_tma_load` construct a 5D descriptor using 64-bit unsigned address wraparound:

$$\text{wrap\_stride} = \frac{2^{40}}{\text{elem\_bytes}} \text{ elements}, \qquad \text{ragged\_rows} = \frac{2^{64}}{\text{wrap\_stride}} \quad (2^{25} = 33,554,432 \text{ rows for FP16/BF16})$$

$$\text{Shape}_{5\text{D}} = \left(\text{features},\; \text{ragged\_rows},\; 2^{30} + 1,\; 2^{31} - 1,\; \text{ragged\_rows} + 1\right)$$

with element strides $(1,\; \text{row\_stride},\; \text{wrap\_stride},\; \text{wrap\_stride},\; 0)$. Because every byte stride is $\le 2^{40}$ bytes and the base pointer remains at the tensor's valid start address, offsetting dimension 1 by $\text{ragged\_rows} - \text{seqlen\_k}$ wraps valid rows $0 \le r < \text{seqlen\_k}$ back to their exact in-memory address modulo $2^{64}$ while mapping every row $r \ge \text{seqlen\_k}$ to coordinate $\ge \text{ragged\_rows}$, where TMA hardware clamps out-of-bounds loads to `0.0` before shared memory or MMA registers:

- **`#2983`**: Protects SM100 forward `V` loads (`194` existing B200 cases + `146` new `NaN`-tail regressions, $\le 2.23\%$ max latency overhead).
- **`#3012`**: Generalizes `create_ragged_tensor_for_tma_load(tensor, rows_dim=0)` for `(rows, features)` layouts and bounds `K`, `V`, and 2-CTA `Kt` loads in `flash_bwd_sm100.py`.
- **`#3013`**: Extends hardware OOB zero-fill to SM90 Hopper forward (`V`) and backward (`K`, `V`), replacing shared-memory partial-tile zeroing (`#2407`).

### 2.2 `verl` Megatron Top-k Distillation `quant_align_size` (`#8325`) & `cp_layout` (`#8332`)

1. **Recipe-Agnostic `get_thd_quant_align_size(config)` (`#8325`)**: Returns `32` for `mxfp8`, `16` for blockwise/other FP8 recipes, and `None` for BF16; pads each sequence to $\text{lcm}(\text{quant\_align\_size}, 2 \cdot \text{CP})$ and total tokens to $2 \cdot \text{CP} \cdot 128$, passing 5/5 steps on 2×B200 (`rollout_corr/kl` `0.0018–0.0026`).
2. **Forwarding `cp_layout` and `local_cp_size` (`#8332`)**: Passes `cp_layout` and `local_cp_size` into `compute_forward_kl_topk` so teacher top-k tensors unpack with the exact same CP layout (`"contiguous"` vs `"zigzag"`) as the student.

### 2.3 `Mooncake@a7496800527b` (`#4720`): Async Cambricon MLU Copies & `ProgressWorker` Polling

- **Per-Context, Per-Direction `MluCopyQueue` (`CNqueue` + `CNnotifier`)**: Pinned (`cnrtHostMalloc`) D2H copies and pinned H2D copies $> 64\text{ KiB}$ run asynchronously via `cnMemcpyAsync_V3` with `CNnotifier` completion tracking. H2D copies $\le 64\text{ KiB}$ run synchronously via `cnMemcpyHtoD` when the context's H2D queue is absent or idle; if prior H2D work is pending, pinned copies join the queue while pageable copies drain the queue before copying synchronously, preserving intra-context H2D order.
- **`ProgressWorker` `poll_until_terminal` & `SegmentTracker` Rollback**: `SubBatch::needsProgressPolling()` opts MLU batches into `ProgressWorker::polling_` so batches without completion interrupts are polled until terminal before caller buffers are released, and `SegmentTracker` serializes registration rollback under its mutex.

---

## 3. Offline Verification Matrix

Validated by `experiments/fa4_sm90_sm100_seqlenk_tma_nan_guard_verl_topk_cp_pack_mooncake_mlu/run.py --check`:

| Subsystem | Scenario | Legacy Behavior | Phase 77 Fixed Behavior | Invariant |
|---|---|---|---|---|
| FA4 5D Ragged TMA (`#2983`, `#3012`, `#3013`) | FP16/BF16 (`b=2`), `seqlen_k=50`, `tile_n=128` with `NaN` tail | `0.0 * NaN` poisons `O` and `dQ` to **`NaN`** | 5D shape `(128, 2^25, 2^30+1, 2^31-1, 2^25+1)` hardware zero-fills OOB rows; output is **`1.0`** | Stride $\le 2^{40}\text{ B}$, zero `NaN` |
| `verl` Top-k Distillation (`#8325`, `#8332`) | `CP=2`, `MXFP8` (`align=32`), 16 seqs of 17 tokens (`256` rows/rank) | Shapes match (`256==256`), silently misaligning **128** (rank 0) & **100%** (rank 1) valid tokens | Forwards `quant_align_size=32` & `cp_layout="contiguous"`, yielding **`0`** misaligned tokens | Zero silent KL corruption |
| Mooncake TENT MLU (`#4720`) | `64 KiB` H2D threshold + `CNnotifier` + `SegmentTracker` | Synchronous blocking; unregister race on sync failure | Idle $\le 64\text{ KiB}$ inline sync; async `CNqueue` + `poll_until_terminal` otherwise | Ordering & lifetime safe |

---

## 4. Architect Exercises & Falsifiable Follow-Ups

1. **Wraparound Derivation Exercise**: Compute the maximum supported `ragged_rows` in FlashAttention-4's 5D TMA descriptor for FP8 (`elem_bytes = 1`) and FP32 (`elem_bytes = 4`).
2. **Falsifiable Hypotheses**:
   - **H1**: In `flash_bwd_sm100.py`, bounding 2-CTA `Kt` loads via the 5D ragged TMA descriptor adds less than `1.5%` overhead even at short sequences (`seqlen_k = 256`).
   - **H2**: Lowering Mooncake TENT's inline H2D threshold on Cambricon MLU from `64 KiB` to `16 KiB` under 8 concurrent staging streams reduces P99 transfer tail latency by increasing DMA/network overlap.
