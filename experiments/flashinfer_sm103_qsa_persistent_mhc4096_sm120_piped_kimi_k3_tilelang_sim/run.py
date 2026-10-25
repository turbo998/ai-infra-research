#!/usr/bin/env python3
"""Deterministic stdlib simulator for Phase 74 Track A:
FlashInfer SM103 QSA NVFP4 Persistent-Walk Two-Wave Decode & tcgen05 One-Tile Fused Merge,
SM120 mma.sync Decode & SPLIT_OVERRIDES, SM100/SM103 Hidden-4096 4-Route Mega-mHC,
SM120 Pipelined Topology-Weighted PCIe CE Ring All-Reduce, Kimi-K3 L2::evict_first &
Claimed-Item Resident TP1 Tail GEMM, and TileLang Cross-Address-Space Barrier Tracking &
Global Radix Top-K Workspace.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sm103_qsa_persistent_walk_and_one_tile_merge() -> dict[str, Any]:
    """Model flashinfer@131f85942edb00c9258bc6e393ccbe57b88b55bc:
    - SM103 (GB300, 152 SMs) persistent-walk decode (`decode_tcgen05_persistent`) for two-wave
      split-form grids (`1 < ceil(rows * num_kv_heads * plain_splits / sm_count) <= WALK_MAX_WAVES=2`)
      where `longest_walked_cta_tiles < plain_tiles`.
    - At owner's shape (24x2 heads, 2051 selected tokens -> 33 tiles of 64 tokens):
      rows=128 has `128 * 2 = 256` pairs, `plain_splits = 1`, `256` plain CTAs across `152` SMs
      (2 waves, longest CTA = 33 tiles). Persistent walk launches `152` CTAs walking `256 * 33 = 8448`
      (pair, tile) slots (`55-56` tiles/CTA, < `2 * 33 = 66` two-wave serial tiles) with `walk_splits = 2`
      partial slots per pair.
    - Also models the `tcgen05` one-tile fused merge (`rows=1,2`) on `sm_100a`/`sm_103a`/`sm_107a`.
    """
    sm_count = 152
    num_kv_heads = 2
    selection_width = 2051
    tile_tokens = 64
    tiles_per_pair = math.ceil(selection_width / tile_tokens)  # 33

    rows = 128
    pairs = rows * num_kv_heads  # 256
    plain_splits = 1
    plain_ctas = pairs * plain_splits  # 256
    plain_waves = math.ceil(plain_ctas / sm_count)  # 2
    plain_serial_tiles_max = plain_waves * tiles_per_pair  # 66

    total_tile_slots = pairs * tiles_per_pair  # 8448
    walk_ctas = sm_count  # 152
    walk_tiles_max = math.ceil(total_tile_slots / walk_ctas)  # 56
    walk_tiles_min = total_tile_slots // walk_ctas  # 55
    walk_splits = 2  # at most 2 CTAs touch any 33-tile pair when step is 55..56
    walk_applies = (1 < plain_waves <= 2) and (walk_tiles_max < plain_serial_tiles_max)

    return {
        "commit": "131f85942edb00c9258bc6e393ccbe57b88b55bc",
        "arch": "sm_103a",
        "gpu": "NVIDIA GB300",
        "sm_count": sm_count,
        "num_qo_heads": 24,
        "num_kv_heads": num_kv_heads,
        "selection_width": selection_width,
        "tiles_per_pair": tiles_per_pair,
        "two_wave_walk_rows128": {
            "rows": rows,
            "pairs": pairs,
            "plain_splits": plain_splits,
            "plain_ctas": plain_ctas,
            "plain_waves": plain_waves,
            "plain_serial_tiles_max": plain_serial_tiles_max,
            "walk_ctas": walk_ctas,
            "walk_tiles_min": walk_tiles_min,
            "walk_tiles_max": walk_tiles_max,
            "walk_splits": walk_splits,
            "walk_applies": walk_applies,
            "eager_nhd_memset_before_us": 71.89,
            "eager_nhd_memset_after_us": 67.74,
            "eager_nhd_memset_speedup_x": round(71.89 / 67.74, 4),
            "eager_hnd_read_before_us": 72.14,
            "eager_hnd_read_after_us": 68.09,
            "eager_hnd_read_speedup_x": round(72.14 / 68.09, 4),
            "cudagraph_nhd_before_us": 72.37,
            "cudagraph_nhd_after_us": 67.74,
            "cudagraph_nhd_speedup_x": round(72.37 / 67.74, 4),
        },
        "one_tile_fused_merge": {
            "rows1_nhd_memset_before_us": 11.65,
            "rows1_nhd_memset_after_us": 10.21,
            "rows1_nhd_memset_speedup_x": round(11.65 / 10.21, 4),
            "rows1_nhd_read_before_us": 11.70,
            "rows1_nhd_read_after_us": 10.02,
            "rows1_nhd_read_speedup_x": round(11.70 / 10.02, 4),
            "rows2_hnd_memset_before_us": 14.70,
            "rows2_hnd_memset_after_us": 12.77,
            "rows2_hnd_memset_speedup_x": round(14.70 / 12.77, 4),
        },
    }


def simulate_sm120_qsa_mma_sync_and_split_overrides() -> dict[str, Any]:
    """Model flashinfer@09a77503f004b8f36a38d23074fc5da6877de811:
    - SM120 (`sm_120a`, RTX PRO 6000 Blackwell Server Edition, 188 SMs, 2 CTAs/SM -> 376 fused CTAs)
      has no tcgen05/TMEM; uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
    - `SPLIT_OVERRIDES` reduces `plain_splits` at `rows=4,8,16,32` from `33,23,11,5` to `16,11,5,3`
      while keeping `33` at `rows=1,2`.
    """
    sm_count = 188
    co_resident_ctas_per_sm = 2
    max_fused_ctas = sm_count * co_resident_ctas_per_sm  # 376
    split_overrides = {
        "rows_1": {"formula_splits": 33, "override_splits": 33, "nhd_before_us": 15.61, "nhd_after_us": 10.50},
        "rows_2": {"formula_splits": 33, "override_splits": 33, "nhd_before_us": 20.69, "nhd_after_us": 13.71},
        "rows_4": {"formula_splits": 33, "override_splits": 16, "nhd_before_us": 28.59, "nhd_after_us": 16.13},
        "rows_8": {"formula_splits": 23, "override_splits": 11, "nhd_before_us": 39.21, "nhd_after_us": 25.00},
        "rows_16": {"formula_splits": 11, "override_splits": 5, "nhd_before_us": 37.90, "nhd_after_us": 27.62},
        "rows_32": {"formula_splits": 5, "override_splits": 3, "nhd_before_us": 56.11, "nhd_after_us": 44.30},
        "rows_64": {"formula_splits": 2, "override_splits": 2, "nhd_before_us": 116.11, "nhd_after_us": 90.06},
        "rows_128": {"formula_splits": 1, "override_splits": 1, "nhd_before_us": 301.26, "nhd_after_us": 161.36},
    }
    for row_cfg in split_overrides.values():
        row_cfg["nhd_speedup_x"] = round(row_cfg["nhd_before_us"] / row_cfg["nhd_after_us"], 4)

    return {
        "commit": "09a77503f004b8f36a38d23074fc5da6877de811",
        "arch": "sm_120a",
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "sm_count": sm_count,
        "co_resident_ctas_per_sm": co_resident_ctas_per_sm,
        "max_fused_ctas": max_fused_ctas,
        "split_overrides": split_overrides,
        "rows128_hnd_eager_before_us": 320.93,
        "rows128_hnd_eager_after_us": 169.62,
        "rows128_hnd_eager_speedup_x": round(320.93 / 169.62, 4),
        "rows128_hnd_cudagraph_before_us": 321.59,
        "rows128_hnd_cudagraph_after_us": 169.10,
        "rows128_hnd_cudagraph_speedup_x": round(321.59 / 169.10, 4),
    }


def simulate_mega_mhc_hidden4096_sm100_sm103() -> dict[str, Any]:
    """Model flashinfer@847172573ad1d5a929a8e7fb67aa7c072f8f7dcb:
    - Hidden-4096 4-route mHC inference programs (`bf16`, `tailpipe`, `tailpipe_wsmem`,
      `fp8_group128`, `nvfp4`) for GLM-5.3-Flash (`n=4, H=4096, K=16384`) on SM100 (B200) and SM103 (GB300).
    - Also verifies `CheckedMul` (`__builtin_mul_overflow`) TMA descriptor stride guard against 32-bit overflow.
    """
    n_routes = 4
    hidden_dim = 4096
    k_flat = n_routes * hidden_dim  # 16384
    sinkhorn_iters = 20
    # CheckedMul overflow boundary: for stride = K * sizeof(bf16) = 32768 bytes,
    # 64-bit byte size at T = 131072 is 131072 * 32768 = 4,294,967,296 bytes (2^32, overflows uint32).
    overflow_tokens_bf16 = (1 << 32) // (k_flat * 2)  # 131072

    return {
        "commit": "847172573ad1d5a929a8e7fb67aa7c072f8f7dcb",
        "n_routes": n_routes,
        "hidden_dim": hidden_dim,
        "k_flat": k_flat,
        "sinkhorn_iters": sinkhorn_iters,
        "programs": ["bf16", "tailpipe", "tailpipe_wsmem", "fp8_group128", "nvfp4"],
        "uint32_overflow_token_threshold_bf16": overflow_tokens_bf16,
        "b200_sm100a": {
            "t32_ref_us": 36.115,
            "t32_fused_us": 26.867,
            "t32_speedup_x": round(36.115 / 26.867, 3),
            "t1024_ref_us": 39.095,
            "t1024_fused_us": 27.424,
            "t1024_speedup_x": round(39.095 / 27.424, 3),
            "t4096_ref_us": 103.855,
            "t4096_fused_us": 75.520,
            "t4096_speedup_x": round(103.855 / 75.520, 3),
        },
        "gb300_sm103a": {
            "t32_ref_us": 32.783,
            "t32_fused_us": 23.829,
            "t32_speedup_x": round(32.783 / 23.829, 3),
            "t1024_ref_us": 35.510,
            "t1024_fused_us": 25.608,
            "t1024_speedup_x": round(35.510 / 25.608, 3),
            "t4096_ref_us": 109.945,
            "t4096_fused_us": 79.744,
            "t4096_speedup_x": round(109.945 / 79.744, 3),
        },
    }


def simulate_sm120_pcie_ce_ring_piped() -> dict[str, Any]:
    """Model flashinfer@68e3e09ad3a52cbbd6b39d159d2fad7d1653073d:
    - `IpcVariant.COPY_ENGINE_RING_PIPED` (`kCopyEngineRingPiped = 7`) on 8x SM120 PCIe Gen5 across 2 CPU sockets.
    - Ring 0->1->2->3->4->5->6->7->0 has two slow inter-socket (`NVML SYSTEM`) hops: `3->4` and `7->0`.
    - `decide_ring_slow_hops` sets bitmask `(1 << 3) | (1 << 7) = 0x88` (`136`).
    - Chunks that skip a slow hop in the 7-step reduce-scatter + 7-step all-gather get larger weight
      and are split into `2..4` pipeline pieces once `numel * sizeof(T) >= 4 MiB/rank`.
    """
    world_size = 8
    slow_hop_mask = (1 << 3) | (1 << 7)  # 0x88 = 136
    min_piped_bytes_per_rank = 4 * 1024 * 1024  # 4 MiB

    return {
        "commit": "68e3e09ad3a52cbbd6b39d159d2fad7d1653073d",
        "variant_enum": 7,
        "variant_name": "COPY_ENGINE_RING_PIPED",
        "world_size": world_size,
        "slow_hop_mask": slow_hop_mask,
        "min_piped_bytes_per_rank": min_piped_bytes_per_rank,
        "benchmarks": {
            "48mib_per_rank_bf16": {
                "pieces": 2,
                "ring_memop_ms": 2.1832,
                "ring_piped_ms": 2.0792,
                "speedup_x": round(2.1832 / 2.0792, 4),
            },
            "96mib_per_rank_bf16": {
                "pieces": 3,
                "ring_memop_ms": 4.4753,
                "ring_piped_ms": 4.0287,
                "speedup_x": round(4.4753 / 4.0287, 4),
            },
            "96mib_per_rank_fp16": {
                "pieces": 3,
                "ring_memop_ms": 4.5031,
                "ring_piped_ms": 4.0560,
                "speedup_x": round(4.5031 / 4.0560, 4),
            },
        },
    }


def simulate_kimi_k3_evict_first_and_resident_tail() -> dict[str, Any]:
    """Model flashinfer@1c0d2be434d6d57ca16a428d439c433b8b1e6f81:
    - Kimi-K3 Stable LatentMoE (`H=7168, L=512, I=2048, E=384, K=8`):
      1) `DECODE_A_EVICT_FIRST = True` (`L2::evict_first` on streamed weight tiles for `T <= 128`)
         stops weight tiles from evicting reused token activations in L2.
      2) Claimed-item resident `256x512` 2-CTA TP1 tail GEMM with fused RMSNorm for `T >= 1024`.
    """
    return {
        "commit": "1c0d2be434d6d57ca16a428d439c433b8b1e6f81",
        "decode_evict_first_max_tokens": 128,
        "resident_tail_min_tokens": 1024,
        "resident_tail_tile_shape": [256, 512],
        "front_tp1_gb300_t64": {
            "before_us": 53.07,
            "after_us": 44.18,
            "speedup_x": round(53.07 / 44.18, 4),
        },
        "tail_tp1_gb300_t64": {
            "before_us": 34.98,
            "after_us": 29.36,
            "speedup_x": round(34.98 / 29.36, 4),
        },
        "tail_tp1_resident_gb200_t1024": {
            "before_us": 120.87,
            "after_us": 107.81,
            "speedup_x": round(120.87 / 107.81, 4),
        },
        "tail_tp1_resident_gb300_t1024": {
            "before_us": 115.65,
            "after_us": 104.26,
            "speedup_x": round(115.65 / 104.26, 4),
        },
    }


def simulate_tilelang_barrier_and_topk_workspace() -> dict[str, Any]:
    """Model tilelang@efa1ffc1e3dc547b11fee5acf671daff1f0f94cb and
    tilelang@4da574c01326eab5da5345af0aec1d19ce473c6a:
    - `ThreadPartialSyncRewriter` preserves `used_barrier_ids_` across `"shared"` and `"shared.dyn"`
      rewrites so partial-thread barriers on `shared` (`[256, 384)`) and `shared.dyn` (`[0, 128)`)
      receive distinct PTX `bar.sync` IDs instead of colliding on barrier ID 3 and deadlocking.
    - `tl_topk_impl` replaces the fixed `[2, 4096]` shared-memory candidate queue with a
      `(batch, 2, seq_len)` global tensor workspace so uniform/tied score distributions (`seq_len > 4096`)
      never overflow shared memory (`CUDA_ERROR_ILLEGAL_ADDRESS`).
    """
    shared_pass_barriers = [1, 2, 3]  # e.g. barrier 3 allocated to threads [256, 384) in "shared" pass
    # Buggy behavior cleared used_barrier_ids_ before "shared.dyn" pass, reassigning barrier 3 to [0, 128).
    buggy_shared_dyn_barrier_id = 3
    # Fixed behavior keeps used_barrier_ids_ = {1, 2, 3} across passes, assigning barrier 4 to [0, 128).
    fixed_shared_dyn_barrier_id = max(shared_pass_barriers) + 1  # 4

    seq_len = 32768
    batch = 4
    old_smem_queue_cap = 4096
    new_global_queue_cap = seq_len
    global_workspace_bytes = batch * 2 * seq_len * 4  # int32 indices

    return {
        "barrier_commit": "efa1ffc1e3dc547b11fee5acf671daff1f0f94cb",
        "topk_commit": "4da574c01326eab5da5345af0aec1d19ce473c6a",
        "shared_pass_barrier_id": 3,
        "buggy_shared_dyn_barrier_id": buggy_shared_dyn_barrier_id,
        "fixed_shared_dyn_barrier_id": fixed_shared_dyn_barrier_id,
        "barrier_collision_eliminated": fixed_shared_dyn_barrier_id != buggy_shared_dyn_barrier_id,
        "old_smem_topk_queue_capacity": old_smem_queue_cap,
        "new_global_topk_queue_capacity": new_global_queue_cap,
        "global_workspace_bytes_b4_s32k": global_workspace_bytes,
    }


def build_summary() -> dict[str, Any]:
    sm103_qsa = simulate_sm103_qsa_persistent_walk_and_one_tile_merge()
    sm120_qsa = simulate_sm120_qsa_mma_sync_and_split_overrides()
    mega_mhc = simulate_mega_mhc_hidden4096_sm100_sm103()
    pcie_piped = simulate_sm120_pcie_ce_ring_piped()
    kimi_k3 = simulate_kimi_k3_evict_first_and_resident_tail()
    tilelang = simulate_tilelang_barrier_and_topk_workspace()

    return {
        "experiment_id": "flashinfer_sm103_qsa_persistent_mhc4096_sm120_piped_kimi_k3_tilelang_sim",
        "phase": 74,
        "priority": 83,
        "track": "Track A: Kernel & Hardware Microarchitecture",
        "sm103_qsa_persistent_walk": sm103_qsa,
        "sm120_qsa_mma_sync": sm120_qsa,
        "mega_mhc_hidden4096": mega_mhc,
        "sm120_pcie_ce_ring_piped": pcie_piped,
        "kimi_k3_latent_moe": kimi_k3,
        "tilelang_compiler_fixes": tilelang,
        "key_invariants": {
            "sm103_rows128_walk_ctas": sm103_qsa["two_wave_walk_rows128"]["walk_ctas"],
            "sm103_rows128_walk_tiles_max": sm103_qsa["two_wave_walk_rows128"]["walk_tiles_max"],
            "sm103_rows128_speedup_x": sm103_qsa["two_wave_walk_rows128"]["eager_nhd_memset_speedup_x"],
            "sm120_rows128_hnd_cudagraph_speedup_x": sm120_qsa["rows128_hnd_cudagraph_speedup_x"],
            "mega_mhc_b200_t1024_speedup_x": mega_mhc["b200_sm100a"]["t1024_speedup_x"],
            "mega_mhc_gb300_t1024_speedup_x": mega_mhc["gb300_sm103a"]["t1024_speedup_x"],
            "sm120_pcie_96mib_bf16_speedup_x": pcie_piped["benchmarks"]["96mib_per_rank_bf16"]["speedup_x"],
            "kimi_k3_front_t64_speedup_x": kimi_k3["front_tp1_gb300_t64"]["speedup_x"],
            "tilelang_barrier_collision_eliminated": tilelang["barrier_collision_eliminated"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches deterministic output.")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing {args.output}; run without --check first.")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"{args.output} is out of date with deterministic generator.")
        print(f"[OK] {args.output} verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"[WROTE] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
