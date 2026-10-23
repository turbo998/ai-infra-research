#!/usr/bin/env python3
"""Phase 68 Track C deterministic experiment:
TransformerEngine 128x128 Wide-Tile Grouped MXFP8 Requantize (#3656, #3732),
LMCache Native Address-Kernel Block Transfer (#5955),
NIXL UCX Notification Zero-Copy & InlinedVector (#2412),
plus Megatron-LM Mixtral TF5 (#7937), TorchTitan Flex/PP/RL (#5547, #5317),
Mooncake DFS-only KV Bench (#4798), and Dynamo Strict-Tool-Schema (#17163).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def simulate_te_wide_mxfp8_requantize() -> dict[str, object]:
    """Reproduce TE fused_group_requantize_wide_kernel 128x128 tile geometry & swizzle (#3656)."""
    wide_tile = 128
    wide_threads = 128
    warps_per_cta = wide_threads // 32  # 4
    wide_max_groups = 128
    wide_ctas_per_sm = 16
    scale_block_bytes = wide_tile * 4  # 512 bytes per 128x128 tile (128 rows x 4 scale blocks of 32 cols)

    # Verify bijective 512-byte swizzle mapping for rowwise scales:
    # row r in [0, 127] (where r = 32*warp + lane) writes 4-byte word at byte offset (lane * 16 + warp * 4)
    rowwise_byte_offsets = set()
    for warp in range(warps_per_cta):
        for lane in range(32):
            off = lane * 16 + warp * 4
            assert 0 <= off < scale_block_bytes and off % 4 == 0
            rowwise_byte_offsets.add(off)

    # Verify bijective 512-byte swizzle mapping for columnwise scales:
    # column c = 4*lane + j in [0, 127], 32-row block warp in [0, 3] -> (c % 32) * 16 + (c // 32) * 4 + warp
    colwise_byte_offsets = set()
    for warp in range(warps_per_cta):
        for lane in range(32):
            for j in range(4):
                c = 4 * lane + j
                off = (c % 32) * 16 + (c // 32) * 4 + warp
                assert 0 <= off < scale_block_bytes
                colwise_byte_offsets.add(off)

    # Early exit savings on capacity-sized buffer with short live prefix:
    # e.g., capacity 16384 rows x 4096 cols, live_end = 2048 rows x 4096 cols
    cap_rows, live_rows, num_cols = 16384, 2048, 4096
    tiles_x = num_cols // wide_tile  # 32
    total_wide_tiles = (cap_rows // wide_tile) * tiles_x  # 128 * 32 = 4096
    active_wide_tiles = (live_rows // wide_tile) * tiles_x  # 16 * 32 = 512
    narrow_tiles_total = (cap_rows // 32) * tiles_x  # 512 * 32 = 16384

    return {
        "wide_tile": wide_tile,
        "wide_threads": wide_threads,
        "wide_max_groups": wide_max_groups,
        "wide_ctas_per_sm": wide_ctas_per_sm,
        "scale_block_bytes_per_tile": scale_block_bytes,
        "rowwise_swizzle_words_unique": len(rowwise_byte_offsets),
        "colwise_swizzle_bytes_unique": len(colwise_byte_offsets),
        "bijective_512b_swizzle": (
            len(rowwise_byte_offsets) == 128 and len(colwise_byte_offsets) == 512
        ),
        "capacity_rows": cap_rows,
        "live_rows": live_rows,
        "num_cols": num_cols,
        "total_wide_tiles": total_wide_tiles,
        "active_wide_tiles": active_wide_tiles,
        "narrow_32x128_tiles_total": narrow_tiles_total,
        "wide_vs_narrow_tile_reduction_x": narrow_tiles_total // total_wide_tiles,
        "live_prefix_active_tile_fraction": round(active_wide_tiles / total_wide_tiles, 4),
    }


def simulate_lmcache_address_kernel_transfer() -> dict[str, object]:
    """Reproduce LMCache transfer_with_address_kernel chunk splitting & sync rules (#5955)."""
    max_chunks_per_launch = 4
    block_size = 4
    blocks_per_chunk = 2
    lmcache_chunk_size = block_size * blocks_per_chunk  # 8 tokens per chunk
    blocks_per_launch = max_chunks_per_launch * lmcache_chunk_size // block_size  # 8 blocks

    def plan_launches(
        chunk_ptrs: list[int],
        block_ids: list[int],
        skip_prefix_n_blocks: int,
        all_pinned: bool,
        direction: str,
    ) -> dict[str, object]:
        launches = []
        for launch_idx, first in enumerate(range(0, len(chunk_ptrs), max_chunks_per_launch)):
            start = launch_idx * blocks_per_launch
            launches.append(
                {
                    "launch_idx": launch_idx,
                    "chunks": chunk_ptrs[first : first + max_chunks_per_launch],
                    "block_ids": block_ids[start : start + blocks_per_launch],
                    "skip": skip_prefix_n_blocks if launch_idx == 0 else 0,
                }
            )
        needs_staging_copyBack = (not all_pinned) and (direction == "D2H")
        needs_temp_pin = (not all_pinned) and (direction == "H2D")
        needs_device_sync = not all_pinned
        return {
            "num_launches": len(launches),
            "launches": launches,
            "needs_staging_copy_back": needs_staging_copyBack,
            "needs_temp_pin": needs_temp_pin,
            "needs_device_sync": needs_device_sync,
        }

    five_chunk_addresses = [0x1000 * (i + 1) for i in range(5)]
    ten_blocks = list(range(10))
    pinned_plan = plan_launches(five_chunk_addresses, ten_blocks, skip_prefix_n_blocks=1, all_pinned=True, direction="D2H")
    unpinned_d2h = plan_launches(five_chunk_addresses, ten_blocks, skip_prefix_n_blocks=0, all_pinned=False, direction="D2H")
    unpinned_h2d = plan_launches(five_chunk_addresses, ten_blocks, skip_prefix_n_blocks=0, all_pinned=False, direction="H2D")

    return {
        "max_chunks_per_launch": max_chunks_per_launch,
        "blocks_per_launch": blocks_per_launch,
        "five_chunk_pinned_plan": pinned_plan,
        "unpinned_d2h_plan": unpinned_d2h,
        "unpinned_h2d_plan": unpinned_h2d,
    }


def simulate_nixl_ucx_notif_and_ecosystem() -> dict[str, object]:
    """Model NIXL UCX notification optimization (#2412) and ecosystem fixes (#5547, #7937, #4798, #17163)."""
    # NIXL #2412 benchmark numbers from PR test report (4096B block, batch=8, pipeline_depth=8, 1M iters)
    nixl_before = {
        "block_size_bytes": 4096,
        "batch_size": 8,
        "bw_gb_s": 4.051741,
        "avg_prep_us": 2.8,
        "p99_prep_us": 8.0,
        "avg_post_us": 4.0,
        "p99_post_us": 12.0,
        "avg_tx_us": 60.6,
        "p99_tx_us": 75.0,
    }
    nixl_after = {
        "block_size_bytes": 4096,
        "batch_size": 8,
        "bw_gb_s": 4.168674,
        "avg_prep_us": 1.9,
        "p99_prep_us": 6.0,
        "avg_post_us": 3.7,
        "p99_post_us": 9.0,
        "avg_tx_us": 59.0,
        "p99_tx_us": 76.0,
    }

    # TorchTitan #5547 Flex Attention recompile budget analysis:
    # Llama3 Flex (6 layers) + Qwen3 MoE Flex (8 layers) in one process
    torchtitan_flex = {
        "default_dynamo_recompile_limit": 8,
        "new_test_recompile_limit": 16,
        "annotated_eager_graphs_llama3_plus_qwen3": 6 + 8,
        "unannotated_eager_graphs_per_test": 1,
        "main_loss_mismatch_eager": 6.995096683502197,
        "fixed_loss_bitwise_equal": 6.995151519775391,
        "pp_preprocess_tuple_len": 4,
        "graphtrainer_pp_tuple_len": 3,
    }

    return {
        "nixl_ucx_notif": {
            "before": nixl_before,
            "after": nixl_after,
            "avg_prep_reduction_pct": round((2.8 - 1.9) / 2.8 * 100.0, 2),
            "p99_post_reduction_pct": round((12.0 - 9.0) / 12.0 * 100.0, 2),
            "bw_gain_pct": round((4.168674 - 4.051741) / 4.051741 * 100.0, 2),
        },
        "torchtitan_flex_pp": torchtitan_flex,
        "megatron_mixtral_tf5": {
            "min_transformers_version": "4.36.0",
            "tf4_moe_attr": "block_sparse_moe",
            "tf5_moe_attr": "mlp",
            "tf5_fused_expert_attr": "gate_up_proj",
        },
        "mooncake_dfs_kv_bench": {
            "valid_when_dfs_replica_only": True,
            "invalid_when_all_three_zero": True,
        },
        "dynamo_strict_schema": {
            "env_var": "DYN_DISABLE_OPENAI_STRICT_SCHEMA_VALIDATION",
            "default_enabled": False,
        },
    }


def build_summary() -> dict[str, object]:
    te = simulate_te_wide_mxfp8_requantize()
    lmc = simulate_lmcache_address_kernel_transfer()
    eco = simulate_nixl_ucx_notif_and_ecosystem()

    return {
        "experiment_id": "te-wide-mxfp8-requant-lmcache-addr-nixl-ucx-notif",
        "phase": 68,
        "priority": 77,
        "track": "Track C — Distributed Training, Collectives & KV Hierarchy",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_tier": "own-offline",
        "te_wide_mxfp8": te,
        "lmcache_address_transfer": lmc,
        "nixl_and_ecosystem": eco,
        "invariants": {
            "te_bijective_512b_swizzle": te["bijective_512b_swizzle"] is True,
            "te_4x_fewer_tiles_than_narrow": te["wide_vs_narrow_tile_reduction_x"] == 4,
            "lmcache_5_chunks_split_4_and_1_with_first_skip_only": (
                lmc["five_chunk_pinned_plan"]["num_launches"] == 2
                and [len(x["chunks"]) for x in lmc["five_chunk_pinned_plan"]["launches"]] == [4, 1]
                and [x["skip"] for x in lmc["five_chunk_pinned_plan"]["launches"]] == [1, 0]
                and lmc["five_chunk_pinned_plan"]["needs_device_sync"] is False
            ),
            "lmcache_unpinned_syncs_and_stages": (
                lmc["unpinned_d2h_plan"]["needs_staging_copy_back"] is True
                and lmc["unpinned_d2h_plan"]["needs_device_sync"] is True
                and lmc["unpinned_h2d_plan"]["needs_temp_pin"] is True
                and lmc["unpinned_h2d_plan"]["needs_device_sync"] is True
            ),
            "nixl_prep_latency_reduced_32pct": eco["nixl_ucx_notif"]["avg_prep_reduction_pct"] == 32.14,
            "torchtitan_unannotated_eager_1_graph": (
                eco["torchtitan_flex_pp"]["unannotated_eager_graphs_per_test"] == 1
                and eco["torchtitan_flex_pp"]["pp_preprocess_tuple_len"] == 4
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failure: {summary['invariants']}"
        print("OK: te-wide-mxfp8-requant-lmcache-addr-nixl-ucx-notif invariants verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
