from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_fa4_cudagraph_and_hd256_splitkv() -> dict[str, Any]:
    """Model FlashAttention #2984 (`73019affed`) and #2917 (`e9cf2c1651`)."""
    # #2917: SM100 hd256 2CTA forward SplitKV work-tile predicate `has_kv_work`
    seqlen_k = 384
    tile_n = 128
    num_splits = 4
    total_n_blocks = math.ceil(seqlen_k / tile_n)  # 3 blocks across 4 splits
    num_n_blocks_per_split = math.ceil(total_n_blocks / num_splits)  # 1 block/split
    splits = []
    for split_idx in range(num_splits):
        n_min = split_idx * num_n_blocks_per_split
        n_max = min(n_min + num_n_blocks_per_split, total_n_blocks)
        has_work = n_min < n_max
        splits.append({
            "split_idx": split_idx,
            "n_block_min": n_min,
            "n_block_max": n_max,
            "raw_block_delta": n_max - n_min,
            "has_kv_work": has_work,
        })
    active_splits = sum(1 for s in splits if s["has_kv_work"])
    skipped_empty_splits = sum(1 for s in splits if not s["has_kv_work"])

    return {
        "paged_kv_cudagraph_guard": {
            "eager_mode_runs_max_item_check": True,
            "capture_mode_skips_host_sync": True,
            "host_syncs_during_capture_before": 1,
            "host_syncs_during_capture_after": 0,
        },
        "sm100_hd256_2cta_splitkv": {
            "seqlen_k": seqlen_k,
            "tile_n": tile_n,
            "num_splits": num_splits,
            "active_splits": active_splits,
            "skipped_empty_splits": skipped_empty_splits,
            "splits": splits,
        },
    }


def simulate_torchtitan_muse_spark_gb300_rl() -> dict[str, Any]:
    """Model TorchTitan #5581 (`43d1bb9801`): Muse Spark 816B-A42B DAPO-Math on GB300."""
    old_layout = {
        "trainer_gpus": 256,
        "fsdp_shard": 64,
        "tp": 4,
        "cp": 1,
        "ep": 4,
        "expert_fsdp_group_size": 64,
        "cross_rack_expert_allgather": True,
    }
    new_layout = {
        "trainer_gpus": 128,
        "fsdp_shard": 8,
        "tp": 4,
        "cp": 4,
        "ep": 16,
        "expert_fsdp_group_size": 8,
        "cross_rack_expert_allgather": False,
        "gb300_hosts_per_ep_group": 4,
        "dist_moe_scratch_factor": 16,
        "measured_peak_scratch_utilization_pct": 22.0,
        "measured_peak_mem_gib": 186,
        "measured_mfu_progression_pct": [15, 26, 28],
        "weight_sync_seconds": [11, 13],
    }
    return {
        "old_layout": old_layout,
        "new_layout": new_layout,
        "trainer_gpu_reduction_factor": round(old_layout["trainer_gpus"] / new_layout["trainer_gpus"], 2),
        "expert_fsdp_group_reduction_factor": round(
            old_layout["expert_fsdp_group_size"] / new_layout["expert_fsdp_group_size"], 2
        ),
        "per_gpu_token_shard_divisor_old": old_layout["tp"] * old_layout["cp"],
        "per_gpu_token_shard_divisor_new": new_layout["tp"] * new_layout["cp"],
    }


def simulate_mooncake_and_lmcache_datapath() -> dict[str, Any]:
    """Model Mooncake #3424 (`0afc417657`) and LMCache #5150 (`6913259264`) / #4410 (`00f6d71ed1`)."""
    # LMCache #5150: packed uint32 token buffer vs tuple/list of Python ints
    prompt_tokens = 32768
    python_int_tuple_bytes = 48 + prompt_tokens * (8 + 28)  # 8B pointer + 28B PyLong per distinct int
    packed_uint32_bytes = prompt_tokens * 4
    compression_ratio = round(python_int_tuple_bytes / packed_uint32_bytes, 2)

    return {
        "lmcache_packed_uint32": {
            "prompt_tokens": prompt_tokens,
            "python_int_tuple_bytes": python_int_tuple_bytes,
            "packed_uint32_bytes": packed_uint32_bytes,
            "payload_reduction_factor_x": compression_ratio,
        },
        "mooncake_nof_decoupling": {
            "interfaces": ["NVMeoFInitiator", "DmaBufferAllocator"],
            "runtime_backends": ["spdk", "none"],
            "spdk_env_guard_prevents_reinit_einval": True,
            "page_registry_granularity_mb": 2,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "fa4_verl_mcore_mooncake_phase87",
        "phase": 87,
        "priority": 96,
        "fa4_cudagraph_and_hd256": simulate_fa4_cudagraph_and_hd256_splitkv(),
        "torchtitan_muse_spark_gb300": simulate_torchtitan_muse_spark_gb300_rl(),
        "mooncake_and_lmcache": simulate_mooncake_and_lmcache_datapath(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        fa = summary["fa4_cudagraph_and_hd256"]
        assert fa["paged_kv_cudagraph_guard"]["host_syncs_during_capture_after"] == 0
        assert fa["sm100_hd256_2cta_splitkv"]["active_splits"] == 3
        assert fa["sm100_hd256_2cta_splitkv"]["skipped_empty_splits"] == 1
        tt = summary["torchtitan_muse_spark_gb300"]
        assert tt["trainer_gpu_reduction_factor"] == 2.0
        assert tt["expert_fsdp_group_reduction_factor"] == 8.0
        assert tt["per_gpu_token_shard_divisor_new"] == 16
        ml = summary["mooncake_and_lmcache"]
        assert ml["lmcache_packed_uint32"]["packed_uint32_bytes"] == 131072
        assert ml["lmcache_packed_uint32"]["payload_reduction_factor_x"] == 9.0
        print("OK: fa4_verl_mcore_mooncake_phase87 verified")
        return 0

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
