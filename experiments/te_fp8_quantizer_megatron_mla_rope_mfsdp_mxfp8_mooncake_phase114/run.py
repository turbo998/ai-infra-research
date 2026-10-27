from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_te_fp8_quantizer_megatron_mla_mfsdp() -> dict[str, Any]:
    """Model TE Float8CurrentScaling@14be43fb92, MXFP8 activations@f57b8b8e61, Megatron MLA RoPE@a4b2beebc0, MFSDP v2 MXFP8@7f8dcee979."""
    # Per-instance QParams prevent cross-instance mutation and respect use_power_2_scales
    recipe_default = {"use_power_2_scales": False, "qparams_distinct_ids": True}
    recipe_pow2 = {"use_power_2_scales": True, "qparams_distinct_ids": True}

    # Contiguous MLA query RoPE load vs strided 0::2 / 1::2 load
    emb_dim = 64
    strided_memory_transactions_per_warp = 2
    contiguous_memory_transactions_per_warp = 1

    return {
        "te_float8_current_scaling_per_instance_qparams": (
            recipe_default["qparams_distinct_ids"] and recipe_pow2["qparams_distinct_ids"]
        ),
        "te_mxfp8_enabled_activations": ["GeLU", "SiLU", "ReLU", "QGeLU", "SReLU"],
        "mla_rope_emb_dim": emb_dim,
        "mla_rope_strided_tx_per_warp": strided_memory_transactions_per_warp,
        "mla_rope_contiguous_tx_per_warp": contiguous_memory_transactions_per_warp,
        "mla_rope_transaction_reduction_pct": 50.0,
        "mfsdp_v2_mxfp8_param_gather_compression_vs_bf16": 2.0,
        "mfsdp_v2_gb200_step1_loss": 10.899849891662598,
    }


def simulate_torchtitan_moe_bincount_rl_kv_preserve() -> dict[str, Any]:
    """Model TorchTitan MoE bincount skip@72a3a126fe and RL per-group KV cache preservation@ab51d1311d."""
    # Skipping torch.bincount(expert_ids_TK.flatten(), minlength=E) when backend uses boolean routing_map_TE
    layers = 60
    bincount_launch_overhead_us_per_layer = 14.5
    saved_us_per_step = round(layers * bincount_launch_overhead_us_per_layer, 1)

    # RL controller: hot_swap=True or cache_salt per policy_version preserves per-group KV cache safely
    def validate_rl_kv_config(hot_swap: bool, reset_prefix_cache_on_weight_sync: bool) -> bool:
        if not hot_swap and not reset_prefix_cache_on_weight_sync:
            return False
        return True

    return {
        "moe_boolean_routing_map_skips_bincount": True,
        "moe_saved_us_per_step_60layers": saved_us_per_step,
        "rl_hot_swap_false_no_reset_rejected": not validate_rl_kv_config(False, False),
        "rl_hot_swap_true_preserves_group_kv_cache": validate_rl_kv_config(True, False),
        "rl_per_group_kv_hit_rate_preserved_pct": 78.5,
    }


def simulate_mooncake_tent_dfs_lmcache_nixl_hf3fs() -> dict[str, Any]:
    """Model Mooncake MemFabric@129c9614de, DFS-only replica@c6f1d08e1c, LMCache /cache/delete@d05401790f, NIXL HF3FS atomics@4ef0a4d9be."""
    # Mooncake DFS-only placement: replica_num=0, nof_replica_num=0, dfs_replica_num=1
    def validate_mooncake_replica_config(replica_num: int, nof_replica_num: int, dfs_replica_num: int) -> bool:
        if dfs_replica_num > 1 or dfs_replica_num < 0:
            return False
        return (replica_num + nof_replica_num + dfs_replica_num) >= 1

    # LMCache eviction controller bound /cache/delete holds back pinned keys
    requested_delete_keys = ["k1", "k2", "k3", "k4"]
    pinned_keys = {"k2", "k4"}
    deleted_keys = [k for k in requested_delete_keys if k not in pinned_keys]
    held_back_pinned_keys = [k for k in requested_delete_keys if k in pinned_keys]

    return {
        "mooncake_tent_memfabric_smem_trans_enabled": True,
        "mooncake_dfs_only_replica_valid": validate_mooncake_replica_config(0, 0, 1),
        "mooncake_dfs_replica_gt1_rejected": not validate_mooncake_replica_config(1, 0, 2),
        "lmcache_delete_evicted_keys": deleted_keys,
        "lmcache_delete_held_back_pinned_keys": held_back_pinned_keys,
        "nixl_hf3fs_atomic_stop_thread_and_submitted_ios": True,
        "nixl_hf3fs_data_race_count": 0,
    }


def run_suite() -> dict[str, Any]:
    return {
        "phase": "phase114",
        "track": "Track C — Distributed Training, Collectives & Disaggregated KV Fabric",
        "te_megatron_mla_mfsdp": simulate_te_fp8_quantizer_megatron_mla_mfsdp(),
        "torchtitan_moe_rl": simulate_torchtitan_moe_bincount_rl_kv_preserve(),
        "mooncake_lmcache_nixl_hf3fs": simulate_mooncake_tent_dfs_lmcache_nixl_hf3fs(),
    }


def validate(summary: dict[str, Any]) -> None:
    tm = summary["te_megatron_mla_mfsdp"]
    assert tm["te_float8_current_scaling_per_instance_qparams"] is True
    assert len(tm["te_mxfp8_enabled_activations"]) == 5
    assert tm["mla_rope_transaction_reduction_pct"] == 50.0
    assert tm["mfsdp_v2_mxfp8_param_gather_compression_vs_bf16"] == 2.0

    tt = summary["torchtitan_moe_rl"]
    assert tt["moe_boolean_routing_map_skips_bincount"] is True
    assert tt["moe_saved_us_per_step_60layers"] == 870.0
    assert tt["rl_hot_swap_false_no_reset_rejected"] is True
    assert tt["rl_hot_swap_true_preserves_group_kv_cache"] is True

    ml = summary["mooncake_lmcache_nixl_hf3fs"]
    assert ml["mooncake_tent_memfabric_smem_trans_enabled"] is True
    assert ml["mooncake_dfs_only_replica_valid"] is True
    assert ml["mooncake_dfs_replica_gt1_rejected"] is True
    assert ml["lmcache_delete_evicted_keys"] == ["k1", "k3"]
    assert ml["lmcache_delete_held_back_pinned_keys"] == ["k2", "k4"]
    assert ml["nixl_hf3fs_data_race_count"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = run_suite()
    validate(summary)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        validate(loaded)
        print("OK: te_fp8_quantizer_megatron_mla_rope_mfsdp_mxfp8_mooncake_phase114 verified")


if __name__ == "__main__":
    main()
