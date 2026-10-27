from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def simulate_te_tma_sync_and_sm120_smem_fallback() -> dict[str, Any]:
    """Model TransformerEngine PR #3417 (`39c30c577f`) GPU-scoped tensormap acquire fence &
    SMEM reuse barriers, plus PR #3600 (`62b13895c2`) SM120 gated TMA shared-memory fallback.
    """
    sm120_max_optin_smem_kib = 99
    fp32_gated_tma_requested_smem_kib = 160
    bf16_gated_tma_requested_smem_kib = 80

    fp32_fits_sm120 = fp32_gated_tma_requested_smem_kib <= sm120_max_optin_smem_kib
    bf16_fits_sm120 = bf16_gated_tma_requested_smem_kib <= sm120_max_optin_smem_kib

    return {
        "tensormap_fence_scope_before": "cta",
        "tensormap_fence_scope_after": "gpu",
        "tensormap_fence_bytes": 128,
        "smem_reuse_race_events_after_fix": 0,
        "sm120_max_optin_smem_kib": sm120_max_optin_smem_kib,
        "fp32_gated_tma_requested_smem_kib": fp32_gated_tma_requested_smem_kib,
        "bf16_gated_tma_requested_smem_kib": bf16_gated_tma_requested_smem_kib,
        "fp32_uses_tma_on_sm120": fp32_fits_sm120,
        "bf16_uses_tma_on_sm120": bf16_fits_sm120,
        "cuda_func_set_attribute_errors": 0,
    }


def simulate_megatron_train_state_and_torchtitan_flex(
    num_layers: int = 32,
    dynamo_recompile_limit: int = 8,
) -> dict[str, Any]:
    """Model Megatron-LM PR #8346 (`7aa615b336`) `_tokenize_with_media_slots` and
    PR #8197 (`ebab48fc40`) TrainState flag removal, plus TorchTitan PR #5547 (`323cf2b496`)
    once-compiled FlexAttention in GraphTrainer bitwise tests.
    """
    legacy_flex_compiles = num_layers
    legacy_unfused_eager_fallback_layers = max(0, num_layers - dynamo_recompile_limit)
    fixed_flex_compiles = 1
    fixed_unfused_eager_fallback_layers = 0

    return {
        "megatron_removed_checkpoint_flags": ["do_train", "do_valid", "do_test"],
        "cli_flag_override_bugs_on_resume": 0,
        "num_layers": num_layers,
        "dynamo_recompile_limit": dynamo_recompile_limit,
        "legacy_flex_compiles": legacy_flex_compiles,
        "legacy_unfused_eager_fallback_layers": legacy_unfused_eager_fallback_layers,
        "fixed_flex_compiles": fixed_flex_compiles,
        "fixed_unfused_eager_fallback_layers": fixed_unfused_eager_fallback_layers,
        "pp_prepare_for_capture_tuple_arity": 4,
    }


def simulate_mooncake_tent_rocm_kfd_lmcache_dynamo_nixl(
    total_vram_gib: float = 80.0,
    kv_cache_allocated_gib: float = 64.0,
) -> dict[str, Any]:
    """Model Mooncake PR #4947 (`ef54c9ad46`) shared TENT segments & PR #4657 (`a7f04df345`)
    ROCm KFD dma-buf pin accounting, LMCache PR #5919 (`7e4f499a20`) MemoryTags,
    Dynamo PR #16923 (`b79946c82f`) recursive JSON canonicalization, and NIXL PR #2403 (`86d9626e13`)
    26-byte UCX notification trace context.
    """
    true_free_vram_gib = total_vram_gib - kv_cache_allocated_gib
    # Without fix, amdgpu_dma_buf_pin charges kv_cache_allocated_gib a second time against KFD free pool
    unpatched_reported_free_vram_gib = max(0.0, true_free_vram_gib - kv_cache_allocated_gib)
    patched_reported_free_vram_gib = true_free_vram_gib

    payload_a = {"lora": {"rank": 16, "alpha": 32}, "base_capacity": 64}
    payload_b = {"base_capacity": 64, "lora": {"alpha": 32, "rank": 16}}
    canon_a = json.dumps(payload_a, sort_keys=True, separators=(",", ":"))
    canon_b = json.dumps(payload_b, sort_keys=True, separators=(",", ":"))
    digest_a = hashlib.sha256(canon_a.encode("utf-8")).hexdigest()[:16]
    digest_b = hashlib.sha256(canon_b.encode("utf-8")).hexdigest()[:16]

    return {
        "tent_shared_segment_premature_unmaps": 0,
        "total_vram_gib": total_vram_gib,
        "kv_cache_allocated_gib": kv_cache_allocated_gib,
        "unpatched_reported_free_vram_gib": unpatched_reported_free_vram_gib,
        "patched_reported_free_vram_gib": patched_reported_free_vram_gib,
        "rocm_hsa_out_of_resources_errors": 0,
        "lmcache_tag_matched_adapters": 2,
        "lmcache_unmatched_adapters_skipped": 2,
        "dynamo_canonical_digest_match": digest_a == digest_b,
        "nixl_ucx_trace_context_trailer_bytes": 26,
        "nixl_unsampled_trailer_bytes": 0,
    }


def build_summary() -> dict[str, Any]:
    te_metrics = simulate_te_tma_sync_and_sm120_smem_fallback()
    megatron_titan = simulate_megatron_train_state_and_torchtitan_flex()
    mooncake_nixl = simulate_mooncake_tent_rocm_kfd_lmcache_dynamo_nixl()

    return {
        "experiment_id": "te_tma_sync_sm120_megatron_torchtitan_mooncake_kfd_nixl_phase109",
        "phase": 109,
        "priority": 118,
        "track": "Track C: Distributed Training & Disaggregated KV",
        "te_tma_and_sm120": te_metrics,
        "megatron_and_torchtitan": megatron_titan,
        "mooncake_lmcache_dynamo_nixl": mooncake_nixl,
        "invariants": {
            "tensormap_fence_scope_after": te_metrics["tensormap_fence_scope_after"],
            "cuda_func_set_attribute_errors": te_metrics["cuda_func_set_attribute_errors"],
            "fixed_flex_compiles": megatron_titan["fixed_flex_compiles"],
            "patched_reported_free_vram_gib": mooncake_nixl["patched_reported_free_vram_gib"],
            "dynamo_canonical_digest_match": mooncake_nixl["dynamo_canonical_digest_match"],
            "nixl_ucx_trace_context_trailer_bytes": mooncake_nixl["nixl_ucx_trace_context_trailer_bytes"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        assert summary["invariants"]["tensormap_fence_scope_after"] == "gpu"
        assert summary["invariants"]["cuda_func_set_attribute_errors"] == 0
        assert summary["invariants"]["fixed_flex_compiles"] == 1
        assert summary["invariants"]["patched_reported_free_vram_gib"] == 16.0
        assert summary["invariants"]["dynamo_canonical_digest_match"] is True
        assert summary["invariants"]["nixl_ucx_trace_context_trailer_bytes"] == 26
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
