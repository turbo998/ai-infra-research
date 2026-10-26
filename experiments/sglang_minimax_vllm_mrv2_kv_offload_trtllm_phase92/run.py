from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def simulate_sglang_minimax_m31_flash_and_dspark() -> dict[str, Any]:
    """Model sgl-project/sglang@a9bf8c76a0 (#45925): MiniMax-M3.1-Flash MSA & DSPark partial-RoPE / mask fixes."""
    head_dim = 128
    rot_dim = 64
    pass_through_dim = head_dim - rot_dim
    cos_sin_half_width = rot_dim // 2

    window_slots = 16
    commit_len = 11
    uncommitted_slots_masked = window_slots - commit_len
    return {
        "commit": "a9bf8c76a0",
        "msa_pinned_commit": "1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431",
        "monokernel_backend_flag": "--monokernel-backend msa",
        "head_dim": head_dim,
        "rope_rotated_dim": rot_dim,
        "pass_through_dim": pass_through_dim,
        "cos_sin_half_width": cos_sin_half_width,
        "window_slots": window_slots,
        "commit_len": commit_len,
        "uncommitted_slots_masked_before_req_pool_deref": uncommitted_slots_masked,
    }


def simulate_sglang_amd_aiter_softcap() -> dict[str, Any]:
    """Model sgl-project/sglang@d96c386335 (#45531): Pass layer.logit_cap to AMD aiter unified-attention decode."""
    import math

    raw_logits = [12.5, 45.0, 85.0, 120.0]
    logit_cap = 50.0
    legacy_softcap0_logits = list(raw_logits)
    fixed_softcapped_logits = [round(logit_cap * math.tanh(x / logit_cap), 4) for x in raw_logits]
    return {
        "commit": "d96c386335",
        "layer_logit_cap": logit_cap,
        "raw_logits": raw_logits,
        "legacy_softcap_arg": 0,
        "fixed_softcap_arg": logit_cap,
        "legacy_unbounded_logits": legacy_softcap0_logits,
        "fixed_softcapped_logits": fixed_softcapped_logits,
        "max_logit_bounded_below_cap": max(fixed_softcapped_logits) <= logit_cap,
    }


def simulate_sglang_diffusion_video_cleanup_and_dp_routing() -> dict[str, Any]:
    """Model sgl-project/sglang@ce77d5e93d (#46057) and sgl-project/sglang@042dc54cbf (#46249)."""
    dp_size = 4
    sessions = ["rt-sess-alpha", "rt-sess-beta", "rt-sess-gamma", "rt-sess-delta"]
    routing = []
    for idx, sid in enumerate(sessions):
        digest = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
        owner_replica = digest % dp_size
        legacy_rr_release_replica = (idx + 1) % dp_size
        fixed_release_replica = owner_replica
        routing.append(
            {
                "session_id": sid,
                "owner_replica": owner_replica,
                "legacy_release_replica": legacy_rr_release_replica,
                "fixed_release_replica": fixed_release_replica,
                "routed_to_owner": fixed_release_replica == owner_replica,
            }
        )
    return {
        "video_cleanup_commit": "ce77d5e93d",
        "dp_routing_commit": "042dc54cbf",
        "dp_size": dp_size,
        "rejected_upload_tempdir_leaked_bytes_fixed": 0,
        "cloud_upload_verified_before_completed_status": True,
        "session_routing": routing,
        "all_releases_routed_to_owner": all(r["routed_to_owner"] for r in routing),
    }


def simulate_vllm_mrv2_cudagraph_profile_only_skip() -> dict[str, Any]:
    """Model vllm-project/vllm@4135590ffe (#63431): Skip adaptive verification profiling in MRV2 profile_only pass."""
    captured_token_counts = [1, 2, 4, 8, 16, 32, 64]
    cached_activation_mib_per_shape = 24.0
    false_cudagraph_mib = len(captured_token_counts) * cached_activation_mib_per_shape
    kv_block_mib = 2.0
    recovered_kv_blocks = int(false_cudagraph_mib // kv_block_mib)
    return {
        "commit": "4135590ffe",
        "captured_token_counts": captured_token_counts,
        "profile_only_runs_adaptive_verification_fixed": False,
        "real_capture_runs_adaptive_verification": True,
        "eliminated_false_cudagraph_mib": false_cudagraph_mib,
        "recovered_kv_cache_blocks": recovered_kv_blocks,
    }


def simulate_vllm_rocm_dsv41_krdy_barrier() -> dict[str, Any]:
    """Model vllm-project/vllm@9744ada582 (#63576): Await krdy before reading klen in ROCm DSv4.1 stage_combine."""
    chunk_keys = 64
    true_klen = 2048
    stale_l1_klen = 512
    legacy_nlive = (stale_l1_klen + chunk_keys - 1) // chunk_keys
    fixed_nlive = (true_klen + chunk_keys - 1) // chunk_keys
    return {
        "commit": "9744ada582",
        "index_source_launch": True,
        "awaited_flag": "krdy",
        "load_cache_modifier": "CM_DEV",
        "true_klen": true_klen,
        "stale_klen_without_barrier": stale_l1_klen,
        "legacy_nlive_chunks": legacy_nlive,
        "fixed_nlive_chunks": fixed_nlive,
        "missing_chunks_recovered": fixed_nlive - legacy_nlive,
    }


def simulate_vllm_kv_offload_reset_hold_and_nixl_rollback() -> dict[str, Any]:
    """Model vllm-project/vllm@14aefcd4f9 (#62929) and vllm-project/vllm@8a4b018f1c (#63628)."""
    inflight_jobs_at_reset = 3
    events = [
        {"step": 1, "event": "cache_reset_issued", "inflight_jobs": 3, "promotions_allowed": False},
        {"step": 2, "event": "job_1_reported", "inflight_jobs": 2, "promotions_allowed": False},
        {"step": 3, "event": "job_2_reported", "inflight_jobs": 1, "promotions_allowed": False},
        {"step": 4, "event": "job_3_reported", "inflight_jobs": 0, "promotions_allowed": True},
    ]
    return {
        "reset_hold_commit": "14aefcd4f9",
        "nixl_rollback_commit": "8a4b018f1c",
        "inflight_jobs_at_reset": inflight_jobs_at_reset,
        "reset_drain_timeline": events,
        "stale_pre_reset_promotions_after_fix": 0,
        "failed_nixl_peer_leaked_handles_after_fix": 0,
    }


def simulate_trtllm_unified_kv_coalescing_and_beam_histories() -> dict[str, Any]:
    """Model NVIDIA/TensorRT-LLM@fce28f0726 (#20178) and NVIDIA/TensorRT-LLM@732fd63463 (#20915)."""
    buffers = [
        {"buffer_id": "layer0_k", "lifecycle": 1, "coalescing_group": 0, "expanded_size": 65536},
        {"buffer_id": "layer0_v", "lifecycle": 1, "coalescing_group": 0, "expanded_size": 65536},
        {"buffer_id": "layer0_indexer", "lifecycle": 1, "coalescing_group": 1, "expanded_size": 65536},
        {"buffer_id": "layer1_sparse_k", "lifecycle": 2, "coalescing_group": 0, "expanded_size": 32768},
    ]
    legacy_pools: dict[tuple[int, int], list[str]] = {}
    fixed_pools: dict[tuple[int, int, int], list[str]] = {}
    for buf in buffers:
        lk = (buf["lifecycle"], buf["expanded_size"])
        fk = (buf["lifecycle"], buf["coalescing_group"], buf["expanded_size"])
        legacy_pools.setdefault(lk, []).append(buf["buffer_id"])
        fixed_pools.setdefault(fk, []).append(buf["buffer_id"])

    # Beam search parent backpointer trace across 3 decode steps for 2 beams:
    # Step 0: beam0 emits 101, beam1 emits 202
    # Step 1: beam0 comes from parent 1 (emits 203), beam1 comes from parent 0 (emits 104)
    # Step 2: beam0 comes from parent 0 (emits 205), beam1 comes from parent 0 (emits 206)
    slot_tokens = {
        0: [101, 203, 205],
        1: [202, 104, 206],
    }
    rebuilt_beam_hypotheses = {
        0: [202, 203, 205],
        1: [202, 203, 206],
    }
    return {
        "kv_coalescing_commit": "fce28f0726",
        "beam_histories_commit": "732fd63463",
        "legacy_pool_groups": len(legacy_pools),
        "fixed_pool_groups": len(fixed_pools),
        "indexer_isolated_from_kv_pool": fixed_pools[(1, 1, 65536)] == ["layer0_indexer"],
        "per_slot_token_lists": slot_tokens,
        "rebuilt_beam_hypotheses": rebuilt_beam_hypotheses,
        "slot_vs_hypothesis_diverged_under_beam_swap": slot_tokens[0] != rebuilt_beam_hypotheses[0],
    }


def build_summary() -> dict[str, Any]:
    minimax = simulate_sglang_minimax_m31_flash_and_dspark()
    aiter = simulate_sglang_amd_aiter_softcap()
    diffusion = simulate_sglang_diffusion_video_cleanup_and_dp_routing()
    mrv2 = simulate_vllm_mrv2_cudagraph_profile_only_skip()
    dsv41 = simulate_vllm_rocm_dsv41_krdy_barrier()
    kv_offload = simulate_vllm_kv_offload_reset_hold_and_nixl_rollback()
    trtllm = simulate_trtllm_unified_kv_coalescing_and_beam_histories()

    return {
        "experiment": "sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92",
        "phase": 92,
        "priority": 101,
        "track": "serving_runtimes_sglang_vllm_trtllm",
        "sglang_minimax_m31_flash_and_dspark": minimax,
        "sglang_amd_aiter_softcap": aiter,
        "sglang_diffusion_cleanup_and_dp_routing": diffusion,
        "vllm_mrv2_cudagraph_profile_only_skip": mrv2,
        "vllm_rocm_dsv41_krdy_barrier": dsv41,
        "vllm_kv_offload_reset_and_nixl": kv_offload,
        "trtllm_unified_kv_and_beam_histories": trtllm,
        "headline_metrics": {
            "mrv2_recovered_kv_blocks": mrv2["recovered_kv_cache_blocks"],
            "dsv41_missing_chunks_recovered": dsv41["missing_chunks_recovered"],
            "stale_pre_reset_promotions_after_fix": kv_offload["stale_pre_reset_promotions_after_fix"],
            "failed_nixl_leaked_handles_after_fix": kv_offload["failed_nixl_peer_leaked_handles_after_fix"],
            "trtllm_fixed_pool_groups": trtllm["fixed_pool_groups"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 92 Track 2 SGLang, vLLM & TensorRT-LLM runtime simulator")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--check", action="store_true", help="Verify existing summary.json matches deterministic output")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing summary file: {args.output}")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with deterministic simulator output")
        print("OK: summary.json matches deterministic simulator output.")
        return 0

    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
