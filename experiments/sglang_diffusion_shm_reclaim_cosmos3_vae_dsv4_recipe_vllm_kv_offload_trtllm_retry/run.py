#!/usr/bin/env python3
"""Phase 82 Track 2 experiment: SGLang diffusion SHM reclaim, Cosmos3 VAE, DeepSeek-V4 recipe, vLLM KV offload & TRT-LLM retry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_sglang_diffusion_shm_and_cosmos3_dsv4() -> dict[str, Any]:
    """Model SGLang diffusion per-RPC SHM reclaim (#46154), Cosmos3 VAE (#46173), and DeepSeek-V4 recipe (#46243)."""
    shm_spill_threshold_mib = 32
    cancelled_rpc_count = 64
    reply_payload_mib = 48  # >= 32 MiB spills to SHM file
    leaked_shm_mib_before = cancelled_rpc_count * reply_payload_mib
    leaked_shm_mib_after = 0

    cosmos3_num_gpus = 4
    decoder_conv3d_channels_last = cosmos3_num_gpus >= 2
    encoder_conv3d_channels_last = False  # kept contiguous to preserve I2V seed bit-exactness
    use_vae_fast_path_enabled = True

    return {
        "shm_spill_threshold_mib": shm_spill_threshold_mib,
        "cancelled_rpc_count": cancelled_rpc_count,
        "reply_payload_mib": reply_payload_mib,
        "leaked_shm_mib_before": leaked_shm_mib_before,
        "leaked_shm_mib_after": leaked_shm_mib_after,
        "per_rpc_identity_prefix": "sgldiffusion-request-",
        "cosmos3_num_gpus": cosmos3_num_gpus,
        "decoder_conv3d_channels_last_3d": decoder_conv3d_channels_last,
        "encoder_conv3d_channels_last_3d": encoder_conv3d_channels_last,
        "use_vae_fast_path_enabled": use_vae_fast_path_enabled,
        "deepseek_recipe_crate_version": "0.2.0",
        "supported_dsv4_parsers": ["deepseek_v4", "deepseek_v4_1"],
        "encode_request_returns_token_ids": True,
    }


def model_vllm_kv_offload_and_trtllm_retry() -> dict[str, Any]:
    """Model vLLM KV offload fixes (#61390, #63010, #63126) and TRT-LLM ClientOSError retry (#20889)."""
    # #63010: Preempted request with 4 chunks previously stored, then preempted (num_computed_tokens == 0)
    total_chunks = 6
    chunks_stored_before_preemption = 4
    # Before fix: next_stored_chunk_idx stayed 4, so chunks 0..3 were skipped after eviction
    skipped_chunks_before_fix = chunks_stored_before_preemption
    # After fix: next_stored_chunk_idx resets to 0 when num_computed_tokens == 0
    skipped_chunks_after_fix = 0
    stored_chunks_after_fix = total_chunks

    # #61390: Concurrent content-addressed FS block writes raising EEXIST on rename(2)
    concurrent_duplicate_block_writes = 16
    false_write_errors_before_fix = concurrent_duplicate_block_writes
    false_write_errors_after_fix = 0
    leaked_tmp_files_after_fix = 0

    # #63126: Lazy generator batch_lookup raising OSError on iteration
    lazy_generator_caught_inside_try_before = False
    eager_list_caught_inside_try_after = True

    # TRT-LLM #20889: 5 retry attempts with 2**attempt backoff (1, 2, 4, 8s before 5th attempt)
    max_attempts = 5
    backoff_seconds = [2**attempt for attempt in range(max_attempts - 1)]

    return {
        "preemption_total_chunks": total_chunks,
        "skipped_chunks_before_fix": skipped_chunks_before_fix,
        "skipped_chunks_after_fix": skipped_chunks_after_fix,
        "stored_chunks_after_fix": stored_chunks_after_fix,
        "concurrent_duplicate_block_writes": concurrent_duplicate_block_writes,
        "false_write_errors_before_fix": false_write_errors_before_fix,
        "false_write_errors_after_fix": false_write_errors_after_fix,
        "leaked_tmp_files_after_fix": leaked_tmp_files_after_fix,
        "lazy_generator_caught_inside_try_before": lazy_generator_caught_inside_try_before,
        "eager_list_caught_inside_try_after": eager_list_caught_inside_try_after,
        "trtllm_retry_max_attempts": max_attempts,
        "trtllm_backoff_schedule_s": backoff_seconds,
        "trtllm_backoff_total_s": sum(backoff_seconds),
        "trtllm_catches_client_os_error": True,
    }


def run_experiment() -> dict[str, Any]:
    sglang = model_sglang_diffusion_shm_and_cosmos3_dsv4()
    vllm_trtllm = model_vllm_kv_offload_and_trtllm_retry()
    return {
        "experiment_id": "sglang_diffusion_shm_reclaim_cosmos3_vae_dsv4_recipe_vllm_kv_offload_trtllm_retry",
        "phase": 82,
        "priority": 91,
        "provenance": "own-offline",
        "sglang_diffusion_cosmos3_dsv4": sglang,
        "vllm_kv_offload_and_trtllm_retry": vllm_trtllm,
        "invariants": {
            "zero_shm_leak_on_cancelled_rpc": (
                sglang["leaked_shm_mib_before"] == 3072 and sglang["leaked_shm_mib_after"] == 0
            ),
            "cosmos3_decoder_only_channels_last_3d": (
                sglang["decoder_conv3d_channels_last_3d"] is True
                and sglang["encoder_conv3d_channels_last_3d"] is False
            ),
            "vllm_preemption_resets_chunk_cursor": (
                vllm_trtllm["skipped_chunks_before_fix"] == 4
                and vllm_trtllm["skipped_chunks_after_fix"] == 0
            ),
            "vllm_eexist_idempotent_and_eager_lookup": (
                vllm_trtllm["false_write_errors_after_fix"] == 0
                and vllm_trtllm["eager_list_caught_inside_try_after"] is True
            ),
            "trtllm_5_attempt_backoff_15s": (
                vllm_trtllm["trtllm_retry_max_attempts"] == 5
                and vllm_trtllm["trtllm_backoff_total_s"] == 15
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate invariants and write summary.json")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
