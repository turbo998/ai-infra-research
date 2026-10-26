"""Phase 91 Track B offline simulator: SGLang Rust chat processor & atomic diffusion image publish /
disaggregated GLM pixel preservation & AMD gfx950 HIP chunked KDA prefill, vLLM standalone draft DP
replica index & KV offload same-step resume / P2P dead-session ConnectAck / NIXL rc_gda exclusion, and
TensorRT-LLM 64M-element KV NaN/Inf scan & TTFT-percentile GEN admission + SWA context-chunk replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def simulate_sglang_rust_chat_and_diffusion_and_gfx950_kda() -> dict:
    """Model sglang#45449, #46060, #46193, and #45710."""
    dropped_rust_crates = [
        "dynamo-parsers",
        "dynamo-protocols",
        "dynamo-renderer",
        "dynamo-tokenizers",
        "hf-hub",
        "minijinja",
    ]
    # Atomic temp-file + os.replace vs in-place open("wb") truncation on write failure
    existing_bytes_len = 4096
    in_place_failure_bytes_len = 0
    atomic_replace_failure_bytes_len = existing_bytes_len

    # GLM disaggregated diffusion head pixel preservation across dynamic batch split
    glm_batch_pixel_buffers = [[b"img_req_0"], [b"img_req_1"]]

    # AMD gfx950 (MI350X/MI355X) HIP chunked KDA prefill kernel vs fallback
    seq_len = 8192
    chunk_size = 64
    num_chunks = seq_len // chunk_size
    return {
        "rust_server_dropped_crates_count": len(dropped_rust_crates),
        "rust_server_unified_sglang_processor": True,
        "in_place_truncation_surviving_bytes": in_place_failure_bytes_len,
        "atomic_replace_surviving_bytes": atomic_replace_failure_bytes_len,
        "disagg_glm_head_pixels_preserved": len(glm_batch_pixel_buffers) == 2,
        "gfx950_kda_chunk_size": chunk_size,
        "gfx950_kda_num_chunks_at_8k": num_chunks,
    }


def simulate_vllm_spec_dp_and_kv_offload_p2p() -> dict:
    """Model vllm#62301 (draft DP index), #60574 (same-step offload reload), #63664 (ConnectAck), #63498 (TLS=^rc_gda)."""
    target_dp_indices = [0, 1, 2, 3]
    legacy_draft_dp_indices = [0, 0, 0, 0]
    fixed_draft_dp_indices = list(target_dp_indices)

    # Same-step prefix cache reset + preempt + reload
    preempted_old_transfer_id = 101
    reloaded_new_transfer_id = 102
    legacy_surviving_jobs_on_same_step_reset = []
    fixed_surviving_jobs_on_same_step_reset = [reloaded_new_transfer_id]

    # P2P NIXL UCX engine_config with TLS=^rc_gda
    ucx_tls_rule = "^rc_gda"
    return {
        "legacy_nonzero_dp_mismatch_count": sum(
            1 for t, d in zip(target_dp_indices, legacy_draft_dp_indices) if t != d
        ),
        "fixed_nonzero_dp_mismatch_count": sum(
            1 for t, d in zip(target_dp_indices, fixed_draft_dp_indices) if t != d
        ),
        "preempted_old_transfer_id": preempted_old_transfer_id,
        "legacy_surviving_reload_jobs": len(legacy_surviving_jobs_on_same_step_reset),
        "fixed_surviving_reload_jobs": len(fixed_surviving_jobs_on_same_step_reset),
        "connect_ack_failure_marks_session_dead": True,
        "nixl_p2p_ucx_tls_filter": ucx_tls_rule,
        "nixl_p2p_gpu_gda_lanes_excluded": ucx_tls_rule == "^rc_gda",
    }


def simulate_trtllm_kv_scan_and_gen_admission() -> dict:
    """Model TensorRT-LLM#20928 (64M-element KV NaN/Inf scan) and #20736 (TTFT-percentile GEN admission + SWA replay)."""
    # 32 GiB FP8 KV buffer = 34,359,738,368 elements; page = 64 tokens * 128 head_dim * 2 = 16,384 elements
    total_elements = 1 << 31  # 2,147,483,648 elements per layer group sample
    elements_per_page = 16384
    legacy_pages_per_slice = 256
    legacy_elements_per_call = legacy_pages_per_slice * elements_per_page  # 4,194,304
    fixed_elements_per_call = 1 << 26  # _INVALID_VALUE_SCAN_ELEMS = 67,108,864

    legacy_kernel_calls = math.ceil(total_elements / legacy_elements_per_call)
    fixed_kernel_calls = math.ceil(total_elements / fixed_elements_per_call)
    # Also FP8 skips unsupported torch.isinf check after caching capability on first slice
    legacy_checks_per_slice = 2  # isnan + isinf (raises/caught each slice)
    fixed_checks_per_slice_fp8 = 1  # isnan only after capability cache

    # Disaggregated GEN admission queue: compare FCFS on GEN enqueue vs TTFT-percentile (orig_elapsed_s)
    # Requests: (req_id, orig_arrival_s, ctx_duration_s, gen_enqueue_s)
    requests = [
        ("req_long_ctx", 0.00, 0.85, 0.85),
        ("req_short_ctx_1", 0.30, 0.10, 0.40),
        ("req_short_ctx_2", 0.45, 0.10, 0.55),
    ]
    # At t = 0.90s when all 3 are waiting in GEN queue:
    fcfs_gen_order = [r[0] for r in sorted(requests, key=lambda x: x[3])]
    p99_orig_arrival_order = [r[0] for r in sorted(requests, key=lambda x: x[1])]

    # SWA scratch reuse pregrown context chunk replay
    swa_window_tokens = 1024
    pregrown_chunk_tokens = 2048
    unfixed_shrunk_replay_tokens = 1536
    missing_window_tokens_if_shrunk = pregrown_chunk_tokens - unfixed_shrunk_replay_tokens
    return {
        "invalid_value_scan_elems": fixed_elements_per_call,
        "legacy_scan_calls_for_2g_elems": legacy_kernel_calls,
        "fixed_scan_calls_for_2g_elems": fixed_kernel_calls,
        "scan_call_reduction_factor_x": legacy_kernel_calls // fixed_kernel_calls,
        "legacy_fp8_checks_per_slice": legacy_checks_per_slice,
        "fixed_fp8_checks_per_slice": fixed_checks_per_slice_fp8,
        "fcfs_gen_enqueue_order": fcfs_gen_order,
        "p99_orig_arrival_order": p99_orig_arrival_order,
        "oldest_request_promoted_first_at_p99": p99_orig_arrival_order[0] == "req_long_ctx",
        "swa_window_tokens": swa_window_tokens,
        "pregrown_chunk_tokens": pregrown_chunk_tokens,
        "missing_window_tokens_if_shrunk": missing_window_tokens_if_shrunk,
        "exact_pregrown_chunk_replay_tokens": pregrown_chunk_tokens,
    }


def run_simulation() -> dict:
    sglang = simulate_sglang_rust_chat_and_diffusion_and_gfx950_kda()
    vllm = simulate_vllm_spec_dp_and_kv_offload_p2p()
    trtllm = simulate_trtllm_kv_scan_and_gen_admission()
    return {
        "experiment": "sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91",
        "phase": 91,
        "priority": 100,
        "sglang_rust_diffusion_amd_kda": sglang,
        "vllm_spec_dp_and_kv_offload_p2p": vllm,
        "trtllm_kv_scan_and_gen_admission": trtllm,
        "invariants": {
            "atomic_image_publish_preserves_existing_file": (
                sglang["atomic_replace_surviving_bytes"] == 4096
            ),
            "vllm_draft_dp_replica_aligned": vllm["fixed_nonzero_dp_mismatch_count"] == 0,
            "vllm_same_step_reload_preserved": vllm["fixed_surviving_reload_jobs"] == 1,
            "vllm_nixl_p2p_excludes_rc_gda": vllm["nixl_p2p_gpu_gda_lanes_excluded"],
            "trtllm_scan_16x_fewer_slices": trtllm["scan_call_reduction_factor_x"] == 16,
            "trtllm_p99_admits_oldest_orig_arrival": trtllm["oldest_request_promoted_first_at_p99"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    summary = run_simulation()
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing {args.output}; run without --check first.")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py.")
        print("OK: Phase 91 Track B summary.json verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
