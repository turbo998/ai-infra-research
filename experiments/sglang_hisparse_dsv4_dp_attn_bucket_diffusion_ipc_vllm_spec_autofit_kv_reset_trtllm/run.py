from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
from typing import Any


def simulate_hisparse_dsv4_token_granular_backup() -> dict[str, Any]:
    """Model sglang#45104 (c00e7e12b64b): DSV4 HiSparse scattered decode token backup."""
    slot_page_size = 64
    value_dim = 512
    scale_dim = 8

    # Construct 2 pages of C4 storage: each page stores [values(64*512), scales(64*8)]
    # Batch has 64 decode requests, 1 new token per request, scattered at odd slots across 2 pages:
    # page 0 slots 0..31, page 1 slots 0..31 (non-contiguous across pages!).
    device_pages = {}
    for page_id in (10, 25):
        values = [float(page_id * 1000 + slot) for slot in range(slot_page_size)]
        scales = [float(page_id * 100 + slot) / 10.0 for slot in range(slot_page_size)]
        device_pages[page_id] = {"values": values, "scales": scales}

    scattered_locs = [(10, slot) for slot in range(32)] + [(25, slot) for slot in range(32)]
    num_tokens = len(scattered_locs)

    # Legacy bug: backup_from_device_all_layer checks numel() % slot_page_size == 0 (64 % 64 == 0)
    # and reshapes(-1, 64)[:, 0] -> page_slots = [page10_slot0], copying whole page 10 and missing page 25!
    legacy_mod_zero_triggers_page_copy = (num_tokens % slot_page_size) == 0
    legacy_copied_tokens = [
        (device_pages[10]["values"][slot], device_pages[10]["scales"][slot])
        for slot in range(slot_page_size)
    ]

    # Fixed: backup_tokens_from_device_all_layer always invokes transfer_cache_dsv4_mla per token
    fixed_copied_tokens = [
        (device_pages[page_id]["values"][slot], device_pages[page_id]["scales"][slot])
        for page_id, slot in scattered_locs
    ]

    expected_tokens = fixed_copied_tokens
    corrupted_count_legacy = sum(
        1 for a, b in zip(legacy_copied_tokens, expected_tokens) if a != b
    )
    corrupted_count_fixed = sum(
        1 for a, b in zip(fixed_copied_tokens, expected_tokens) if a != b
    )

    return {
        "slot_page_size": slot_page_size,
        "value_dim": value_dim,
        "scale_dim": scale_dim,
        "decode_batch_tokens": num_tokens,
        "legacy_mod_zero_triggers_page_copy": legacy_mod_zero_triggers_page_copy,
        "corrupted_tokens_legacy": corrupted_count_legacy,
        "corrupted_tokens_fixed": corrupted_count_fixed,
        "corrupted_fraction_legacy": round(corrupted_count_legacy / num_tokens, 4),
    }


def simulate_dp_attention_bucket_and_custom_ar() -> dict[str, Any]:
    """Model sglang#46138 (9b230c1dae7e): DP-attention prefill bucket rounding & decode Custom-AR."""
    prefill_cg_bs = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    raw_global_tokens = [0, 17, 12, 9]  # 4 DP ranks; rank 0 is idle (0 real tokens), max is 17

    legacy_max_len = max(raw_global_tokens)
    legacy_bucket_aligned = legacy_max_len in prefill_cg_bs

    target_bucket = min(
        (size for size in prefill_cg_bs if size >= legacy_max_len),
        default=legacy_max_len,
    )
    fixed_padded_tokens = [target_bucket for _ in raw_global_tokens]
    fixed_bucket_aligned = all(t in prefill_cg_bs for t in fixed_padded_tokens)

    # Idle DP rank 0 metadata skip when skips_prefill_metadata_on_idle_dp_rank=True
    idle_rank_skipped_init_forward_metadata = raw_global_tokens[0] == 0

    # Decode SUM_LEN custom all-reduce threshold alignment
    decode_tokens = [8, 8, 8, 8]  # sum = 32 tokens * 7168 * 2 bytes = 458,752 bytes <= 512 KiB custom AR threshold
    sum_tokens = sum(decode_tokens)
    custom_ar_max_bytes = 524288
    hidden_bytes_per_token = 7168 * 2
    fits_custom_ar = (sum_tokens * hidden_bytes_per_token) <= custom_ar_max_bytes
    legacy_cuda_graph_padding_mode = "MAX_LEN"
    fixed_cuda_graph_padding_mode = "SUM_LEN" if fits_custom_ar else "MAX_LEN"

    return {
        "prefill_cg_buckets": prefill_cg_bs,
        "raw_global_tokens": raw_global_tokens,
        "legacy_max_len": legacy_max_len,
        "legacy_bucket_aligned": legacy_bucket_aligned,
        "fixed_target_bucket": target_bucket,
        "fixed_padded_tokens": fixed_padded_tokens,
        "fixed_bucket_aligned": fixed_bucket_aligned,
        "idle_rank_skipped_init_forward_metadata": idle_rank_skipped_init_forward_metadata,
        "decode_sum_tokens": sum_tokens,
        "fits_custom_all_reduce": fits_custom_ar,
        "legacy_cuda_graph_padding_mode": legacy_cuda_graph_padding_mode,
        "fixed_cuda_graph_padding_mode": fixed_cuda_graph_padding_mode,
    }


def simulate_diffusion_ipc_cleanup() -> dict[str, Any]:
    """Model sglang#46154 (c05fb1bd04d2): per-request /dev/shm IPC directory cleanup across 7 outcomes."""
    outcomes = [
        {"case": "success", "spilled_mib": 64, "legacy_leaked_mib": 0, "fixed_leaked_mib": 0},
        {"case": "bad_pickle", "spilled_mib": 64, "legacy_leaked_mib": 64, "fixed_leaked_mib": 0},
        {"case": "bad_array", "spilled_mib": 64, "legacy_leaked_mib": 64, "fixed_leaked_mib": 0},
        {"case": "timeout_before", "spilled_mib": 0, "legacy_leaked_mib": 0, "fixed_leaked_mib": 0},
        {"case": "timeout_after", "spilled_mib": 64, "legacy_leaked_mib": 64, "fixed_leaked_mib": 0},
        {"case": "cancel_before", "spilled_mib": 0, "legacy_leaked_mib": 0, "fixed_leaked_mib": 0},
        {"case": "cancel_after", "spilled_mib": 64, "legacy_leaked_mib": 64, "fixed_leaked_mib": 0},
    ]
    return {
        "min_file_ref_bytes": 32 * 1024 * 1024,
        "outcomes": outcomes,
        "legacy_total_leaked_mib": sum(o["legacy_leaked_mib"] for o in outcomes),
        "fixed_total_leaked_mib": sum(o["fixed_leaked_mib"] for o in outcomes),
    }


def simulate_vllm_spec_autofit_kv_reset_and_trtllm_retry() -> dict[str, Any]:
    """Model vllm#60513, vllm#60810, vllm#63476, and TensorRT-LLM#20889."""
    # 1. vllm#60513: Speculator dynamic max_model_len property after auto-fit
    initial_max_model_len = 32768
    draft_max_seq_len = 32768
    autofit_max_model_len = 8192
    legacy_speculator_max_seq_len = min(draft_max_seq_len, initial_max_model_len)
    fixed_speculator_max_seq_len = min(draft_max_seq_len, autofit_max_model_len)

    # 2. vllm#60810: Streaming session (resumable=True) KV offload block_ids reset
    # Chunk 1 allocates blocks [0, 1]; chunk 2 resumes as scheduled_new_reqs with full table [0, 1, 2, 3, 4]
    # and num_computed_tokens = 2 blocks (32 tokens at block_size=16).
    chunk1_blocks = [0, 1]
    resumed_full_table = [0, 1, 2, 3, 4]
    computed_blocks = 2

    # Legacy: is_full_table=False on scheduled_new_reqs -> appends full table onto existing [0, 1] -> [0, 1, 0, 1, 2, 3, 4]
    # Then slices uncomputed 3 blocks starting at index 2 -> [0, 1, 2] instead of [2, 3, 4]!
    legacy_accumulated = list(chunk1_blocks) + list(resumed_full_table)
    legacy_chunk2_stored_blocks = legacy_accumulated[computed_blocks : computed_blocks + 3]

    # Fixed: is_full_table=True clears block_ids before extending -> [0, 1, 2, 3, 4]
    fixed_accumulated = list(resumed_full_table)
    fixed_chunk2_stored_blocks = fixed_accumulated[computed_blocks : computed_blocks + 3]

    # 3. vllm#63476: DBO microbatch chronological error reporting
    # Microbatch 1 fails first with RuntimeError("OOM in expert MLP"), then microbatch 0 fails on recv hook
    errors_dict = {1: "RuntimeError: OOM in expert MLP", 0: "RuntimeError: peer recv hook broken"}
    legacy_reported_ubatch = min(errors_dict)
    fixed_reported_ubatch = next(iter(errors_dict))

    # 4. TensorRT-LLM#20889: aiohttp.ClientOSError(errno.ECONNRESET) transient TCP budget
    transient_tcp_budget = 5
    default_max_retries = 0
    retry_cases = [
        {
            "error": "ClientOSError(ECONNRESET)",
            "errno": errno.ECONNRESET,
            "is_transient_tcp": True,
            "legacy_effective_max_retries": default_max_retries,
            "fixed_effective_max_retries": max(default_max_retries, transient_tcp_budget),
        },
        {
            "error": "ClientOSError(EPIPE)",
            "errno": errno.EPIPE,
            "is_transient_tcp": False,
            "legacy_effective_max_retries": default_max_retries,
            "fixed_effective_max_retries": default_max_retries,
        },
    ]

    return {
        "speculator_autofit": {
            "initial_max_model_len": initial_max_model_len,
            "autofit_max_model_len": autofit_max_model_len,
            "legacy_speculator_max_seq_len": legacy_speculator_max_seq_len,
            "fixed_speculator_max_seq_len": fixed_speculator_max_seq_len,
        },
        "streaming_kv_offload": {
            "chunk1_blocks": chunk1_blocks,
            "resumed_full_table": resumed_full_table,
            "expected_chunk2_new_blocks": [2, 3, 4],
            "legacy_chunk2_stored_blocks": legacy_chunk2_stored_blocks,
            "fixed_chunk2_stored_blocks": fixed_chunk2_stored_blocks,
        },
        "dbo_microbatch_error_order": {
            "insertion_order_ubatches": list(errors_dict.keys()),
            "legacy_reported_ubatch": legacy_reported_ubatch,
            "fixed_reported_ubatch": fixed_reported_ubatch,
            "fixed_reports_root_cause": errors_dict[fixed_reported_ubatch] == "RuntimeError: OOM in expert MLP",
        },
        "trtllm_client_oserror_retry": retry_cases,
    }


def run_experiment() -> dict[str, Any]:
    hisparse = simulate_hisparse_dsv4_token_granular_backup()
    dp_attn = simulate_dp_attention_bucket_and_custom_ar()
    ipc_cleanup = simulate_diffusion_ipc_cleanup()
    vllm_trtllm = simulate_vllm_spec_autofit_kv_reset_and_trtllm_retry()
    return {
        "experiment": "sglang_hisparse_dsv4_dp_attn_bucket_diffusion_ipc_vllm_spec_autofit_kv_reset_trtllm",
        "phase": 77,
        "priority": 86,
        "hisparse_dsv4_backup": hisparse,
        "dp_attention_alignment": dp_attn,
        "diffusion_ipc_cleanup": ipc_cleanup,
        "vllm_and_trtllm": vllm_trtllm,
        "invariants": {
            "hisparse_eliminates_32_of_64_corrupted_tokens": (
                hisparse["corrupted_tokens_legacy"] == 32
                and hisparse["corrupted_tokens_fixed"] == 0
            ),
            "dp_attn_aligns_prefill_bucket_and_decode_sum_len": (
                not dp_attn["legacy_bucket_aligned"]
                and dp_attn["fixed_bucket_aligned"]
                and dp_attn["fixed_target_bucket"] == 24
                and dp_attn["fixed_cuda_graph_padding_mode"] == "SUM_LEN"
            ),
            "diffusion_ipc_reclaims_all_256_mib_across_faults": (
                ipc_cleanup["legacy_total_leaked_mib"] == 256
                and ipc_cleanup["fixed_total_leaked_mib"] == 0
            ),
            "vllm_spec_autofit_and_streaming_kv_reset_exact": (
                vllm_trtllm["speculator_autofit"]["fixed_speculator_max_seq_len"] == 8192
                and vllm_trtllm["streaming_kv_offload"]["legacy_chunk2_stored_blocks"] == [0, 1, 2]
                and vllm_trtllm["streaming_kv_offload"]["fixed_chunk2_stored_blocks"] == [2, 3, 4]
                and vllm_trtllm["dbo_microbatch_error_order"]["fixed_reports_root_cause"]
            ),
            "trtllm_econnreset_gets_5_retries_and_epipe_stays_0": (
                vllm_trtllm["trtllm_client_oserror_retry"][0]["fixed_effective_max_retries"] == 5
                and vllm_trtllm["trtllm_client_oserror_retry"][1]["fixed_effective_max_retries"] == 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 77 Track B experiment.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate invariants and exit non-zero on failure.",
    )
    args = parser.parse_args()
    summary = run_experiment()
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not all(summary["invariants"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
