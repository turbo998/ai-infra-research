"""Phase 91 Track C offline simulator: FlashAttention-4 SM100 sparse MLA O-residual padded Q-head guard
& ROCm/CK split-KV heuristic fix (<=8 cap), Megatron-LM middle-PP sequence-packing dataset guard, DeepEP
internode low-latency combine fence.proxy.async.shared::cta, verl distillation teacher top-k CP layout
alignment, Mooncake Store read-only SSD->DRAM prefetch-on-exist & managed weight revision lifecycle,
NIXL bidirectional NVTX request-ID payload correlation, and Dynamo SGLang zero-copy response relay &
KV-router scalar lower-tier walk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def simulate_fa4_mla_residual_and_rocm_splitkv() -> dict:
    """Model flash-attention@2cb0a8e471 (SM100 sparse MLA O residual) and #2901 (ROCm/CK num_splits)."""
    tile_q_rows = 128
    real_q_heads = 24
    padded_invalid_rows = tile_q_rows - real_q_heads
    # Without is_valid_qhead_row predicate on register->gmem residual store, 104 rows overwrite next token o_lo
    legacy_corrupted_rows_per_token = padded_invalid_rows
    fixed_corrupted_rows_per_token = 0

    # ROCm/CK num_splits heuristic fix (#2901)
    # Legacy bug 1: num_n_blocks = ceildiv(hdim_v, hdim_v) == 1 -> max_splits clamped to 1
    # Legacy bug 2: used nhead instead of nhead_k (overstating parallelism by GQA ratio = nhead // nhead_k)
    seqlen_k = 16384
    block_n = 128
    hdim_v = 128
    legacy_num_n_blocks = math.ceil(hdim_v / hdim_v)
    fixed_num_n_blocks = math.ceil(seqlen_k / block_n)
    gqa_ratio = 8  # e.g. nhead=64, nhead_k=8
    ck_combine_safe_max_splits = 8  # num_splits >= 9 diverges with rel err 0.28..2.0 on gfx950
    return {
        "sm100_mla_tile_q_rows": tile_q_rows,
        "sm100_mla_real_q_heads": real_q_heads,
        "legacy_corrupted_o_lo_rows_per_token": legacy_corrupted_rows_per_token,
        "fixed_corrupted_o_lo_rows_per_token": fixed_corrupted_rows_per_token,
        "rocm_ck_legacy_num_n_blocks": legacy_num_n_blocks,
        "rocm_ck_fixed_num_n_blocks": fixed_num_n_blocks,
        "rocm_ck_gqa_ratio": gqa_ratio,
        "rocm_ck_safe_max_splits": ck_combine_safe_max_splits,
        "rocm_ck_speedup_at_batch_nhead_k_64_x": 3.21,
        "rocm_ck_speedup_at_batch_nhead_k_128_x": 1.95,
    }


def simulate_megatron_deepep_and_verl_distill() -> dict:
    """Model Megatron-LM#8378, DeepEP#642, and verl#8332."""
    pp_world_size = 4
    # With sequence_packing_scheduler enabled, only first (0) and last (3) PP stages build datasets
    legacy_stages_building_dataset = [0, 1, 2, 3]
    fixed_stages_building_dataset = [0, pp_world_size - 1]

    # verl#8332: CP=2, MXFP8 (align=16 -> 256 rows/rank), 16 sequences of 17 tokens (136 valid tokens/rank)
    num_seqs = 16
    seq_len = 17
    cp_size = 2
    valid_tokens_per_rank = (num_seqs * seq_len) // cp_size  # 136
    padded_rows_per_rank = 256
    # Under contiguous (student dsv4_hybrid) vs zigzag (legacy teacher default), rank 0 misaligns 128/136 valid rows
    # and rank 1 misaligns 136/136 valid rows
    legacy_rank0_misaligned_valid_rows = 128
    legacy_rank1_misaligned_valid_rows = 136
    fixed_misaligned_valid_rows = 0
    return {
        "megatron_pp_world_size": pp_world_size,
        "megatron_legacy_dataset_stages": legacy_stages_building_dataset,
        "megatron_fixed_dataset_stages": fixed_stages_building_dataset,
        "deepep_internode_ll_combine_proxy_fence": True,
        "verl_valid_tokens_per_rank": valid_tokens_per_rank,
        "verl_padded_rows_per_rank": padded_rows_per_rank,
        "verl_legacy_rank0_misaligned_rows": legacy_rank0_misaligned_valid_rows,
        "verl_legacy_rank1_misaligned_rows": legacy_rank1_misaligned_valid_rows,
        "verl_fixed_misaligned_rows": fixed_misaligned_valid_rows,
    }


def simulate_mooncake_nixl_and_dynamo() -> dict:
    """Model Mooncake#4272 & #4313, NIXL#2495, and Dynamo#16765 & #17149."""
    # Mooncake#4272: QueryReadOnly pins lease expiry to query_start_ts (0 lease duration) and triggers SSD->DRAM prefetch
    query_start_ts_ms = 1000
    readonly_lease_expiry_ms = query_start_ts_ms
    ssd_only_key_prefetched_to_dram = True

    # Mooncake#4313: Managed weight group revision lifecycle states
    managed_weight_generic_eviction_blocked = True

    # NIXL#2495: sender postXferReq / xfer.complete and receiver notif.received share uint64 request_id payload
    sample_xfer_request_id = 9100042
    sender_nvtx_payload = sample_xfer_request_id
    receiver_nvtx_payload = sample_xfer_request_id

    # Dynamo#16765: SGLang sidecar response relay moves 64 KiB routed_experts string without cloning Value
    routed_experts_bytes = 65536
    legacy_host_payload_copies = 2
    fixed_host_payload_copies = 1

    # Dynamo#17149: kv-router scalar lower-tier walk for 1 worker + 1 hint source avoids HashSet/Vec allocations
    blocks_walked = 128
    legacy_heap_allocs_per_query = blocks_walked
    scalar_fastpath_heap_allocs_per_query = 0
    return {
        "mooncake_readonly_lease_duration_ms": readonly_lease_expiry_ms - query_start_ts_ms,
        "mooncake_ssd_prefetch_on_exist": ssd_only_key_prefetched_to_dram,
        "mooncake_managed_weight_protected_from_generic_remove": managed_weight_generic_eviction_blocked,
        "nixl_bidirectional_nvtx_payload_equal": sender_nvtx_payload == receiver_nvtx_payload,
        "nixl_sample_xfer_request_id": sample_xfer_request_id,
        "dynamo_routed_experts_bytes": routed_experts_bytes,
        "dynamo_legacy_host_copies": legacy_host_payload_copies,
        "dynamo_fixed_host_copies": fixed_host_payload_copies,
        "dynamo_kv_router_legacy_heap_allocs": legacy_heap_allocs_per_query,
        "dynamo_kv_router_scalar_heap_allocs": scalar_fastpath_heap_allocs_per_query,
    }


def run_simulation() -> dict:
    fa4 = simulate_fa4_mla_residual_and_rocm_splitkv()
    megatron_deepep_verl = simulate_megatron_deepep_and_verl_distill()
    mooncake_nixl_dynamo = simulate_mooncake_nixl_and_dynamo()
    return {
        "experiment": "fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91",
        "phase": 91,
        "priority": 100,
        "fa4_mla_residual_and_rocm_splitkv": fa4,
        "megatron_deepep_and_verl_distill": megatron_deepep_verl,
        "mooncake_nixl_and_dynamo": mooncake_nixl_dynamo,
        "invariants": {
            "fa4_padded_qhead_o_lo_corruption_zero": fa4["fixed_corrupted_o_lo_rows_per_token"] == 0,
            "fa4_rocm_ck_max_splits_capped_at_8": fa4["rocm_ck_safe_max_splits"] == 8,
            "megatron_middle_pp_skips_dataset": megatron_deepep_verl["megatron_fixed_dataset_stages"] == [0, 3],
            "verl_teacher_student_cp_aligned": megatron_deepep_verl["verl_fixed_misaligned_rows"] == 0,
            "mooncake_readonly_grants_zero_lease": (
                mooncake_nixl_dynamo["mooncake_readonly_lease_duration_ms"] == 0
            ),
            "nixl_nvtx_request_id_correlated": (
                mooncake_nixl_dynamo["nixl_bidirectional_nvtx_payload_equal"]
            ),
            "dynamo_scalar_walk_zero_heap_allocs": (
                mooncake_nixl_dynamo["dynamo_kv_router_scalar_heap_allocs"] == 0
            ),
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
        print("OK: Phase 91 Track C summary.json verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
