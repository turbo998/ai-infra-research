#!/usr/bin/env python3
"""Phase 59 Track C deterministic offline models for Megatron-LM, TE, TorchTitan, LMCache & Mooncake.

Models:
1. Megatron-LM Fused GDP Prefill & Decode Prepare Kernel (`megatron@b291235082`)
2. Megatron-LM Batched Packed MTP Context-Parallel Boundary Exchanges (`megatron@50f94464f6`)
3. Megatron-LM Live-Row Bounded Batch-Invariant MoE & Mamba2 Target-Row Decode (`megatron@77232c387a` & `aa46a57229`)
4. TransformerEngine cuBLASMp 0.11.0+ All-Gathered `grad_output` Reuse in `wgrad` (`te@1aaede2f9e`)
5. TorchTitan RL Generator Dist-MoE Fused SwiGLU MXFP8 Mega Kernel (`torchtitan@ed3c713afe`)
6. LMCache L1/L2 Memory Tags Affinity & DAX Range-Based `[offset, offset+size)` Ownership (`lmcache@7e4f499a20` & `8afb6da466`)
7. Mooncake EFA 2 MiB THP PTE Budgeting & Three-Tier Fallback (`mooncake@088d599e62`)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def model_megatron_gdp_prepare() -> dict[str, Any]:
    """Model Megatron-LM fused GDP prefill & decode prepare kernel (`megatron@b291235082`)."""
    eager_prefill_launches_per_layer = 14
    fused_prefill_launches_per_layer = 1
    num_gdp_layers = 32

    return {
        "commit": "b291235082",
        "eager_prefill_launches_per_layer": eager_prefill_launches_per_layer,
        "fused_prefill_launches_per_layer": fused_prefill_launches_per_layer,
        "launch_reduction_factor_x": eager_prefill_launches_per_layer / fused_prefill_launches_per_layer,
        "saved_launches_32_layers": (eager_prefill_launches_per_layer - fused_prefill_launches_per_layer) * num_gdp_layers,
        "decode_grid_axis": "one_program_per_head",
        "prefill_grid_axis": "one_program_per_token_copy",
        "uses_libdevice_transcendentals_for_bitwise_parity": True,
    }


def model_megatron_mtp_cp_boundary_exchange() -> dict[str, Any]:
    """Model Megatron-LM batched packed MTP CP roll boundary exchanges (`megatron@50f94464f6`)."""
    cp_size = 4
    # Interior rank (0 < local_rank < cp_size - 1) exchanges first_boundary and second_boundary
    # with both previous_rank and next_rank in one batch_isend_irecv on cp_group
    interior_p2p_ops = 4
    boundary_rank_p2p_ops = 2

    # Padded boundary translation example from unit test
    cu_seqlens_unpadded = [0, 4, 8, 8, 8]
    translated_cp0_tp0 = [0, 4, 4, 4, 8, 8, 8, 8, 8]
    translated_cp0_tp1 = [0, 0, 0, 0, 0, 4, 4, 4, 8]

    return {
        "commit": "50f94464f6",
        "cp_size": cp_size,
        "batched_irecv_isend_calls_per_roll": 1,
        "interior_rank_p2p_ops_in_batch": interior_p2p_ops,
        "edge_rank_p2p_ops_in_batch": boundary_rank_p2p_ops,
        "uses_cp_group_not_world": True,
        "cu_seqlens_unpadded": cu_seqlens_unpadded,
        "translated_cp0_tp0": translated_cp0_tp0,
        "translated_cp0_tp1": translated_cp0_tp1,
    }


def model_megatron_live_row_moe_and_mamba2() -> dict[str, Any]:
    """Model Megatron-LM live-row batch-invariant MoE & Mamba2 target-row decode (`77232c387a` & `aa46a57229`)."""
    max_rows = 16384
    K = 2688
    live_rows = 4992  # 39 full 128-row scale tiles
    scale_tile_rows = 128
    k_groups = (K // 32 + 3) // 4
    live_scale_bytes = (live_rows // scale_tile_rows) * k_groups * 512
    worst_case_scale_bytes = (max_rows // scale_tile_rows) * k_groups * 512

    # Mamba2 batch-invariant decode without z replay buffer
    chunk_size = 256
    block_size_m = 64
    mma_row_tile = 16
    nchunks = 64
    nheads = 64
    headdim = 64
    bytes_per_elem = 2  # bf16

    legacy_z_replay_bytes = nchunks * chunk_size * nheads * headdim * bytes_per_elem
    target_row_z_bytes = nchunks * 1 * nheads * headdim * bytes_per_elem

    return {
        "commit_moe_live_rows": "77232c387a",
        "commit_mamba2_rows": "aa46a57229",
        "moe_max_rows": max_rows,
        "moe_live_rows": live_rows,
        "moe_work_reduction_ratio": round(max_rows / live_rows, 4),
        "moe_live_scale_bytes": live_scale_bytes,
        "moe_worst_case_scale_bytes": worst_case_scale_bytes,
        "mamba2_chunk_size": chunk_size,
        "mamba2_training_block_m": block_size_m,
        "mamba2_target_mma_tile_m": mma_row_tile,
        "mamba2_legacy_z_replay_bytes": legacy_z_replay_bytes,
        "mamba2_target_row_z_bytes": target_row_z_bytes,
        "mamba2_z_buffer_reduction_x": chunk_size,
    }


def model_te_cublasmp_wgrad_reuse() -> dict[str, Any]:
    """Model TransformerEngine cuBLASMp 0.11.0+ all-gathered grad_output reuse (`te@1aaede2f9e`)."""
    min_cublasmp_version = 1100  # 0.11.0 -> 0*10000 + 11*100 + 0
    tokens = 8192
    hidden = 8192
    bytes_per_elem = 2  # BF16
    gathered_grad_output_mib = (tokens * hidden * bytes_per_elem) / (1024 * 1024)  # 128 MiB

    return {
        "commit": "1aaede2f9e",
        "min_cublasmp_version_int": min_cublasmp_version,
        "row_parallel_linear_bwd_allgathers_before": 1,
        "row_parallel_linear_bwd_allgathers_after": 0,
        "layernorm_mlp_bwd_allgathers_before": 2,
        "layernorm_mlp_bwd_allgathers_after": 1,
        "saved_allgather_mib_per_layer": gathered_grad_output_mib,
        "fallback_below_0_11_0_without_exception": True,
    }


def model_torchtitan_dist_moe_mega_swiglu() -> dict[str, Any]:
    """Model TorchTitan RL generator Dist-MoE MXFP8 fused SwiGLU mega kernel (`torchtitan@ed3c713afe`)."""
    return {
        "commit": "ed3c713afe",
        "model": "Qwen3-30B-A3B",
        "staged_pipeline_speedup_min_x": 0.90,
        "staged_pipeline_speedup_max_x": 1.35,
        "mega_fused_speedup_min_x": 1.50,
        "mega_fused_speedup_max_x": 1.70,
        "mxfp8_vs_bf16_measured_rel_error": 0.07,
        "mxfp8_rel_tol_gate": 0.12,
        "swapped_gate_up_rel_error": 0.90,
        "interleaves_gate_up_rows_for_mega": True,
    }


def model_lmcache_tags_and_dax_ranges() -> dict[str, Any]:
    """Model LMCache L1/L2 memory tags & DAX range-based ownership (`lmcache@7e4f499a20` & `8afb6da466`)."""
    l1_managers = [
        {"id": "l1_cxl_0", "tags": ("cxl", "local")},
        {"id": "l1_dram_0", "tags": ("dram", "local")},
    ]
    wanted_tags_priority = [("dram", "local"), ("cxl", "local")]

    def resolve_l1_affinity(managers: list[dict[str, Any]], wanted: list[tuple[str, str]]) -> str:
        for pref in wanted:
            for mgr in managers:
                if mgr["tags"] == pref:
                    return mgr["id"]
        raise ValueError("No matching L1 manager")

    selected_l1 = resolve_l1_affinity(l1_managers, wanted_tags_priority)

    # DAX range overlap verification on /dev/dax0.0
    active_ranges = [
        {"tier": "L1", "path": "/dev/dax0.0", "offset_gb": 0, "size_gb": 256},
        {"tier": "L2", "path": "/dev/dax0.0", "offset_gb": 256, "size_gb": 256},
    ]

    def overlaps(r1_off: int, r1_sz: int, r2_off: int, r2_sz: int) -> bool:
        return max(r1_off, r2_off) < min(r1_off + r1_sz, r2_off + r2_sz)

    candidate_disjoint = {"path": "/dev/dax0.0", "offset_gb": 512, "size_gb": 256}
    candidate_conflict = {"path": "/dev/dax0.0", "offset_gb": 128, "size_gb": 256}

    disjoint_ok = not any(
        overlaps(r["offset_gb"], r["size_gb"], candidate_disjoint["offset_gb"], candidate_disjoint["size_gb"])
        for r in active_ranges
    )
    conflict_rejected = any(
        overlaps(r["offset_gb"], r["size_gb"], candidate_conflict["offset_gb"], candidate_conflict["size_gb"])
        for r in active_ranges
    )

    return {
        "commit_memory_tags": "7e4f499a20",
        "commit_dax_offset": "8afb6da466",
        "selected_l1_for_host_adapter": selected_l1,
        "disjoint_same_device_l1_l2_allowed": disjoint_ok,
        "overlapping_range_returns_http_409": conflict_rejected,
    }


def model_mooncake_efa_thp_budget() -> dict[str, Any]:
    """Model Mooncake EFA 2 MiB THP PTE budgeting & 3-tier fallback (`mooncake@088d599e62`)."""
    max_pte_entries = 22 * 1024 * 1024  # 22M PTEs per EFA NIC
    page_4kb = 4 * 1024
    page_2mb = 2 * 1024 * 1024

    max_bytes_per_nic_4kb = max_pte_entries * page_4kb  # 88 GiB
    max_bytes_per_nic_2mb = max_pte_entries * page_2mb  # 44 TiB

    max_gib_4kb = max_bytes_per_nic_4kb / (1024**3)  # 88.0 GiB
    max_tib_2mb = max_bytes_per_nic_2mb / (1024**4)  # 44.0 TiB

    # 1 TiB buffer on a 2-NIC host
    buffer_gib = 1024
    num_nics = 2
    chunks_needed_4kb = math.ceil(buffer_gib / max_gib_4kb)  # 12 chunks

    # smaps parser validation: "AnonHugePages: 2048 kB" starts with hex char 'A'
    smaps_lines = [
        "7f8a00000000-7f8a40000000 rw-p 00000000 00:00 0",
        "Size:            1048576 kB",
        "KernelPageSize:        4 kB",
        "AnonHugePages:   1048576 kB",
    ]
    vma_headers_detected = sum(1 for line in smaps_lines if "-" in line.split()[0])

    return {
        "commit": "088d599e62",
        "max_pte_entries_per_nic": max_pte_entries,
        "max_nic_registration_gib_at_4kb": max_gib_4kb,
        "max_nic_registration_tib_at_2mb": max_tib_2mb,
        "pte_capacity_multiplier_x": page_2mb // page_4kb,
        "chunks_needed_for_1tib_at_4kb": chunks_needed_4kb,
        "num_nics_example": num_nics,
        "three_tier_plans": ["kAuto_2MB_full", "kThpPartitioned_2MB_sharded", "kBasePages_4KB_partitioned"],
        "smaps_vma_headers_detected": vma_headers_detected,
    }


def build_summary() -> dict[str, Any]:
    return {
        "phase": 59,
        "priority": 68,
        "track": "Track C — Distributed Training, Collective Overlap & Disaggregated KV Storage",
        "megatron_gdp_prepare": model_megatron_gdp_prepare(),
        "megatron_mtp_cp_boundary_exchange": model_megatron_mtp_cp_boundary_exchange(),
        "megatron_live_row_moe_and_mamba2": model_megatron_live_row_moe_and_mamba2(),
        "te_cublasmp_wgrad_reuse": model_te_cublasmp_wgrad_reuse(),
        "torchtitan_dist_moe_mega_swiglu": model_torchtitan_dist_moe_mega_swiglu(),
        "lmcache_tags_and_dax_ranges": model_lmcache_tags_and_dax_ranges(),
        "mooncake_efa_thp_budget": model_mooncake_efa_thp_budget(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches model")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of sync with run.py")
        print("Phase 59 Track C summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
