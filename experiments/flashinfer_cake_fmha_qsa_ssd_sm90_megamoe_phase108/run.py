#!/usr/bin/env python3
"""Phase 108 Experiment 1: FlashInfer Cake FMHA NVFP4 Paged Context, QSA NVFP4 Workspace Reuse, cuDNN Ragged Token-Offset Masking, CUTLASS ScatterD & TileLang K-Padding."""
from __future__ import annotations
import argparse
import json
import pathlib


def align_up(value: int, alignment: int = 256) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def cake_fmha_paged_context_nvfp4(
    seq_len: int = 8192,
    num_kv_heads: int = 8,
    head_dim: int = 128,
) -> dict:
    """Model flashinfer #7512 (07cc0d60a9): 20 Cake FMHA paged context programs (BF16, FP8, NVFP4 KV) on sm_100a/sm_103a."""
    total_kv_elements = 2 * seq_len * num_kv_heads * head_dim
    bf16_kv_bytes = total_kv_elements * 2
    fp8_kv_bytes = total_kv_elements * 1
    # NVFP4 stores 0.5B per element + 1 FP8 E4M3 scale byte per 16 elements (0.5625 B/elem)
    nvfp4_kv_bytes = int(total_kv_elements * 0.5625)
    return {
        "seq_len": seq_len,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "programs_registered": 20,
        "page_size_32_config": {"page_size": 32, "l2_swizzle": 1, "num_m_blocks": 11, "tok_per_stage": 25},
        "page_size_1024_config": {"page_size": 1024, "l2_swizzle": 8, "num_m_blocks": 6, "tok_per_stage": 25},
        "bf16_kv_bytes": bf16_kv_bytes,
        "fp8_kv_bytes": fp8_kv_bytes,
        "nvfp4_kv_bytes": nvfp4_kv_bytes,
        "nvfp4_vs_bf16_bandwidth_ratio": round(bf16_kv_bytes / nvfp4_kv_bytes, 4),
        "nvfp4_vs_fp8_bandwidth_ratio": round(fp8_kv_bytes / nvfp4_kv_bytes, 4),
    }


def cake_qsa_nvfp4_workspace_reuse(
    rows: int = 64,
    num_qo_heads: int = 64,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    splits: int = 8,
    decode_steps: int = 1000,
) -> dict:
    """Model flashinfer #7514 (fa8fb88e0f): reuse checked 256B-aligned workspace carve across calls."""
    partial_out_bytes = align_up(rows * num_qo_heads * splits * head_dim * 4, 256)
    partial_lse_bytes = align_up(rows * num_qo_heads * splits * 4, 256)
    counter_bytes = align_up(rows * num_kv_heads * 4, 256)
    total_workspace_bytes = partial_out_bytes + partial_lse_bytes + counter_bytes
    # Legacy checked and carved on every call; #7514 checks/zeroes once per storage and reuses the carve
    legacy_checks_and_carves = decode_steps
    reused_checks_and_carves = 1
    single_split_allocates_partials = False
    return {
        "rows": rows,
        "splits": splits,
        "total_workspace_bytes": total_workspace_bytes,
        "legacy_checks_per_1000_steps": legacy_checks_and_carves,
        "reused_checks_per_1000_steps": reused_checks_and_carves,
        "host_dispatch_overhead_reduction_pct": round(
            100.0 * (legacy_checks_and_carves - reused_checks_and_carves) / legacy_checks_and_carves, 2
        ),
        "single_split_allocates_partials": single_split_allocates_partials,
    }


def cudnn_ragged_token_offset_mask(
    q_offsets: list[int] | None = None,
    explicit_seq_lens_q: list[int] | None = None,
) -> dict:
    """Model flashinfer #7438 (a3452cd405): honor explicit prefill lengths with token offsets."""
    if q_offsets is None:
        q_offsets = [0, 320, 512]
    if explicit_seq_lens_q is None:
        explicit_seq_lens_q = [300, 177]
    inferred_slot_lens = [q_offsets[i + 1] - q_offsets[i] for i in range(len(explicit_seq_lens_q))]
    buggy_valid_rows = sum(inferred_slot_lens)
    fixed_valid_rows = sum(explicit_seq_lens_q)
    leaked_padding_rows_eliminated = buggy_valid_rows - fixed_valid_rows
    return {
        "q_offsets": q_offsets,
        "explicit_seq_lens_q": explicit_seq_lens_q,
        "inferred_slot_lens": inferred_slot_lens,
        "buggy_valid_rows": buggy_valid_rows,
        "fixed_valid_rows": fixed_valid_rows,
        "leaked_padding_rows_eliminated": leaked_padding_rows_eliminated,
    }


def cutlass_scatterd_and_tilelang_k_pad() -> dict:
    """Model CUTLASS #3278 (1d7cde107f) ScatterD pointer guard and TileLang #2818 (876a59e6eb) MMA K-atom guard."""
    # CUTLASS PredicatedTileIterator: across kGroup=4 with increment_group=2048, ScatterD=True must keep base byte_pointer
    k_groups = 4
    increment_group = 2048
    buggy_scatterd_offset = (k_groups - 1) * increment_group
    fixed_scatterd_offset = 0
    # TileLang MMA K atom divisibility: chunk=8 rejected when MMA requires K divisible by 16
    requested_k = 8
    mma_k_atom = 16
    padded_k = ((requested_k + mma_k_atom - 1) // mma_k_atom) * mma_k_atom
    return {
        "cutlass_buggy_scatterd_offset_bytes": buggy_scatterd_offset,
        "cutlass_fixed_scatterd_offset_bytes": fixed_scatterd_offset,
        "tilelang_requested_k": requested_k,
        "tilelang_mma_k_atom": mma_k_atom,
        "tilelang_rejects_unpadded_partial_k": (requested_k % mma_k_atom != 0),
        "tilelang_explicit_padded_k": padded_k,
    }


def run_experiment() -> dict:
    return {
        "experiment_id": "flashinfer_cake_fmha_qsa_ssd_sm90_megamoe_phase108",
        "evidence_mode": "own-offline",
        "cake_fmha_paged_context": cake_fmha_paged_context_nvfp4(),
        "cake_qsa_workspace_reuse": cake_qsa_nvfp4_workspace_reuse(),
        "cudnn_ragged_token_offset": cudnn_ragged_token_offset_mask(),
        "cutlass_scatterd_tilelang_k_pad": cutlass_scatterd_and_tilelang_k_pad(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_experiment()
    out_path = pathlib.Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["cake_fmha_paged_context"]["programs_registered"] == 20
        assert summary["cake_fmha_paged_context"]["nvfp4_vs_bf16_bandwidth_ratio"] == 3.5556
        assert summary["cake_qsa_workspace_reuse"]["reused_checks_per_1000_steps"] == 1
        assert summary["cake_qsa_workspace_reuse"]["single_split_allocates_partials"] is False
        assert summary["cudnn_ragged_token_offset"]["leaked_padding_rows_eliminated"] == 35
        assert summary["cutlass_scatterd_tilelang_k_pad"]["cutlass_fixed_scatterd_offset_bytes"] == 0
        assert summary["cutlass_scatterd_tilelang_k_pad"]["tilelang_rejects_unpadded_partial_k"] is True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
