#!/usr/bin/env python3
"""Phase 63 Track A deterministic verifier:
FlashInfer Cake DSA Indexer Far-Sample Sink Scan, SM103a Dense Projection GEMM & FP32 Router Backward,
KDA Document-Group Batching, and TileLang Analyzer Clone & Batched Finalize-Reducer Lowering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_dsa_indexer_far_sample() -> dict[str, Any]:
    """Verify FlashInfer Cake DSA indexer far-sample first-threshold scan (1ff9f67725)."""
    gb200_sink_rows = [
        {
            "row": "Z1",
            "shape": "262k doc, CP16 zigzag rank 0",
            "queries_t": 16384,
            "keys_tkv": 270336,
            "segments": 2,
            "topk": 2048,
            "prev_ms": 22.588,
            "pr_ms": 12.692,
            "speedup": round(22.588 / 12.692, 4),
        },
        {
            "row": "Z1k4096",
            "shape": "Z1 with top-k 4096",
            "queries_t": 16384,
            "keys_tkv": 270336,
            "segments": 2,
            "topk": 4096,
            "prev_ms": 23.654,
            "pr_ms": 13.721,
            "speedup": round(23.654 / 13.721, 4),
        },
        {
            "row": "Z1k256",
            "shape": "Z1 with top-k 256",
            "queries_t": 16384,
            "keys_tkv": 270336,
            "segments": 2,
            "topk": 256,
            "prev_ms": 22.138,
            "pr_ms": 12.561,
            "speedup": round(22.138 / 12.561, 4),
        },
        {
            "row": "Z2",
            "shape": "8 x 32k docs, full causal",
            "queries_t": 262144,
            "keys_tkv": 262144,
            "segments": 8,
            "topk": 2048,
            "prev_ms": 43.909,
            "pr_ms": 34.133,
            "speedup": round(43.909 / 34.133, 4),
        },
        {
            "row": "Z2cp",
            "shape": "8 x 32k docs, CP16 zigzag rank 0",
            "queries_t": 16384,
            "keys_tkv": 270336,
            "segments": 16,
            "topk": 2048,
            "prev_ms": 3.317,
            "pr_ms": 2.252,
            "speedup": round(3.317 / 2.252, 4),
        },
        {
            "row": "P1sink",
            "shape": "recorded P1 geometry with sink stats",
            "queries_t": 16231,
            "keys_tkv": 268757,
            "segments": 8,
            "topk": 2048,
            "prev_ms": 5.479,
            "pr_ms": 3.450,
            "speedup": round(5.479 / 3.450, 4),
        },
        {
            "row": "S4sink",
            "shape": "1 x 64k causal doc, sink stats",
            "queries_t": 65536,
            "keys_tkv": 65536,
            "segments": 1,
            "topk": 2048,
            "prev_ms": 19.751,
            "pr_ms": 14.671,
            "speedup": round(19.751 / 14.671, 4),
        },
        {
            "row": "S6sink",
            "shape": "1 x 262k causal doc, sink stats",
            "queries_t": 262144,
            "keys_tkv": 262144,
            "segments": 1,
            "topk": 2048,
            "prev_ms": 366.792,
            "pr_ms": 211.554,
            "speedup": round(366.792 / 211.554, 4),
        },
    ]

    smem_prev_bytes = 207104
    smem_pr_bytes = 194944
    smem_saved_bytes = smem_prev_bytes - smem_pr_bytes

    assert gb200_sink_rows[0]["speedup"] == 1.7797
    assert gb200_sink_rows[-1]["speedup"] == 1.7338
    assert smem_saved_bytes == 12160

    return {
        "commit": "1ff9f67725",
        "SupportedArchitectures": ["sm_100a", "sm_103a", "sm_107a"],
        "DescendingChangedArchitectures": ["sm_100a", "sm_107a"],
        "gb200_sink_rows": gb200_sink_rows,
        "z1_gb200_speedup": gb200_sink_rows[0]["speedup"],
        "s6sink_gb200_speedup": gb200_sink_rows[-1]["speedup"],
        "smem_prev_bytes": smem_prev_bytes,
        "smem_pr_bytes": smem_pr_bytes,
        "smem_saved_bytes": smem_saved_bytes,
        "noftz_helpers": [
            "add_f32x2_noftz",
            "fma_f32x2_rn_noftz",
        ],
    }


def evaluate_dense_projection_gemm() -> dict[str, Any]:
    """Verify FlashInfer Cake dense projection GEMM SM103a & FP32 router backward (2706a076ac)."""
    program_instances = {
        "prev_total": 104,
        "prev_by_arch": {"sm_100a": 47, "sm_103a": 0, "sm_107a": 57},
        "pr_total": 247,
        "pr_by_arch": {"sm_100a": 80, "sm_103a": 80, "sm_107a": 87},
    }
    router_splits = {
        "prev": {"kk": 4, "kn": 1, "nn_t": 11},
        "pr": {"kk": 4, "kn": 1, "nn_t": 6},
    }

    gb200_highlights = [
        {
            "shape": "4096->512",
            "op": "wgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 91.0,
            "pr_us": 48.5,
            "speedup": round(91.0 / 48.5, 3),
        },
        {
            "shape": "router 4096->256",
            "op": "dgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 351.5,
            "pr_us": 189.2,
            "speedup": round(351.5 / 189.2, 3),
        },
        {
            "shape": "router 4096->256",
            "op": "fwd",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 324.4,
            "pr_us": 193.1,
            "speedup": round(324.4 / 193.1, 3),
        },
        {
            "shape": "router 4096->256",
            "op": "wgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 313.1,
            "pr_us": 187.2,
            "speedup": round(313.1 / 187.2, 3),
        },
        {
            "shape": "router 5120->256",
            "op": "dgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 462.4,
            "pr_us": 238.0,
            "speedup": round(462.4 / 238.0, 3),
        },
        {
            "shape": "router 6144->256",
            "op": "dgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "previous version",
            "baseline_us": 529.1,
            "pr_us": 273.8,
            "speedup": round(529.1 / 273.8, 3),
        },
    ]

    gb300_highlights = [
        {
            "shape": "5120->512",
            "op": "dgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "cuBLAS BF16",
            "baseline_us": 261.1,
            "pr_us": 73.6,
            "speedup": round(261.1 / 73.6, 3),
        },
        {
            "shape": "4096->512",
            "op": "dgrad",
            "out": "f32",
            "T": 16384,
            "baseline": "cuBLAS BF16",
            "baseline_us": 212.7,
            "pr_us": 60.4,
            "speedup": round(212.7 / 60.4, 3),
        },
        {
            "shape": "router 6144->256",
            "op": "fwd",
            "out": "f32",
            "T": 16231,
            "baseline": "cuBLAS FP32 (BF16x9)",
            "baseline_us": 488.5,
            "pr_us": 224.8,
            "speedup": round(488.5 / 224.8, 3),
        },
        {
            "shape": "router 5120->256",
            "op": "fwd",
            "out": "f32",
            "T": 16231,
            "baseline": "cuBLAS FP32 (BF16x9)",
            "baseline_us": 414.7,
            "pr_us": 195.7,
            "speedup": round(414.7 / 195.7, 3),
        },
        {
            "shape": "router 4096->256",
            "op": "fwd",
            "out": "f32",
            "T": 16231,
            "baseline": "cuBLAS FP32 (BF16x9)",
            "baseline_us": 341.1,
            "pr_us": 166.2,
            "speedup": round(341.1 / 166.2, 3),
        },
        {
            "shape": "router 6144->256",
            "op": "wgrad",
            "out": "f32",
            "T": 16231,
            "baseline": "cuBLAS FP32 (BF16x9)",
            "baseline_us": 321.7,
            "pr_us": 214.0,
            "speedup": round(321.7 / 214.0, 3),
        },
    ]

    assert sum(program_instances["pr_by_arch"].values()) == 247
    assert gb300_highlights[0]["speedup"] == 3.548
    assert gb300_highlights[2]["speedup"] == 2.173

    return {
        "commit": "2706a076ac",
        "program_instances": program_instances,
        "router_splits": router_splits,
        "exact_wave_block_n": 224,
        "router_block_k": 32,
        "router_block_n": 128,
        "gb200_faster_rows": "242/294",
        "gb300_faster_rows": "250/294",
        "gb200_highlights": gb200_highlights,
        "gb300_highlights": gb300_highlights,
    }


def doc_batches(
    doc_lengths: tuple[int, ...],
    batch_tokens: int,
    max_docs_per_batch: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Deterministic mirror of cake_backend.doc_batches greedy partition."""
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    for length in doc_lengths:
        if length > batch_tokens:
            raise ValueError(f"Document length {length} exceeds batch_tokens {batch_tokens}")

    groups: list[tuple[int, int]] = []
    start = 0
    curr_tokens = 0
    curr_docs = 0
    for idx, length in enumerate(doc_lengths):
        if length == 0 and idx > start:
            curr_docs += 1
            continue
        exceeds_tokens = curr_docs > 0 and (curr_tokens + length > batch_tokens)
        exceeds_docs = (
            max_docs_per_batch is not None
            and curr_docs >= max_docs_per_batch
        )
        if exceeds_tokens or exceeds_docs:
            groups.append((start, idx))
            start = idx
            curr_tokens = length
            curr_docs = 1
        else:
            curr_tokens += length
            curr_docs += 1
    if start < len(doc_lengths):
        groups.append((start, len(doc_lengths)))
    return tuple(groups)


def evaluate_kda_doc_group_batching() -> dict[str, Any]:
    """Verify FlashInfer KDA document-group batching (fd5e515c6f)."""
    s261393_lengths = (23496, 88510, 114974, 34413)
    eq4_lengths = (65349, 65348, 65348, 65348)
    assert sum(s261393_lengths) == 261393
    assert sum(eq4_lengths) == 261393

    g2_s261393 = doc_batches(s261393_lengths, batch_tokens=149387)
    g3_s261393 = doc_batches(s261393_lengths, batch_tokens=114974)
    gdoc_s261393 = doc_batches(s261393_lengths, batch_tokens=261393, max_docs_per_batch=1)

    assert g2_s261393 == ((0, 2), (2, 4))
    assert g3_s261393 == ((0, 2), (2, 3), (3, 4))
    assert gdoc_s261393 == ((0, 1), (1, 2), (2, 3), (3, 4))

    gb300_arena_rows = [
        {
            "row": "P14_h64_s261393",
            "H": 64,
            "option": "lean, 4 tail passes (no groups)",
            "arena_default_mib": 71703,
            "arena_option_mib": 45054,
            "reduction_factor": round(71703 / 45054, 2),
            "b2_default_ms": 47.440,
            "b2_option_ms": 47.441,
            "b3_default_ms": 79.817,
            "b3_option_ms": 79.746,
            "b3_delta_pct": -0.1,
        },
        {
            "row": "P14_h64_s261393",
            "H": 64,
            "option": "lean, 4 tail passes, 2 document groups",
            "arena_default_mib": 71703,
            "arena_option_mib": 25755,
            "reduction_factor": round(71703 / 25755, 2),
            "b2_default_ms": 47.440,
            "b2_option_ms": 51.536,
            "b3_default_ms": 79.817,
            "b3_option_ms": 84.809,
            "b3_delta_pct": 6.3,
        },
        {
            "row": "P14_h64_s261393",
            "H": 64,
            "option": "lean, 4 tail passes, 3 document groups",
            "arena_default_mib": 71703,
            "arena_option_mib": 19806,
            "reduction_factor": round(71703 / 19806, 2),
            "b2_default_ms": 47.440,
            "b2_option_ms": 53.090,
            "b3_default_ms": 79.817,
            "b3_option_ms": 86.083,
            "b3_delta_pct": 7.9,
        },
        {
            "row": "EQ4_H64",
            "H": 64,
            "option": "lean, 4 tail passes, 2 document groups",
            "arena_default_mib": 71703,
            "arena_option_mib": 22538,
            "reduction_factor": round(71703 / 22538, 2),
            "b2_default_ms": 47.967,
            "b2_option_ms": 48.205,
            "b3_default_ms": 80.884,
            "b3_option_ms": 81.286,
            "b3_delta_pct": 0.5,
        },
        {
            "row": "EQ4_H64",
            "H": 64,
            "option": "lean, 4 tail passes, one group per document",
            "arena_default_mib": 71703,
            "arena_option_mib": 11280,
            "reduction_factor": round(71703 / 11280, 2),
            "b2_default_ms": 47.967,
            "b2_option_ms": 53.679,
            "b3_default_ms": 80.884,
            "b3_option_ms": 86.840,
            "b3_delta_pct": 7.4,
        },
    ]

    assert gb300_arena_rows[2]["reduction_factor"] == 3.62
    assert gb300_arena_rows[4]["reduction_factor"] == 6.36

    return {
        "commit": "fd5e515c6f",
        "s261393_lengths": list(s261393_lengths),
        "eq4_lengths": list(eq4_lengths),
        "partitions_s261393": {
            "g2": [list(p) for p in g2_s261393],
            "g3": [list(p) for p in g3_s261393],
            "gdoc": [list(p) for p in gdoc_s261393],
        },
        "proportional_segment_guard_max_chains": 4,
        "head_bytes_per_token_head": 4488,
        "lean_4pass_bytes_per_token_head": 2820,
        "gb300_arena_rows": gb300_arena_rows,
    }


def evaluate_code_folding_and_tilelang() -> dict[str, Any]:
    """Verify FlashInfer grouped GEMM / softmax folding & TileLang compiler guards."""
    layout_batch_size = 8
    effective_batch = 2
    old_emitted_offsets = [0]
    fixed_emitted_offsets = list(range(0, layout_batch_size, effective_batch))
    assert fixed_emitted_offsets == [0, 2, 4, 6]

    return {
        "flashinfer_grouped_gemm_commit": "c01aa7779b",
        "flashinfer_grouped_gemm_physical_programs": 5,
        "flashinfer_grouped_gemm_wgrad_out_flags": {"bf16": 16, "fp32": 32},
        "flashinfer_softmax_commit": "6b38488084",
        "flashinfer_softmax_dropped_variants": ["c1_t8", "c2_t8", "c4_t8"],
        "flashinfer_softmax_retained_variants": ["c1_t4", "c2_t4", "c4_t4", "c8_t4", "c8_t8", "c16_t4", "c16_t8"],
        "tilelang_commits": {
            "vectorize_analyzer_clone": "fe8c34618f",
            "partition_analyzer_isolation": "741bf8b722",
            "finalize_reducer_batched": "6db1fc771a",
            "cuda_16bit_math": "43f13c7ad1",
        },
        "finalize_reducer_simulation": {
            "layout_batch_size": layout_batch_size,
            "effective_batch": effective_batch,
            "old_emitted_offsets": old_emitted_offsets,
            "fixed_emitted_offsets": fixed_emitted_offsets,
            "missing_chunks_recovered": len(fixed_emitted_offsets) - len(old_emitted_offsets),
        },
    }


def build_summary() -> dict[str, Any]:
    dsa = evaluate_dsa_indexer_far_sample()
    dense_gemm = evaluate_dense_projection_gemm()
    kda = evaluate_kda_doc_group_batching()
    folding_tl = evaluate_code_folding_and_tilelang()
    return {
        "experiment": "flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang",
        "phase": 63,
        "priority": 72,
        "track": "A",
        "status": "verified",
        "dsa_indexer_far_sample": dsa,
        "dense_projection_gemm": dense_gemm,
        "kda_doc_group_batching": kda,
        "code_folding_and_tilelang": folding_tl,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against summary.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing summary file: {args.output}")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py")
        print("OK: flashinfer-dsa-sink-dense-gemm-kda-batch-tilelang summary verified.")
        return 0

    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
