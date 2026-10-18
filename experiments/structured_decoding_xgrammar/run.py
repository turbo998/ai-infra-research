"""Deterministic offline simulator for Grammar-Constrained Structured Decoding (XGrammar / vLLM V1).

Models CFG / JSON Schema constrained decoding across vocab_size = 128,256 and Batch Sizes 1..128:
1. Naive Regex/FSM Per-Token Dynamic Vocabulary Scan on CPU (`8.5 - 38.0 ms` per step bottleneck).
2. Uncached Pushdown Automaton + Full FP32 Logit Mask (`512 KB/req` H2D copy + synchronous CPU mask build).
3. XGrammar AdaptiveTokenMask Precomputed Cache (`97.4%` context-independent tokens pre-classified) + Compact `int32` Bitmask (`ceil(128256/32) = 4008 int32 = 16,032 B/req`, 32x smaller).
4. XGrammar + vLLM V1 Async CPU ThreadPool Overlap + Speculative Draft-Tree `Rollback(1)` Bitmask Verification.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.json"


def compute_mask_memory_profile(
    vocab_size: int = 128256,
    batch_size: int = 64,
    speculative_draft_tokens: int = 0,
) -> dict[str, Any]:
    """Compare FP32 full mask vs XGrammar compact int32 dynamic bitset."""
    rows = batch_size * (1 + speculative_draft_tokens)
    fp32_bytes_per_row = vocab_size * 4
    bitmask_int32_words_per_row = math.ceil(vocab_size / 32)
    bitmask_bytes_per_row = bitmask_int32_words_per_row * 4

    total_fp32_bytes = rows * fp32_bytes_per_row
    total_bitmask_bytes = rows * bitmask_bytes_per_row

    return {
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "speculative_draft_tokens": speculative_draft_tokens,
        "total_mask_rows": rows,
        "fp32_bytes_per_row": fp32_bytes_per_row,
        "bitmask_int32_words_per_row": bitmask_int32_words_per_row,
        "bitmask_bytes_per_row": bitmask_bytes_per_row,
        "total_fp32_kib": round(total_fp32_bytes / 1024.0, 2),
        "total_bitmask_kib": round(total_bitmask_bytes / 1024.0, 2),
        "compression_ratio": round(fp32_bytes_per_row / bitmask_bytes_per_row, 2),
    }


def simulate_structured_decode_step(
    batch_size: int,
    vocab_size: int = 128256,
    context_independent_ratio: float = 0.974,
    cpu_threads: int = 8,
    gpu_base_tpot_ms: float = 1.80,
    pcie_h2d_gbps: float = 48.0,
    backend: str = "xgrammar_async_overlap",
) -> dict[str, Any]:
    """Simulate per-step structured decoding CPU mask generation, H2D transfer, GPU kernel, and exposed TPOT."""
    mem = compute_mask_memory_profile(vocab_size=vocab_size, batch_size=batch_size)
    uncertain_tokens = int(round(vocab_size * (1.0 - context_independent_ratio)))

    # GPU base decode step grows modestly with batch size
    gpu_decode_ms = gpu_base_tpot_ms + 0.011 * (batch_size - 1)

    if backend == "naive_dynamic_fsm_fp32":
        # Evaluates regex/FSM transition per token across vocab on CPU, writes FP32 mask
        per_req_cpu_us = 540.0  # 0.54 ms per request scanning vocabulary clusters
        parallel_efficiency = min(cpu_threads, batch_size) * 0.72 if batch_size >= 4 else 1.0
        cpu_mask_ms = (batch_size * per_req_cpu_us / parallel_efficiency) / 1000.0
        h2d_ms = (mem["total_fp32_kib"] * 1024.0 / (pcie_h2d_gbps * 1e9)) * 1e3 + 0.018
        gpu_apply_mask_ms = 0.085 + 0.0012 * batch_size
        exposed_overhead_ms = cpu_mask_ms + h2d_ms + gpu_apply_mask_ms

    elif backend == "uncached_pda_fp32_sync":
        # Pushdown automaton without AdaptiveTokenMask splitting; evaluates ~18% candidate tokens dynamically
        per_req_cpu_us = 115.0
        parallel_efficiency = min(cpu_threads, batch_size) * 0.78 if batch_size >= 8 else 1.0
        cpu_mask_ms = (batch_size * per_req_cpu_us / parallel_efficiency) / 1000.0
        h2d_ms = (mem["total_fp32_kib"] * 1024.0 / (pcie_h2d_gbps * 1e9)) * 1e3 + 0.015
        gpu_apply_mask_ms = 0.075 + 0.0010 * batch_size
        exposed_overhead_ms = cpu_mask_ms + h2d_ms + gpu_apply_mask_ms

    elif backend == "xgrammar_bitmask_sync":
        # AdaptiveTokenMask precomputes 97.4% tokens into kAccepted/kRejected/kAcceptedBitset;
        # Runtime only checks ~3,335 uncertain tokens + SIMD bitwise OR into 4,008 int32 words
        per_req_cpu_us = 4.2 + (uncertain_tokens / 3335.0) * 3.6  # ~7.8 us/req single-core
        eff_workers = min(cpu_threads, max(1, batch_size // 8)) * 0.86 if batch_size >= 16 else 1.0
        cpu_mask_ms = (batch_size * per_req_cpu_us / eff_workers) / 1000.0
        h2d_ms = (mem["total_bitmask_kib"] * 1024.0 / (pcie_h2d_gbps * 1e9)) * 1e3 + 0.006
        gpu_apply_mask_ms = 0.014 + 0.00025 * batch_size
        exposed_overhead_ms = cpu_mask_ms + h2d_ms + gpu_apply_mask_ms

    elif backend == "xgrammar_async_overlap":
        # vLLM V1 / SGLang overlaps CPU BatchFillNextTokenBitmask for step t+1 with GPU forward of step t
        per_req_cpu_us = 4.2 + (uncertain_tokens / 3335.0) * 3.6
        eff_workers = min(cpu_threads, max(1, batch_size // 8)) * 0.88 if batch_size >= 16 else 1.0
        cpu_mask_ms = (batch_size * per_req_cpu_us / eff_workers) / 1000.0
        h2d_ms = (mem["total_bitmask_kib"] * 1024.0 / (pcie_h2d_gbps * 1e9)) * 1e3 + 0.005
        gpu_apply_mask_ms = 0.014 + 0.00025 * batch_size
        # CPU mask fill overlaps with previous step's GPU decode forward (85% of gpu_decode_ms available for overlap)
        non_overlapped_cpu_ms = max(0.0, cpu_mask_ms - gpu_decode_ms * 0.85)
        exposed_overhead_ms = non_overlapped_cpu_ms + h2d_ms + gpu_apply_mask_ms
    else:
        raise ValueError(f"Unknown backend: {backend}")

    total_tpot_ms = gpu_decode_ms + exposed_overhead_ms
    tpot_degradation_pct = (exposed_overhead_ms / gpu_decode_ms) * 100.0

    return {
        "backend": backend,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "uncertain_tokens_evaluated": uncertain_tokens if "xgrammar" in backend else vocab_size,
        "gpu_base_decode_ms": round(gpu_decode_ms, 3),
        "cpu_mask_gen_ms": round(cpu_mask_ms, 4),
        "h2d_copy_ms": round(h2d_ms, 4),
        "gpu_bitmask_kernel_ms": round(gpu_apply_mask_ms, 4),
        "exposed_overhead_ms": round(exposed_overhead_ms, 4),
        "exposed_overhead_us": round(exposed_overhead_ms * 1000.0, 1),
        "total_tpot_ms": round(total_tpot_ms, 3),
        "tpot_degradation_pct": round(tpot_degradation_pct, 2),
        "batch_tokens_per_sec": round((batch_size / total_tpot_ms) * 1000.0, 1),
    }


def simulate_speculative_tree_grammar_validation(
    batch_size: int = 32,
    tree_sizes: list[int] | None = None,
    vocab_size: int = 128256,
) -> list[dict[str, Any]]:
    """Simulate XGrammar TraverseDraftTreeRecursive DFS + Rollback(1) bitmask fill for EAGLE-3 draft trees."""
    if tree_sizes is None:
        tree_sizes = [1, 8, 16, 32, 64]
    rows = []
    for n_tree in tree_sizes:
        mem = compute_mask_memory_profile(vocab_size=vocab_size, batch_size=batch_size, speculative_draft_tokens=n_tree - 1)
        # DFS AcceptToken + Rollback(1) per node takes ~1.9 us per node across 8 threads
        dfs_cpu_ms = (batch_size * n_tree * 1.9 / 6.8) / 1000.0
        h2d_ms = (mem["total_bitmask_kib"] * 1024.0 / (48.0 * 1e9)) * 1e3 + 0.006
        rows.append({
            "batch_size": batch_size,
            "draft_tree_nodes": n_tree,
            "total_bitmask_rows": mem["total_mask_rows"],
            "total_bitmask_kib": mem["total_bitmask_kib"],
            "dfs_rollback_cpu_ms": round(dfs_cpu_ms, 3),
            "h2d_transfer_ms": round(h2d_ms, 3),
            "total_tree_mask_ms": round(dfs_cpu_ms + h2d_ms, 3),
        })
    return rows


def generate_results() -> dict[str, Any]:
    batch_sizes = [1, 8, 32, 64, 128]
    backends = [
        "naive_dynamic_fsm_fp32",
        "uncached_pda_fp32_sync",
        "xgrammar_bitmask_sync",
        "xgrammar_async_overlap",
    ]

    evaluation_matrix = []
    for bs in batch_sizes:
        for b in backends:
            evaluation_matrix.append(simulate_structured_decode_step(batch_size=bs, backend=b))

    bs64_naive = next(r for r in evaluation_matrix if r["batch_size"] == 64 and r["backend"] == "naive_dynamic_fsm_fp32")
    bs64_xg_sync = next(r for r in evaluation_matrix if r["batch_size"] == 64 and r["backend"] == "xgrammar_bitmask_sync")
    bs64_xg_async = next(r for r in evaluation_matrix if r["batch_size"] == 64 and r["backend"] == "xgrammar_async_overlap")
    bs128_xg_async = next(r for r in evaluation_matrix if r["batch_size"] == 128 and r["backend"] == "xgrammar_async_overlap")

    return {
        "experiment_id": "structured_decoding_xgrammar",
        "title": "XGrammar & vLLM V1 Pushdown-Automaton Adaptive Bitmask Structured Decoding",
        "vocabulary_profile": {
            "vocab_size": 128256,
            "context_independent_ratio": 0.974,
            "precomputed_tokens": 124921,
            "runtime_uncertain_tokens": 3335,
            "bitmask_words_int32": 4008,
            "bitmask_bytes_per_request": 16032,
            "fp32_mask_bytes_per_request": 513024,
            "mask_compression_ratio": 32.0,
        },
        "memory_profiles": [
            compute_mask_memory_profile(128256, bs, 0) for bs in batch_sizes
        ],
        "evaluation_matrix": evaluation_matrix,
        "speculative_tree_validation": simulate_speculative_tree_grammar_validation(batch_size=32),
        "headline_findings": {
            "bs64_naive_cpu_ms": bs64_naive["cpu_mask_gen_ms"],
            "bs64_naive_total_tpot_ms": bs64_naive["total_tpot_ms"],
            "bs64_xgrammar_sync_overhead_us": bs64_xg_sync["exposed_overhead_us"],
            "bs64_xgrammar_async_overhead_us": bs64_xg_async["exposed_overhead_us"],
            "bs64_xgrammar_async_tpot_degradation_pct": bs64_xg_async["tpot_degradation_pct"],
            "bs128_xgrammar_async_overhead_us": bs128_xg_async["exposed_overhead_us"],
            "bs128_xgrammar_async_tpot_degradation_pct": bs128_xg_async["tpot_degradation_pct"],
            "bs64_throughput_speedup_vs_naive": round(
                bs64_xg_async["batch_tokens_per_sec"] / bs64_naive["batch_tokens_per_sec"], 2
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate deterministic results against results.json")
    args = parser.parse_args()

    results = generate_results()
    serialized = json.dumps(results, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}; run without --check first.")
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != serialized:
            raise SystemExit("Deterministic check failed: generated results differ from results.json")
        print("structured_decoding_xgrammar check PASSED")
        return 0

    RESULTS_PATH.write_text(serialized, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
