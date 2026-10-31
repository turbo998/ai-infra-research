from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

def compute_summary() -> dict:
    # 1. PyTorch CUDA vectorized dropout Philox offset reservation (pytorch@fe58682)
    vec_size = 4
    unroll_loops = 4
    buggy_reserved_steps = 1
    fixed_reserved_steps = unroll_loops
    buggy_overlap_fraction = round((unroll_loops - buggy_reserved_steps) / unroll_loops, 4)
    fixed_overlap_fraction = round((unroll_loops - fixed_reserved_steps) / unroll_loops, 4)

    # 2. LMCache GPU paged KV compaction after token dropping (LMCache@fa34a70)
    seq_len = 8192
    page_size = 16
    keep_ratio = 0.40
    orig_pages = math.ceil(seq_len / page_size)
    compacted_pages = math.ceil(math.ceil(seq_len * keep_ratio) / page_size)
    reclaimed_pages_pct = round((orig_pages - compacted_pages) / orig_pages * 100.0, 2)

    # 3. LMCache L1 full-hit short-circuit skipping L2 lookup (LMCache@eed45ed)
    l1_full_hit_rate = 0.75
    l2_lookup_us = 400.0
    l1_check_us = 20.0
    baseline_prefetch_lookup_us = l1_check_us + l2_lookup_us
    shortcircuit_prefetch_lookup_us = l1_check_us + (1.0 - l1_full_hit_rate) * l2_lookup_us
    prefetch_lookup_speedup = round(baseline_prefetch_lookup_us / shortcircuit_prefetch_lookup_us, 2)

    return {
        "experiment": "megatron_kda_pytorch_philox_lmcache_dynamo_phase137",
        "evidence_tier": "own-offline",
        "metrics": {
            "vec_size": vec_size,
            "buggy_overlap_fraction": buggy_overlap_fraction,
            "fixed_overlap_fraction": fixed_overlap_fraction,
            "orig_pages": orig_pages,
            "compacted_pages": compacted_pages,
            "reclaimed_pages_pct": reclaimed_pages_pct,
            "baseline_prefetch_lookup_us": baseline_prefetch_lookup_us,
            "shortcircuit_prefetch_lookup_us": shortcircuit_prefetch_lookup_us,
            "prefetch_lookup_speedup": prefetch_lookup_speedup,
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = compute_summary()
    out = Path(__file__).with_name("summary.json")
    if args.check:
        existing = json.loads(out.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        return
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
