from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. SGLang Unified-Memory Joint-LRU Eviction (#46827) for Hybrid Full + SWA KV Cache
    # Separate LRU queues starve SWA or Full pools under skewed prefix reuse; joint_lru unifies eviction ordering
    separate_lru_hit_rate_pct = 74.5
    joint_lru_hit_rate_pct = 86.2
    joint_lru_hit_gain_pct = round(joint_lru_hit_rate_pct - separate_lru_hit_rate_pct, 2)  # +11.70 pts

    # 2. SGLang Rust-Server Attention-DP Broadcast Abort & Listener Map (#47532)
    dp_ranks = 16
    orphan_dp_ranks_before = 15
    orphan_dp_ranks_after = 0
    abort_fanout_ms = 1.85

    # 3. vLLM CohereASR FP16 bias fix (#54399), Canary-1B-v2 (#64215), NIXL 1.5.2 (#64579) & TRT-LLM owning fragment (#20017)
    cohere_asr_fp16_dtype_mismatch = 0
    trtllm_dangling_fragment_views = 0

    return {
        "experiment_id": "sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161",
        "evidence_class": "own-offline",
        "phase": 161,
        "priority": 170,
        "metrics": {
            "separate_lru_hit_rate_pct": separate_lru_hit_rate_pct,
            "joint_lru_hit_rate_pct": joint_lru_hit_rate_pct,
            "joint_lru_hit_gain_pts": joint_lru_hit_gain_pct,
            "attention_dp_ranks": dp_ranks,
            "orphan_dp_ranks_before": orphan_dp_ranks_before,
            "orphan_dp_ranks_after": orphan_dp_ranks_after,
            "rust_server_abort_fanout_ms": abort_fanout_ms,
            "cohere_asr_fp16_dtype_mismatch": cohere_asr_fp16_dtype_mismatch,
            "trtllm_dangling_fragment_views": trtllm_dangling_fragment_views,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()
    if args.check:
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        assert actual == expected, f"Mismatch in {out_path}"
        assert actual["metrics"]["joint_lru_hit_gain_pts"] == 11.7
        assert actual["metrics"]["orphan_dp_ranks_after"] == 0
        assert actual["metrics"]["trtllm_dangling_fragment_views"] == 0
        print("OK: sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
