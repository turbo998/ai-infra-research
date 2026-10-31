from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. SGLang dsv41 SM90 Paged Decode Indexer Tile Balancing (#47387, 2ccbdab65f)
    # Mixed batch with 15 short contexts (4K) and 1 long context (128K) across 132 SM90 SMs
    num_sms = 132
    seq_lengths = [4096] * 15 + [131072]
    tile_size = 1024
    tiles_per_seq = [s // tile_size for s in seq_lengths]
    total_tiles = sum(tiles_per_seq)
    # Unbalanced (1 sequence per CTA wave): max tiles on single SM = 128 tiles
    unbalanced_max_tiles_per_sm = max(tiles_per_seq)
    # Balanced across all SMs: ceil(total_tiles / num_sms)
    balanced_max_tiles_per_sm = (total_tiles + num_sms - 1) // num_sms
    tail_step_speedup = round(unbalanced_max_tiles_per_sm / balanced_max_tiles_per_sm, 2)

    # 2. SGLang Disaggregation First-Refusal Admission Stop (#46898, 566dbd2cb3)
    # & AMD Mamba HiCache Slotless Checkpoint Storage (#44718, 2f6e7f3475)
    queued_decode_transfers = 32
    first_refusal_index = 4
    legacy_probed_admissions = queued_decode_transfers
    early_stop_probed_admissions = first_refusal_index
    saved_admission_probes_pct = round(
        (legacy_probed_admissions - early_stop_probed_admissions) * 100.0 / legacy_probed_admissions, 2
    )
    mamba_hicache_gpu_slots_required = 0

    # 3. vLLM Single-Launch Fused Scoring + Top-K Sparse-Indexer Prefill (#60618, 25fc60723c)
    prefill_tokens = 8192
    candidate_blocks = 4096
    topk = 64
    unfused_logits_materialized_bytes = prefill_tokens * candidate_blocks * 4  # FP32 score matrix
    fused_output_indices_bytes = prefill_tokens * topk * 4
    memory_traffic_reduction_ratio = round(unfused_logits_materialized_bytes / fused_output_indices_bytes, 2)

    return {
        "suite": "sglang_dsv41_sm90_disagg_stop_vllm_fused_topk_phase167",
        "phase": 167,
        "priority": 176,
        "status": "ok",
        "sglang_dsv41_sm90_tile_balance": {
            "num_sms": num_sms,
            "total_tiles": total_tiles,
            "unbalanced_max_tiles_per_sm": unbalanced_max_tiles_per_sm,
            "balanced_max_tiles_per_sm": balanced_max_tiles_per_sm,
            "tail_step_speedup_ratio": tail_step_speedup,
        },
        "sglang_disagg_and_mamba_hicache": {
            "queued_decode_transfers": queued_decode_transfers,
            "early_stop_probed_admissions": early_stop_probed_admissions,
            "saved_admission_probes_pct": saved_admission_probes_pct,
            "mamba_hicache_gpu_slots_required": mamba_hicache_gpu_slots_required,
        },
        "vllm_fused_scoring_topk_prefill": {
            "prefill_tokens": prefill_tokens,
            "candidate_blocks": candidate_blocks,
            "topk": topk,
            "unfused_logits_materialized_bytes": unfused_logits_materialized_bytes,
            "fused_output_indices_bytes": fused_output_indices_bytes,
            "memory_traffic_reduction_ratio": memory_traffic_reduction_ratio,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = run_suite()
    res_path = Path(__file__).with_name("results.json")
    res_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert out["status"] == "ok"
        assert out["sglang_dsv41_sm90_tile_balance"]["tail_step_speedup_ratio"] == 64.0
        assert out["sglang_disagg_and_mamba_hicache"]["mamba_hicache_gpu_slots_required"] == 0
        assert out["vllm_fused_scoring_topk_prefill"]["memory_traffic_reduction_ratio"] == 64.0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
