from __future__ import annotations
import json
from pathlib import Path


def evaluate_pp_schedules_memory_and_bubble() -> dict:
    # Workload: 80-layer LLM across PP=8 stages, M=32 microbatches, 4.0 GB activation per full-stage microbatch
    tiers = [
        {
            "tier_id": "tier1_standard_1f1b_8_to_1_stage0_memory_imbalance",
            "name": "Standard 1F1B (Stage 0 Holds 8 Microbatches = 32 GB, Stage 7 Holds 1 = 4 GB)",
            "stage0_peak_activation_gb": 32.0,
            "pipeline_bubble_pct": 18.0,
            "relative_p2p_comm_volume_x": 1.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Severe 8:1 memory imbalance across stages (Stage 0 hits 32.0 GB while Stage 7 uses 4.0 GB) plus an 18.0% pipeline bubble!"
        },
        {
            "tier_id": "tier2_interleaved_1f1b_vpp4_p2p_and_memory_inflation",
            "name": "Interleaved 1F1B (VPP = 4 Virtual Stages, 4x P2P Volume & +25% Stage-0 Memory)",
            "stage0_peak_activation_gb": 39.0,
            "pipeline_bubble_pct": 5.2,
            "relative_p2p_comm_volume_x": 4.0,
            "action": "Send to a human",
            "status": "warn",
            "note": "Cuts bubble by 4x to 5.2%, but multiplies cross-stage P2P traffic by 4.0x and raises Stage-0 peak memory to 39.0 GB -> Stage 0 OOM!"
        },
        {
            "tier_id": "tier3_v_shape_folded_1f1b_uniform_memory",
            "name": "V-Shape Folded 1F1B (2 Chunks/Stage Co-Locating Chunk s and 2P-1-s)",
            "stage0_peak_activation_gb": 18.0,
            "pipeline_bubble_pct": 9.4,
            "relative_p2p_comm_volume_x": 2.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Folds the pipeline into a V-shape so every stage holds exactly 9 half-chunks (18.0 GB uniform across all 8 GPUs) and Emb/LM-Head co-locate on Stage 0!"
        },
        {
            "tier_id": "tier4_zb_v_dualpipe_bidirectional_split_bw_uniform_memory",
            "name": "ZB-V / DualPipe Bidirectional V-Schedule + Split Backward (B=dX, W=dW)",
            "stage0_peak_activation_gb": 16.0,
            "pipeline_bubble_pct": 1.8,
            "relative_p2p_comm_volume_x": 2.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Balances activation memory uniformly at 16.0 GB across all 8 stages (-59% vs VPP=4), co-locates Emb & LM Head on Stage 0, and cuts bubble to 1.8%!"
        },
    ]
    return {
        "experiment_id": "llm_megatron_pp_virtual_pipeline_interleaved_1f1b_v_schedule_selection",
        "round": 290,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "pp_stages": 8,
            "microbatches_m": 32,
            "activation_per_stage_mb_gb": 4.0
        },
        "tiers": tiers,
        "summary": {
            "stage0_memory_reduction_vs_vpp4_pct": round((39.0 - 16.0) / 39.0 * 100.0, 1),
            "bubble_reduction_x": round(18.0 / 1.8, 1),
            "recommended_stage0_memory_gb": 16.0,
            "recommended_tier": "tier4_zb_v_dualpipe_bidirectional_split_bw_uniform_memory"
        }
    }


if __name__ == "__main__":
    data = evaluate_pp_schedules_memory_and_bubble()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
