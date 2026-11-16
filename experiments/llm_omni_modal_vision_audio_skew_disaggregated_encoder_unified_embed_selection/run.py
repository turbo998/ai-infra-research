from __future__ import annotations
import json
from pathlib import Path


def evaluate_omni_modal_disaggregated_encoders() -> dict:
    # Workload: 72B Omni-Modal LLM (ViT-2B + Whisper-1.5B + 70B LLM) serving mixed 60% text, 25% image, 15% audio traffic
    tiers = [
        {
            "tier_id": "tier1_monolithic_inline_vit_audio_in_tp8_llm_step",
            "name": "Monolithic Inline Execution (`ViT` + `Audio Encoder` Inside `TP=8` LLM Forward Step)",
            "text_request_p99_ttft_ms": 890.0,
            "cluster_throughput_req_per_sec": 14.2,
            "encoder_tp8_sync_bubble_pct": 62.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Runs small ViT/Audio encoders across TP=8 (tiny per-GPU tiles + heavy All-Reduce) inline before LLM prefill, blocking text requests for 890ms!"
        },
        {
            "tier_id": "tier2_colocated_data_parallel_encoders_blocking_llm_step",
            "name": "Colocated `DP=8` Encoders on Same GPUs (Rank-Skewed Image/Audio Counts Stall `TP=8` Barrier)",
            "text_request_p99_ttft_ms": 465.0,
            "cluster_throughput_req_per_sec": 21.8,
            "encoder_tp8_sync_bubble_pct": 44.5,
            "action": "Send to a human",
            "status": "fail",
            "note": "Rank 0 gets 4 high-res images while Ranks 1..7 get text-only -> Ranks 1..7 sit idle at the first LLM TP=8 All-Reduce barrier!"
        },
        {
            "tier_id": "tier3_separate_modality_queues_without_encoder_disaggregation",
            "name": "Separate Modality Batching Queues on Monolithic Replicas (Fragmented KV Pool)",
            "text_request_p99_ttft_ms": 95.0,
            "cluster_throughput_req_per_sec": 28.5,
            "encoder_tp8_sync_bubble_pct": 21.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Isolates text-only batches from image/audio batches, but duplicates ViT+Audio weights across all LLM GPUs and suffers burst starvation."
        },
        {
            "tier_id": "tier4_disaggregated_tp1_vision_audio_encoder_pool_plus_unified_embed_rdma",
            "name": "Disaggregated `TP=1` Vision/Audio Encoder Micro-Pool + Unified `FP8` Embedding Slot Injection (`EPD`)",
            "text_request_p99_ttft_ms": 52.0,
            "cluster_throughput_req_per_sec": 40.3,
            "encoder_tp8_sync_bubble_pct": 1.8,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Runs ViT/Audio on dedicated TP=1 GPUs, streams compact projector embeddings into LLM `inputs_embeds` -> 17.1x lower text P99 TTFT & 2.84x throughput!"
        },
    ]
    return {
        "experiment_id": "llm_omni_modal_vision_audio_skew_disaggregated_encoder_unified_embed_selection",
        "round": 309,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "model": "72B Omni-Modal (ViT-2B + Audio-1.5B + 70B LLM)",
            "traffic_mix": "60% Text, 25% High-Res Image, 15% Long Audio"
        },
        "tiers": tiers,
        "summary": {
            "text_p99_ttft_speedup_x": round(890.0 / 52.0, 2),
            "throughput_speedup_x": round(40.3 / 14.2, 2),
            "recommended_bubble_pct": 1.8,
            "recommended_tier": "tier4_disaggregated_tp1_vision_audio_encoder_pool_plus_unified_embed_rdma"
        }
    }


if __name__ == "__main__":
    data = evaluate_omni_modal_disaggregated_encoders()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
