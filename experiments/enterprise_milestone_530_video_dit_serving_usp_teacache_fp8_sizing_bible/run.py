from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "single_gpu_bf16_50step_video_dit",
            "name": "Anti-Pattern: Single-GPU BF16 50-Step Uncached Video DiT (720p 5s Clip)",
            "clip_generation_seconds": 340.0,
            "vbench_visual_quality_score": 84.2,
            "clips_per_hour_per_8gpu_node": 84.7,
            "cost_per_video_clip_usd": 0.293,
            "action": "Send to a human",
        },
        {
            "arch_id": "usp8_sequence_parallel_bf16",
            "name": "Baseline: 8-GPU xDiT Unified Sequence Parallelism (Ulysses=4 × Ring=2, BF16 50 Steps)",
            "clip_generation_seconds": 46.5,
            "vbench_visual_quality_score": 84.2,
            "clips_per_hour_per_8gpu_node": 77.4,
            "cost_per_video_clip_usd": 0.320,
            "action": "Confirm",
        },
        {
            "arch_id": "usp8_teacache_sageattention_fp8_cfg_parallel",
            "name": "Optimal: USP=8 (NVLink) + TeaCache (2.3x Step Skip) + SageAttention2 FP8",
            "clip_generation_seconds": 11.2,
            "vbench_visual_quality_score": 83.9,
            "clips_per_hour_per_8gpu_node": 321.4,
            "cost_per_video_clip_usd": 0.077,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_milestone_530_video_dit_serving_usp_teacache_fp8_sizing_bible",
        "title": "Milestone #530: Enterprise Video DiT Serving (Wan2.1 / HunyuanVideo) — xDiT USP Sequence Parallelism + TeaCache + SageAttention2 FP8 Bible",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "latency_speedup_factor_vs_single_gpu": round(340.0 / 11.2, 2),
            "node_throughput_gain_factor": round(321.4 / 84.7, 2),
            "cost_reduction_pct_per_clip": round((1.0 - 0.077 / 0.293) * 100, 1),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
