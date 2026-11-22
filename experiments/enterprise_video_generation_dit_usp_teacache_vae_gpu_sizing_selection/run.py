from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    dit_stacks = [
        {
            "stack": "Tier 1: Single-GPU BF16 Sequential Denoising (50 Steps × 2 CFG Passes + Full-Frame 3D VAE)",
            "video_clip_generation_latency_sec": 384.0,
            "peak_vram_per_gpu_gb": 79.4,
            "cost_per_5sec_720p_video_usd": 0.42,
            "videos_per_hour_per_8gpu_node": 75,
        },
        {
            "stack": "Tier 2: 8-GPU Tensor Parallelism (TP=8 AllReduce per DiT Block, BF16)",
            "video_clip_generation_latency_sec": 74.5,
            "peak_vram_per_gpu_gb": 34.0,
            "cost_per_5sec_720p_video_usd": 0.65,
            "videos_per_hour_per_8gpu_node": 48,
        },
        {
            "stack": "Tier 3: 8-GPU xDiT (CFG=2 × Ulysses=4 Sequence Parallelism + Tiled 3D VAE, BF16)",
            "video_clip_generation_latency_sec": 49.2,
            "peak_vram_per_gpu_gb": 28.5,
            "cost_per_5sec_720p_video_usd": 0.43,
            "videos_per_hour_per_8gpu_node": 73,
        },
        {
            "stack": "Tier 4: xDiT (CFG=2 × Ulysses=4) + FP8 Attention/GEMM + TeaCache (2.1x Step Reuse) + Parallel Tiled VAE",
            "video_clip_generation_latency_sec": 14.8,
            "peak_vram_per_gpu_gb": 19.2,
            "cost_per_5sec_720p_video_usd": 0.13,
            "videos_per_hour_per_8gpu_node": 243,
        },
    ]
    return {
        "experiment_id": "enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "model_reference": "14B Diffusion Transformer (Wan2.1 / HunyuanVideo Class, 720p 81 Frames)",
        "spatiotemporal_latent_tokens": 32400,
        "dit_stacks": dit_stacks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
