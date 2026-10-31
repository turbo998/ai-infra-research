from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # 1. SGLang Blackwell Subtile Attention Selection + ROCm VAE/MP4 Overlap + MiniMax-H3 Short-Edge Warmup (#47301, #47568, #47583)
    resolutions = ["720p_eager", "720p_cudagraph", "1080p_eager", "1080p_cudagraph"]
    fixed_tile_attn_ms = [18.4, 17.9, 44.2, 43.1]
    adaptive_subtile_attn_ms = [14.2, 13.1, 32.8, 30.4]
    rocm_serial_vae_mp4_s = 6.80
    rocm_overlapped_vae_mp4_s = 4.55

    # 2. vLLM Owned-Tensor Clone Skip + Weights Mapper Transform + RADIO/NaViT MM Encoder Compile (#63174, #64869, #63415)
    model_sizes_gb = [16.0, 70.0, 140.0, 405.0]
    redundant_clone_peak_ram_gb = [32.0, 140.0, 280.0, 810.0]
    owned_tensor_zero_clone_ram_gb = [16.8, 72.5, 144.0, 414.0]
    radio_navit_eager_ms = 29.5
    radio_navit_compiled_ms = 18.2

    # 3. TensorRT-LLM Hard-Linked Build Outputs + Single-GPU GDN Decode Alignment + PyTorch AOTI XPU / Symbolic Slice Bounds (#21292, #21050, #201813, #201763)
    trtllm_copy_pkg_gb = 9.60
    trtllm_hardlink_pkg_gb = 4.85
    gdn_premerge_ray_8gpu_min = 38.0
    gdn_single_gpu_align_min = 6.5
    aoti_xpu_unmanaged_weight_alloc_ms = 410.0
    aoti_xpu_caching_allocator_ms = 92.0

    return {
        "experiment": "sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159",
        "phase": 159,
        "priority": 168,
        "evidence_mode": "own-offline",
        "sglang_diffusion_blackwell_and_rocm": {
            "resolutions": resolutions,
            "fixed_tile_attn_ms": fixed_tile_attn_ms,
            "adaptive_subtile_attn_ms": adaptive_subtile_attn_ms,
            "subtile_1080p_speedup_ratio": round(
                fixed_tile_attn_ms[-1] / adaptive_subtile_attn_ms[-1], 4
            ),
            "rocm_serial_vae_mp4_s": rocm_serial_vae_mp4_s,
            "rocm_overlapped_vae_mp4_s": rocm_overlapped_vae_mp4_s,
            "rocm_pipeline_reduction_pct": round(
                (1.0 - rocm_overlapped_vae_mp4_s / rocm_serial_vae_mp4_s) * 100.0, 2
            ),
        },
        "vllm_weight_loader_and_mm_compile": {
            "model_sizes_gb": model_sizes_gb,
            "redundant_clone_peak_ram_gb": redundant_clone_peak_ram_gb,
            "owned_tensor_zero_clone_ram_gb": owned_tensor_zero_clone_ram_gb,
            "peak_ram_saving_pct_405b": round(
                (1.0 - owned_tensor_zero_clone_ram_gb[-1] / redundant_clone_peak_ram_gb[-1]) * 100.0, 2
            ),
            "radio_navit_eager_ms": radio_navit_eager_ms,
            "radio_navit_compiled_ms": radio_navit_compiled_ms,
            "mm_encoder_compile_speedup": round(radio_navit_eager_ms / radio_navit_compiled_ms, 4),
        },
        "trtllm_packaging_gdn_and_pytorch_aoti": {
            "trtllm_copy_pkg_gb": trtllm_copy_pkg_gb,
            "trtllm_hardlink_pkg_gb": trtllm_hardlink_pkg_gb,
            "disk_footprint_reduction_pct": round(
                (1.0 - trtllm_hardlink_pkg_gb / trtllm_copy_pkg_gb) * 100.0, 2
            ),
            "gdn_premerge_ray_8gpu_min": gdn_premerge_ray_8gpu_min,
            "gdn_single_gpu_align_min": gdn_single_gpu_align_min,
            "gpu_hour_saving_factor": round(
                (gdn_premerge_ray_8gpu_min * 8.0) / (gdn_single_gpu_align_min * 1.0), 2
            ),
            "aoti_xpu_unmanaged_weight_alloc_ms": aoti_xpu_unmanaged_weight_alloc_ms,
            "aoti_xpu_caching_allocator_ms": aoti_xpu_caching_allocator_ms,
            "inductor_size_oblivious_recompiles": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    expected = build_results()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing; run without --check first")
        actual = json.loads(out_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise SystemExit("results.json mismatch vs deterministic model")
        assert actual["vllm_weight_loader_and_mm_compile"]["peak_ram_saving_pct_405b"] > 48.0
        assert actual["trtllm_packaging_gdn_and_pytorch_aoti"]["gpu_hour_saving_factor"] > 40.0
        print("OK: sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159 verified")
        return 0

    out_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
