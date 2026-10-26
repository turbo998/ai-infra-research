#!/usr/bin/env python3
"""Phase 89 Track B offline model:
SGLang CuTe-DSL Multi-Chunk Norm, One-Pass Spatial-Parallel Causal Conv3D & GLM Streaming Parser,
vLLM Persistent TopK Exact Overflow Recovery, ModelOpt NVFP4 Dummy Scales, ROCm AITER W4A16 EP,
MRV2 Warm-Up Sampler Profiling & H200 FP8 MoE Config, and TensorRT-LLM Native UPB Protobuf / Canonical CuTeDSL MoE IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary() -> dict:
    # 1. SGLang diffusion CuTe-DSL multi-chunk norm (#46227), one-pass spatial-parallel causal conv3d (#46174),
    # and GLM streaming tool-call tail scanner (#45143)
    chunk_cols = 32 * 8  # 256 columns per 128-bit load
    hidden_dim = 5120
    num_chunks = hidden_dim // chunk_cols  # 20 chunks per row
    glm_chunks = 1000
    rescan_char_comparisons = sum(3 * (i + 1) for i in range(glm_chunks))
    tail_scan_char_comparisons = 3 * glm_chunks

    sglang_diffusion_and_parser = {
        "cutedsl_norm_chunk_cols": chunk_cols,
        "cutedsl_norm_hidden_dim": hidden_dim,
        "cutedsl_norm_chunks_per_row": num_chunks,
        "cutedsl_norm_smem_acc_slots": num_chunks + 1,
        "cutedsl_norm_bitwise_invariant_across_chunks_per_warp": True,
        "spatial_conv3d_fused_halo_cat_pad_passes_before": 2,
        "spatial_conv3d_fused_halo_cat_pad_passes_after": 1,
        "glm_streaming_rescan_work_units": rescan_char_comparisons,
        "glm_streaming_tail_only_work_units": tail_scan_char_comparisons,
        "glm_streaming_scan_reduction_factor": round(rescan_char_comparisons / tail_scan_char_comparisons, 1),
    }

    # 2. vLLM Persistent TopK exact overflow recovery (#62936), ModelOpt NVFP4 dummy scales (#63698),
    # ROCm AITER W4A16 naive EP (#59255), MRV2 warm-up sampler profiling (#57926), and H200 FP8 MoE (#62763)
    medium_hist_bytes = 2 * (256 + 128) * 4  # 3072 bytes
    medium_scalars_bytes = 5 * 4             # 20 bytes
    exact_candidate_offset_bytes = 16 * 1024

    vllm_topk_and_moe = {
        "persistent_topk_radix": 256,
        "persistent_topk_medium_hist_bytes": medium_hist_bytes,
        "persistent_topk_medium_scalars_bytes": medium_scalars_bytes,
        "persistent_topk_exact_candidate_offset_bytes": exact_candidate_offset_bytes,
        "persistent_topk_vec4_requires_16b_base_alignment": True,
        "nvfp4_dummy_scales": {
            "w13_weight_scale": 1.0,
            "w2_weight_scale": 1.0,
            "w13_weight_scale_2": 1e-3,
            "w2_weight_scale_2": 1e-3,
            "w13_input_scale": 1e-2,
            "w2_input_scale": 1e-2,
            "gate_up_global_scales_matched": True,
        },
        "rocm_aiter_w4a16_naive_ep_gate_valid_mask": True,
        "mrv2_warmup_profiles_both_default_and_active_sampling_params": True,
        "h200_fp8_moe_tuned_config": {
            "num_experts": 256,
            "shard_intermediate_size_n": 320,
            "small_batch_block_m": 16,
            "small_batch_block_n": 64,
            "small_batch_block_k": 128,
        },
    }

    # 3. TensorRT-LLM native protobuf UPB (#19760) & canonical CuteDSL MoE identities (#20929)
    trtllm_grpc_and_moe = {
        "default_protobuf_implementation": "upb",
        "accepted_native_protobuf_implementations": ["upb", "cpp"],
        "warns_on_python_protobuf": True,
        "canonical_cutedsl_id_pattern": "trtllm.cutedsl.<kernel>.<quant>",
        "dense_gemm_nvfp4_class": "TrtllmCutedslDenseGemmNvfp4Impl",
    }

    return {
        "experiment_id": "sglang_diffusion_vllm_topk_nvfp4_kv_offload_trtllm_grpc_phase89",
        "phase": 89,
        "priority": 98,
        "track": "B",
        "sglang_diffusion_and_parser": sglang_diffusion_and_parser,
        "vllm_topk_and_moe": vllm_topk_and_moe,
        "trtllm_grpc_and_moe": trtllm_grpc_and_moe,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(f"Drift detected in {out_path}")
        print("verified")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
