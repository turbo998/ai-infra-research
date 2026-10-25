"""Reproducible stdlib model for Phase 75 Track B inference engine mechanics.

Covers:
1. TensorRT-LLM (#20213, 7d391a06d7fb; #20663, abb94ab8fc1c): Shared `MultimodalEncoderMixin`
   CUDA graph capture (`EncoderGraphIOAdapter`) for Qwen3-VL / RADIO (`vision`) / GLM5-Next (`image`)
   with isolated 1-token dummy padding context (`total_tokens + 1`, `num_contexts + 1`), plus
   Gemma4 multimodal encoder item budgeting (`_padded_patches_per_frame = max_soft_tokens * pooling_kernel_size^2`,
   `_AUDIO_SUBSAMPLING_FACTOR = 4`).
2. vLLM (#63451, b3466364b69d; #61019, fc0cfcdc05b1; #60084, a55937f25528; #60810, 921ed7fa11c4):
   `WeightCacheDaemon.release_distributed()` calling `torch._C._cuda_clearCublasWorkspaces()` after
   warmup; `OffloadingConnector` partial-tail serving/storing with EAGLE/MTP proof margin
   (`boundary + cache_hit_alignment_tokens` vs `boundary + block_size`); ROCm `set_triton_ptr_range`
   bounding layer KV cache views to `< 2 GiB` for 32-bit `buffer_load`/`buffer_store`.
3. SGLang (#46139, 174bd2976d34; #46140, a99c7d050fae; #45104, c00e7e12b64b; #39729, 5d11819f3a7a):
   `MSA-v1` decode & NVFP4 KV cache kernels (`kNvfp4BlockSize = 16`, `rcp.approx.ftz.f32` matching
   `flashinfer::nvfp4_kv_quantize` vs IEEE division `~0.1%` nibble drift); `MiniMaxM3Nvfp4Config`
   (`weight_block_size=[1, 32]` MXFP8 projections converted to `[128, 128]` block-FP8 off-Blackwell,
   `group_size=16` NVFP4 experts); `hisparse` token-by-token DeepSeek-V4 backup copy; disaggregated
   KV checksum disabled on decode radix-cache hits and layout-signature gated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_trtllm_mm_encoder_graph_and_gemma4() -> dict[str, Any]:
    """Model TensorRT-LLM `EncoderGraphIOAdapter` (#20213) and Gemma4 MM item budgeting (#20663)."""
    buckets = [
        {"real_tokens": 256, "real_contexts": 1},
        {"real_tokens": 1024, "real_contexts": 4},
        {"real_tokens": 4096, "real_contexts": 16},
    ]
    captured_keys = [
        {
            "real_tokens": b["real_tokens"],
            "real_contexts": b["real_contexts"],
            "captured_total_tokens": b["real_tokens"] + 1,
            "captured_num_contexts": b["real_contexts"] + 1,
            "dummy_padding_tokens": 1,
            "isolated_dummy_context": True,
        }
        for b in buckets
    ]
    max_soft_tokens = 256
    pooling_kernel_size = 3
    padded_patches = max_soft_tokens * (pooling_kernel_size**2)
    audio_mel_frames = 3000
    audio_subsampling_factor = 4
    subsampled_audio_tokens = audio_mel_frames // audio_subsampling_factor
    return {
        "supported_encoders": ["Qwen3-VL", "RADIO", "GLM5-Next"],
        "dummy_padding_token_reserve": 1,
        "dummy_padding_context_reserve": 1,
        "captured_graph_keys": captured_keys,
        "gemma4_budgeting": {
            "max_soft_tokens": max_soft_tokens,
            "pooling_kernel_size": pooling_kernel_size,
            "padded_patches_per_frame": padded_patches,
            "audio_subsampling_factor": audio_subsampling_factor,
            "sample_audio_mel_frames": audio_mel_frames,
            "subsampled_audio_tokens": subsampled_audio_tokens,
        },
    }


def model_vllm_cublas_tail_and_rocm_ptr_range() -> dict[str, Any]:
    """Model vLLM daemon cuBLAS workspace release (#63451), EAGLE partial-tail (#61019), and ROCm 32-bit range (#60084)."""
    mamba_boundary = 24
    block_size = 16
    eagle_cases = []
    for alignment in (4, 16, None):
        margin = alignment if alignment is not None else block_size
        eagle_cases.append(
            {
                "cache_hit_alignment_tokens": alignment,
                "mamba_checkpoint_tokens": mamba_boundary,
                "attention_proof_tokens": mamba_boundary + margin,
            }
        )

    num_layers = 64
    layer_kv_bytes = 512 * 1024 * 1024  # 512 MiB per layer
    total_storage_bytes = num_layers * layer_kv_bytes  # 32 GiB shared allocation
    rocm_32bit_limit_bytes = (1 << 31) - 1
    return {
        "daemon_workspace_cleanup": {
            "clears_process_groups": True,
            "clears_cublas_workspaces": True,
            "api_symbol": "torch._C._cuda_clearCublasWorkspaces",
        },
        "eagle_partial_tail_proof_cases": eagle_cases,
        "rocm_triton_ptr_range": {
            "num_layers": num_layers,
            "layer_kv_bytes": layer_kv_bytes,
            "total_storage_bytes": total_storage_bytes,
            "rocm_32bit_limit_bytes": rocm_32bit_limit_bytes,
            "unbounded_view_uses_32bit_buffer_ops": total_storage_bytes <= rocm_32bit_limit_bytes,
            "bounded_layer_view_uses_32bit_buffer_ops": layer_kv_bytes <= rocm_32bit_limit_bytes,
        },
    }


def model_sglang_msa_nvfp4_and_disagg_checksum() -> dict[str, Any]:
    """Model SGLang MSA-v1 NVFP4 KV kernels (#46139), MiniMax-M3.1 NVFP4 (#46140), and disagg checksum (#39729)."""
    tokens = 4096
    head_dim = 128
    bf16_bytes = tokens * head_dim * 2
    nvfp4_block_size = 16
    nvfp4_payload_bytes = (tokens * head_dim) // 2
    nvfp4_scale_bytes = (tokens * head_dim) // nvfp4_block_size
    nvfp4_total_bytes = nvfp4_payload_bytes + nvfp4_scale_bytes
    compression_ratio = round(bf16_bytes / nvfp4_total_bytes, 4)

    return {
        "msa_v1_nvfp4_kv": {
            "block_size": nvfp4_block_size,
            "packed_block_bytes": nvfp4_block_size // 2,
            "scale_dtype": "UE4M3",
            "reciprocal_instruction": "rcp.approx.ftz.f32",
            "ieee_div_nibble_drift_pct": 0.1,
            "bf16_bytes_for_4k_tokens": bf16_bytes,
            "nvfp4_total_bytes_for_4k_tokens": nvfp4_total_bytes,
            "kv_compression_ratio_vs_bf16": compression_ratio,
        },
        "minimax_m3_1_nvfp4": {
            "projection_mxfp8_block_size": [1, 32],
            "non_blackwell_fallback_block_fp8_size": [128, 128],
            "expert_nvfp4_group_size": 16,
            "index_k_quant_type": "nvfp4",
            "kv_cache_quant_algo": "FP8",
        },
        "disagg_kv_checksum": {
            "disabled_when_decode_radix_cache_enabled": True,
            "requires_matching_kv_layout_signature": True,
            "per_rank_mismatch_rng_seeded_with_tp_rank": True,
        },
    }


def build_summary() -> dict[str, Any]:
    trtllm = model_trtllm_mm_encoder_graph_and_gemma4()
    vllm = model_vllm_cublas_tail_and_rocm_ptr_range()
    sglang = model_sglang_msa_nvfp4_and_disagg_checksum()
    return {
        "experiment_id": "trtllm_qwen3vl_graph_gemma4_vllm_cublas_tail_sglang_msa_nvfp4",
        "phase": 75,
        "priority": 84,
        "track": "Track B — Inference Engine",
        "trtllm_mm_encoder_graph_and_gemma4": trtllm,
        "vllm_cublas_tail_and_rocm_ptr_range": vllm,
        "sglang_msa_nvfp4_and_disagg_checksum": sglang,
        "invariants": {
            "dummy_padding_adds_one_token_and_context": (
                trtllm["captured_graph_keys"][0]["captured_total_tokens"] == 257
                and trtllm["captured_graph_keys"][0]["captured_num_contexts"] == 2
            ),
            "gemma4_padded_patches_2304": trtllm["gemma4_budgeting"]["padded_patches_per_frame"] == 2304,
            "eagle_proof_tokens_match_alignment": [
                c["attention_proof_tokens"] for c in vllm["eagle_partial_tail_proof_cases"]
            ]
            == [28, 40, 40],
            "rocm_ptr_range_restores_32bit_buffer_addressing": (
                not vllm["rocm_triton_ptr_range"]["unbounded_view_uses_32bit_buffer_ops"]
                and vllm["rocm_triton_ptr_range"]["bounded_layer_view_uses_32bit_buffer_ops"]
            ),
            "sglang_nvfp4_kv_compression_ratio_3_5556x": (
                sglang["msa_v1_nvfp4_kv"]["kv_compression_ratio_vs_bf16"] == 3.5556
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output path for summary.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the existing summary.json matches build_summary()",
    )
    args = parser.parse_args()
    summary = build_summary()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of date with build_summary()")
        return 0
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
