from __future__ import annotations

import json
from pathlib import Path


def evaluate_sdpa_fallback_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unaligned Explicit attn_mask[B, 1, S, S] (Silent SDPA Math Fallback)",
            "attention_latency_16k_ms": 52.40,
            "attention_workspace_hbm_gb": 32.00,
            "silent_fallback_risk": True,
            "fits_80gb_h100_at_16k": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: 16-Byte Padded Explicit attn_mask (Cutlass Memory-Efficient Backend)",
            "attention_latency_16k_ms": 11.80,
            "attention_workspace_hbm_gb": 1.07,
            "silent_fallback_risk": True,
            "fits_80gb_h100_at_16k": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 3: FlexAttention BlockMask (128x128 Sparse Tile Mask for Custom Masks)",
            "attention_latency_16k_ms": 4.10,
            "attention_workspace_hbm_gb": 0.26,
            "silent_fallback_risk": False,
            "fits_80gb_h100_at_16k": True,
            "decision": "Act automatically",
        },
        {
            "tier": "Tier 4: Strict sdpa_kernel(FLASH_ATTENTION) + is_causal=True & Varlen cu_seqlens",
            "attention_latency_16k_ms": 3.80,
            "attention_workspace_hbm_gb": 0.25,
            "silent_fallback_risk": False,
            "fits_80gb_h100_at_16k": True,
            "decision": "Act automatically",
        },
    ]

    fallback_triggers = [
        {
            "trigger": "Passing explicit `attn_mask` tensor instead of `is_causal=True`",
            "rejected_backend": "FLASH_ATTENTION",
            "fallback_penalty": "Drops to Cutlass (3.1x slower) or Math (13.8x slower + 32GB OOM)",
        },
        {
            "trigger": "Unaligned sequence length `S = 4099` (`stride * elem_size % 16 != 0`) on `attn_mask`",
            "rejected_backend": "FLASH_ATTENTION & EFFICIENT_ATTENTION",
            "fallback_penalty": "Drops to C++ MATH backend (allocates full [B, H, S, S] FP32 matrix!)",
        },
        {
            "trigger": "RoPE / QK-Norm outputs `q` in `float32` while `v` is `bfloat16`",
            "rejected_backend": "FLASH_ATTENTION (fp16/bf16 only)",
            "fallback_penalty": "Upcasts or falls back to slow FP32 path without warning",
        },
        {
            "trigger": "Expanded broadcast view `attn_mask.expand(B, H, S, S)` with `stride == 0`",
            "rejected_backend": "EFFICIENT_ATTENTION Backward",
            "fallback_penalty": "Crashes or falls back to O(S^2) Math backward",
        },
    ]

    summary = {
        "benchmark": "pytorch_sdpa_silent_math_fallback_attn_mask_stride_alignment_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "batch_size": 2,
        "num_heads": 32,
        "seq_len": 16384,
        "math_fallback_latency_ms": 52.40,
        "flash_strict_latency_ms": 3.80,
        "latency_speedup_x": round(52.40 / 3.80, 1),
        "math_fallback_hbm_gb": 32.00,
        "flash_strict_hbm_gb": 0.25,
        "hbm_reduction_x": round(32.00 / 0.25, 1),
        "tiers": tiers,
        "fallback_triggers": fallback_triggers,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_sdpa_fallback_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
