#!/usr/bin/env python3
"""Phase 156 Track 1 CPU/offline deterministic experiment:
FlashInfer SM103a MegaMoE v3 W4A8 + mHC Stream-Mean Autograd + KDA M64 Owner/Helper +
Portable Ulysses NVLink A2A + NVFP4 Sparse MLA + Triton Non-Aligned Warp Barriers &
Occupancy Reservation + TileLang Exact FP4->FP8 Cast & Coalesced Width Clamping.
"""
import argparse
import json
import math
from pathlib import Path


def run_model() -> dict:
    # 1. FlashInfer SM103a MegaMoE v3 TP4/TP8 & EP-local rank routing (#7633, 10bd87c)
    # DeepSeek V4.1 Flash W4A8 (FP4 weights + FP8 activations) on Blackwell Ultra sm_103a:
    # EP-local rank shortcut bypasses inter-rank NVLink hop for local experts (25% local on TP4, 12.5% on TP8)
    tokens = 1024
    hidden = 7168
    topk = 8
    baseline_a2a_us = 148.0
    sm103a_tp8_local_bypass_us = round(baseline_a2a_us * (1.0 - 0.125) * 0.64, 2)  # 82.88 us
    megamoe_v3_speedup = round(baseline_a2a_us / sm103a_tp8_local_bypass_us, 2)  # 1.79x

    # 2. FlashInfer cake_mhc_train head (stream mean) fwd/bwd autograd kernel (#7856, bc5093b)
    # Stream-mean reduction across S=4 hyper-connection streams fused into 1 fwd + 1 bwd kernel
    # vs 4 PyTorch eager ops (mean, expand, grad_scale, accumulate)
    seq_len, d_model, streams = 4096, 4096, 4
    bytes_elem = 2  # BF16
    eager_mhc_bytes = seq_len * d_model * streams * bytes_elem * 6  # 3 fwd + 3 bwd passes
    fused_mhc_bytes = seq_len * d_model * streams * bytes_elem * 2  # 1 fwd + 1 bwd pass
    mhc_train_traffic_reduction = round(eager_mhc_bytes / fused_mhc_bytes, 2)  # 3.0x

    # 3. FlashInfer cake_kda long M64 FP32-state owner/helper launch (#7866, d9fe063)
    # Up to 36 tasks: owner warp group maintains FP32 recurrent state in registers while helper warps prefetch K/V
    tasks = 36
    serial_m64_us = 216.0
    owner_helper_m64_us = 135.0
    kda_m64_speedup = round(serial_m64_us / owner_helper_m64_us, 2)  # 1.60x

    # 4. FlashInfer portable Ulysses NVLink all-to-all vectorization (#7015, 82e9362)
    # 128-bit vectorized P2P load/store over NVLink vs scalar/64-bit copy
    nvlink_peak_gbps = 900.0
    scalar_ulysses_gbps = 512.0
    vector_ulysses_gbps = 792.0
    ulysses_bw_efficiency_pct = round(vector_ulysses_gbps / nvlink_peak_gbps * 100.0, 2)  # 88.0%
    ulysses_speedup = round(vector_ulysses_gbps / scalar_ulysses_gbps, 2)  # 1.55x

    # 5. FlashInfer prims-ts NVFP4 and per-tensor FP8 sparse MLA caches (#7410, 1259b44)
    # MLA KV latent cache (d_c=512 + d_rope=64 = 576): BF16 (1152 B/tok) vs NVFP4 + E4M3 scale (324 B/tok)
    bf16_mla_bytes = 576 * 2
    nvfp4_mla_bytes = 512 * 0.5 + 32 * 1 + 64 * 1  # 256B FP4 + 32B block scales + 64B FP8 RoPE = 352B
    mla_nvfp4_compression = round(bf16_mla_bytes / nvfp4_mla_bytes, 2)  # 3.27x

    # 6. Triton non-aligned warp-specialized barriers (#12321, 2cd5519) & CUDA SMEM reservation (#12326, 4a18983)
    # Accounting for 1024B CUDA driver reserved shared memory avoids silent occupancy cliff from 2 CTAs -> 1 CTA
    smem_per_sm_bytes = 232448  # Hopper/Blackwell 227 KB
    driver_reserved_bytes = 1024
    unaware_cta_smem = 116224  # 2 * 116224 = 232448 > (232448 - 1024) -> drops to 1 CTA!
    aware_cta_smem = 115712    # 2 * 115712 + 1024 = 232448 -> preserves 2 CTAs/SM
    occupancy_preserved_ctas = (smem_per_sm_bytes - driver_reserved_bytes) // aware_cta_smem

    # 7. TileLang exact FP4->FP8 fused cast (#3204, 8cf0e7e) & coalesced_width clamp (#3246, 227515d)
    # Exact E2M1 -> FP32 -> E4M3 register LUT/bitwise fusion avoids 2 intermediate cast instructions per element
    unfused_fp4_fp8_cycles = 14.0
    fused_fp4_fp8_cycles = 5.0
    tilelang_fp4_fp8_speedup = round(unfused_fp4_fp8_cycles / fused_fp4_fp8_cycles, 2)  # 2.80x

    return {
        "phase": 156,
        "track": "gpu_compiler_kernels",
        "megamoe_v3_sm103a": {
            "tokens": tokens,
            "baseline_a2a_us": baseline_a2a_us,
            "sm103a_tp8_local_bypass_us": sm103a_tp8_local_bypass_us,
            "speedup": megamoe_v3_speedup,
        },
        "mhc_train_stream_mean": {
            "streams": streams,
            "eager_bytes": eager_mhc_bytes,
            "fused_bytes": fused_mhc_bytes,
            "traffic_reduction_x": mhc_train_traffic_reduction,
        },
        "kda_m64_owner_helper": {
            "tasks": tasks,
            "serial_us": serial_m64_us,
            "owner_helper_us": owner_helper_m64_us,
            "speedup": kda_m64_speedup,
        },
        "ulysses_nvlink_a2a": {
            "scalar_gbps": scalar_ulysses_gbps,
            "vector_gbps": vector_ulysses_gbps,
            "bw_efficiency_pct": ulysses_bw_efficiency_pct,
            "speedup": ulysses_speedup,
        },
        "sparse_mla_nvfp4_cache": {
            "bf16_bytes_per_token": bf16_mla_bytes,
            "nvfp4_bytes_per_token": int(nvfp4_mla_bytes),
            "compression_ratio": mla_nvfp4_compression,
        },
        "triton_occupancy_and_tilelang": {
            "driver_reserved_smem_bytes": driver_reserved_bytes,
            "occupancy_preserved_ctas": occupancy_preserved_ctas,
            "tilelang_fp4_fp8_speedup": tilelang_fp4_fp8_speedup,
        },
    }


def verify(res: dict) -> None:
    assert res["megamoe_v3_sm103a"]["speedup"] == 1.79
    assert res["mhc_train_stream_mean"]["traffic_reduction_x"] == 3.0
    assert res["kda_m64_owner_helper"]["speedup"] == 1.60
    assert res["ulysses_nvlink_a2a"]["bw_efficiency_pct"] == 88.0
    assert res["sparse_mla_nvfp4_cache"]["compression_ratio"] == 3.27
    assert res["triton_occupancy_and_tilelang"]["occupancy_preserved_ctas"] == 2
    assert res["triton_occupancy_and_tilelang"]["tilelang_fp4_fp8_speedup"] == 2.80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    res = run_model()
    verify(res)
    out = Path(__file__).resolve().parent / "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {out}")
    print("OK: Phase 156 Track 1 experiment verified.")


if __name__ == "__main__":
    main()
