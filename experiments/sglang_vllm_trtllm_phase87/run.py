from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def floor_chunked_tokens(trunc_len: int, remaining: int, grid: int) -> int:
    """Reproduce SGLang #41701 (`8252e3af63`): keep chunked prefill cut on Mamba chunk grid for DCP."""
    if grid <= 1 or trunc_len >= remaining or trunc_len < grid:
        return trunc_len
    return (trunc_len // grid) * grid


def simulate_sglang_mamba_dcp_grid() -> dict[str, Any]:
    grid = 64
    dsa_align = 24
    combined_lcm = math.lcm(dsa_align, grid)
    cases = [
        {"trunc_len": 100, "remaining": 8192, "cut_tokens": floor_chunked_tokens(100, 8192, grid)},
        {"trunc_len": 8010, "remaining": 100000, "cut_tokens": floor_chunked_tokens(8010, 100000, grid)},
        {"trunc_len": 8192, "remaining": 100000, "cut_tokens": floor_chunked_tokens(8192, 100000, grid)},
        {"trunc_len": 37, "remaining": 37, "cut_tokens": floor_chunked_tokens(37, 37, grid)},
        {"trunc_len": 37, "remaining": 8192, "cut_tokens": floor_chunked_tokens(37, 8192, grid)},
    ]
    # Simulate 4-chunk prefill of 1024 tokens with unaligned budget 260 vs grid-floored budget
    unaligned_prefix = 0
    unaligned_donations = 0
    aligned_prefix = 0
    aligned_donations = 0
    rem_u = 1024
    rem_a = 1024
    while rem_u > 0:
        step = min(260, rem_u)
        unaligned_prefix += step
        rem_u -= step
        if unaligned_prefix % grid == 0:
            unaligned_donations += 1
    while rem_a > 0:
        step = floor_chunked_tokens(min(260, rem_a), rem_a, grid)
        aligned_prefix += step
        rem_a -= step
        if aligned_prefix % grid == 0:
            aligned_donations += 1
    return {
        "mamba_chunk_grid": grid,
        "dsa_kpool_align": dsa_align,
        "combined_lcm_align": combined_lcm,
        "test_cases": cases,
        "unaligned_dcp_donations_1024": unaligned_donations,
        "grid_aligned_dcp_donations_1024": aligned_donations,
    }


def simulate_sglang_k3_sm107_splitk() -> dict[str, Any]:
    """Model SGLang #46464 (`1b112fd62a`): FP32 warp split-K GEMM for Kimi-K3 MoE front projection on SM107."""
    # _K3_FRONT_WARP_SPLITK = {8: (8, 11), 16: (16, 9), 32: (32, 6)}
    tactics = {8: (8, 11), 16: (16, 9), 32: (32, 6)}
    n, k = 6016, 7168
    router_slice = (1536, 2432)
    router_experts = router_slice[1] - router_slice[0]
    top_k = 16
    compute_warps = 4
    a_loader_warps = 2
    rows = []
    for m, (cta_m, stages) in tactics.items():
        flops = 2 * m * n * k
        weight_bytes_bf16 = n * k * 2
        out_bytes_fp32 = m * n * 4
        rows.append({
            "m": m,
            "cta_m": cta_m,
            "ring_stages": stages,
            "in_flight_smem_kb": stages * 8,
            "flops": flops,
            "weight_bytes_bf16": weight_bytes_bf16,
            "out_bytes_fp32": out_bytes_fp32,
        })
    return {
        "target_sm": 107,
        "shape_n_k": [n, k],
        "router_logit_slice": list(router_slice),
        "router_num_experts": router_experts,
        "router_top_k": top_k,
        "compute_warps": compute_warps,
        "a_loader_warps": a_loader_warps,
        "tactics": rows,
    }


def simulate_vllm_and_trtllm_guards() -> dict[str, Any]:
    """Model vLLM #63460/#63496 and TensorRT-LLM #17147/#20270."""
    # TensorRT-LLM #17147: Ulysses packed QKV post-unscatter [P, B, Sp, 3, H, D] -> 3 x [B, P*Sp, H, D]
    p, b, sp, h, d = 2, 1, 2048, 24, 128
    vec = 8
    ps_p = p * sp
    grid_x = 3 * ps_p
    vec_per_row = d // vec
    block_x = h * vec_per_row
    total_bytes_moved = 2 * (p * b * sp * 3 * h * d * 2)  # BF16 read + write

    # vLLM #63460: Shared NCCL trainer rendezvous payload
    nccl_rendezvous = {
        "trainer_rank": 0,
        "worker_rank_offset": 1,
        "dense_packed_supported": True,
        "sparse_packed_default": False,
        "concurrent_rpc_and_endpoint_open": True,
    }

    # vLLM #63496: LoRA config validation rejects MegaMoE backends early
    rejected_moe_backends = ["deep_gemm_mega_moe", "aiter_mega_moe"]

    return {
        "trtllm_ulysses_packed_qkv": {
            "p": p,
            "b": b,
            "sp": sp,
            "h": h,
            "d": d,
            "vec_width_elements": vec,
            "uint4_bytes": 16,
            "grid_dim_x": grid_x,
            "block_dim_x": block_x,
            "total_bf16_bytes_moved": total_bytes_moved,
            "separate_launches_before": 3,
            "fused_launches_after": 1,
        },
        "vllm_nccl_rendezvous": nccl_rendezvous,
        "vllm_lora_rejected_megamoe_backends": rejected_moe_backends,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "sglang_vllm_trtllm_phase87",
        "phase": 87,
        "priority": 96,
        "sglang_mamba_dcp_grid": simulate_sglang_mamba_dcp_grid(),
        "sglang_k3_sm107_splitk": simulate_sglang_k3_sm107_splitk(),
        "vllm_and_trtllm_guards": simulate_vllm_and_trtllm_guards(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        dcp = summary["sglang_mamba_dcp_grid"]
        assert dcp["combined_lcm_align"] == 192
        assert dcp["unaligned_dcp_donations_1024"] == 1
        assert dcp["grid_aligned_dcp_donations_1024"] == 4
        k3 = summary["sglang_k3_sm107_splitk"]
        assert k3["router_num_experts"] == 896
        assert k3["shape_n_k"] == [6016, 7168]
        vt = summary["vllm_and_trtllm_guards"]
        assert vt["trtllm_ulysses_packed_qkv"]["grid_dim_x"] == 12288
        assert vt["trtllm_ulysses_packed_qkv"]["fused_launches_after"] == 1
        print("OK: sglang_vllm_trtllm_phase87 verified")
        return 0

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
