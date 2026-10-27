"""Phase 111 Track A stdlib experiment:
FlashInfer Frost W8A16 MoE, Hopper MSA Top-K Select, SM120 Quant Out-Proj,
GB300 DSA Indexer Shift, SM12x >2GiB Static DLPack, SSU Non-Multiple-of-64 HeadDim,
FlashAttention-4 SM100 Sparse MLA In-Kernel Recompute-P + Token-Chunked Backward,
and TileLang call_extern Bare Data Var Write Annotation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_frost_w8a16_and_ep_dispatch() -> dict[str, Any]:
    """Model FlashInfer Frost W8A16 MoE candidates (d42c276bd872), Hopper MSA Top-K (3d01e3fec63f),
    and MoEEpCommunication runtime_max_tokens_per_rank dynamic bound (47e64b0e96c4).
    """
    # Frost W8A16 schedule selection across SM90, SM103, SM120
    arch_schedules = {
        "sm90_t4_bf16_geglu": {"tile_k": 256, "serial_gather": False, "latency_us": 19.2},
        "sm103_t4_bf16_geglu": {"tile_k": 128, "serial_gather": True, "latency_us": 14.8},
        "sm120_t8_bf16_geglu": {"tile_k": 128, "group_size": 8, "num_stages": 2, "latency_us": 26.5},
    }

    # Hopper cake_hopper_msa msa_topk_select single-pass exact schedule (3d01e3fec63f)
    hopper_msa_programs = {
        "plain_1chunk_128w_11lvl": {"hash": "276e272a54b4efd20df6", "chunks": 1, "width": 128, "levels": 11, "speedup": 1.06},
        "masked_1chunk_128w_11lvl": {"hash": "7bbffa90057de145b905", "chunks": 1, "width": 128, "levels": 11, "speedup": 1.05},
        "masked_2chunk_32w_9lvl": {"hash": "e9b601567cc3a6dd8033", "chunks": 2, "width": 32, "levels": 9, "speedup": 1.04},
    }

    # MoEEpCommunication dynamic runtime_max_tokens_per_rank (47e64b0e96c4)
    max_tokens_per_rank = 4096
    runtime_max_tokens_per_rank = 512
    ep_world_size = 8
    hidden_size = 7168
    bytes_per_token = hidden_size * 2  # BF16
    static_comm_bytes = max_tokens_per_rank * ep_world_size * bytes_per_token
    dynamic_comm_bytes = runtime_max_tokens_per_rank * ep_world_size * bytes_per_token
    comm_volume_reduction_ratio = static_comm_bytes / dynamic_comm_bytes

    return {
        "arch_schedules": arch_schedules,
        "hopper_msa_programs": hopper_msa_programs,
        "max_tokens_per_rank": max_tokens_per_rank,
        "runtime_max_tokens_per_rank": runtime_max_tokens_per_rank,
        "static_comm_bytes": static_comm_bytes,
        "dynamic_comm_bytes": dynamic_comm_bytes,
        "comm_volume_reduction_ratio": round(comm_volume_reduction_ratio, 4),
    }


def simulate_sm120_quant_out_proj_and_gb300_dsa_shift() -> dict[str, Any]:
    """Model SM120 MiniMax-H3 quant_out_proj Program 1 routing (73c83aac3125),
    GB300 cake_dsa_indexer sample_shift_rule=[4096, 375] (2b5a5ffff99a),
    SM12x W4A16 >2GiB static DLPack layout (9c4aacf1bfd6), and SSU non-64 head_dim (0c3d03c7656a).
    """
    program_1_sms = {0: (188,), 1: (188, 170)}
    rtx5090_nvfp4_routed_to_program_1 = 170 in program_1_sms[1]
    m4097_speedup = round(346.3 / 338.8, 6)
    m33472_speedup = round(2316.9 / 2267.1, 6)

    # GB300 DSA indexer sample_shift_rule = [4096, 375]
    def select_shift_permille(top_k: int) -> int:
        return 375 if top_k >= 4096 else 250

    gb300_shift_k2048 = select_shift_permille(2048)
    gb300_shift_k4096 = select_shift_permille(4096)
    gb300_z1k4096_speedup = 1.13

    # SM12x W4A16 >2GiB static DLPack layout
    num_experts = 256
    inter_x2 = 2048
    packed_k = 5120
    w13_bytes = num_experts * inter_x2 * packed_k  # 2,684,354,560 bytes (> 2^31 - 1)
    int32_max = (1 << 31) - 1
    overflows_dynamic_i32_stride = w13_bytes > int32_max

    # SSU non-multiple-of-64 head_dim CTA dispatch (0c3d03c7656a)
    def dispatch_ctas_per_head(k_dim: int, k_min_rows: int = 16) -> int:
        for ctas in (4, 2, 1):
            if k_dim % ctas == 0 and (k_dim // ctas) % k_min_rows == 0:
                return ctas
        return 1

    ssu_ctas = {dim: dispatch_ctas_per_head(dim) for dim in (64, 80, 96, 128, 288)}

    return {
        "rtx5090_nvfp4_routed_to_program_1": rtx5090_nvfp4_routed_to_program_1,
        "m4097_speedup": m4097_speedup,
        "m33472_speedup": m33472_speedup,
        "gb300_shift_k2048_permille": gb300_shift_k2048,
        "gb300_shift_k4096_permille": gb300_shift_k4096,
        "gb300_z1k4096_speedup": gb300_z1k4096_speedup,
        "w13_bytes": w13_bytes,
        "overflows_dynamic_i32_stride": overflows_dynamic_i32_stride,
        "ssu_ctas_per_head": ssu_ctas,
    }


def simulate_fa4_sparse_mla_recompute_and_tilelang() -> dict[str, Any]:
    """Model FlashAttention-4 SM100 Sparse MLA backward recompute-P + token-chunked backward (294cfb3593ed),
    SM100 HD256 varlen grid bound (38ddafeb35e5), and TileLang call_extern bare data var write (f3ffeaa6b8ee).
    """
    q_heads = 128
    topk_len = 2048
    # Per token saved activations in default path:
    # p: bf16 [128, 2048] = 524,288 B = 512 KiB; row_max: fp32 [128, 16] = 8 KiB -> 520 KiB/token
    p_bytes_per_token = q_heads * topk_len * 2
    row_max_bytes_per_token = 8192
    saved_bytes_per_token_default = p_bytes_per_token + row_max_bytes_per_token

    seqlen_64k = 65536
    saved_gib_64k_default = (seqlen_64k * p_bytes_per_token) / (1024**3)  # 32.0 GiB of p
    ds_transient_gib_64k_unchunked = (seqlen_64k * p_bytes_per_token) / (1024**3)  # 32.0 GiB of dS

    token_chunk = 4096
    saved_gib_64k_recompute_p = 0.0  # only out+lse saved, p+row_max eliminated
    ds_transient_gib_chunked = (token_chunk * p_bytes_per_token) / (1024**3)  # 2.0 GiB per chunk

    # Causal chunked k_end limit per chunk [tok0, tok1): k_end = seqlen_k - seqlen_q + tok1
    seqlen_q = 65536
    seqlen_k = 65536
    chunk_0_k_end = seqlen_k - seqlen_q + token_chunk  # 4096

    # FA4 SM100 HD256 varlen grid over-launch reduction (38ddafeb35e5)
    batch_size = 16
    seqlens = [512] * 15 + [8192]
    total_q = sum(seqlens)
    max_seqlen_q = max(seqlens)
    tile_m = 128
    overlaunched_blocks = batch_size * math.ceil(total_q / tile_m)
    bounded_max_blocks = batch_size * math.ceil(max_seqlen_q / tile_m)
    flat_varlen_blocks = sum(math.ceil(s / tile_m) for s in seqlens)

    # TileLang AnnotateReadOnlyParams (f3ffeaa6b8ee)
    tilelang_cases = {
        "call_extern_bare_data_var": {"op": "call_extern", "arg_kind": "bare_data_var", "marked_written": True, "readonly_attr": False},
        "call_extern_address_of": {"op": "call_extern", "arg_kind": "address_of", "marked_written": True, "readonly_attr": False},
        "pure_isnullptr_bare_data_var": {"op": "tirx.isnullptr", "arg_kind": "bare_data_var", "marked_written": False, "readonly_attr": True},
    }

    return {
        "saved_gib_64k_default": round(saved_gib_64k_default, 4),
        "saved_gib_64k_recompute_p": saved_gib_64k_recompute_p,
        "ds_transient_gib_64k_unchunked": round(ds_transient_gib_64k_unchunked, 4),
        "ds_transient_gib_chunked": round(ds_transient_gib_chunked, 4),
        "fwd_8k_ms_default": 4.43,
        "fwd_8k_ms_recompute_p": 3.77,
        "chunk_0_k_end": chunk_0_k_end,
        "hd256_overlaunched_blocks": overlaunched_blocks,
        "hd256_bounded_max_blocks": bounded_max_blocks,
        "hd256_flat_varlen_blocks": flat_varlen_blocks,
        "tilelang_cases": tilelang_cases,
    }


def run_all() -> dict[str, Any]:
    return {
        "phase": 111,
        "track": "A",
        "title": "FlashInfer Frost W8A16 MoE & Hopper MSA Top-K, FA4 SM100 Sparse MLA Recompute-P & Token-Chunked Bwd, and TileLang call_extern Write Analysis",
        "frost_w8a16_and_ep": simulate_frost_w8a16_and_ep_dispatch(),
        "sm120_and_gb300": simulate_sm120_quant_out_proj_and_gb300_dsa_shift(),
        "fa4_sparse_mla_and_tilelang": simulate_fa4_sparse_mla_recompute_and_tilelang(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["frost_w8a16_and_ep"]["comm_volume_reduction_ratio"] == 8.0
        assert summary["sm120_and_gb300"]["rtx5090_nvfp4_routed_to_program_1"] is True
        assert summary["sm120_and_gb300"]["gb300_shift_k4096_permille"] == 375
        assert summary["sm120_and_gb300"]["overflows_dynamic_i32_stride"] is True
        assert summary["sm120_and_gb300"]["ssu_ctas_per_head"][96] == 2
        assert summary["sm120_and_gb300"]["ssu_ctas_per_head"][80] == 1
        assert summary["fa4_sparse_mla_and_tilelang"]["saved_gib_64k_default"] == 32.0
        assert summary["fa4_sparse_mla_and_tilelang"]["ds_transient_gib_chunked"] == 2.0
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
