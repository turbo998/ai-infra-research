#!/usr/bin/env python3
"""Phase 107 Experiment 1: FlashInfer MXFP4/MXFP8 UE8M0 Subnormal Quantization, Rubin 2-CTA MoE Tactics, Qwen 3.8 QSA Decode & TileLang Barrier Uniqueness."""
from __future__ import annotations
import argparse
import json
import math
import pathlib
import struct

def float_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]

def bits_to_float(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]

def float_to_e8m0_rp_u8(x: float) -> int:
    """Bit-exact CuTile/CUDA UE8M0 round-toward-positive-infinity conversion (flashinfer #7501)."""
    if x <= 0.0:
        return 0
    bits = float_to_bits(x)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    round_up = 1 if (mantissa != 0 and (exponent != 0 or mantissa > 0x400000)) else 0
    return min(exponent + round_up, 254)

def ue8m0_inverse_scale(exp_byte: int) -> float:
    """Bit-exact inverse scale matching flashinfer #7501 ue8m0_inverse_scale."""
    if exp_byte == 0:
        return 0.0
    shifted = max(254 - int(exp_byte), 0) << 23
    return bits_to_float(shifted)

def rubin_moe_cluster_covers_one_tile(
    tile_size: int,
    mma_tiler_m: int,
    mma_inst_m: int,
    cluster_m: int,
    cluster_n: int,
) -> dict:
    """Model flashinfer #7027 Rubin SM107 MoE cluster-to-routing-tile mapping and multicast."""
    atom_cta_m = 2 if mma_inst_m == 256 else 1
    valid = (mma_tiler_m == tile_size) and (cluster_m == atom_cta_m)
    # Cluster N multicasts A and SFA across cluster_n CTAs sharing the same M routing tile
    a_bandwidth_reduction_factor = float(cluster_n) if valid else 1.0
    # When valid, every CTA in the cluster maps to floor(cta_m / atom_cta_m), so 0 extra padding tiles needed
    metadata_padding_tiles = 0 if valid else max(cluster_m - atom_cta_m, 0)
    return {
        "tile_size": tile_size,
        "mma_tiler_m": mma_tiler_m,
        "mma_inst_m": mma_inst_m,
        "cluster_shape_mn": [cluster_m, cluster_n],
        "atom_cta_m": atom_cta_m,
        "valid_single_routing_tile": valid,
        "a_sfa_multicast_factor": round(a_bandwidth_reduction_factor, 2),
        "metadata_padding_tiles_required": metadata_padding_tiles,
    }

def qwen38_qsa_compressed_key_traffic(seq_len: int, top_k_blocks: int, d_head: int = 256, compress_stride: int = 4) -> dict:
    """Model flashinfer #7136 Qwen 3.8 Flash-Next QSA decode indexer compressed-key + sparse KV traffic."""
    num_compressed_blocks = seq_len // compress_stride
    ring_tail_tokens = seq_len % compress_stride
    # Dense decode reads all seq_len K+V rows (2 * seq_len * d_head * 2 bytes in BF16)
    dense_kv_bytes = 2 * seq_len * d_head * 2
    # QSA reads compressed indexer keys (128-dim BF16 per 4-token block) + top_k_blocks * 4 tokens of K+V
    indexer_scan_bytes = (num_compressed_blocks + ring_tail_tokens) * 128 * 2
    selected_tokens = min(seq_len, top_k_blocks * compress_stride + ring_tail_tokens)
    sparse_kv_bytes = 2 * selected_tokens * d_head * 2
    total_qsa_bytes = indexer_scan_bytes + sparse_kv_bytes
    return {
        "seq_len": seq_len,
        "num_compressed_blocks": num_compressed_blocks,
        "dense_kv_bytes": dense_kv_bytes,
        "total_qsa_bytes": total_qsa_bytes,
        "bandwidth_speedup_ratio": round(dense_kv_bytes / total_qsa_bytes, 4),
    }

def tilelang_cross_scope_barrier_allocation(shared_groups: int, shared_dyn_groups: int) -> dict:
    """Model TileLang #3643 partial-sync barrier ID uniqueness across shared and shared.dyn."""
    buggy_shared_ids = list(range(1, shared_groups + 1))
    buggy_shared_dyn_ids = list(range(1, shared_dyn_groups + 1))
    collisions = sorted(set(buggy_shared_ids).intersection(buggy_shared_dyn_ids))
    fixed_shared_ids = list(range(1, shared_groups + 1))
    fixed_shared_dyn_ids = list(range(shared_groups + 1, shared_groups + shared_dyn_groups + 1))
    fixed_collisions = sorted(set(fixed_shared_ids).intersection(fixed_shared_dyn_ids))
    return {
        "buggy_collision_count": len(collisions),
        "fixed_collision_count": len(fixed_collisions),
        "fixed_shared_ids": fixed_shared_ids,
        "fixed_shared_dyn_ids": fixed_shared_dyn_ids,
    }

def run_experiment() -> dict:
    subnormal_amax = bits_to_float(0x00400000)  # FP32 subnormal 2^-127
    scale_float = subnormal_amax / 448.0
    e8m0_zero = float_to_e8m0_rp_u8(scale_float)
    inv_zero = ue8m0_inverse_scale(e8m0_zero)
    normal_amax = 448.0 * (2.0 ** -10)
    e8m0_normal = float_to_e8m0_rp_u8(normal_amax / 448.0)
    inv_normal = ue8m0_inverse_scale(e8m0_normal)

    tactics = [
        rubin_moe_cluster_covers_one_tile(128, 128, 128, 1, 1),
        rubin_moe_cluster_covers_one_tile(256, 256, 128, 1, 2),
        rubin_moe_cluster_covers_one_tile(256, 256, 256, 2, 2),
        rubin_moe_cluster_covers_one_tile(128, 128, 128, 2, 1),
    ]
    qsa = qwen38_qsa_compressed_key_traffic(seq_len=32768, top_k_blocks=512, d_head=256, compress_stride=4)
    barriers = tilelang_cross_scope_barrier_allocation(shared_groups=2, shared_dyn_groups=2)

    return {
        "experiment_id": "flashinfer_mxfp_ue8m0_rubin_2cta_moe_qwen38_qsa_phase107",
        "evidence_mode": "own-offline",
        "mxfp_ue8m0_subnormal": {
            "subnormal_amax": subnormal_amax,
            "subnormal_e8m0_byte": e8m0_zero,
            "subnormal_inverse_scale": inv_zero,
            "normal_e8m0_byte": e8m0_normal,
            "normal_inverse_scale": inv_normal,
        },
        "rubin_moe_tactics": tactics,
        "qwen38_qsa_decode": qsa,
        "tilelang_barrier_uniqueness": barriers,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_experiment()
    out_path = pathlib.Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["mxfp_ue8m0_subnormal"]["subnormal_e8m0_byte"] == 0
        assert summary["mxfp_ue8m0_subnormal"]["subnormal_inverse_scale"] == 0.0
        assert summary["rubin_moe_tactics"][2]["valid_single_routing_tile"] is True
        assert summary["rubin_moe_tactics"][2]["a_sfa_multicast_factor"] == 2.0
        assert summary["rubin_moe_tactics"][3]["valid_single_routing_tile"] is False
        assert summary["qwen38_qsa_decode"]["bandwidth_speedup_ratio"] == 8.0
        assert summary["tilelang_barrier_uniqueness"]["buggy_collision_count"] == 2
        assert summary["tilelang_barrier_uniqueness"]["fixed_collision_count"] == 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
