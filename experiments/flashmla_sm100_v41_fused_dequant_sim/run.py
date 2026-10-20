"""Deterministic simulation of DeepSeek FlashMLA SM100 V4.1 KV cache formats,
bank-conflict-free warpgroup dequantization (`KVBlockDequantizer`), fused
`Norm+RoPE+Attn+RoPE+Cast` TMEM column packing, weight permutation bijections
(`permute_q_b_proj`, `permute_wv_proj`), and 128-bit `st.async` LLP64 safety.

Verified against `deepseek-ai/FlashMLA` commits:
- `07a1089857b63e74e3133630c02b083b75e8d4b2`:
  - `csrc/kernels/kv_cache_format.h`
  - `csrc/kernels/sm100/dequant_utils.cuh`
  - `csrc/kernels/sm100/helpers.h`
  - `csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/config.h`
  - `csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu`
  - `csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu`
- `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`:
  - `csrc/kerutils/include/kerutils/device/sm90/intrinsics.cuh`
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).resolve().parent / "results" / "summary.json"


def compute_kv_cache_format(model_type: str) -> dict[str, Any]:
    """Exact mirror of `KVCacheFormat<MT>` in `csrc/kernels/kv_cache_format.h`."""
    is_v32 = model_type == "V32"
    is_fp4 = model_type == "V41_FP4"
    d_qk = 576 if is_v32 else 512
    d_rope = 64
    d_nope = d_qk - d_rope
    d_fp4 = d_qk if is_fp4 else 0
    d_fp8 = 0 if is_fp4 else (d_qk if model_type == "V41" else d_nope)
    d_bf16 = d_qk - d_fp4 - d_fp8
    quant_tile_size = (
        16 if is_fp4 else (32 if model_type == "V41" else (64 if model_type == "V4" else 128))
    )
    num_scales_each_token = (d_nope if is_v32 else d_qk) // quant_tile_size
    # V4 has 7 scales of 64 dims for 448 NoPE dims + 1 byte of padding = 8 scale bytes
    if model_type == "V4":
        num_scales_each_token = 8
    scale_bytes = 4 if is_v32 else 1
    quant_bytes = d_fp4 // 2 + d_fp8
    tma_k_stride = (
        quant_bytes
        + (num_scales_each_token * scale_bytes if is_v32 else 0)
        + 2 * d_bf16
    )
    bytes_per_token = tma_k_stride + (0 if is_v32 else num_scales_each_token * scale_bytes)
    return {
        "model_type": model_type,
        "d_qk": d_qk,
        "d_rope": d_rope,
        "d_nope": d_nope,
        "d_fp4": d_fp4,
        "d_fp8": d_fp8,
        "d_bf16": d_bf16,
        "quant_tile_size": quant_tile_size,
        "num_scales_each_token": num_scales_each_token,
        "scale_bytes": scale_bytes,
        "quant_bytes": quant_bytes,
        "tma_k_stride": tma_k_stride,
        "bytes_per_token": bytes_per_token,
        "tma_stride_ge_256": tma_k_stride >= 256,
    }


def simulate_kv_block_dequantizer_banks() -> dict[str, Any]:
    """Verify SMEM bank conflict counts for `KVBlockDequantizer` (`dequant_utils.cuh`)
    and `fused_norm_rope_attn_rope_cast_fwd` (`core_attn/config.h`).
    """
    # 1. KVBlockDequantizer LDS.32 on gather4 4-row groups (32 banks, 4 B per bank = 128 B wrap)
    # In a gather4 group, 4 rows (r = 0, 1, 2, 3) are separated by RAW_TOKEN_SMEM_STRIDE.
    # For 8 threads per row (idx_in_group = 0..7, reading 4 B each = 32 B = 8 banks):
    # Within 4 rows (32 threads = 1 warp), thread (r, idx) accesses byte offset:
    #   offset = r * stride + idx * 4 -> bank = (offset // 4) % 32.
    def max_lds32_bank_way(stride_bytes: int) -> int:
        banks = [((r * stride_bytes + idx * 4) // 4) % 32 for r in range(4) for idx in range(8)]
        counts: dict[int, int] = {}
        for b in banks:
            counts[b] = counts.get(b, 0) + 1
        return max(counts.values())

    # Unpadded FP4 row bytes: 256 B (1-CTA) or 128 B (2-CTA) -> stride / 4 % 16 == 0 -> 4-way conflict!
    # Padded FP4 row bytes: 256 + 32 = 288 B (1-CTA) or 128 + 32 = 160 B (2-CTA) -> stride / 4 % 16 == 8 -> 1-way (conflict-free)!
    fp4_1cta_unpadded_way = max_lds32_bank_way(256)
    fp4_1cta_padded_way = max_lds32_bank_way(288)
    fp4_2cta_unpadded_way = max_lds32_bank_way(128)
    fp4_2cta_padded_way = max_lds32_bank_way(160)

    # 2. SW128 K-major output tile STS.128 wavefront (8 lanes, 16 B per lane = 4 banks per lane)
    # `dst_offset = (group_idx // 8) * 1024 + row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16`
    # For a fixed `row_in_atom` (or across 8 rows for a fixed column in a wavefront), check 16 B bank groups:
    sw128_wavefront_conflict_free = True
    for row_in_atom in range(8):
        bank_groups = {
            ((row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16) // 16) % 8
            for idx_in_group in range(8)
        }
        if len(bank_groups) != 8:
            sw128_wavefront_conflict_free = False
    # Also across 8 rows (group_idx = 0..7) for the same idx_in_group (column wavefront):
    for idx_in_group in range(8):
        bank_groups = {
            ((row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16) // 16) % 8
            for row_in_atom in range(8)
        }
        if len(bank_groups) != 8:
            sw128_wavefront_conflict_free = False

    # 3. `fused_norm_rope_attn_rope_cast_fwd` common dequant path (`LDS.128`, +16 B pad):
    # `RAW_TOKEN_SMEM_STRIDE = RAW_TOKEN_DATA_BYTES + 16`:
    # V41_FP4: 1-CTA 256+16=272 B (272/16=17 odd), 2-CTA 128+16=144 B (144/16=9 odd)
    # V41 FP8: 1-CTA 512+16=528 B (528/16=33 odd), 2-CTA 256+16=272 B (272/16=17 odd)
    common_strides = {
        "fp4_1cta": 256 + 16,
        "fp4_2cta": 128 + 16,
        "fp8_1cta": 512 + 16,
        "fp8_2cta": 256 + 16,
    }
    common_strides_odd_16b = all((s // 16) % 2 == 1 for s in common_strides.values())

    return {
        "fp4_1cta_unpadded_lds32_way": fp4_1cta_unpadded_way,
        "fp4_1cta_padded_288b_lds32_way": fp4_1cta_padded_way,
        "fp4_2cta_unpadded_lds32_way": fp4_2cta_unpadded_way,
        "fp4_2cta_padded_160b_lds32_way": fp4_2cta_padded_way,
        "sw128_sts128_wavefront_conflict_free": sw128_wavefront_conflict_free,
        "common_lds128_strides_bytes": common_strides,
        "common_lds128_strides_odd_16b": common_strides_odd_16b,
    }


def verify_e2m1_times_e4m3_exact_in_bf16() -> dict[str, Any]:
    """Verify the invariant in `dequant_utils.cuh`:
    `data_bf16[i] = __hmul2(data_bf16[i], scale); // Exact: e2m1 x e4m3 has at most 2 + 4 significant bits`
    Check all finite non-zero (e2m1, e4m3) pairs against exact FP64 product rounded to BF16 vs FP64.
    """
    # Positive E2M1 values (1 sign, 2 exp, 1 mantissa, bias=1):
    # 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    e2m1_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    e2m1_signed = e2m1_vals + [-v for v in e2m1_vals]

    # All finite positive E4M3 values (exp 0..15, mantissa 0..7, bias=7, excluding exp=15,mant=7 NaN)
    e4m3_vals: list[float] = []
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue
            if exp == 0:
                val = (mant / 8.0) * (2.0 ** -6)
            else:
                val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
            e4m3_vals.append(val)

    def float_to_bf16_exact(x: float) -> float:
        b = struct.pack(">f", x)
        # Check that low 16 bits of IEEE-754 FP32 are identically zero!
        low16 = struct.unpack(">H", b[2:4])[0]
        assert low16 == 0, f"Value {x} is not exact in BF16 (low16={low16:#x})"
        return struct.unpack(">f", b[:2] + b"\x00\x00")[0]

    total_pairs = 0
    inexact_pairs = 0
    for a in e2m1_signed:
        for s in e4m3_vals:
            prod = a * s
            b = struct.pack(">f", prod)
            low16 = struct.unpack(">H", b[2:4])[0]
            total_pairs += 1
            if low16 != 0:
                inexact_pairs += 1
            _ = float_to_bf16_exact(prod)

    return {
        "total_finite_pairs_checked": total_pairs,
        "inexact_bf16_pairs": inexact_pairs,
        "all_products_exact_in_bf16": inexact_pairs == 0,
    }


def simulate_fused_sm100_tmem_and_permutations() -> dict[str, Any]:
    """Verify `tmem_cols` in `core_attn/config.h` and the bijectivity of
    `permute_q_b_proj_kernel` and `permute_wv_proj_kernel`.
    """
    d_qk = 512
    d_vo = 512
    h_q_per_cta = 64
    fold_factor = 128 // h_q_per_cta  # 2
    num_mrgemm_rails = 2

    tmem_layouts: dict[str, Any] = {}
    for h_q in (64, 128):
        cluster_size = (h_q + 63) // 64
        b_topk = 96 if cluster_size == 2 else 64
        num_p_bufs = 1 if cluster_size == 2 else 2
        need_tp_empty_bar = num_p_bufs == 1
        col_o = 0
        col_q = col_o + d_vo // fold_factor  # 256
        col_p0 = col_q + d_qk // num_mrgemm_rails // 2  # 256 + 128 = 384
        p_width = b_topk * num_mrgemm_rails // fold_factor  # 64 (1-CTA) or 96 (2-CTA)
        col_p1 = col_p0 + p_width
        total_cols = (col_p1 if num_p_bufs == 2 else col_p0) + p_width
        tmem_layouts[f"hq_{h_q}"] = {
            "h_q": h_q,
            "cluster_size": cluster_size,
            "b_topk": b_topk,
            "num_p_bufs": num_p_bufs,
            "need_tp_empty_bar": need_tp_empty_bar,
            "O_range": [col_o, col_q],
            "Q_range": [col_q, col_p0],
            "P0_range": [col_p0, col_p0 + p_width],
            "P1_range": [col_p1, col_p1 + p_width] if num_p_bufs == 2 else None,
            "total_tmem_cols": total_cols,
            "fits_512_cols": total_cols <= 512,
        }

    # Verify `permute_q_b_proj_kernel` bijection for H_Q = 128, D_Q = 512 (65,536 rows)
    h_q = 128
    d_q = 512
    q_b_out_rows = set()
    for row_idx in range(h_q * d_q):
        head_idx = row_idx // d_q
        head_dim_idx = row_idx % d_q
        out_row_idx = (head_dim_idx % 16) + head_idx * 16 + (head_dim_idx // 16) * (h_q * 16)
        q_b_out_rows.add(out_row_idx)
    q_b_bijective = len(q_b_out_rows) == h_q * d_q and min(q_b_out_rows) == 0 and max(q_b_out_rows) == h_q * d_q - 1

    # Verify `permute_wv_proj_kernel` bijection for wv_group_size = 8, D_O = 512, CHUNK_SIZE = 32
    wv_group_size = 8
    d_o = 512
    chunk_size = 32
    num_chunks = wv_group_size * (d_o // chunk_size)  # 8 * 16 = 128
    wv_out_chunks = set()
    for i in range(num_chunks):
        head_idx = i // (d_o // chunk_size)
        head_dim_chunk_idx = i % (d_o // chunk_size)
        chunk_idx_in_permuted_row = head_dim_chunk_idx * wv_group_size + head_idx
        wv_out_chunks.add(chunk_idx_in_permuted_row)
    wv_bijective = len(wv_out_chunks) == num_chunks

    # Verify LLP64 (`long2` = 8B on Windows MSVC vs `longlong2` = 16B) for 128-bit `st.async`
    llp64_long2_bytes = 2 * 4
    llp64_longlong2_bytes = 2 * 8

    return {
        "tmem_layouts": tmem_layouts,
        "permute_q_b_proj_rows": h_q * d_q,
        "permute_q_b_proj_bijective": q_b_bijective,
        "permute_wv_proj_chunks_per_group": num_chunks,
        "permute_wv_proj_bijective": wv_bijective,
        "llp64_long2_bytes": llp64_long2_bytes,
        "llp64_longlong2_bytes": llp64_longlong2_bytes,
        "llp64_long2_truncates_upper_64b": llp64_long2_bytes == 8 and llp64_longlong2_bytes == 16,
    }


def build_summary() -> dict[str, Any]:
    formats = {mt: compute_kv_cache_format(mt) for mt in ("V32", "V4", "V41", "V41_FP4")}
    dequant_banks = simulate_kv_block_dequantizer_banks()
    exact_bf16 = verify_e2m1_times_e4m3_exact_in_bf16()
    fused_sm100 = simulate_fused_sm100_tmem_and_permutations()
    return {
        "experiment": "flashmla_sm100_v41_fused_dequant_sim",
        "source_commits": [
            "deepseek-ai/FlashMLA@07a1089857b63e74e3133630c02b083b75e8d4b2",
            "deepseek-ai/FlashMLA@ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
        ],
        "kv_cache_formats": formats,
        "compression_ratios": {
            "v41_fp4_vs_v32": round(
                formats["V32"]["bytes_per_token"] / formats["V41_FP4"]["bytes_per_token"], 4
            ),
            "v41_fp4_vs_v41_fp8": round(
                formats["V41"]["bytes_per_token"] / formats["V41_FP4"]["bytes_per_token"], 4
            ),
        },
        "dequant_banks": dequant_banks,
        "e2m1_e4m3_bf16_exactness": exact_bf16,
        "fused_sm100": fused_sm100,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify against committed summary.json")
    args = parser.parse_args()

    summary = build_summary()
    assert summary["kv_cache_formats"]["V32"]["bytes_per_token"] == 656
    assert summary["kv_cache_formats"]["V4"]["bytes_per_token"] == 584
    assert summary["kv_cache_formats"]["V41"]["bytes_per_token"] == 528
    assert summary["kv_cache_formats"]["V41_FP4"]["bytes_per_token"] == 288
    assert summary["kv_cache_formats"]["V41_FP4"]["tma_k_stride"] == 256
    assert summary["dequant_banks"]["fp4_1cta_unpadded_lds32_way"] == 4
    assert summary["dequant_banks"]["fp4_1cta_padded_288b_lds32_way"] == 1
    assert summary["dequant_banks"]["fp4_2cta_padded_160b_lds32_way"] == 1
    assert summary["dequant_banks"]["sw128_sts128_wavefront_conflict_free"] is True
    assert summary["e2m1_e4m3_bf16_exactness"]["all_products_exact_in_bf16"] is True
    assert summary["fused_sm100"]["tmem_layouts"]["hq_64"]["total_tmem_cols"] == 512
    assert summary["fused_sm100"]["tmem_layouts"]["hq_128"]["total_tmem_cols"] == 480
    assert summary["fused_sm100"]["permute_q_b_proj_bijective"] is True
    assert summary["fused_sm100"]["permute_wv_proj_bijective"] is True

    if args.check:
        if not SUMMARY_PATH.exists():
            print(f"Missing {SUMMARY_PATH}", file=sys.stderr)
            return 1
        committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        if committed != summary:
            print("Mismatch between computed and committed summary.json", file=sys.stderr)
            return 1
        print("flashmla_sm100_v41_fused_dequant_sim: OK")
        return 0

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
