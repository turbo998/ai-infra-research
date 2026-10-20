#!/usr/bin/env python3
"""Deterministic simulation of vLLM V1 FlashMLA Sparse Multi-Gen KV Formats, HiSparse Hot-Cache, and GDN Checkpoints.

Models `vllm-project/vllm` (`commit 5285b9b8646d6d0e8b869c9f5c6d277ccfce637b`):
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`:
  Exact KV cache record layouts across DeepSeek V3.2 (`656 B`), DeepSeek V4 (`584 B`, 576B split-page data TMA stride),
  DeepSeek V4.1 (`528 B` MXFP8 with quantized RoPE), and SM100 `nvfp4_ds_mla` (`352 B` with `8x4 -> 4x8` permuted scale
  offset `8 * (s & 3) + (s >> 2)`).
- `vllm/v1/hisparse/runtime.py` & `vllm/v1/attention/backends/mla/index_group.py`:
  `ResolvedHiSparseConfig` (`device_buffer_size = (max_query_len + 1) * model_top_k`, `int16` slot limit `32768`),
  `_check_residency_shared_memory` (`required = 4 * (2 * (max_union_rows + 1) + 3 * ceil(hot_size / 32) + 7) + 2 * hot_size`),
  and `SparseMLAIndexGroup._convert_once` group-leader side-stream conversion.
- `vllm/v1/attention/backends/gdn_attn.py`:
  `plan_flashinfer_checkpoints` (`FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64`, `every_n_tokens = gcd(*offsets)`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64


def nvfp4_ds_mla_scale_byte_offset(s: int) -> int:
    """Exact `8x4 -> 4x8` transpose offset from `flashmla_sparse.py`: `8 * (s & 3) + (s >> 2)`."""
    assert 0 <= s < 32
    return 8 * (s & 3) + (s >> 2)


def hisparse_residency_smem_bytes(max_query_len: int, model_top_k: int, hot_size: int | None = None) -> dict[str, int]:
    """Exact shared-memory requirement from `_check_residency_shared_memory` in `vllm/v1/hisparse/runtime.py`."""
    max_union_rows = max_query_len * model_top_k
    if hot_size is None:
        hot_size = (max_query_len + 1) * model_top_k
    num_chunks = (hot_size + 31) // 32
    required_bytes = 4 * (2 * (max_union_rows + 1) + 3 * num_chunks + 7) + 2 * hot_size
    return {
        "max_query_len": max_query_len,
        "model_top_k": model_top_k,
        "max_union_rows": max_union_rows,
        "device_buffer_size": hot_size,
        "num_chunks": num_chunks,
        "required_smem_bytes": required_bytes,
    }


def plan_flashinfer_checkpoints(
    offsets: list[int], query_lens: list[int]
) -> dict[str, Any]:
    """Exact reproduction of `plan_flashinfer_checkpoints` in `vllm/v1/attention/backends/gdn_attn.py`."""
    every_n_tokens = math.gcd(*offsets)
    cu_starts = [0]
    for query_len in query_lens:
        cu_starts.append(cu_starts[-1] + query_len // every_n_tokens)
    first = len(offsets) - len(query_lens)
    rows = [0] * len(offsets)
    for i, offset in enumerate(offsets[first:]):
        if offset:
            rows[first + i] = cu_starts[i] + offset // every_n_tokens - 1
    return {
        "every_n_tokens": every_n_tokens,
        "num_checkpoints": cu_starts[-1],
        "cu_starts": cu_starts,
        "rows": rows,
    }


def simulate_vllm_v1_sparse_mla_hisparse() -> dict[str, Any]:
    # 1. Multi-generation FlashMLA Sparse KV formats (61 layers, 128K context, batch=32)
    num_layers = 61
    seq_len = 131072  # 128K context
    batch_size = 32
    total_tokens = seq_len * batch_size  # 4,194,304 tokens

    kv_formats = {
        "bf16_mla_baseline": {
            "nope_bytes": 512 * 2,
            "rope_bytes": 64 * 2,
            "scale_bytes": 0,
            "bytes_per_token": 1152,
            "tma_stride_bytes": 1152,
        },
        "deepseek_v3_2_fp8_ds_mla": {
            "nope_bytes": 512,
            "scale_bytes": 16,  # 4x float32 (1 per 128 elements)
            "rope_bytes": 128,  # 64x bfloat16 unquantized
            "bytes_per_token": 656,
            "tma_stride_bytes": 656,
        },
        "deepseek_v4_fp8_split_page": {
            "nope_bytes": 448,
            "rope_bytes": 128,  # 64x bfloat16 unquantized
            "scale_bytes": 8,  # 7x ue8m0 + 1B pad (1 per 64 elements)
            "bytes_per_token": 584,
            "tma_stride_bytes": 576,  # data rows first (576B stride), scale rows after
        },
        "deepseek_v4_1_mxfp8": {
            "nope_and_rope_fp8_bytes": 512,  # all 512 dims (including 64 RoPE) as float8_e4m3
            "scale_bytes": 16,  # 16x ue8m0 (1 per 32 elements, MXFP8)
            "bytes_per_token": 528,
            "tma_stride_bytes": 528,
        },
        "sm100_nvfp4_ds_mla": {
            "nope_e2m1_packed_bytes": 256,  # 512 e2m1 packed 2/byte
            "rope_e4m3_unscaled_bytes": 64,  # 64 float8_e4m3 unscaled RoPE
            "scale_e4m3_permuted_bytes": 32,  # 32 float8_e4m3 (1 per 16 elements, 8x4->4x8 transpose)
            "bytes_per_token": 352,
            "tma_stride_bytes": 352,
        },
    }

    format_metrics = {}
    for name, spec in kv_formats.items():
        bpt = spec["bytes_per_token"]
        total_gib = (total_tokens * num_layers * bpt) / (1024**3)
        compression_vs_bf16_x = round(1152.0 / bpt, 3)
        format_metrics[name] = {
            **spec,
            "cluster_kv_gib_32x128k": round(total_gib, 2),
            "compression_vs_bf16_x": compression_vs_bf16_x,
        }

    # Verify the `8x4 -> 4x8` scale permutation in `nvfp4_ds_mla` (`8 * (s & 3) + (s >> 2)`):
    # A thread dequantizing every 4th scale block (s = 0, 4, 8, 12, 16, 20, 24, 28) reads contiguous bytes 0..7!
    thread0_scale_blocks = [0, 4, 8, 12, 16, 20, 24, 28]
    thread0_byte_offsets = [nvfp4_ds_mla_scale_byte_offset(s) for s in thread0_scale_blocks]

    # 2. HiSparse host-offload & hot-cache residency (`vllm/v1/hisparse/runtime.py`)
    # Model top_k = 2048 tokens; compare standard decode (`max_query_len = 1`) vs MTP spec decode (`max_query_len = 4`)
    hisparse_q1 = hisparse_residency_smem_bytes(max_query_len=1, model_top_k=2048)
    hisparse_q4 = hisparse_residency_smem_bytes(max_query_len=4, model_top_k=2048)
    # Device HBM savings when offloading 128K context to pinned host pool while keeping `(max_query_len + 1) * top_k` hot rows on GPU:
    hot_rows_q4 = hisparse_q4["device_buffer_size"]  # 10,240 rows per request
    hbm_resident_ratio_q4 = hot_rows_q4 / seq_len  # 10,240 / 131,072 = 7.8125%
    hbm_saving_vs_full_128k_x = round(seq_len / hot_rows_q4, 2)  # 12.8x reduction in GPU HBM per active 128K sequence!

    # Group-leader index conversion (`SparseMLAIndexGroup._convert_once`):
    # Without index group: 61 Triton `triton_convert_req_index_to_global_index` launches per step
    # With `SparseMLAIndexGroup`: 1 launch on `side_stream` at `layer_index == 0`
    per_layer_convert_us = 9.5
    unshared_convert_ms = (num_layers * per_layer_convert_us) / 1e3
    grouped_convert_ms = (1 * per_layer_convert_us) / 1e3

    # 3. GatedDeltaNet (`gdn_attn.py`) FlashInfer checkpoint plan
    gdn_plan = plan_flashinfer_checkpoints(
        offsets=[0, 128, 256, 192],
        query_lens=[256, 512, 256],
    )

    return {
        "experiment": "vllm_v1_sparse_mla_hisparse_sim",
        "source_commit": "5285b9b8646d6d0e8b869c9f5c6d277ccfce637b",
        "sparse_mla_kv_formats": format_metrics,
        "nvfp4_scale_permutation": {
            "formula": "8 * (s & 3) + (s >> 2)",
            "thread0_logical_blocks": thread0_scale_blocks,
            "thread0_physical_byte_offsets": thread0_byte_offsets,
            "is_contiguous_8byte_load": thread0_byte_offsets == list(range(8)),
        },
        "hisparse_hot_cache": {
            "single_token_decode_q1": hisparse_q1,
            "spec_decode_mtp_q4": hisparse_q4,
            "int16_max_slot_limit": 32768,
            "hbm_saving_vs_full_128k_x": hbm_saving_vs_full_128k_x,
            "hbm_resident_pct_at_128k_q4": round(hbm_resident_ratio_q4 * 100.0, 4),
            "index_group_convert_saving": {
                "num_layers": num_layers,
                "unshared_convert_ms": round(unshared_convert_ms, 4),
                "grouped_side_stream_convert_ms": round(grouped_convert_ms, 4),
                "convert_launch_reduction_x": num_layers,
            },
        },
        "gdn_flashinfer_checkpoint_plan": gdn_plan,
    }


def _check(results: dict[str, Any]) -> None:
    fmts = results["sparse_mla_kv_formats"]
    assert fmts["deepseek_v3_2_fp8_ds_mla"]["bytes_per_token"] == 656
    assert fmts["deepseek_v4_fp8_split_page"]["bytes_per_token"] == 584
    assert fmts["deepseek_v4_1_mxfp8"]["bytes_per_token"] == 528
    assert fmts["sm100_nvfp4_ds_mla"]["bytes_per_token"] == 352
    perm = results["nvfp4_scale_permutation"]
    assert perm["is_contiguous_8byte_load"] is True
    assert perm["thread0_physical_byte_offsets"] == [0, 1, 2, 3, 4, 5, 6, 7]
    hs = results["hisparse_hot_cache"]
    assert hs["single_token_decode_q1"]["device_buffer_size"] == 4096
    assert hs["spec_decode_mtp_q4"]["device_buffer_size"] == 10240
    assert hs["spec_decode_mtp_q4"]["required_smem_bytes"] == 89892
    assert hs["hbm_saving_vs_full_128k_x"] == 12.8
    gdn = results["gdn_flashinfer_checkpoint_plan"]
    assert gdn["every_n_tokens"] == 64
    assert gdn["num_checkpoints"] == 16
    assert gdn["cu_starts"] == [0, 4, 12, 16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against results.json")
    args = parser.parse_args()

    results = simulate_vllm_v1_sparse_mla_hisparse()
    _check(results)

    out_path = Path(__file__).resolve().parent / "results.json"
    if args.check:
        expected = json.loads(out_path.read_text(encoding="utf-8"))
        assert results == expected, "Mismatch with committed results.json"
        print("Check passed: vllm_v1_sparse_mla_hisparse_sim matches results.json")
    else:
        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
