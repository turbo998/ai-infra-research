"""Phase 65 Track C deterministic experiment:
1. TransformerEngine Fused `parallel_cross_entropy(..., return_argmax=True)` with NaN-Safe Tie-Break (39a8b879df / #3671)
   + Grouped MXFP8 Varying-Last-Dim Scale Base Fix (76ae4b0981 / #3599):
   - Models single-pass forward cross-entropy + argmax over TP-sharded vocabulary without a second
     full-logits HBM read, including `_fold_block_argmax` where NaN counts as maximum and first
     global tie/NaN wins across TP ranks.
   - Models `tensor_base_for_scales = is_single_tensor ? tensor_start_offset : 0` to eliminate
     double-offset scale pointer corruption in multi-tensor grouped MXFP8 quantization.
2. LMCache 2-D Batch Copy (`cudaMemcpy3DBatchAsync`) for Block-Outermost KV Layouts (0d441be069 / #5506):
   - Models `uniform_layer_pitch` detection for block-outermost KV caches (e.g., vLLM `BLNHC` for
     DeepSeek-V4) where `pitch >= block_bytes` and `(nl - 1) * pitch + block_bytes <= block_stride`,
     collapsing `nl` 1-D copy descriptors per block into a single 2-D copy (`width = block_bytes`, `height = nl`).
3. NIXL Multipart S3-over-RDMA for >4 GiB Objects (99d469be5d / #2406) & GPUNetIO Peer-VRAM Two-SGE WRITE (14e329032b / #2172):
   - Models splitting large VRAM arenas (9 GiB and 64 GiB) into `<= 4 GiB - 64 KiB` (`CUOBJ_MAX_MEMORY_REG_SIZE`)
     RDMA parts with independent `cuMemObjGetRDMAToken` tokens and ranged GETs, plus GPUNetIO 2-SGE WQE coalescing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CUOBJ_MAX_MEMORY_REG_SIZE = (4 * 1024 * 1024 * 1024) - (64 * 1024)  # 4 GiB - 64 KiB = 4,294,901,760 bytes


def simulate_te_parallel_ce_argmax_and_mxfp8() -> dict[str, Any]:
    # Vocabulary = 128,256 across TP = 8 ranks (16,032 cols/rank), batch_tokens = 4096, BF16/FP32
    tokens = 4096
    vocab_size = 128_256
    tp_size = 8
    cols_per_rank = vocab_size // tp_size
    bytes_per_logit = 2  # BF16

    # Baseline: parallel_cross_entropy reads logits once + standalone torch.argmax reads logits a second time
    logits_bytes_per_rank = tokens * cols_per_rank * bytes_per_logit
    extra_argmax_hbm_read_saved_bytes = logits_bytes_per_rank
    # Extra all-gather payload: 1 FP32 column per token per rank
    extra_allgather_bytes_per_rank = tokens * 4

    # NaN-safe & tie-breaking verification (`_fold_block_argmax` vs naive `tl.max`)
    # Case A: [8.0, -10.0, NaN, -10.0] -> naive non-NaN-safe comparison returns index 1 or 0; torch.argmax returns 2 (first NaN)
    # Case B: [3.5, 9.0, 9.0, 2.0, NaN, NaN] -> first NaN is index 4
    # Case C: [1.0, 7.5, 7.5, 4.0] across 2 TP ranks (cols_per_rank=2): rank 0 has [1.0, 7.5] (local idx 1 -> global 1),
    #         rank 1 has [7.5, 4.0] (local idx 0 -> global 2). Strict `>` keeps rank 0's global index 1!
    def fold_row_argmax(row_vals: list[float]) -> int:
        run_max = float("-inf")
        run_idx = 0
        run_nan = False
        for idx, val in enumerate(row_vals):
            is_nan = math.isnan(val)
            if not run_nan:
                if is_nan:
                    run_idx = idx
                    run_nan = True
                elif val > run_max:
                    run_max = val
                    run_idx = idx
        return run_idx

    nan_row_idx = fold_row_argmax([8.0, -10.0, float("nan"), -10.0])
    multi_nan_row_idx = fold_row_argmax([3.5, 9.0, 9.0, 2.0, float("nan"), float("nan")])
    tie_row_idx = fold_row_argmax([1.0, 7.5, 7.5, 4.0])

    # Grouped MXFP8 scale pointer fix (#3599):
    # For multi-tensor grouped quantization (`is_single_tensor = False`), `scales_rowwise` already adds `tensor_base / 32`.
    # Buggy code also set `tensor_base_for_scales = tensor_base`, double-counting the offset!
    tensor_base = 65_536
    scale_dim_x = 32
    buggy_scale_elem_offset = (tensor_base // scale_dim_x) + (tensor_base // scale_dim_x)
    fixed_scale_elem_offset = (tensor_base // scale_dim_x) + 0

    return {
        "tokens": tokens,
        "vocab_size": vocab_size,
        "tp_size": tp_size,
        "saved_second_logits_read_bytes_per_rank": extra_argmax_hbm_read_saved_bytes,
        "extra_allgather_index_bytes_per_rank": extra_allgather_bytes_per_rank,
        "net_memory_traffic_saved_ratio": round(
            extra_argmax_hbm_read_saved_bytes / extra_allgather_bytes_per_rank, 1
        ),
        "nan_row_argmax_index": nan_row_idx,
        "multi_nan_first_wins_index": multi_nan_row_idx,
        "tie_break_first_wins_index": tie_row_idx,
        "grouped_mxfp8_buggy_scale_offset": buggy_scale_elem_offset,
        "grouped_mxfp8_fixed_scale_offset": fixed_scale_elem_offset,
    }


def simulate_lmcache_2d_direct_copy(
    num_layers: int = 61,
    num_blocks_in_chunk: int = 32,
    kv_planes: int = 1,
    block_bytes: int = 36_864,
) -> dict[str, Any]:
    # Case 1: Block-outermost layout (vLLM BLNHC for DeepSeek-V4 MLA)
    # Each block packs `num_layers` pages side by side: pitch = block_bytes, block_stride = num_layers * block_bytes
    blnhc_pitch = block_bytes
    blnhc_block_stride = num_layers * block_bytes
    blnhc_fits = (
        blnhc_pitch >= block_bytes
        and (num_layers - 1) * blnhc_pitch + block_bytes <= blnhc_block_stride
    )
    detected_pitch_blnhc = blnhc_pitch if blnhc_fits else 0

    # Case 2: Layer-outermost layout (standard NLBHC / LBNHC)
    # Consecutive layers are `total_blocks * block_bytes` apart (> block_stride)
    total_engine_blocks = 4096
    layer_outermost_pitch = total_engine_blocks * block_bytes
    layer_outermost_block_stride = block_bytes
    layer_outermost_fits = (
        layer_outermost_pitch >= block_bytes
        and (num_layers - 1) * layer_outermost_pitch + block_bytes <= layer_outermost_block_stride
    )
    detected_pitch_layer_outermost = layer_outermost_pitch if layer_outermost_fits else 0

    entries_1d_before = kv_planes * num_layers * num_blocks_in_chunk
    entries_2d_after_blnhc = kv_planes * num_blocks_in_chunk

    return {
        "num_layers": num_layers,
        "num_blocks_in_chunk": num_blocks_in_chunk,
        "block_bytes": block_bytes,
        "blnhc_uniform_layer_pitch": detected_pitch_blnhc,
        "layer_outermost_uniform_layer_pitch": detected_pitch_layer_outermost,
        "batch_memcpy_entries_1d_before": entries_1d_before,
        "batch_memcpy_entries_2d_after": entries_2d_after_blnhc,
        "descriptor_reduction_factor_x": round(entries_1d_before / entries_2d_after_blnhc, 2),
    }


def simulate_nixl_s3_rdma_multipart_and_gpunetio() -> dict[str, Any]:
    def plan_s3_rdma_object(size_gib: int) -> dict[str, Any]:
        total_bytes = size_gib * (1024**3)
        num_parts = math.ceil(total_bytes / CUOBJ_MAX_MEMORY_REG_SIZE)
        parts = []
        offset = 0
        for part_idx in range(num_parts):
            part_size = min(CUOBJ_MAX_MEMORY_REG_SIZE, total_bytes - offset)
            parts.append(
                {
                    "part_number": part_idx + 1,
                    "offset_bytes": offset,
                    "size_bytes": part_size,
                    "within_cuobj_max": part_size <= CUOBJ_MAX_MEMORY_REG_SIZE,
                }
            )
            offset += part_size
        return {
            "size_gib": size_gib,
            "total_bytes": total_bytes,
            "uses_multipart": total_bytes > CUOBJ_MAX_MEMORY_REG_SIZE,
            "num_parts": num_parts,
            "last_part_bytes": parts[-1]["size_bytes"],
            "all_parts_within_cuobj_limit": all(p["within_cuobj_max"] for p in parts),
            "total_bytes_reconstructed": sum(p["size_bytes"] for p in parts) == total_bytes,
        }

    obj_3gib = plan_s3_rdma_object(3)
    obj_9gib = plan_s3_rdma_object(9)
    obj_64gib = plan_s3_rdma_object(64)

    # GPUNetIO two-SGE WRITE coalescing (#2172):
    # Pairs of eligible non-contiguous descriptors sharing a remote contiguous range combine into 1 WQE with 2 SGEs
    descriptors = 64
    wqes_single_sge = descriptors
    wqes_two_sge = descriptors // 2

    return {
        "cuobj_max_memory_reg_size_bytes": CUOBJ_MAX_MEMORY_REG_SIZE,
        "transfer_3gib": obj_3gib,
        "transfer_9gib": obj_9gib,
        "transfer_64gib": obj_64gib,
        "gpunetio_descriptors": descriptors,
        "gpunetio_wqes_before": wqes_single_sge,
        "gpunetio_wqes_two_sge_after": wqes_two_sge,
        "gpunetio_peer_vram_supported": True,
    }


def run_experiment() -> dict[str, Any]:
    te = simulate_te_parallel_ce_argmax_and_mxfp8()
    lmcache = simulate_lmcache_2d_direct_copy()
    nixl = simulate_nixl_s3_rdma_multipart_and_gpunetio()

    return {
        "experiment": "te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma",
        "phase": 65,
        "priority": 74,
        "track": "Track C — Distributed Training, KV Cache & RDMA Storage",
        "transformer_engine": te,
        "lmcache_2d_copy": lmcache,
        "nixl_s3_rdma_and_gpunetio": nixl,
        "invariants": {
            "te_nan_counts_as_max_and_first_wins": (
                te["nan_row_argmax_index"] == 2 and te["multi_nan_first_wins_index"] == 4
            ),
            "te_tie_break_first_rank_wins": te["tie_break_first_wins_index"] == 1,
            "te_mxfp8_double_offset_eliminated": (
                te["grouped_mxfp8_fixed_scale_offset"] * 2 == te["grouped_mxfp8_buggy_scale_offset"]
            ),
            "lmcache_blnhc_uses_2d_copy": lmcache["blnhc_uniform_layer_pitch"] == 36_864,
            "lmcache_layer_outermost_falls_back_to_1d": lmcache["layer_outermost_uniform_layer_pitch"] == 0,
            "nixl_9gib_uses_3_parts": nixl["transfer_9gib"]["num_parts"] == 3,
            "nixl_64gib_uses_17_parts": nixl["transfer_64gib"]["num_parts"] == 17,
            "nixl_all_parts_valid": (
                nixl["transfer_9gib"]["all_parts_within_cuobj_limit"]
                and nixl["transfer_64gib"]["all_parts_within_cuobj_limit"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against summary.json")
    args = parser.parse_args()

    result = run_experiment()
    summary_path = Path(__file__).with_name("summary.json")

    if args.check:
        expected = json.loads(summary_path.read_text(encoding="utf-8"))
        assert result == expected, "Summary mismatch!"
        assert all(result["invariants"].values()), "Invariant failed!"
        print("OK: te-parallel-ce-argmax-lmcache-2d-copy-nixl-s3-rdma verified.")
        return

    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
