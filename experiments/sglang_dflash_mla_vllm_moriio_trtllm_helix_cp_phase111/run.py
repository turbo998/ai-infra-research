"""Phase 111 Track B stdlib experiment:
SGLang DFlash SpecForge MLA Drafts & Async NaN Probe,
vLLM ROCm RDNA PagedAttention 2D, SM90 GDN Tile, MORI IO Strided Block Len & Overlaid KV Copy,
and TensorRT-LLM Helix CP Attention-DP Consensus & MiniMax-H3 Fused QK-Norm+RoPE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_sglang_dflash_mla_and_async_nan() -> dict[str, Any]:
    """Model SGLang DFlash MLA draft serving & set_mla_kv_buffer_prefix_valid (b068417bfcc1),
    fused single-kernel async NaN check (ee5443f53026), NUMA physical CPU budget (302d5e7b8ca9),
    and DiT BCG caller-allocated output buffers (c43abd6d9e38).
    """
    # set_mla_kv_buffer_prefix_valid: rows past commit_lens route to slot 0
    loc_2d = [
        [101, 102, 103, 104],
        [201, 202, 203, 204],
    ]
    commit_lens = [3, 1]
    flattened_loc: list[int] = []
    for row, c_len in zip(loc_2d, commit_lens):
        for col_idx, slot in enumerate(row):
            flattened_loc.append(slot if col_idx < c_len else 0)

    # Fused async NaN check (ee5443f53026): replaces isnan -> any -> _assert_async (3 kernels) with 1 Triton kernel
    eager_kernels_per_probe = 3
    fused_kernels_per_probe = 1
    probes_per_step = 4
    kernel_launches_saved_per_step = (eager_kernels_per_probe - fused_kernels_per_probe) * probes_per_step

    return {
        "flattened_loc_with_slot0_mask": flattened_loc,
        "valid_committed_slots": sum(1 for s in flattened_loc if s != 0),
        "masked_slot0_writes": sum(1 for s in flattened_loc if s == 0),
        "eager_kernels_per_probe": eager_kernels_per_probe,
        "fused_kernels_per_probe": fused_kernels_per_probe,
        "kernel_launches_saved_per_step": kernel_launches_saved_per_step,
    }


def simulate_vllm_rdna_gdn_moriio_and_overlaid_kv() -> dict[str, Any]:
    """Model vLLM ROCm RDNA paged attention 2d launch (9af236a33281),
    SM90 GDN decode BV=4 for FP32 state (94f56ce92cc4),
    MORI IO strided block_len (3d81a510f9eb),
    and copy_kv_cache_blocks_inplace geometry-keyed overlaid KV views (8701c9b7513d).
    """
    # RDNA launch config for kernel_paged_attention_2d (9af236a33281)
    def rdna_launch_config(triton_block_size: int, head_size_pow2: int) -> dict[str, int]:
        tile_size = triton_block_size * head_size_pow2
        if triton_block_size >= 32 or tile_size >= 4096:
            num_warps = 16 if tile_size >= 8192 else 8 if tile_size >= 4096 else 4
            return {"num_warps": num_warps, "num_stages": 1}
        return {"num_warps": 4, "num_stages": 2}

    rdna_tile_4096 = rdna_launch_config(32, 128)
    rdna_tile_8192 = rdna_launch_config(64, 128)

    # SM90 GDN packed decode BV selection (94f56ce92cc4)
    sm90_gdn_bv_fp32 = 4
    sm90_gdn_bv_bf16 = 32

    # MORI IO stride-aware block_len (3d81a510f9eb)
    # Example padded MLA block: shape=(num_blocks, 1, 64, 576), stride=(38912, 38912, 608, 1), elem=1
    shape = (100, 1, 64, 576)
    stride = (38912, 38912, 608, 1)
    element_size = 1
    naive_block_len = shape[2] * shape[1] * shape[3] * element_size  # 36,864 B (misses row padding!)
    strided_block_len = (1 + sum((s - 1) * st for s, st in zip(shape[1:], stride[1:]))) * element_size  # 38,880 B

    # Overlaid Mamba + Attention KV block copy (8701c9b7513d)
    block_stride = 64
    mamba_state_bytes = 24
    # Old key (device, data_ptr) only copied first view (24B), leaving 40B uncopied!
    old_copied_bytes_per_block = mamba_state_bytes
    # New key includes (device, data_ptr, shape, stride, element_size) -> copies both views -> full 64B covered
    new_copied_bytes_per_block = block_stride
    tail_corruption_bytes_prevented = new_copied_bytes_per_block - old_copied_bytes_per_block

    return {
        "rdna_tile_4096": rdna_tile_4096,
        "rdna_tile_8192": rdna_tile_8192,
        "sm90_gdn_bv_fp32": sm90_gdn_bv_fp32,
        "sm90_gdn_bv_bf16": sm90_gdn_bv_bf16,
        "naive_block_len": naive_block_len,
        "strided_block_len": strided_block_len,
        "old_copied_bytes_per_block": old_copied_bytes_per_block,
        "new_copied_bytes_per_block": new_copied_bytes_per_block,
        "tail_corruption_bytes_prevented": tail_corruption_bytes_prevented,
    }


def simulate_trtllm_helix_cp_and_minimax_rope() -> dict[str, Any]:
    """Model TensorRT-LLM align_gen_admission_across_cp under Helix CP + Attention-DP (94348c4de6eb)
    and MiniMax-H3 BF16 fused per-head QK-norm + partial split-half RoPE (af32051e1926).
    """
    # Helix CP rank admission consensus: rank 0 admits {r1, r2, r3}, rank 1 admits {r1, r2, r4}
    # R1 routes to dp=0 on both; R2 routes to dp=0 on rank 0 but dp=1 on rank 1 (divergent DP route!)
    cp_rank0_admitted = {"req_1": 0, "req_2": 0, "req_3": 1}
    cp_rank1_admitted = {"req_1": 0, "req_2": 1, "req_4": 1}

    consensus_admitted = [
        req_id
        for req_id, dp0 in cp_rank0_admitted.items()
        if req_id in cp_rank1_admitted and cp_rank1_admitted[req_id] == dp0
    ]
    deferred_for_retry = sorted(
        (set(cp_rank0_admitted) | set(cp_rank1_admitted)) - set(consensus_admitted)
    )

    # MiniMax-H3 fused QK-norm + partial split-half RoPE (af32051e1926)
    eager_bf16_roundings = 5
    fused_bf16_roundings = 1
    rel_l2_vs_fp32 = 3.5e-3
    lpips_threshold_unified = 0.30

    return {
        "consensus_admitted": consensus_admitted,
        "deferred_for_retry": deferred_for_retry,
        "eager_bf16_roundings": eager_bf16_roundings,
        "fused_bf16_roundings": fused_bf16_roundings,
        "rel_l2_vs_fp32": rel_l2_vs_fp32,
        "lpips_threshold_unified": lpips_threshold_unified,
    }


def run_all() -> dict[str, Any]:
    return {
        "phase": 111,
        "track": "B",
        "title": "SGLang DFlash SpecForge MLA & Async NaN Probe, vLLM RDNA/MORI-IO/Overlaid KV, and TRT-LLM Helix CP Consensus & MiniMax-H3 Fused RoPE",
        "sglang": simulate_sglang_dflash_mla_and_async_nan(),
        "vllm": simulate_vllm_rdna_gdn_moriio_and_overlaid_kv(),
        "trtllm": simulate_trtllm_helix_cp_and_minimax_rope(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert summary["sglang"]["masked_slot0_writes"] == 4
        assert summary["sglang"]["kernel_launches_saved_per_step"] == 8
        assert summary["vllm"]["rdna_tile_8192"] == {"num_warps": 16, "num_stages": 1}
        assert summary["vllm"]["strided_block_len"] == 38880
        assert summary["vllm"]["tail_corruption_bytes_prevented"] == 40
        assert summary["trtllm"]["consensus_admitted"] == ["req_1"]
        assert summary["trtllm"]["fused_bf16_roundings"] == 1
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
