"""Phase 86 Track 2: SGLang Custom-AllReduce & Out-of-Tree Linkers, vLLM MRV2 2D Staged Writes & RDNA3 W4A8, TRT-LLM Compact Stop Words & QuACK SwiGLU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sglang_allreduce_and_diffusion_sync_free() -> dict[str, Any]:
    """Model SGLang PR #34543 oversized deterministic all-reduce slicing and PR #46047/40595."""
    max_size_bytes = 8 * 1024 * 1024  # 8 MiB custom all-reduce buffer cap
    elem_bytes = 2  # BF16
    chunk_elems = max_size_bytes // elem_bytes  # 4,194,304 elements
    oversized_tokens = 16384
    hidden_dim = 4096
    total_elems = oversized_tokens * hidden_dim  # 67,108,864 elements (128 MiB)

    num_unreg_slices = math.ceil(total_elems / chunk_elems)
    leaked_ipc_handles_before_per_step = 1
    leaked_ipc_handles_after_per_step = 0

    # PR #46047: Video diffusion per-step host sync elimination across 50 denoise steps
    denoise_steps = 50
    h2d_uploads_before = denoise_steps * 6  # RoPE, mask, PDD weight, PDD bias, AdaLN index, coords
    h2d_uploads_after = 6  # Uploaded/cached once on step 0

    return {
        "max_size_bytes": max_size_bytes,
        "oversized_payload_bytes": total_elems * elem_bytes,
        "unreg_slices_per_call": num_unreg_slices,
        "leaked_ipc_handles_before_per_step": leaked_ipc_handles_before_per_step,
        "leaked_ipc_handles_after_per_step": leaked_ipc_handles_after_per_step,
        "diffusion_denoise_steps": denoise_steps,
        "diffusion_h2d_uploads_before": h2d_uploads_before,
        "diffusion_h2d_uploads_after": h2d_uploads_after,
        "diffusion_upload_reduction_ratio": round(
            h2d_uploads_before / h2d_uploads_after, 1
        ),
        "unified_cache_linker_backends": ["mooncake", "mori", "dynamic"],
    }


def simulate_vllm_mrv2_staged_writes_and_rdna3_w4a8() -> dict[str, Any]:
    """Model vLLM PR #62698 2D staged block-table writes, PR #63495 Clef H2D coalescing, PR #60413 RDNA3 W4A8 GEMV."""
    write_block_size = 1024
    staged_max_len = 2500
    num_requests = 64
    chunks_along_dim1 = math.ceil(staged_max_len / write_block_size)
    grid_1d_programs = num_requests
    grid_2d_programs = num_requests * chunks_along_dim1

    # PR #63495: Clef joint schema coalesces 7 separate async_tensor_h2d calls into 1 packed H2D transfer
    clef_h2d_calls_before = 7
    clef_h2d_calls_after = 1

    # PR #60413: RDNA3/RDNA3.5 (gfx11) W4A8 int8-dot MXFP4 GEMV (1 <= M <= 8, group=32)
    # 32 elements reduced via 8 v_dot4_i32_iu8 instructions + 1 FP32 scale multiply
    group_size = 32
    v_dot4_per_group = group_size // 4
    max_w4a8_decode_batch = 8

    return {
        "write_block_size": write_block_size,
        "staged_max_len": staged_max_len,
        "num_write_chunks_dim1": chunks_along_dim1,
        "grid_1d_programs": grid_1d_programs,
        "grid_2d_programs": grid_2d_programs,
        "clef_h2d_calls_before": clef_h2d_calls_before,
        "clef_h2d_calls_after": clef_h2d_calls_after,
        "rdna3_w4a8_group_size": group_size,
        "rdna3_v_dot4_instructions_per_group": v_dot4_per_group,
        "rdna3_max_w4a8_decode_batch": max_w4a8_decode_batch,
    }


def simulate_trtllm_compact_stop_words_and_quack_swiglu() -> dict[str, Any]:
    """Model TensorRT-LLM PR #20679 per-request compact stop-word allocation and PR #19773 QuACK SwiGLU epilogue."""
    # PR #20679: Previously global stop_words_cuda [max_num_stop_words, max_stop_word_length, max_num_sequences]
    # and past_tokens_cuda [max_stop_word_length, max_num_sequences, max_beam_width] grew monotonically
    # to the worst-case request ever seen and never shrank.
    max_num_sequences = 1024
    max_beam_width = 4
    outlier_stop_words = 128
    outlier_stop_len = 256
    int32_bytes = 4

    global_padded_bytes = (
        outlier_stop_words * outlier_stop_len * max_num_sequences * int32_bytes
        + outlier_stop_len * max_num_sequences * max_beam_width * int32_bytes
    )
    # Compact per-slot metadata row is 5 x int64 (40 bytes) + per-request allocation freed on release_request
    active_stop_word_reqs = 4
    avg_rules = 3
    avg_rule_len = 4
    compact_metadata_bytes = max_num_sequences * 5 * 8
    compact_active_alloc_bytes = (
        active_stop_word_reqs
        * (avg_rules * (avg_rule_len + 1) + avg_rule_len * max_beam_width)
        * int32_bytes
    )
    total_compact_bytes = compact_metadata_bytes + compact_active_alloc_bytes

    # PR #19773: Fuse SwiGLU into BF16 gate/up GEMM epilogue via QuACK gemm_act on SM100/SM103
    # Eliminates writing [M, 2I] BF16 intermediate and reading [M, 2I] BF16 in standalone SwiGLU kernel
    m_tokens = 8192
    intermediate_i = 14336
    bf16_bytes = 2
    unfused_intermediate_traffic_bytes = (
        m_tokens * (2 * intermediate_i) * bf16_bytes * 2  # 1 write + 1 read of [M, 2I]
    )
    fused_intermediate_traffic_bytes = 0

    return {
        "global_padded_stop_buffer_bytes": global_padded_bytes,
        "total_compact_stop_buffer_bytes": total_compact_bytes,
        "stop_buffer_memory_reduction_x": round(
            global_padded_bytes / total_compact_bytes, 1
        ),
        "unfused_gate_up_intermediate_traffic_mib": round(
            unfused_intermediate_traffic_bytes / (1024 * 1024), 2
        ),
        "fused_gate_up_intermediate_traffic_mib": fused_intermediate_traffic_bytes,
    }


def run_all() -> dict[str, Any]:
    sglang = simulate_sglang_allreduce_and_diffusion_sync_free()
    vllm = simulate_vllm_mrv2_staged_writes_and_rdna3_w4a8()
    trtllm = simulate_trtllm_compact_stop_words_and_quack_swiglu()
    return {
        "experiment": "sglang_vllm_trtllm_phase86",
        "phase": 86,
        "priority": 95,
        "sglang_allreduce_and_diffusion": sglang,
        "vllm_mrv2_and_rdna3_w4a8": vllm,
        "trtllm_stop_words_and_quack_swiglu": trtllm,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = run_all()
    summary_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not summary_path.exists():
            raise SystemExit("summary.json does not exist")
        if summary_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("summary.json is out of date")
        return 0

    summary_path.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
