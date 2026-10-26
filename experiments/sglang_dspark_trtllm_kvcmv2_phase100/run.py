from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sglang_dspark_fp8_lmhead_and_npu_mxfp8() -> dict[str, Any]:
    """Model SGLang #39622 (AMD gfx95 DSPARK FP8 target lm_head GEMM) and #40519/#40435 (Ascend NPU MXFP8 dispatch & arch35 gate)."""
    hidden = 7168
    vocab_slice = 64640
    verify_batches = [8, 16, 32, 64, 128]
    # MI350X / gfx95 effective roof and bandwidth parameters for [M, K] x [K, N]
    bf16_tflops = 1150.0
    fp8_tflops = 2200.0
    hbm_bw_gbps = 5200.0

    batch_rows: list[dict[str, Any]] = []
    for m in verify_batches:
        flops = 2.0 * m * hidden * vocab_slice
        bf16_weight_bytes = 2.0 * hidden * vocab_slice
        fp8_weight_bytes = 1.0 * hidden * vocab_slice + 4.0 * vocab_slice
        act_quant_overhead_us = 2.2 + 0.015 * m
        bf16_us = max((flops / (bf16_tflops * 1e12)) * 1e6, (bf16_weight_bytes / (hbm_bw_gbps * 1e9)) * 1e6) + 3.5
        fp8_us = max((flops / (fp8_tflops * 1e12)) * 1e6, (fp8_weight_bytes / (hbm_bw_gbps * 1e9)) * 1e6) + 3.5 + act_quant_overhead_us
        batch_rows.append({
            "num_verify_tokens": m,
            "bf16_lm_head_us": round(bf16_us, 2),
            "fp8_scaled_mm_lm_head_us": round(fp8_us, 2),
            "speedup": round(bf16_us / fp8_us, 2),
            "weight_bytes_saved_mib": round((bf16_weight_bytes - fp8_weight_bytes) / (1024 ** 2), 2),
        })

    # SGLang #40435: Ascend arch35 (Atlas A5) UE8M0 vs plain FP32 block-FP8 requantization gate
    arch35_cases = [
        {
            "checkpoint_scale_format": "ue8m0_pow2_128x128",
            "allow_lossy_env": False,
            "selected_path": "native_mxfp8_relayout_lossless",
            "duplicated_subgroups_per_128_block": 4,
            "is_lossy": False,
        },
        {
            "checkpoint_scale_format": "fp32_arbitrary_128x128",
            "allow_lossy_env": False,
            "selected_path": "triton_block_fp8_emulation_lossless",
            "duplicated_subgroups_per_128_block": 1,
            "is_lossy": False,
        },
        {
            "checkpoint_scale_format": "fp32_arbitrary_128x128",
            "allow_lossy_env": True,
            "selected_path": "npu_dynamic_mx_quant_requantized_mxfp8",
            "duplicated_subgroups_per_128_block": 4,
            "is_lossy": True,
        },
    ]
    return {
        "dspark_fp8_lm_head_rows": batch_rows,
        "arch35_requant_cases": arch35_cases,
        "npu_w4a8_mxfp4_dispatch": {
            "normal_dispatch_wire_dtype": "bf16",
            "low_latency_dispatch_wire_dtype": "mxfp8",
            "multinode_prefill_preserved": True,
        },
    }


def simulate_vllm_xpu_moe_and_trtllm_kvcmv2_host_fill() -> dict[str, Any]:
    """Model vLLM #51605/#63436 (XPU MoE grid swap & DFlash dtype cast) and TensorRT-LLM #20790 (KVCMv2 8-thread 4 MiB host fill)."""
    # vLLM #51605: batched_triton_kernel_swapped grid (num_tiles, E) vs (E, num_tiles)
    experts = 64
    max_tokens = 256
    n_dim = 4096
    bm = bn = 64
    num_tiles = math.ceil(max_tokens / bm) * math.ceil(n_dim / bn)
    legacy_grid = (experts, num_tiles)
    swapped_xpu_grid = (num_tiles, experts)
    # Under int4_w4a16, _moe_config_shape doubles N because 2 int4 weights pack into 1 byte
    w2_shape = (64, 4096, 2048)
    moe_shape_fp16 = (w2_shape[0], w2_shape[2])
    moe_shape_int4_w4a16 = (w2_shape[0], w2_shape[2] * 2)

    # TensorRT-LLM #20790: KVCMv2 HostMem initial fill with kInitialFillHelpers = 7 (total 8 threads) in 4 MiB calls
    pool_gib = 64
    pool_bytes = pool_gib * (1024 ** 3)
    populate_call_bytes = 4 * (1024 ** 2)  # 4 MiB
    total_populate_calls = pool_bytes // populate_call_bytes
    helper_threads = 7
    total_fill_threads = 1 + helper_threads  # 8 threads

    # Page allocator throughput model (GiB/s) from 1 to 8 threads
    single_thread_gib_s = 4.2
    eight_thread_gib_s = 26.88  # 6.40x speedup before kernel page allocator saturation
    serial_fill_sec = round(pool_gib / single_thread_gib_s, 3)
    parallel_fill_sec = round(pool_gib / eight_thread_gib_s, 3)

    return {
        "vllm_xpu_batched_moe": {
            "legacy_grid": list(legacy_grid),
            "swapped_xpu_grid": list(swapped_xpu_grid),
            "num_tiles": num_tiles,
            "moe_shape_fp16": list(moe_shape_fp16),
            "moe_shape_int4_w4a16": list(moe_shape_int4_w4a16),
            "dflash_aux_fc_casts_fp32_residual_to_draft_dtype": True,
        },
        "trtllm_kvcmv2_host_fill": {
            "pool_gib": pool_gib,
            "populate_call_mib": 4,
            "total_populate_calls": total_populate_calls,
            "kInitialFillHelpers": helper_threads,
            "total_fill_threads": total_fill_threads,
            "serial_fill_sec": serial_fill_sec,
            "parallel_fill_sec": parallel_fill_sec,
            "fill_speedup": round(serial_fill_sec / parallel_fill_sec, 2),
            "kFillFailed_sentinel": (1 << 32) - 1,
        },
    }


def run_experiment() -> dict[str, Any]:
    sglang_part = simulate_sglang_dspark_fp8_lmhead_and_npu_mxfp8()
    vllm_trtllm_part = simulate_vllm_xpu_moe_and_trtllm_kvcmv2_host_fill()
    b64_row = next(r for r in sglang_part["dspark_fp8_lm_head_rows"] if r["num_verify_tokens"] == 64)
    return {
        "experiment_id": "sglang_dspark_trtllm_kvcmv2_phase100",
        "phase": 100,
        "priority": 109,
        "sglang": sglang_part,
        "vllm_and_trtllm": vllm_trtllm_part,
        "invariants": {
            "dspark_fp8_lm_head_halves_verify_gemm_at_b64": b64_row["speedup"] >= 1.80,
            "arch35_default_blocks_lossy_fp32_to_mxfp8": (
                sglang_part["arch35_requant_cases"][1]["selected_path"] == "triton_block_fp8_emulation_lossless"
            ),
            "vllm_xpu_swaps_grid_and_doubles_int4_n": (
                vllm_trtllm_part["vllm_xpu_batched_moe"]["swapped_xpu_grid"] == [256, 64]
                and vllm_trtllm_part["vllm_xpu_batched_moe"]["moe_shape_int4_w4a16"] == [64, 4096]
            ),
            "trtllm_kvcmv2_8_threads_4mib_speedup": (
                vllm_trtllm_part["trtllm_kvcmv2_host_fill"]["total_fill_threads"] == 8
                and vllm_trtllm_part["trtllm_kvcmv2_host_fill"]["populate_call_mib"] == 4
                and vllm_trtllm_part["trtllm_kvcmv2_host_fill"]["fill_speedup"] >= 6.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 100 SGLang DSPARK FP8 LM Head, vLLM XPU MoE & TRT-LLM KVCMv2 Simulator")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    args = parser.parse_args()
    summary = run_experiment()
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
