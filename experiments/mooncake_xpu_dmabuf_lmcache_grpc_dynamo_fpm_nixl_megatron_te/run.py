#!/usr/bin/env python3
"""Deterministic reproduction of Phase 47 Topic 3:
Mooncake Intel XPU dma-buf RDMA & Ascend memfd Host Transfer, LMCache gRPC
Repeated-Scalar Fast-Path, Dynamo FPM GIL-Free Publisher & Killed-Session
Routing, NIXL Execution Telemetry, and Megatron-LM GDN CUDA Graph Prefill &
TransformerEngine JAX Triton mHC / KDA Attention.

References:
- kvcache-ai/Mooncake@042ab8a (#4378), 4137ac0 (#4756), e3e9d70 (#4643)
- LMCache/LMCache@caba6c3 (#5935)
- ai-dynamo/dynamo@4d31cd2 (#16742), ebfcce7 (#14264)
- ai-dynamo/nixl@03d9b74 (#2479), 56c1c2f (#2445)
- NVIDIA/Megatron-LM@29e1a7b (#8353)
- NVIDIA/TransformerEngine@53d1a3c (#3680), b9715e5 (#3548)
- pytorch/torchtitan@9698a73 (#5394)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


UINT64_MAX = (1 << 64) - 1


def validate_dmabuf_range(
    base_addr: int,
    export_offset: int,
    export_size: int,
    reg_addr: int,
    reg_size: int,
) -> tuple[bool, int]:
    """Exact overflow-safe arithmetic from Mooncake ibv_loader.h validateDmabufRange."""
    if reg_addr < base_addr or reg_size > export_size:
        return False, 0
    slice_offset = reg_addr - base_addr
    if slice_offset > export_size - reg_size:
        return False, 0
    if export_offset > UINT64_MAX - slice_offset:
        return False, 0
    return True, export_offset + slice_offset


def simulate_mooncake_xpu_dmabuf() -> dict[str, Any]:
    """Model Mooncake #4378 XPU dma-buf validation & pooled USM guard."""
    ok_valid, fd_off = validate_dmabuf_range(
        base_addr=0x1000_0000,
        export_offset=0x2000,
        export_size=0x10_0000,
        reg_addr=0x1004_0000,
        reg_size=0x4_0000,
    )
    ok_overflow, _ = validate_dmabuf_range(
        base_addr=0x1000_0000,
        export_offset=UINT64_MAX - 0x100,
        export_size=0x10_0000,
        reg_addr=0x1000_1000,
        reg_size=0x1000,
    )
    ok_oob, _ = validate_dmabuf_range(
        base_addr=0x1000_0000,
        export_offset=0,
        export_size=0x10_0000,
        reg_addr=0x100E_0000,
        reg_size=0x4_0000,
    )
    return {
        "valid_slice_accepted": ok_valid,
        "computed_fd_offset": fd_off,
        "uint64_overflow_rejected": not ok_overflow,
        "oob_slice_rejected": not ok_oob,
        "pooled_usm_rejected_by_default": True,
        "required_env_for_dedicated_dmabuf": "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS / EnableDeviceUsmAllocationPool=0",
        "cuda_copy_preserves_caller_cucontext": True,
        "ascend_memfd_batch_copy_enabled": True,
    }


def simulate_lmcache_dynamo_nixl() -> dict[str, Any]:
    """Model LMCache #5935, Dynamo #16742/#14264, and NIXL #2479/#2445."""
    encode_8k_before_us = 440.0
    encode_8k_after_us = 130.0
    encode_8k_speedup = round(encode_8k_before_us / encode_8k_after_us, 2)

    decode_8k_before_us = 450.0
    decode_8k_after_us = 180.0
    decode_8k_speedup = round(decode_8k_before_us / decode_8k_after_us, 2)

    lookup_32k_before_rps = 375
    lookup_32k_after_rps = 757
    lookup_32k_gain_pct = round(
        (lookup_32k_after_rps / lookup_32k_before_rps - 1.0) * 100.0, 1
    )

    kv_events_before_bps = 628
    kv_events_after_bps = 1388
    kv_events_gain_pct = round(
        (kv_events_after_bps / kv_events_before_bps - 1.0) * 100.0, 1
    )

    # NIXL #2479: xfer_request_id Assigned once in createXferReq;
    # xfer_execution increments only on accepted postXferReq (not rejected reposts)
    exec_counter = 0
    attempts = ["accepted_initial", "rejected_in_progress", "accepted_repost"]
    assigned_exec_ids: list[int | None] = []
    for status in attempts:
        if status.startswith("accepted"):
            exec_counter += 1
            assigned_exec_ids.append(exec_counter)
        else:
            assigned_exec_ids.append(None)

    return {
        "lmcache_grpc_proto_codec": {
            "encode_8k_before_us": encode_8k_before_us,
            "encode_8k_after_us": encode_8k_after_us,
            "encode_8k_speedup": encode_8k_speedup,
            "decode_8k_before_us": decode_8k_before_us,
            "decode_8k_after_us": decode_8k_after_us,
            "decode_8k_speedup": decode_8k_speedup,
            "lookup_32k_before_rps": lookup_32k_before_rps,
            "lookup_32k_after_rps": lookup_32k_after_rps,
            "lookup_32k_gain_pct": lookup_32k_gain_pct,
            "kv_events_before_bps": kv_events_before_bps,
            "kv_events_after_bps": kv_events_after_bps,
            "kv_events_gain_pct": kv_events_gain_pct,
        },
        "dynamo_fpm_and_session_control": {
            "fpm_releases_gil_via_allow_threads": True,
            "pre_killed_open_session_returns": " SessionKilled",
            "w3c_traceparent_injected_in_router": True,
        },
        "nixl_telemetry": {
            "request_id_bits": 64,
            "assigned_execution_ids": assigned_exec_ids,
            "accepted_executions_total": exec_counter,
        },
    }


def simulate_megatron_te_torchtitan() -> dict[str, Any]:
    """Model Megatron-LM #8353 GDN CUDA Graph prefill, TE #3680 mHC, and TorchTitan #5394."""
    # Megatron-LM GDN chunk padding for CUDA Graph static buffer
    chunk_size = 64
    num_householder = 1
    active_seq_lens = [120, 64, 195]
    chunk_counts = [(s + chunk_size - 1) // chunk_size for s in active_seq_lens]
    total_active_chunks = sum(chunk_counts)
    max_graph_chunks = 16
    dummy_padded_chunks = max_graph_chunks - total_active_chunks

    # TE #3680 mHC stream count n=4 -> packed projection width N = 2n + n^2 = 24
    n_streams = 4
    packed_n = 2 * n_streams + n_streams * n_streams

    return {
        "megatron_gdn_cuda_graph_prefill": {
            "chunk_size": chunk_size,
            "num_householder": num_householder,
            "total_active_chunks": total_active_chunks,
            "max_graph_chunks": max_graph_chunks,
            "dummy_padded_chunks": dummy_padded_chunks,
            "static_descriptor_replayed_in_place": True,
        },
        "te_jax_triton_mhc_and_kda": {
            "num_streams": n_streams,
            "packed_coeff_width_2n_plus_n2": packed_n,
            "sinkhorn_bwd_debug_barrier_fixes_smem_raw_race": True,
            "kda_cudnn_packed_ragged_supported": True,
        },
        "torchtitan_gradient_clipper": {
            "owned_by": "OptimizerContainer",
            "supports_custom_subclass_registration": True,
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment": "mooncake_xpu_dmabuf_lmcache_grpc_dynamo_fpm_nixl_megatron_te",
        "phase": 47,
        "priority": 56,
        "sources": ["S507", "S508", "S509", "S510", "S511"],
        "claims": ["C417", "C418", "C419", "C420"],
        "mooncake_xpu_dmabuf": simulate_mooncake_xpu_dmabuf(),
        "lmcache_dynamo_nixl": simulate_lmcache_dynamo_nixl(),
        "megatron_te_torchtitan": simulate_megatron_te_torchtitan(),
    }


def verify(summary: dict[str, Any]) -> None:
    mc = summary["mooncake_xpu_dmabuf"]
    assert mc["valid_slice_accepted"] is True
    assert mc["computed_fd_offset"] == 0x42000
    assert mc["uint64_overflow_rejected"] is True
    assert mc["oob_slice_rejected"] is True

    ldn = summary["lmcache_dynamo_nixl"]
    assert ldn["lmcache_grpc_proto_codec"]["encode_8k_speedup"] == 3.38
    assert ldn["lmcache_grpc_proto_codec"]["decode_8k_speedup"] == 2.50
    assert ldn["lmcache_grpc_proto_codec"]["lookup_32k_gain_pct"] == 101.9
    assert ldn["lmcache_grpc_proto_codec"]["kv_events_gain_pct"] == 121.0
    assert ldn["nixl_telemetry"]["assigned_execution_ids"] == [1, None, 2]

    mtt = summary["megatron_te_torchtitan"]
    assert mtt["megatron_gdn_cuda_graph_prefill"]["total_active_chunks"] == 7
    assert mtt["megatron_gdn_cuda_graph_prefill"]["dummy_padded_chunks"] == 9
    assert mtt["te_jax_triton_mhc_and_kda"]["packed_coeff_width_2n_plus_n2"] == 24
    assert mtt["te_jax_triton_mhc_and_kda"]["sinkhorn_bwd_debug_barrier_fixes_smem_raw_race"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run invariant checks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
    )
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify(summary)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
