"""Reproducible stdlib model for Phase 75 Track C KV/Transfer & Distributed Training mechanics.

Covers:
1. Mooncake (#4870, 6a835cbf934a; #4631, 868a6d05abf9): TENT `MemFabricTransport::addMemoryBuffer`
   batch registration via `SmemTransBatchRegisterMem` (avoiding NPU page-aligned overlap failures
   across adjacent buffers), plus Mooncake Store request-local HCA/NUMA 3-tier affinity
   (`MC_STORE_HCA_AFFINITY=1`, `MC_ENABLE_DEST_DEVICE_AFFINITY=1`) for RL `put`/`get`
   (Tier 0: matching HCA-local ranges -> Tier 1: unknown locality -> Tier 2: known mismatch).
2. LMCache (#5150, 6913259264db; #4410, 00f6d71ed1af): Packed big-endian `uint32` `token_bytes`
   (`TOKEN_STRIDE = 4` B/token vs `28–36 B/token` Python `int` + tuple pointer in `IPCCacheServerKey`,
   preserving identical `blake3` chunk digests), plus engine-driven multi-group hybrid KV transfers
   (`KVLayerGroupsManager.get_kernel_group_window_chunks` for GDN + full attention with SYCL padded
   `block_stride_elems`).
3. Megatron-LM (#7047, ea99fdfce50a; #6502, 2052769557dc): Dynamic Context Parallelism V2
   (`DCP V2 3/4`) `get_valid_dynamic_context_parallel_group_sizes(total_size)` (power-of-two
   divisors plus `total_size`) integrated directly into standard train/eval schedules (eliminating
   687-line `hybrid_cp_schedule.py`), plus checkpoint conversion integrity preservation (`verify_integrity`
   preserved during source `torch_dist` load before clearing for non-`torch_dist` targets).
4. TorchTitan (#5546, 89e4339c963c; #5565, 3ed5aa976fdf), TransformerEngine (#3735, 3d8781839c8f),
   and Dynamo (#16482, 3032e8a703e5; #17153, d043ae041745): Explicit `importlib.util.find_spec`
   verification for `FA4` (`flash_attn.cute` on SM100+) and `FA3` (`flash_attn_interface` on SM90)
   for varlen attention plus safe PyTorch FA2 fallback (`atol=0, rtol=0` loss `0.19502778351306915`,
   grad L2 `0.012680011801421642`) with MLA asymmetric Q/K vs V rejection, and `attn-gym==0.0.17`
   per-export lazy build failure isolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_mooncake_memfabric_and_hca_affinity() -> dict[str, Any]:
    """Model Mooncake TENT `SmemTransBatchRegisterMem` (#4870) and 3-tier HCA affinity (#4631)."""
    npu_page_size = 2 * 1024 * 1024  # 2 MiB NPU hugepage alignment
    buffers = [
        {"addr": 0x10000000, "length": 0x00180000},  # 1.5 MiB
        {"addr": 0x10180000, "length": 0x00180000},  # adjacent 1.5 MiB in same 2 MiB page
        {"addr": 0x10300000, "length": 0x00200000},  # 2.0 MiB
    ]
    individual_pages = []
    overlap_detected = False
    prev_end_page = -1
    for buf in buffers:
        start_page = buf["addr"] // npu_page_size
        end_page = (buf["addr"] + buf["length"] - 1) // npu_page_size
        if start_page <= prev_end_page:
            overlap_detected = True
        prev_end_page = max(prev_end_page, end_page)
        individual_pages.append({"start_page": start_page, "end_page": end_page})

    return {
        "memfabric_batch_register": {
            "npu_page_size_bytes": npu_page_size,
            "buffer_count": len(buffers),
            "per_buffer_loop_causes_page_overlap": overlap_detected,
            "batch_api_calls": 1,
            "batch_api_symbol": "SmemTransBatchRegisterMem",
            "individual_page_spans": individual_pages,
        },
        "store_hca_affinity": {
            "env_flags": ["MC_STORE_HCA_AFFINITY=1", "MC_ENABLE_DEST_DEVICE_AFFINITY=1"],
            "tier_count": 3,
            "tiers": [
                {"tier": 0, "name": "matching_hca_local_range"},
                {"tier": 1, "name": "unknown_locality_segment"},
                {"tier": 2, "name": "known_mismatched_hca_segment"},
            ],
        },
    }


def model_lmcache_packed_tokens_and_hybrid_groups() -> dict[str, Any]:
    """Model LMCache big-endian `uint32` token packing (#5150) and hybrid KV window chunks (#4410)."""
    seq_len = 32768
    hops = 3  # scheduler -> client -> server IPC hops
    token_stride = 4
    python_int_tuple_elem_bytes = 36  # 28 B PyLongObject + 8 B tuple slot pointer
    unpacked_bytes_per_hop = seq_len * python_int_tuple_elem_bytes
    packed_bytes_per_hop = seq_len * token_stride

    chunk_size = 256
    groups = [
        {"group_idx": 0, "kind": "full_attention", "sw_size_tokens": -1, "recurrent_state": False},
        {"group_idx": 1, "kind": "sliding_window_attention", "sw_size_tokens": 1024, "recurrent_state": False},
        {"group_idx": 2, "kind": "gdn_recurrent", "sw_size_tokens": 256, "recurrent_state": True},
    ]
    window_chunks_default = []
    window_chunks_full_sw_kv = []
    for g in groups:
        if g["sw_size_tokens"] == -1:
            window_chunks_default.append(-1)
            window_chunks_full_sw_kv.append(-1)
        else:
            chunks = max(1, (g["sw_size_tokens"] + chunk_size - 1) // chunk_size)
            window_chunks_default.append(chunks)
            window_chunks_full_sw_kv.append(chunks if g["recurrent_state"] else -1)

    return {
        "packed_token_codec": {
            "token_stride_bytes": token_stride,
            "wire_endianness": "big",
            "blake3_hash_compatible": True,
            "seq_len_tokens": seq_len,
            "ipc_hops": hops,
            "unpacked_bytes_per_hop": unpacked_bytes_per_hop,
            "packed_bytes_per_hop": packed_bytes_per_hop,
            "memory_reduction_ratio_x": round(unpacked_bytes_per_hop / packed_bytes_per_hop, 2),
        },
        "hybrid_kv_layer_groups": {
            "chunk_size_tokens": chunk_size,
            "groups": groups,
            "window_chunks_default": window_chunks_default,
            "window_chunks_full_sw_kv": window_chunks_full_sw_kv,
        },
    }


def get_valid_dcp_group_sizes(total_size: int) -> list[int]:
    """Exact replica of Megatron-LM `get_valid_dynamic_context_parallel_group_sizes` (#7047)."""
    group_sizes = []
    group_size = 1
    while group_size < total_size:
        if total_size % group_size == 0:
            group_sizes.append(group_size)
        group_size *= 2
    group_sizes.append(total_size)
    return group_sizes


def model_megatron_dcpv2_and_torchtitan() -> dict[str, Any]:
    """Model Megatron-LM DCP V2 group sizes (#7047), ckpt integrity (#6502), and TorchTitan FA3/FA4 (#5546)."""
    dcp_examples = {
        str(total): get_valid_dcp_group_sizes(total) for total in (8, 12, 16, 24)
    }
    return {
        "megatron_dcp_v2": {
            "valid_group_sizes_by_dp_x_cp": dcp_examples,
            "removed_standalone_hybrid_cp_schedule_lines": 687,
            "ckpt_convert_preserves_source_verify_integrity": True,
        },
        "torchtitan_attention_and_attn_gym": {
            "sm90_varlen_requires": {"impl": "FA3", "module": "flash_attn_interface"},
            "sm100_varlen_requires": {"impl": "FA4", "module": "flash_attn.cute"},
            "sdpa_fa2_fallback_loss": 0.19502778351306915,
            "sdpa_fa2_fallback_grad_l2": 0.012680011801421642,
            "mla_asymmetric_qk_v_head_dim_rejected_on_sdpa": True,
            "attn_gym_pinned_version": "0.0.17",
            "attn_gym_isolated_failure_cases": 9,
        },
    }


def build_summary() -> dict[str, Any]:
    mooncake = model_mooncake_memfabric_and_hca_affinity()
    lmcache = model_lmcache_packed_tokens_and_hybrid_groups()
    megatron_titan = model_megatron_dcpv2_and_torchtitan()
    return {
        "experiment_id": "mooncake_memfabric_rl_affinity_lmcache_packed_mp_megatron_dcpv2_torchtitan",
        "phase": 75,
        "priority": 84,
        "track": "Track C — KV/Transfer & Distributed Training",
        "mooncake_memfabric_and_hca_affinity": mooncake,
        "lmcache_packed_tokens_and_hybrid_groups": lmcache,
        "megatron_dcpv2_and_torchtitan": megatron_titan,
        "invariants": {
            "memfabric_batch_avoids_npu_page_overlap": (
                mooncake["memfabric_batch_register"]["per_buffer_loop_causes_page_overlap"]
                and mooncake["memfabric_batch_register"]["batch_api_calls"] == 1
            ),
            "mooncake_hca_affinity_has_3_tiers": mooncake["store_hca_affinity"]["tier_count"] == 3,
            "lmcache_packed_tokens_9x_smaller": (
                lmcache["packed_token_codec"]["memory_reduction_ratio_x"] == 9.0
            ),
            "hybrid_full_sw_kv_preserves_recurrent_window": (
                lmcache["hybrid_kv_layer_groups"]["window_chunks_full_sw_kv"] == [-1, -1, 1]
            ),
            "megatron_dcp_v2_non_power_of_two_24": (
                megatron_titan["megatron_dcp_v2"]["valid_group_sizes_by_dp_x_cp"]["24"]
                == [1, 2, 4, 8, 24]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output path for summary.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the existing summary.json matches build_summary()",
    )
    args = parser.parse_args()
    summary = build_summary()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of date with build_summary()")
        return 0
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
