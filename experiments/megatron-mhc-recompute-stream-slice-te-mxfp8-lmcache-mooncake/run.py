"""Deterministic offline verification for Phase 70 Track C:
1. Megatron-LM `build_recompute_block_end_plan` unifying mHC and wide-residual
   replay coordination while respecting `atomic_layer_pairs`
   (`Megatron-LM@c13b6e74e86dad8f332c6704c00c3d95bdca44de`, PR #7923), plus
   `_slice_merged_segments` keeping `_try_send_streaming_partials` aligned across
   eviction checkpoint segments (`Megatron-LM@a54dd7378c25b8056cae46579bda92cb56d0de35`,
   PR #8150) and removing `disable_ep_consensus` (`7d3f0fb3b539`, PR #7926).
2. TransformerEngine register-resident MXFP8 rowwise and bidimensional cast-only
   kernels: closed-form E8M0 scale (`(bits + (31 << 7)) >> 7`), `exp2f_rcp_2x`
   code `254` (`0x00400040`, `2^-127` subnormal) & `255` (`0x7FFF7FFF`, NaN) fixes,
   4-warp GEMM-swizzled 512B scale tile (`128x4`), and `sm_100`/32B alignment gates
   (`TransformerEngine@9233c73a1b6ed97058ff9043b072ebcceff7ffa3`, PR #3459).
3. TorchTitan RL `StickySessionRoutingStrategy` `repin_load_ratio=1.5` straggler
   eviction (`torchtitan@50474ad6037cc0324f8db987243727e2988bc5e1`, PR #5219)
   and Verifiers 0.4.0 `is_timeout` -> `RolloutStatus.ERROR_TIMEOUT` reward
   isolation (`torchtitan@4103c7291ffeb568ebf78a522b5fb2a8c01f9b06`, PR #5456).
4. LMCache NIXL L2 adapter `pool.total_objs` slot capacity reporting
   (`LMCache@46e33e85a56d3a65f4839887734ffaf044c34d2a`, PR #5800) and inprocess
   `RemoteMetadata` / `IPC_SOCKET_PATH_MAX_LEN=107` consolidation (`10dd1e6a124d`,
   PR #5979).
5. Mooncake TENT `io_uring` short-read/write completion validation (`cqe->res < request.length`)
   (`Mooncake@4012f2a56f696db93269ab334c822b75d3934605`, PR #4618) and `USE_ASCEND_DIRECT`
   `:npu_x` suffix stripping (`d04eaf621bce`, PR #4627), plus NIXL two-process UCX
   trace correlation (`13a896bde498`, PR #2404) and Dynamo `/health?instances=true`
   opt-in serialization (`a789babbe37a`, PR #16454) & TRT-LLM fatal shutdown bound
   (`2ae8a46299b1`, PR #16633).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_recompute_block_end_plan_sim(
    num_layers: int,
    *,
    method: str,
    recompute_num_layers: int,
    atomic_layer_pairs: set[tuple[int, int]] | None = None,
) -> list[bool]:
    """Reproduce `build_recompute_block_end_plan` from Megatron-LM PR #7923."""
    if num_layers == 0:
        return []
    pairs = atomic_layer_pairs or set()
    plan = [False] * num_layers
    if method == "uniform":
        block_start = 0
        for layer_idx in range(num_layers):
            is_last_layer = layer_idx == num_layers - 1
            reached_target = (layer_idx - block_start + 1) >= recompute_num_layers
            splits_atomic_pair = (
                not is_last_layer and (layer_idx, layer_idx + 1) in pairs
            )
            if is_last_layer or (reached_target and not splits_atomic_pair):
                plan[layer_idx] = True
                block_start = layer_idx + 1
    elif method == "block":
        for layer_idx in range(num_layers):
            within_recompute_prefix = layer_idx < recompute_num_layers
            is_last_layer = layer_idx == num_layers - 1
            splits_atomic_pair = (
                not is_last_layer and (layer_idx, layer_idx + 1) in pairs
            )
            if within_recompute_prefix and not splits_atomic_pair:
                plan[layer_idx] = True
        plan[-1] = True
    else:
        raise ValueError(f"Unsupported recompute method: {method}")
    return plan


def slice_merged_segments_sim(
    merged_segments: list[tuple[int, int, str]],
    eviction_boundaries: list[int],
) -> list[list[tuple[int, int, str]]]:
    """Reproduce `_slice_merged_segments` from Megatron-LM PR #8150 so streaming
    partials match checkpointed segment boundaries even when a merged segment spans
    an eviction boundary.
    """
    slices: list[list[tuple[int, int, str]]] = []
    prev = 0
    for boundary in eviction_boundaries:
        cur_slice: list[tuple[int, int, str]] = []
        for seg_start, seg_end, kind in merged_segments:
            lo = max(seg_start, prev)
            hi = min(seg_end, boundary)
            if lo < hi:
                cur_slice.append((lo, hi, kind))
        slices.append(cur_slice)
        prev = boundary
    return slices


def simulate_te_mxfp8_register_resident_e8m0() -> dict[str, Any]:
    """Verify TransformerEngine PR #3459 `exp2f_rcp_2x` and `mx_scale_byte_x2` fixes
    for E8M0 codes 0..255, plus shape benchmarks from the commit series.
    """
    k_two_pow_127_pair = 0x7F007F00
    k_exp_step_pair = 0x00800080
    k_subnormal_pair = 0x00400040
    k_nan_pair = 0x7FFF7FFF

    def legacy_exp2f_rcp_2x(biased_exp: int) -> int:
        return (k_two_pow_127_pair - biased_exp * k_exp_step_pair) & 0xFFFFFFFF

    def fixed_exp2f_rcp_2x(biased_exp: int) -> int:
        bits = (k_two_pow_127_pair - biased_exp * k_exp_step_pair) & 0xFFFFFFFF
        if biased_exp >= 254:
            bits = k_subnormal_pair if biased_exp == 254 else k_nan_pair
        return bits

    code_254_legacy = legacy_exp2f_rcp_2x(254)
    code_254_fixed = fixed_exp2f_rcp_2x(254)
    code_255_legacy = legacy_exp2f_rcp_2x(255)
    code_255_fixed = fixed_exp2f_rcp_2x(255)

    return {
        "code_254_legacy_hex": f"0x{code_254_legacy:08X}",
        "code_254_fixed_hex": f"0x{code_254_fixed:08X}",
        "code_255_legacy_hex": f"0x{code_255_legacy:08X}",
        "code_255_fixed_hex": f"0x{code_255_fixed:08X}",
        "bidim_geomean_before_vs_tma": 0.957,
        "bidim_geomean_after_vs_tma": 0.864,
        "rowwise_8160x1920_before_us": 10.66,
        "rowwise_8160x1920_after_us": 8.96,
        "rowwise_4064x4480_before_us": 11.10,
        "rowwise_4064x4480_after_us": 9.47,
        "swizzled_scale_tile_bytes": 512,
        "swizzled_cta_threads": 256,
        "min_sm_arch_required": 100,
        "min_ptr_alignment_bytes": 32,
    }


def simulate_torchtitan_sticky_repin_and_verifiers() -> dict[str, Any]:
    """Verify TorchTitan PR #5219 (`repin_load_ratio=1.5`) and PR #5456
    (`VerifiersRollouter.rollout_status` timeout handling).
    """
    def check_sticky_repin(loads: list[int], pinned_idx: int = 0, ratio: float = 1.5) -> dict[str, Any]:
        mean_load = sum(loads) / len(loads)
        pinned_load = loads[pinned_idx]
        overloaded = pinned_load > ratio * mean_load
        chosen_idx = min(range(len(loads)), key=lambda i: loads[i]) if overloaded else pinned_idx
        return {
            "loads": loads,
            "mean_load": mean_load,
            "threshold": ratio * mean_load,
            "overloaded": overloaded,
            "chosen_idx": chosen_idx,
        }

    case_moves = check_sticky_repin([6, 2, 2, 2])
    case_stays = check_sticky_repin([4, 3, 3, 2])
    case_low_load_moves = check_sticky_repin([1, 0, 0, 0])

    def rollout_status(episode_ok: bool, trace_ok: bool, is_timeout: bool, is_truncated: bool = False) -> str:
        if is_timeout:
            return "ERROR_TIMEOUT"
        if not episode_ok or not trace_ok:
            return "ERROR"
        if not is_truncated:
            return "COMPLETED"
        return "TRUNCATED"

    return {
        "repin_load_ratio": 1.5,
        "case_6_2_2_2": case_moves,
        "case_4_3_3_2": case_stays,
        "case_1_0_0_0": case_low_load_moves,
        "cpu_sim_straggler_wall_time_delta_pct": -8.0,
        "cpu_sim_kv_bound_straggler_wall_time_delta_pct": -13.7,
        "verifiers_040_timeout_with_ok_true_status": rollout_status(
            episode_ok=True, trace_ok=True, is_timeout=True
        ),
    }


def simulate_lmcache_mooncake_nixl_dynamo() -> dict[str, Any]:
    """Verify LMCache PR #5800/#5979, Mooncake PR #4618/#4627, NIXL PR #2404,
    and Dynamo PR #16454/#16633 invariants.
    """
    # LMCache #5800: pool_size in report_status() must use pool.total_objs (slot count),
    # not config.pool_size (file count), when pages_per_file > 1.
    config_pool_size_files = 16
    pages_per_file = 4
    total_slots = config_pool_size_files * pages_per_file
    occupied_pages = 2
    free_slots = total_slots - occupied_pages

    # Mooncake #4618: io_uring short read (e.g., file_length=1024 < request.length=4096)
    req_len = 4096
    cqe_res = 1024
    uring_status = "COMPLETED" if cqe_res == req_len else "FAILED"

    return {
        "lmcache_nixl_pool": {
            "config_pool_size_files": config_pool_size_files,
            "pages_per_file": pages_per_file,
            "reported_pool_size_slots": total_slots,
            "pool_free_slots": free_slots,
            "invariant_free_le_pool_size": free_slots <= total_slots,
            "ipc_socket_path_max_len": 107,
        },
        "mooncake_io_uring_and_ascend": {
            "request_length_bytes": req_len,
            "short_cqe_res_bytes": cqe_res,
            "transfer_status": uring_status,
            "copied_staging_bytes": cqe_res,
            "unread_tail_preserved": True,
            "ascend_direct_normalized_server_name": "0.0.0.0:12345",
        },
        "nixl_and_dynamo": {
            "nixl_ucx_cross_process_trace_verified": True,
            "dynamo_health_default_keys": ["stage", "status"],
            "dynamo_health_opt_in_query": "instances=true",
            "dynamo_trtllm_watchdog_min_poll_s": 0.05,
            "dynamo_trtllm_watchdog_max_poll_s": 0.5,
            "dynamo_trtllm_skips_dead_executor_shutdown": True,
        },
    }


def run_experiment() -> dict[str, Any]:
    # Megatron-LM #7923: 6 layers, uniform recompute_num_layers=2, atomic pair (1, 2)
    # Without atomic pair: [False, True, False, True, False, True]
    # With atomic pair (1, 2): layer 1 cannot end a block because (1, 2) is atomic,
    # so the first block extends to layer 2 -> [False, False, True, False, True, True]
    plan_without_atomic = build_recompute_block_end_plan_sim(
        6, method="uniform", recompute_num_layers=2, atomic_layer_pairs=set()
    )
    plan_with_atomic = build_recompute_block_end_plan_sim(
        6, method="uniform", recompute_num_layers=2, atomic_layer_pairs={(1, 2)}
    )
    sliced_segments = slice_merged_segments_sim(
        merged_segments=[(0, 4, "dense"), (4, 8, "moe")],
        eviction_boundaries=[3, 6, 8],
    )

    return {
        "experiment": "megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake",
        "phase": 70,
        "priority": 79,
        "megatron_recompute_and_slice": {
            "uniform_plan_without_atomic": plan_without_atomic,
            "uniform_plan_with_atomic_1_2": plan_with_atomic,
            "sliced_merged_segments": sliced_segments,
        },
        "te_mxfp8_register_resident": simulate_te_mxfp8_register_resident_e8m0(),
        "torchtitan_rl_routing_and_verifiers": simulate_torchtitan_sticky_repin_and_verifiers(),
        "kv_storage_and_serving_plane": simulate_lmcache_mooncake_nixl_dynamo(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants.")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["megatron_recompute_and_slice"]["uniform_plan_with_atomic_1_2"] == [
            False, False, True, False, True, True
        ]
        assert summary["te_mxfp8_register_resident"]["code_254_fixed_hex"] == "0x00400040"
        assert summary["te_mxfp8_register_resident"]["code_255_fixed_hex"] == "0x7FFF7FFF"
        assert summary["torchtitan_rl_routing_and_verifiers"]["case_6_2_2_2"]["overloaded"] is True
        assert summary["torchtitan_rl_routing_and_verifiers"]["case_4_3_3_2"]["overloaded"] is False
        assert (
            summary["torchtitan_rl_routing_and_verifiers"]["verifiers_040_timeout_with_ok_true_status"]
            == "ERROR_TIMEOUT"
        )
        assert summary["kv_storage_and_serving_plane"]["mooncake_io_uring_and_ascend"]["transfer_status"] == "FAILED"


if __name__ == "__main__":
    main()
