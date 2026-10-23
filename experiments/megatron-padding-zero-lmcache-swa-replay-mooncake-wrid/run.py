#!/usr/bin/env python3
"""Deterministic CPU/offline model for Phase 66 Track C:
1. Megatron-LM CUDA-Graph Safe Device-Pointer Padding Zeroing (`zero_padding_rows`) & Global-Batch K3 Quantile Balancing (Megatron-LM@f9d9bdcc6e09, Megatron-LM@92641a421f40)
2. TransformerEngine `torch.compile` FP8 Weight Workspace Caching across Microbatches (TransformerEngine@9ea05cf2e8dc)
3. LMCache SWA Bounded Prefix-Replay Chunk-Aligned Full-Prompt Hit (LMCache@a0ff4701284d)
4. Mooncake TENT Post-`ibv_post_send` `notify_send_wr_id_` Advancement & NVLink VMM/PID-Namespace Diagnostics (Mooncake@dae22d472e31, Mooncake@b66bb2b5f650)
5. NIXL UCX 16-Byte `treq` Wire Notification Tag & Repost/Release Telemetry Accounting (nixl@f3673f6be3f4, nixl@70badd6aad36)
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


def model_megatron_padding_zero_and_quantile_balancing() -> dict[str, Any]:
    """Model Megatron-LM device-pointer padding zeroing under CUDA graph replay & K3 quantile balancing."""
    # Suppose a CUDA graph is captured with max_tokens = 128 (full dummy batch at capture time: real_tokens = 128).
    # Python slice `padding_slice = slice(128, 128)` is frozen as empty at capture time!
    # On replay with real_token_count = 72, rows [72, 128) contain stale uninitialized/garbage activations
    # (e.g., max garbage abs value = 64.0 while real tokens have max abs value = 2.0).
    captured_rows = 128
    replay_real_tokens = 72
    real_amax = 2.0
    stale_padding_amax = 64.0
    fp8_e4m3_max = 448.0

    # Frozen Python slice leaves [72, 128) untouched -> amax = 64.0 (32x dynamic range loss!)
    frozen_slice_amax = max(real_amax, stale_padding_amax)
    frozen_slice_scale = fp8_e4m3_max / frozen_slice_amax

    # Device-pointer zero_padding_rows reads gpu_view.real_token_count[0] == 72 on device
    # and zeroes rows where row + first_token_idx >= real_token_count.
    device_ptr_amax = real_amax
    device_ptr_scale = fp8_e4m3_max / device_ptr_amax
    snr_preservation_ratio = round(device_ptr_scale / frozen_slice_scale, 2)

    # Sequence-parallel shard check: TP=4, 32 rows per rank -> ranks 0,1 (tokens 0..63) keep all 32 rows;
    # rank 2 (first_token_idx=64) keeps rows 0..7 and zeroes rows 8..31; rank 3 (first_token_idx=96) zeroes all 32 rows.
    tp_size = 4
    rows_per_rank = captured_rows // tp_size
    sp_zeroed_rows_per_rank = []
    for sp_rank in range(tp_size):
        first_idx = sp_rank * rows_per_rank
        zeroed = sum(
            1 for r in range(rows_per_rank) if (r + first_idx) >= replay_real_tokens
        )
        sp_zeroed_rows_per_rank.append(zeroed)

    return {
        "captured_rows": captured_rows,
        "replay_real_tokens": replay_real_tokens,
        "frozen_python_slice_zeroed_rows_on_replay": 0,
        "device_ptr_kernel_zeroed_rows_on_replay": captured_rows - replay_real_tokens,
        "sp_zeroed_rows_per_rank": sp_zeroed_rows_per_rank,
        "frozen_slice_fp8_scale": frozen_slice_scale,
        "device_ptr_fp8_scale": device_ptr_scale,
        "fp8_precision_factor_recovered": snr_preservation_ratio,
        "quantile_balancing_global_batch_allreduce_buffer": "qb_histogram [num_layers, num_experts, num_bins] int32",
    }


def model_te_compiled_fp8_weight_caching() -> dict[str, Any]:
    """Model TransformerEngine torch.compile FP8 weight caching across microbatches."""
    # Upstream TransformerEngine@9ea05cf2e8dc:
    # On microbatch 0 (`is_first_microbatch=True`), compiled Linear passes `weight_workspace=None`
    # into the functional custom op so it creates a fresh quantized FP8 workspace without mutating
    # the old workspace in-place, then stores it in `self._fp8_workspaces["weight"]`.
    # On microbatches 1..M-1 (`is_first_microbatch=False`), compiled Linear passes the cached workspace.
    num_microbatches = 8
    uncompiled_or_uncached_quantizations = num_microbatches
    compiled_cached_quantizations = 1
    quantization_calls_saved_pct = round(
        (uncompiled_or_uncached_quantizations - compiled_cached_quantizations)
        / uncompiled_or_uncached_quantizations
        * 100.0,
        2,
    )
    return {
        "num_microbatches": num_microbatches,
        "uncached_fp8_weight_casts_per_step": uncompiled_or_uncached_quantizations,
        "compiled_cached_fp8_weight_casts_per_step": compiled_cached_quantizations,
        "weight_cast_reduction_pct": quantization_calls_saved_pct,
        "four_4096_layers_peak_mem_eager_mib": 452,
        "four_4096_layers_peak_mem_compiled_mib": 539,
    }


def model_lmcache_swa_prefix_replay_alignment() -> dict[str, Any]:
    """Model LMCache SWA bounded prefix-replay full-prompt hit alignment (LMCache@a0ff4701284d)."""
    # Prompt has 12 tokens, chunk_size = 4, num_computed_tokens = 0, full lookup hit ret = 12.
    prompt_tokens = 12
    chunk_size = 4
    ret = 12

    # Without SWA prefix replay (prefix_replay_tokens == 0):
    # Classic rule: need_to_load = ret - 1 = 11; vLLM allocates ceil(11/4) = 3 blocks (12 token slots).
    # Retrieve op spans [0, 12), which fits in 3 * 4 = 12 slots.
    no_replay_need = ret - 1
    no_replay_allocated_blocks = 3
    no_replay_retrieve_end = 12

    # With SWA bounded prefix replay (prefix_replay_tokens = 128 > 0, e.g., DeepSeek V4.1):
    # vLLM scheduler floors external hit to block boundary before allocating:
    # If connector returned need_to_load = 11, vLLM floors 11 -> 8 tokens (2 blocks = 8 slots),
    # while legacy tracker kept num_lmcache_hit_tokens = 12 -> retrieve [0, 12) overflows 2 blocks by 4 tokens!
    legacy_swa_allocated_blocks = (ret - 1) // chunk_size  # 2 blocks (8 slots)
    legacy_swa_retrieve_end = 12
    legacy_swa_overflow_tokens = legacy_swa_retrieve_end - (
        legacy_swa_allocated_blocks * chunk_size
    )

    # Fixed SWA rule (LMCache@a0ff4701284d):
    # aligned = (ret - 1) // chunk * chunk = 8; pending_free_ranges.append((8, 12));
    # tracker.num_lmcache_hit_tokens = 8; need_to_load = 8.
    fixed_swa_aligned = ((ret - 1) // chunk_size) * chunk_size
    fixed_swa_allocated_blocks = fixed_swa_aligned // chunk_size
    fixed_swa_retrieve_end = fixed_swa_aligned
    fixed_swa_overflow_tokens = fixed_swa_retrieve_end - (
        fixed_swa_allocated_blocks * chunk_size
    )
    fixed_swa_released_lock_range = [fixed_swa_aligned, ret]

    return {
        "prompt_tokens": prompt_tokens,
        "chunk_size": chunk_size,
        "no_replay": {
            "need_to_load": no_replay_need,
            "allocated_blocks": no_replay_allocated_blocks,
            "retrieve_end": no_replay_retrieve_end,
        },
        "legacy_swa_replay": {
            "allocated_blocks": legacy_swa_allocated_blocks,
            "retrieve_end": legacy_swa_retrieve_end,
            "buffer_overflow_tokens": legacy_swa_overflow_tokens,
        },
        "fixed_swa_replay": {
            "aligned_hit_tokens": fixed_swa_aligned,
            "allocated_blocks": fixed_swa_allocated_blocks,
            "retrieve_end": fixed_swa_retrieve_end,
            "buffer_overflow_tokens": fixed_swa_overflow_tokens,
            "released_lookup_lock_range": fixed_swa_released_lock_range,
        },
    }


def model_mooncake_wrid_and_nixl_treq() -> dict[str, Any]:
    """Model Mooncake RdmaEndPoint::sendNotification wr_id slot safety & NIXL UCX treq wire tag."""
    # Mooncake RdmaEndPoint::sendNotification has kNotifyMaxPendingSends = 256 slots indexed by wr_id % 256.
    # Suppose 255 notifications (wr_id 0..254) are in flight (uncompleted).
    # Attempt #256 fails in ibv_post_send() (returns EINVAL).
    # Legacy code did `wr.wr_id = notify_send_wr_id_++` BEFORE `ibv_post_send()`, advancing wr_id to 256
    # while notify_pending_count_ stayed 255 (< 256).
    # The next retry then picked wr_id = 256 -> slot 256 % 256 == 0, overwriting in-flight slot 0!
    k_max_slots = 256
    inflight_wrs = 255  # wr_ids 0..254

    legacy_wr_id_after_failed_post = inflight_wrs + 1  # 256
    legacy_retry_slot = legacy_wr_id_after_failed_post % k_max_slots  # 0 (corrupts in-flight wr_id 0!)
    legacy_overwrites_inflight_slot_0 = legacy_retry_slot == 0

    fixed_wr_id_after_failed_post = inflight_wrs  # 255
    fixed_retry_slot = fixed_wr_id_after_failed_post % k_max_slots  # 255 (free slot!)
    fixed_overwrites_inflight_slot_0 = fixed_retry_slot == 0

    # NIXL UCX notification wire format (nixl@f3673f6be3f4):
    # Carries 26-byte TraceContext ("tctx") + 16-byte little-endian requestTag ("treq": uint64 requestId, uint64 execution)
    req_id = 0x0102030405060708
    exec_num = 3
    treq_wire = struct.pack("<QQ", req_id, exec_num)
    unpacked_req_id, unpacked_exec_num = struct.unpack("<QQ", treq_wire)

    return {
        "mooncake_notify_max_slots": k_max_slots,
        "inflight_uncompleted_wrs": inflight_wrs,
        "legacy_wr_id_after_failed_post": legacy_wr_id_after_failed_post,
        "legacy_retry_slot": legacy_retry_slot,
        "legacy_overwrites_inflight_slot_0": legacy_overwrites_inflight_slot_0,
        "fixed_wr_id_after_failed_post": fixed_wr_id_after_failed_post,
        "fixed_retry_slot": fixed_retry_slot,
        "fixed_overwrites_inflight_slot_0": fixed_overwrites_inflight_slot_0,
        "nixl_tctx_wire_bytes": 26,
        "nixl_treq_wire_bytes": len(treq_wire),
        "nixl_treq_roundtrip_ok": (unpacked_req_id == req_id and unpacked_exec_num == exec_num),
    }


def run_experiment() -> dict[str, Any]:
    megatron = model_megatron_padding_zero_and_quantile_balancing()
    te = model_te_compiled_fp8_weight_caching()
    lmcache = model_lmcache_swa_prefix_replay_alignment()
    fabric = model_mooncake_wrid_and_nixl_treq()

    summary: dict[str, Any] = {
        "experiment": "megatron-padding-zero-lmcache-swa-replay-mooncake-wrid",
        "phase": 66,
        "priority": 75,
        "track": "Track C — Megatron CUDA-Graph Padding Zeroing, TE Compiled FP8 Cache, LMCache SWA Replay & Mooncake/NIXL",
        "megatron_padding_zero_and_quantile_balancing": megatron,
        "te_compiled_fp8_weight_caching": te,
        "lmcache_swa_prefix_replay_alignment": lmcache,
        "mooncake_wrid_and_nixl_treq": fabric,
        "invariants": {
            "megatron_fp8_precision_factor_recovered": megatron["fp8_precision_factor_recovered"],
            "te_weight_cast_reduction_pct": te["weight_cast_reduction_pct"],
            "lmcache_legacy_swa_overflow_tokens": lmcache["legacy_swa_replay"][
                "buffer_overflow_tokens"
            ],
            "lmcache_fixed_swa_overflow_tokens": lmcache["fixed_swa_replay"][
                "buffer_overflow_tokens"
            ],
            "mooncake_legacy_overwrites_slot_0": fabric["legacy_overwrites_inflight_slot_0"],
            "mooncake_fixed_overwrites_slot_0": fabric["fixed_overwrites_inflight_slot_0"],
            "nixl_treq_wire_bytes": fabric["nixl_treq_wire_bytes"],
        },
    }
    return summary


def verify_summary(summary: dict[str, Any]) -> None:
    inv = summary["invariants"]
    assert inv["megatron_fp8_precision_factor_recovered"] == 32.0
    assert inv["te_weight_cast_reduction_pct"] == 87.5
    assert inv["lmcache_legacy_swa_overflow_tokens"] == 4
    assert inv["lmcache_fixed_swa_overflow_tokens"] == 0
    assert inv["mooncake_legacy_overwrites_slot_0"] is True
    assert inv["mooncake_fixed_overwrites_slot_0"] is False
    assert inv["nixl_treq_wire_bytes"] == 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    verify_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.check:
        print(f"[PASS] {summary['experiment']} invariants verified -> {out_path}")


if __name__ == "__main__":
    main()
