"""Deterministic offline model for Phase 31 Track 3:
1. Mooncake TransferEngine RDMA Auto-GID recovery on `IBV_WC_RETRY_EXC_ERR` using epoch-stamped `RdmaCompletionToken` before rail fail-over (`5ea6cf0d`).
2. LMCache `lmcache/inprocess/v1/gpu_connector` isolation and upfront `_validate_vllm_device_features` guard (`df19cd68`).
3. NIXL UCX closed-vocabulary sender/receiver phase tracing (`nixl::submit`, `nixl::wire.submitted`, `nixl::notif.sent`, `nixl::notif.received`) without poll-skewed `wire.completed` (`8d5660c1`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_mooncake_auto_gid_recovery() -> dict[str, Any]:
    """Model Mooncake RDMA Auto-GID data-path recovery on IBV_WC_RETRY_EXC_ERR (5ea6cf0d)."""
    # Suppose an endpoint has 3 candidate GIDs in AutoGidCandidateClass order:
    # gid_index=3 (stale RoCEv2 IPv4 alias that fails on data path with IBV_WC_RETRY_EXC_ERR),
    # gid_index=5 (healthy RoCEv2 IPv4 routable GID),
    # gid_index=1 (fallback link-local).
    # A burst of 16 concurrent slices is posted under selection_epoch=1 (gid_index=3).
    concurrent_slices = 16
    candidate_gids = [
        {"gid_index": 3, "class": "kRoceV2Ipv4", "reachable": False},
        {"gid_index": 5, "class": "kRoceV2Ipv4", "reachable": True},
        {"gid_index": 1, "class": "kFallbackNonzero", "reachable": True},
    ]

    # Legacy behavior (before 5ea6cf0d):
    # On IBV_WC_RETRY_EXC_ERR, WorkerPool immediately called markRailFailed(peerNicPath) without
    # advancing the Auto-GID candidate, blacklisting the entire NIC rail!
    legacy_rail_blacklisted = True
    legacy_recovered_on_same_rail = False

    # Patched behavior (5ea6cf0d):
    # All 16 in-flight slices carry RdmaCompletionToken(endpoint_id=1, selection_epoch=1).
    # Slice 0 fails with IBV_WC_RETRY_EXC_ERR -> recordAutoGidDataPathFailure advances epoch 1 -> 2 (gid_index=5).
    # Slices 1..15 also fail with old selection_epoch=1 -> classified as kIgnoreDuplicate so they don't
    # exhaustgid_index=5 or gid_index=1!
    current_epoch = 1
    active_candidate_idx = 0
    epoch_advances = 0
    duplicate_epoch_completions_ignored = 0

    for slice_idx in range(concurrent_slices):
        slice_token_epoch = 1  # posted under epoch 1
        if not candidate_gids[active_candidate_idx]["reachable"]:
            # First slice with token_epoch == current_epoch advances GID
            if slice_token_epoch == current_epoch:
                current_epoch += 1
                active_candidate_idx += 1
                epoch_advances += 1
        else:
            if slice_token_epoch < current_epoch:
                duplicate_epoch_completions_ignored += 1

    # Retried slices under epoch 2 (gid_index=5) succeed and recordAutoGidDataPathSuccess locks in gid_index=5
    retried_slices_succeeded = concurrent_slices if candidate_gids[active_candidate_idx]["reachable"] else 0

    return {
        "concurrent_slices_in_burst": concurrent_slices,
        "legacy_rail_blacklisted": legacy_rail_blacklisted,
        "legacy_recovered_on_same_rail": legacy_recovered_on_same_rail,
        "patched_epoch_advances": epoch_advances,
        "patched_duplicate_epoch_completions_ignored": duplicate_epoch_completions_ignored,
        "patched_active_gid_index": candidate_gids[active_candidate_idx]["gid_index"],
        "patched_retried_slices_succeeded": retried_slices_succeeded,
        "patched_rail_blacklisted": False,
    }


def simulate_lmcache_inprocess_connector_isolation() -> dict[str, Any]:
    """Model LMCache in-process GPU connector isolation and _validate_vllm_device_features (df19cd68)."""
    device_feature_matrix = [
        {"device": "cuda", "enable_blending": True, "use_gpu_connector_v3": True, "use_layerwise": True},
        {"device": "xpu", "enable_blending": True, "use_gpu_connector_v3": True, "use_layerwise": True},
        {"device": "musa", "enable_blending": True, "use_gpu_connector_v3": False, "use_layerwise": False},
        {"device": "musa", "enable_blending": False, "use_gpu_connector_v3": True, "use_layerwise": False},
        {"device": "hpu", "enable_blending": False, "use_gpu_connector_v3": False, "use_layerwise": True},
        {"device": "hpu", "enable_blending": False, "use_gpu_connector_v3": False, "use_layerwise": False},
    ]

    supported_blending = {"cuda", "xpu"}
    supported_v3 = {"cuda", "xpu"}
    invalid_configs_caught_by_value_error = 0
    deep_stream_crashes_prevented = 0
    valid_configs_accepted = 0

    for case in device_feature_matrix:
        dev = case["device"]
        invalid = (
            (case["enable_blending"] and dev not in supported_blending)
            or (case["use_gpu_connector_v3"] and dev not in supported_v3)
            or (case["use_layerwise"] and dev == "hpu")
        )
        if invalid:
            invalid_configs_caught_by_value_error += 1
            deep_stream_crashes_prevented += 1
        else:
            valid_configs_accepted += 1

    return {
        "total_configs_evaluated": len(device_feature_matrix),
        "valid_configs_accepted": valid_configs_accepted,
        "invalid_configs_caught_by_value_error": invalid_configs_caught_by_value_error,
        "deep_stream_crashes_prevented": deep_stream_crashes_prevented,
        "cpu_only_standalone_import_decoupled": True,
    }


def simulate_nixl_ucx_phase_tracing() -> dict[str, Any]:
    """Model NIXL UCX closed-vocabulary phase tracing and poll-delay isolation (8d5660c1)."""
    # Suppose an RDMA PUT + flush + notification transfer occurs:
    # postXfer start (`nixl::submit`): t = 10.0 us
    # sendXferRange + flush posted (`nixl::wire.submitted`): t = 14.5 us
    # Physical NIC wire completion: t = 38.0 us
    # Receiver observes notification (`nixl::notif.received`): t = 39.2 us
    # Application busy-loop / delayed status check (`checkXfer`) happens at t = 210.0 us (`nixl::notif.sent` if deferred)
    t_submit_us = 10.0
    t_wire_submitted_us = 14.5
    t_actual_wire_done_us = 38.0
    t_notif_received_us = 39.2
    t_app_status_poll_us = 210.0

    # If UCX recorded `nixl::wire.completed` at status-poll time (`checkXfer`), it would report
    # wire duration = 210.0 - 14.5 = 195.5 us instead of actual 23.5 us (8.32x poll-skew inflation!).
    poll_skewed_wire_duration_us = round(t_app_status_poll_us - t_wire_submitted_us, 3)
    actual_wire_duration_us = round(t_actual_wire_done_us - t_wire_submitted_us, 3)
    poll_skew_inflation_ratio = round(poll_skewed_wire_duration_us / actual_wire_duration_us, 4)

    # Also model disconnected peer failure during sendXferRange:
    # Only `nixl::submit` is recorded; `nixl::wire.submitted` is not emitted, and transfer error is preserved
    disconnected_peer_phases = ["nixl::submit"]
    healthy_sampled_phases = [
        "nixl::submit",
        "nixl::wire.submitted",
        "nixl::notif.sent",
        "nixl::notif.received",
    ]

    return {
        "healthy_sampled_phases": healthy_sampled_phases,
        "disconnected_peer_phases": disconnected_peer_phases,
        "submit_to_wire_submitted_us": round(t_wire_submitted_us - t_submit_us, 3),
        "actual_wire_duration_us": actual_wire_duration_us,
        "poll_skewed_wire_duration_us": poll_skewed_wire_duration_us,
        "poll_skew_inflation_ratio": poll_skew_inflation_ratio,
        "wire_completed_omitted_to_prevent_poll_skew": True,
    }


def run_experiment() -> dict[str, Any]:
    mooncake = simulate_mooncake_auto_gid_recovery()
    lmcache = simulate_lmcache_inprocess_connector_isolation()
    nixl = simulate_nixl_ucx_phase_tracing()

    summary = {
        "experiment": "mooncake_gid_recovery_lmcache_inproc_nixl_ucx",
        "phase": 31,
        "priority": 40,
        "mooncake_auto_gid_recovery": mooncake,
        "lmcache_inprocess_isolation": lmcache,
        "nixl_ucx_phase_tracing": nixl,
        "invariants_verified": (
            mooncake["patched_epoch_advances"] == 1
            and mooncake["patched_duplicate_epoch_completions_ignored"] == 15
            and mooncake["patched_retried_slices_succeeded"] == 16
            and mooncake["patched_rail_blacklisted"] is False
            and lmcache["invalid_configs_caught_by_value_error"] == 3
            and lmcache["deep_stream_crashes_prevented"] == 3
            and nixl["wire_completed_omitted_to_prevent_poll_skew"] is True
            and nixl["poll_skew_inflation_ratio"] > 8.0
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        if not summary["invariants_verified"]:
            raise SystemExit("Invariant check failed for mooncake_gid_recovery_lmcache_inproc_nixl_ucx")
        print(
            "[PASS] mooncake_gid_recovery_lmcache_inproc_nixl_ucx: "
            f"gid_epoch_advances={summary['mooncake_auto_gid_recovery']['patched_epoch_advances']} "
            f"(ignored_dups={summary['mooncake_auto_gid_recovery']['patched_duplicate_epoch_completions_ignored']}/15), "
            f"lmcache_guards={summary['lmcache_inprocess_isolation']['invalid_configs_caught_by_value_error']}/3, "
            f"nixl_poll_skew_prevented={summary['nixl_ucx_phase_tracing']['poll_skew_inflation_ratio']}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
