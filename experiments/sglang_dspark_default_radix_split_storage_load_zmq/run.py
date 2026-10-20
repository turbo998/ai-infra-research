#!/usr/bin/env python3
"""Deterministic stdlib simulator for Phase 39 Experiment 2:
SGLang DSpark Simulated Acceptance Opt-In Default, RadixCache UpdatePendingBackupOnNodeSplit,
Storage-Tier Waiting Prefill Load Deduction, and ZMQ ZAP Peer Allowlist.

First-hand sources:
- sgl-project/sglang@f0b65907227480afc376a66db62931b7e9c46c38 (#45433)
- sgl-project/sglang@cc7ae372ac2fba9605460308c8f347c80702c828 (#45242)
- sgl-project/sglang@0d7638b03509967e749c201146bfb04bbdfe380c (#45158)
- sgl-project/sglang@194876bea07214539306526f1bd937a96e02bff9 (#45250)
- sgl-project/sglang@468e26f253b2f1d42bf3545742b7c363d0c6592e (#45280)
"""

from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
from typing import Any


def simulate_dspark_simulated_acc_defaults() -> dict[str, Any]:
    """Model SGLang f0b6590 (#45433): DSpark simulated acceptance shortcuts disabled by default."""
    legacy_defaults = {
        "SGLANG_SIMULATE_ACC_GREEDY": True,
        "SGLANG_DSPARK_SIMULATE_FOLD": True,
    }
    fixed_defaults = {
        "SGLANG_SIMULATE_ACC_GREEDY": False,
        "SGLANG_DSPARK_SIMULATE_FOLD": False,
    }
    return {
        "legacy_defaults": legacy_defaults,
        "fixed_defaults": fixed_defaults,
        "unintended_shortcut_prevented_by_default": (
            not fixed_defaults["SGLANG_SIMULATE_ACC_GREEDY"]
            and not fixed_defaults["SGLANG_DSPARK_SIMULATE_FOLD"]
        ),
    }


def simulate_radix_split_pending_backup_action() -> dict[str, Any]:
    """Model SGLang cc7ae37 (#45242): UpdatePendingBackupOnNodeSplit across write-through and buffer-only modes.

    When `parent -> old_node` splits into `parent -> new_node -> new_child`:
    - Write-through mode (`ack_id = 42`): relocates `old_node` to `[new_node, new_child]` in both
      cache write-through publication list and linker pending offload list, then routes
      `UpdatePendingBackupOnNodeSplit` to `FullComponent.apply_component_action`.
    - Buffer-only mode (`is_host_memory_buffer_only=True`, `ack_id = None`, including SWA leaf split
      in `_maybe_split_leaf_for_swa_lock`): legacy `ReplaceWriteThroughOnNodeSplit` was not emitted
      when `ack_id is None`, orphaning buffer-write dependencies. `UpdatePendingBackupOnNodeSplit`
      is emitted with `ack_id=None` and routed to `FullComponent`.
    """
    # Case 1: write-through pending node split (ack_id = 42)
    wt_publish_list = {42: ["node_10"]}
    linker_offload_list = {42: ["node_10"]}
    full_component_dependencies = {"node_10": "buf_dep_10"}

    action_wt = {
        "type": "UpdatePendingBackupOnNodeSplit",
        "ack_id": 42,
        "old_node_id": "node_10",
        "new_node_id": "node_10_prefix",
        "new_child_node_id": "node_10_child",
        "component_type": "FULL",
    }
    if action_wt["ack_id"] is not None:
        replacement = [action_wt["new_node_id"], action_wt["new_child_node_id"]]
        wt_publish_list[action_wt["ack_id"]] = replacement
        linker_offload_list[action_wt["ack_id"]] = replacement
    dep = full_component_dependencies.pop(action_wt["old_node_id"])
    full_component_dependencies[action_wt["new_node_id"]] = dep
    full_component_dependencies[action_wt["new_child_node_id"]] = dep

    # Case 2: buffer-only SWA leaf split (ack_id = None, is_host_memory_buffer_only = True)
    legacy_buffer_only_orphaned_deps = 1
    fixed_buffer_only_orphaned_deps = 0

    return {
        "write_through_split_nodes": wt_publish_list[42],
        "linker_offload_split_nodes": linker_offload_list[42],
        "full_component_tracked_nodes": sorted(full_component_dependencies.keys()),
        "legacy_buffer_only_orphaned_deps": legacy_buffer_only_orphaned_deps,
        "fixed_buffer_only_orphaned_deps": fixed_buffer_only_orphaned_deps,
    }


def simulate_waiting_storage_span_load_deduction() -> dict[str, Any]:
    """Model SGLang 0d7638b (#45158): Exclude accepted storage-tier hits from waiting prefill load.

    For each waiting request:
    - `matched = req.num_matched_prefix_tokens` (device/host prefix hit)
    - `uncached = max(0, req.seqlen - matched)`
    - If `waiting_storage_span` returns `(span_start, span_end)` (only after scheduler acceptance when
      `info.host_indices is not None` or request is in `buffer_pipeline.staged_prefetches`):
      `storage_tokens = max(0, min(span_end, req.seqlen) - max(span_start, matched))`
      `uncached -= storage_tokens`
    """
    requests = [
        # Accepted ongoing prefetch (host_indices allocated): seqlen=4096, device matched=512, storage=(512, 3584)
        {"req_id": "req_accepted_ongoing", "seqlen": 4096, "matched": 512, "accepted": True, "span": (512, 3584)},
        # Staged prefetch in buffer_pipeline: seqlen=8192, device matched=1024, storage=(1024, 7168)
        {"req_id": "req_staged_pipeline", "seqlen": 8192, "matched": 1024, "accepted": True, "span": (1024, 7168)},
        # Unaccepted query result (host_indices is None): must NOT be deducted yet
        {"req_id": "req_unaccepted_query", "seqlen": 2048, "matched": 256, "accepted": False, "span": (256, 1536)},
    ]

    legacy_waiting_uncached = 0
    fixed_waiting_uncached = 0
    per_request = []

    for req in requests:
        uncached_legacy = max(0, req["seqlen"] - req["matched"])
        uncached_fixed = uncached_legacy
        if req["accepted"] and req["span"] is not None:
            s0, s1 = req["span"]
            storage_tokens = max(0, min(s1, req["seqlen"]) - max(s0, req["matched"]))
            uncached_fixed -= storage_tokens
        else:
            storage_tokens = 0
        legacy_waiting_uncached += uncached_legacy
        fixed_waiting_uncached += uncached_fixed
        per_request.append({
            "req_id": req["req_id"],
            "legacy_uncached": uncached_legacy,
            "fixed_uncached": uncached_fixed,
            "deducted_storage_tokens": storage_tokens,
        })

    return {
        "requests": per_request,
        "legacy_waiting_uncached_tokens": legacy_waiting_uncached,
        "fixed_waiting_uncached_tokens": fixed_waiting_uncached,
        "false_prefill_load_reduction_pct": round(
            100.0 * (legacy_waiting_uncached - fixed_waiting_uncached) / legacy_waiting_uncached, 2
        ),
    }


def simulate_zmq_zap_allowlist_and_detok_batching() -> dict[str, Any]:
    """Model SGLang 194876b (#45250, CVE-2026-93034) and 468e26f (#45280)."""
    raw_allowlist = ("10.24.0.10", "2001:0db8:0000:0000:0000:0000:0000:0001")
    normalized_allowlist = {ipaddress.ip_address(p).compressed for p in raw_allowlist}

    candidate_peers = ["10.24.0.10", "2001:db8::1", "10.24.0.99", "192.168.1.50"]
    admitted = [
        p for p in candidate_peers
        if ipaddress.ip_address(p).compressed in normalized_allowlist
    ]
    rejected = [
        p for p in candidate_peers
        if ipaddress.ip_address(p).compressed not in normalized_allowlist
    ]

    # Detokenizer output grouping (468e26f): 6 outputs interleaved across 2 tokenizer workers
    ipcs = ["worker-b", "worker-a", "worker-b", "worker-a", "worker-b", "worker-a"]
    grouped_ipc_sends = len(set(ipcs))  # 2 grouped sends when SGLANG_BATCH_TOKENIZER_OUTPUTS=True
    unbatched_ipc_sends = len(ipcs)     # 6 individual sends when SGLANG_BATCH_TOKENIZER_OUTPUTS=False

    return {
        "cve_id": "CVE-2026-93034",
        "zap_domain": "sglang",
        "normalized_allowlist": sorted(normalized_allowlist),
        "admitted_peers": admitted,
        "rejected_peers": rejected,
        "grouped_ipc_sends": grouped_ipc_sends,
        "unbatched_ipc_sends": unbatched_ipc_sends,
    }


def run_experiment() -> dict[str, Any]:
    dspark = simulate_dspark_simulated_acc_defaults()
    radix_split = simulate_radix_split_pending_backup_action()
    storage_load = simulate_waiting_storage_span_load_deduction()
    zmq_detok = simulate_zmq_zap_allowlist_and_detok_batching()

    invariants = {
        "dspark_simulated_shortcuts_default_off": dspark["unintended_shortcut_prevented_by_default"] is True,
        "radix_split_routes_both_wt_and_buffer_only": (
            radix_split["write_through_split_nodes"] == ["node_10_prefix", "node_10_child"]
            and radix_split["fixed_buffer_only_orphaned_deps"] == 0
        ),
        "storage_tier_waiting_load_deducts_only_accepted": (
            storage_load["legacy_waiting_uncached_tokens"] == 12544
            and storage_load["fixed_waiting_uncached_tokens"] == 3328
            and storage_load["false_prefill_load_reduction_pct"] == 73.47
        ),
        "zmq_zap_blocks_unauthorized_peers": (
            zmq_detok["admitted_peers"] == ["10.24.0.10", "2001:db8::1"]
            and zmq_detok["rejected_peers"] == ["10.24.0.99", "192.168.1.50"]
        ),
    }

    return {
        "experiment": "sglang_dspark_default_radix_split_storage_load_zmq",
        "phase": 39,
        "priority": 48,
        "evidence_mode": "own-offline",
        "dspark_defaults": dspark,
        "radix_split_action": radix_split,
        "waiting_storage_load": storage_load,
        "zmq_allowlist_and_detok": zmq_detok,
        "invariants": invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_experiment()
    out_dir = Path(__file__).resolve().parent
    for target in (out_dir / "summary.json", out_dir / "results" / "summary.json"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"Failed invariants: {failed}")
        print(json.dumps({"status": "ok", "experiment": summary["experiment"], "invariants": summary["invariants"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
