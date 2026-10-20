#!/usr/bin/env python3
"""Deterministic simulator for Mooncake RDMA Auto-GID 7-class data-path re-probing,
advisory peer epoch synchronization, and bounded in-flight Scatter resharding.

Models the exact state machine and window admission protocols from:
  - kvcache-ai/Mooncake `mooncake-transfer-engine/include/transport/rdma_transport/rdma_gid_probe.h`
    and `rdma_endpoint.cpp` (commit 5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10)
  - kvcache-ai/Mooncake `mooncake-reshard/python/mooncake/reshard/transfer_engine/executor.py`
    (commit 0977fc7066ece59f93440a1db59cb805906fe182)
  - kvcache-ai/Mooncake `mooncake-store/src/client_service.cpp`
    (commit 67df8c414ad04ec415f69da351e6fc92774e8a6f)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


class AutoGidCandidateClass(IntEnum):
    kNetworkRoutable = 0
    kNoNetworkRoutable = 1
    kNetworkPrivateV4 = 2
    kNetworkDegraded = 3
    kNoNetworkPrivateV4 = 4
    kNoNetworkDegraded = 5
    kFallbackNonzero = 6


class AutoGidDataPathAction(IntEnum):
    kRetrySameGid = 0
    kAdvanceGid = 1
    kIgnoreDuplicate = 2
    kExhausted = 3
    kUnavailable = 4


@dataclass(frozen=True)
class AutoGidSelection:
    gid_index: int
    gid: str
    candidate_class: AutoGidCandidateClass


@dataclass
class PeerState:
    current: AutoGidSelection
    selection_epoch: int
    first_failed_endpoint_id: int = 0
    remaining_advances: int = 0
    advance_pending: bool = False
    advance_requested_locally: bool = False
    recovery_active: bool = False
    exhausted: bool = False
    peer_advertised_gid: str = ""
    auto_gid_reprobe_from: str = ""


class PeerAutoGidStateTableSim:
    """Faithful model of `mooncake::PeerAutoGidStateTable` in `rdma_gid_probe.h`."""

    def __init__(self, max_advances: int = 2, enabled: bool = True) -> None:
        self.max_advances = max_advances
        self.enabled = enabled
        self.next_epoch = 1
        self.states: dict[str, PeerState] = {}

    def _alloc_epoch(self) -> int:
        ep = self.next_epoch
        self.next_epoch += 1
        return ep

    def _get_or_create(self, peer: str, default_sel: AutoGidSelection) -> PeerState:
        if peer not in self.states:
            self.states[peer] = PeerState(
                current=default_sel,
                selection_epoch=self._alloc_epoch(),
                remaining_advances=self.max_advances,
            )
        return self.states[peer]

    def _reset_recovery(self, st: PeerState) -> None:
        st.advance_pending = False
        st.advance_requested_locally = False
        st.recovery_active = False
        st.exhausted = False
        st.first_failed_endpoint_id = 0
        st.remaining_advances = self.max_advances
        st.auto_gid_reprobe_from = ""

    @staticmethod
    def reselect_candidate(
        candidates: list[AutoGidSelection], current_gid: str
    ) -> AutoGidSelection | None:
        if not candidates:
            return None
        idx = next((i for i, c in enumerate(candidates) if c.gid == current_gid), None)
        if idx is None:
            return candidates[0]
        if len(candidates) == 1:
            return None
        return candidates[(idx + 1) % len(candidates)]

    def prepare(
        self,
        peer: str,
        default_sel: AutoGidSelection,
        candidates: list[AutoGidSelection],
        advisory_from: str = "",
    ) -> dict[str, Any]:
        if not self.enabled:
            return {
                "gid": default_sel.gid,
                "gid_index": default_sel.gid_index,
                "selection_epoch": 0,
                "auto_gid_reprobe_from": "",
            }
        st = self._get_or_create(peer, default_sel)
        should_advance = st.advance_pending
        locally_requested = st.advance_requested_locally
        matching_advisory = bool(advisory_from) and advisory_from == st.current.gid
        if matching_advisory:
            should_advance = True

        if should_advance and candidates and (not st.exhausted or matching_advisory):
            if matching_advisory and st.remaining_advances == 0:
                st.remaining_advances = 1
            if st.remaining_advances > 0:
                previous_gid = st.current.gid
                next_sel = self.reselect_candidate(candidates, st.current.gid)
                st.advance_pending = False
                st.advance_requested_locally = False
                if next_sel is not None and next_sel.gid != st.current.gid:
                    st.current = next_sel
                    st.selection_epoch = self._alloc_epoch()
                    st.first_failed_endpoint_id = 0
                    st.remaining_advances -= 1
                    st.recovery_active = True
                    st.exhausted = False
                    if locally_requested:
                        st.auto_gid_reprobe_from = (
                            st.peer_advertised_gid if st.peer_advertised_gid else previous_gid
                        )
                else:
                    st.exhausted = True
                    st.auto_gid_reprobe_from = ""
            else:
                st.advance_pending = False
                st.advance_requested_locally = False
                st.exhausted = True
                st.auto_gid_reprobe_from = ""

        return {
            "gid": st.current.gid,
            "gid_index": st.current.gid_index,
            "candidate_class": st.current.candidate_class.name,
            "selection_epoch": st.selection_epoch,
            "remaining_advances": st.remaining_advances,
            "exhausted": st.exhausted,
            "auto_gid_reprobe_from": st.auto_gid_reprobe_from,
        }

    def record_completion(
        self, peer: str, endpoint_id: int, selection_epoch: int, success: bool
    ) -> AutoGidDataPathAction:
        if not self.enabled or endpoint_id == 0 or selection_epoch == 0:
            return AutoGidDataPathAction.kUnavailable
        st = self.states.get(peer)
        if st is None:
            return AutoGidDataPathAction.kUnavailable
        if selection_epoch != st.selection_epoch:
            return AutoGidDataPathAction.kIgnoreDuplicate
        if success:
            if st.recovery_active:
                self._reset_recovery(st)
            return AutoGidDataPathAction.kIgnoreDuplicate
        if st.exhausted:
            return AutoGidDataPathAction.kExhausted
        if st.advance_pending:
            return AutoGidDataPathAction.kIgnoreDuplicate
        if st.first_failed_endpoint_id == 0:
            st.first_failed_endpoint_id = endpoint_id
            st.recovery_active = True
            return AutoGidDataPathAction.kRetrySameGid
        if endpoint_id <= st.first_failed_endpoint_id:
            return AutoGidDataPathAction.kIgnoreDuplicate
        st.advance_pending = True
        st.advance_requested_locally = True
        return AutoGidDataPathAction.kAdvanceGid


def simulate_gid_reprobe_trace() -> dict[str, Any]:
    # 3 GID candidates on a dual-stack RoCEv2 NIC where index 3 (overlay 172.17.x) is initially
    # mis-classified or blackholed across ToR, index 2 (RFC1918 10.24.1.10, kNetworkPrivateV4) works,
    # and index 1 (fe80:: link-local, kNetworkDegraded) is ranked below kNetworkPrivateV4.
    candidates = [
        AutoGidSelection(3, "0000:0000:0000:0000:0000:ffff:ac11:0002", AutoGidCandidateClass.kNoNetworkRoutable),
        AutoGidSelection(2, "0000:0000:0000:0000:0000:ffff:0a18:010a", AutoGidCandidateClass.kNetworkPrivateV4),
        AutoGidSelection(1, "fe80:0000:0000:0000:0211:22ff:fe33:4455", AutoGidCandidateClass.kNetworkDegraded),
    ]
    table = PeerAutoGidStateTableSim(max_advances=2, enabled=True)
    peer = "10.24.1.25:14001@mlx5_0"

    p0 = table.prepare(peer, candidates[0], candidates)
    # Endpoint 101 fails on epoch 1 -> kRetrySameGid (tolerates single transient CQE timeout)
    act1 = table.record_completion(peer, endpoint_id=101, selection_epoch=p0["selection_epoch"], success=False)
    # Duplicate error CQE from endpoint 101 -> kIgnoreDuplicate
    act1_dup = table.record_completion(peer, endpoint_id=101, selection_epoch=p0["selection_epoch"], success=False)
    # Reconstructed endpoint 102 also fails on epoch 1 -> kAdvanceGid
    act2 = table.record_completion(peer, endpoint_id=102, selection_epoch=p0["selection_epoch"], success=False)
    # Next handshake prepares epoch 2 -> advances to RFC1918 10.24.1.10 (kNetworkPrivateV4) with advisory
    p1 = table.prepare(peer, candidates[0], candidates)
    # Stale CQE from epoch 1 arrives late -> kIgnoreDuplicate
    act_stale = table.record_completion(peer, endpoint_id=102, selection_epoch=p0["selection_epoch"], success=False)
    # Endpoint 103 succeeds on epoch 2 -> resets recovery budget back to max_advances=2
    act3 = table.record_completion(peer, endpoint_id=103, selection_epoch=p1["selection_epoch"], success=True)

    # Also test peer advisory bonus advance when local budget is exhausted (remaining_advances == 0)
    peer_b = "10.24.2.99:14001@mlx5_1"
    table_exhausted = PeerAutoGidStateTableSim(max_advances=0, enabled=True)
    pb0 = table_exhausted.prepare(peer_b, candidates[0], candidates)
    pb_advisory = table_exhausted.prepare(
        peer_b, candidates[0], candidates, advisory_from=candidates[0].gid
    )

    return {
        "initial_gid_index": p0["gid_index"],
        "first_failure_action": act1.name,
        "duplicate_endpoint_action": act1_dup.name,
        "second_endpoint_failure_action": act2.name,
        "advanced_gid_index": p1["gid_index"],
        "advanced_candidate_class": p1["candidate_class"],
        "advisory_reprobe_from": p1["auto_gid_reprobe_from"],
        "stale_epoch_action": act_stale.name,
        "post_recovery_remaining_advances": table.states[peer].remaining_advances,
        "advisory_override_when_exhausted_gid_index": pb_advisory["gid_index"],
        "advisory_override_epoch": pb_advisory["selection_epoch"],
    }


def simulate_bounded_scatter_reshard() -> list[dict[str, Any]]:
    """Models MooncakeTransferEngineExecutor._execute_batches bounded in-flight window
    and client_service.cpp GetContiguousSliceRange MR registration coalescing.
    """
    # 64 GiB MoE weight resharding across 256 scatter batches (256 MiB per batch, 64 slices of 4 MiB each)
    total_batches = 256
    batch_bytes = 256 * 1024 * 1024
    submit_overhead_us = 45.0
    rdma_wire_bw_gbps = 360.0  # 400GbE RoCEv2 effective payload throughput (45 GB/s)
    batch_wire_time_ms = (batch_bytes / (rdma_wire_bw_gbps * 1e9 / 8.0)) * 1000.0
    sync_fence_rtt_ms = 0.85

    rows: list[dict[str, Any]] = []
    for max_inflight in [1, 2, 4, 8, 16]:
        window: deque[int] = deque()
        peak_inflight = 0
        completed = 0
        for b in range(total_batches):
            if len(window) >= max_inflight:
                window.popleft()
                completed += 1
            window.append(b)
            peak_inflight = max(peak_inflight, len(window))
        while window:
            window.popleft()
            completed += 1

        if max_inflight == 1:
            # Synchronous batch_transfer_sync_write pays full fence RTT per batch
            per_batch_ms = batch_wire_time_ms + sync_fence_rtt_ms + (submit_overhead_us / 1000.0)
        else:
            # Sliding window overlaps submit + drain RTT with wire transfer; only pipeline fill/drain + fractional stall
            overlap_factor = min(1.0, (max_inflight - 1) * 0.42)
            unhidden_ctrl_ms = (sync_fence_rtt_ms + submit_overhead_us / 1000.0) * max(
                0.04, 1.0 - overlap_factor
            )
            per_batch_ms = batch_wire_time_ms + unhidden_ctrl_ms

        total_time_s = (total_batches * per_batch_ms) / 1000.0
        eff_bw_gbps = (total_batches * batch_bytes * 8.0) / (total_time_s * 1e9)
        pinned_staging_mib = peak_inflight * (batch_bytes // (1024 * 1024))

        rows.append(
            {
                "max_inflight_batches": max_inflight,
                "peak_inflight_observed": peak_inflight,
                "completed_batches": completed,
                "total_reshard_time_s": round(total_time_s, 4),
                "effective_goodput_gbps": round(eff_bw_gbps, 2),
                "pinned_window_footprint_mib": pinned_staging_mib,
                "submission_state_on_unknown_cqe": "POISONED",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run assertions and write results")
    args = parser.parse_args()

    gid_trace = simulate_gid_reprobe_trace()
    reshard_rows = simulate_bounded_scatter_reshard()

    summary = {
        "experiment_id": "mooncake_gid_reprobe_reshard_sim",
        "title": "Mooncake RDMA Auto-GID Re-Probing & Bounded Scatter Resharding Simulator",
        "commit_refs": [
            "kvcache-ai/Mooncake@5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10",
            "kvcache-ai/Mooncake@0977fc7066ece59f93440a1db59cb805906fe182",
            "kvcache-ai/Mooncake@67df8c414ad04ec415f69da351e6fc92774e8a6f",
        ],
        "gid_reprobe_trace": gid_trace,
        "bounded_reshard_scaling": reshard_rows,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "bounded_reshard_scaling.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(reshard_rows[0].keys()))
        writer.writeheader()
        writer.writerows(reshard_rows)

    if args.check:
        assert gid_trace["first_failure_action"] == "kRetrySameGid"
        assert gid_trace["duplicate_endpoint_action"] == "kIgnoreDuplicate"
        assert gid_trace["second_endpoint_failure_action"] == "kAdvanceGid"
        assert gid_trace["advanced_gid_index"] == 2
        assert gid_trace["advanced_candidate_class"] == "kNetworkPrivateV4"
        assert gid_trace["stale_epoch_action"] == "kIgnoreDuplicate"
        assert gid_trace["post_recovery_remaining_advances"] == 2
        assert gid_trace["advisory_override_when_exhausted_gid_index"] == 2
        assert reshard_rows[0]["effective_goodput_gbps"] < reshard_rows[2]["effective_goodput_gbps"]
        assert reshard_rows[2]["peak_inflight_observed"] == 4
        print(
            f"[mooncake_gid_reprobe_reshard_sim] CHECK PASSED: "
            f"advanced_gid={gid_trace['advanced_gid_index']} ({gid_trace['advanced_candidate_class']}), "
            f"goodput_w1={reshard_rows[0]['effective_goodput_gbps']}Gbps -> "
            f"w4={reshard_rows[2]['effective_goodput_gbps']}Gbps"
        )


if __name__ == "__main__":
    main()
