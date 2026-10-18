#!/usr/bin/env python3
"""CPU/offline analytical and discrete-event model for UEC 1.1 / UET / PCM fabric congestion & spraying.

Covers:
1. Header overhead & wire efficiency across RoCEv2 (IPv4/UDP/BTH/RETH), UET over IPv4 (`uet_pkt_hdr.h`:
   `uet_entropy` + `uet_pds_req`/`uet_pds_rudi_req` + `uet_ses_req_*` + optional `uet_sec_ssi` + 4B CRC32C),
   and UEC 1.1 scale-up Unified Forwarding Header (`UFH` saving 20B over IP + up to 11B IPG reduction).
2. UET NSCC / SmaRTTrack (`ultraethernet/uet-ref-prov/cc/uet_cc_st.c` & `spcl/pcm/algorithms/nscc.c`) vs
   SMaRTT (`spcl/pcm/algorithms/smartt.c`) vs Swift (`spcl/pcm/algorithms/swift.c`) four-quadrant ACK/ECN/delay
   reactions and Quick Adapt (`quick_adapt`) under 16-to-1 AI All-to-All / Incast bursts.
3. Multi-path load balancing & packet spraying across `UET_CC_NO_SPRAY` (ECMP hash collision),
   `UET_CC_OBLIVIOUS_SPRAY` (round-robin spraying onto degraded path), `UET_CC_SELECTIVE_SPRAY`
   (`uet_cc_st.c:110-287` 2-bit per-EV state `GOOD/BAD/SKIP`), and PCM `REPS` (`spcl/pcm/algorithms/reps.c`
   circular ACK-EV cache recycling + hash exploration).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return val


def evaluate_header_efficiency(case: dict[str, Any]) -> dict[str, Any]:
    """Compare wire header overhead and small/medium packet goodput efficiency."""
    payload_bytes = _validate_positive_int("payload_bytes", case.get("payload_bytes"))
    line_rate_gbps = _validate_positive_float("line_rate_gbps", case.get("line_rate_gbps"))
    pds_mode = str(case.get("pds_mode", "rod"))
    ses_hdr_type = str(case.get("ses_hdr_type", "std"))
    tss_security = bool(case.get("tss_security", False))
    sync_ext = bool(case.get("sync_ext", False))
    scale_up_ufh = bool(case.get("scale_up_ufh", False))

    if pds_mode not in {"rod", "rud", "rudi", "uud"}:
        raise ValueError(f"Unsupported pds_mode: {pds_mode!r}")
    if ses_hdr_type not in {"std", "small_onm", "medium_smsg"}:
        raise ValueError(f"Unsupported ses_hdr_type: {ses_hdr_type!r}")

    # Standard Ethernet framing constants (bytes)
    eth_l2_bytes = 14
    eth_fcs_bytes = 4
    std_preamble_sfd_bytes = 8
    std_ipg_bytes = 12

    # Baseline RoCEv2 over IPv4/UDP (RDMA WRITE/READ with RETH):
    # L2(14) + IPv4(20) + UDP(8) + IB BTH(12) + RETH(16) + ICRC(4) + FCS(4) = 78B frame overhead
    # + Preamble/SFD(8) + IPG(12) = 98B total wire overhead
    rocev2_frame_hdr_bytes = eth_l2_bytes + 20 + 8 + 12 + 16 + 4 + eth_fcs_bytes
    rocev2_wire_overhead_bytes = rocev2_frame_hdr_bytes + std_preamble_sfd_bytes + std_ipg_bytes
    rocev2_total_wire_bytes = payload_bytes + rocev2_wire_overhead_bytes
    rocev2_goodput_gbps = line_rate_gbps * (payload_bytes / rocev2_total_wire_bytes)

    # UET exact struct sizes from ultraethernet/uet-ref-prov/uet_pkt_hdr.h:
    # struct uet_entropy = 4B
    # struct uet_pds_req = 12B (ROD/RUD), struct uet_pds_rudi_req = 8B (RUDI), struct uet_pds_uud_req = 4B (UUD)
    # struct uet_ses_req_cmn = 12B
    #   - uet_ses_req_onm (UET_HDR_REQ_SMALL) = 12 + 8 = 20B
    #   - uet_ses_req_smsg (UET_HDR_REQ_MEDIUM) = 12 + 8 + 4 + 8 = 32B
    #   - uet_ses_req_std (UET_HDR_REQ_STD) = 12 + 8 + 4 + 8 + 8 + 4 = 44B
    # struct uet_ses_sync_ext = 4B (when UET_SYNC_WRITE / UET_SYNC_ATOMIC)
    # struct uet_sec_ssi = 16B + 16B AES-GCM auth tag (replaces 4B CRC32C when TSS security enabled)
    pds_bytes_map = {"rod": 12, "rud": 12, "rudi": 8, "uud": 4}
    ses_bytes_map = {"small_onm": 20, "medium_smsg": 32, "std": 44}

    pds_hdr_bytes = pds_bytes_map[pds_mode]
    ses_hdr_bytes = ses_bytes_map[ses_hdr_type] + (4 if sync_ext else 0)
    sec_or_crc_bytes = (16 + 16) if tss_security else 4

    # Scale-out IPv4 + uet_entropy(4B) vs UEC 1.1 Scale-up UFH (saves 20B vs IPv4 and up to 11B IPG)
    if scale_up_ufh:
        net_and_entropy_bytes = 4  # UFH overlays L2/routing saving 20B compared to IPv4(20B)+entropy(4B)
        ipg_bytes = std_ipg_bytes - 11  # saves up to 11 idle IPG bytes per UEC 1.1 announcement
    else:
        net_and_entropy_bytes = 20 + 4  # IPv4 (20B) + struct uet_entropy (4B)
        ipg_bytes = std_ipg_bytes

    uet_frame_hdr_bytes = (
        eth_l2_bytes + net_and_entropy_bytes + pds_hdr_bytes + ses_hdr_bytes + sec_or_crc_bytes + eth_fcs_bytes
    )
    uet_wire_overhead_bytes = uet_frame_hdr_bytes + std_preamble_sfd_bytes + ipg_bytes
    uet_total_wire_bytes = payload_bytes + uet_wire_overhead_bytes
    uet_goodput_gbps = line_rate_gbps * (payload_bytes / uet_total_wire_bytes)

    return {
        "case_id": str(case["case_id"]),
        "payload_bytes": payload_bytes,
        "pds_mode": pds_mode,
        "ses_hdr_type": ses_hdr_type,
        "tss_security": tss_security,
        "sync_ext": sync_ext,
        "scale_up_ufh": scale_up_ufh,
        "rocev2_wire_overhead_bytes": rocev2_wire_overhead_bytes,
        "uet_frame_hdr_bytes": uet_frame_hdr_bytes,
        "uet_wire_overhead_bytes": uet_wire_overhead_bytes,
        "wire_bytes_saved_vs_rocev2": rocev2_wire_overhead_bytes - uet_wire_overhead_bytes,
        "rocev2_goodput_gbps": round(rocev2_goodput_gbps, 4),
        "uet_goodput_gbps": round(uet_goodput_gbps, 4),
        "goodput_gain_pct_vs_rocev2": round(
            (uet_goodput_gbps - rocev2_goodput_gbps) / rocev2_goodput_gbps * 100.0, 2
        ),
    }


def evaluate_nscc_smartt_swift_reaction(case: dict[str, Any]) -> dict[str, Any]:
    """Evaluate UET NSCC (`uet_cc_st.c` & `spcl/pcm/algorithms/nscc.c`), SMaRTT, and Swift cwnd transitions."""
    mtu_bytes = _validate_positive_int("mtu_bytes", case.get("mtu_bytes"))
    bw_gbps = _validate_positive_float("bw_gbps", case.get("bw_gbps"))
    base_rtt_us = _validate_positive_float("base_rtt_us", case.get("base_rtt_us"))
    trim_supp = bool(case.get("trim_supp", True))
    init_cwnd_bdp_fraction = _validate_positive_float(
        "init_cwnd_bdp_fraction", case.get("init_cwnd_bdp_fraction", 1.0)
    )
    rtt_sample_us = _validate_positive_float("rtt_sample_us", case.get("rtt_sample_us"))
    ecn_marked = bool(case.get("ecn_marked", False))
    packet_trimmed_or_nack = bool(case.get("packet_trimmed_or_nack", False))
    acked_mtus = _validate_positive_int("acked_mtus", case.get("acked_mtus", 8))
    achieved_bytes_in_qa_window = _validate_non_negative_int(
        "achieved_bytes_in_qa_window", case.get("achieved_bytes_in_qa_window", 0)
    )

    bw_mbps = bw_gbps * 1000.0
    # uet_cc_st.c:154-174 exact parameter formulas
    bdp_bytes = int(round(bw_mbps * base_rtt_us / 8.0))
    cwnd_max = int(round(1.25 * bdp_bytes))
    target_qdelay_us = base_rtt_us / 2.0 if trim_supp else base_rtt_us
    qa_thresh_us = (4.0 if trim_supp else 2.0) * target_qdelay_us
    pi_bytes = 5 * mtu_bytes
    scaling_c = float(mtu_bytes) / (target_qdelay_us * pi_bytes)
    alpha = pi_bytes * mtu_bytes * scaling_c
    fd = 0.8
    fi = 1.0
    gamma = 0.8
    max_md_factor = 0.5
    scaling_factor_a = float(bdp_bytes) / 150000.0
    eta = 0.15 * mtu_bytes * scaling_factor_a

    init_cwnd = max(mtu_bytes, int(round(bdp_bytes * init_cwnd_bdp_fraction)))
    newly_acked_bytes = acked_mtus * mtu_bytes
    qdelay_us = max(0.0, rtt_sample_us - base_rtt_us)

    # NSCC branch selection (uet_cc_st.c:480-508 & spcl/pcm/algorithms/nscc.c:88-165)
    nscc_cwnd = float(init_cwnd)
    inc_bytes = 0.0
    dec_bytes = 0.0
    if packet_trimmed_or_nack or qdelay_us > qa_thresh_us:
        nscc_branch = "quick_adapt"
        nscc_cwnd = float(max(mtu_bytes, min(init_cwnd, achieved_bytes_in_qa_window)))
    elif not ecn_marked and qdelay_us == 0.0:
        nscc_branch = "fast_increase"
        nscc_cwnd = float(init_cwnd + newly_acked_bytes)
    elif not ecn_marked and qdelay_us < target_qdelay_us:
        nscc_branch = "proportional_increase"
        inc_bytes = min(
            float(newly_acked_bytes * pi_bytes),
            alpha * (target_qdelay_us - qdelay_us) * newly_acked_bytes / float(mtu_bytes),
        ) + fi * mtu_bytes * newly_acked_bytes
        nscc_cwnd = max(float(mtu_bytes), init_cwnd + (inc_bytes + eta) / init_cwnd)
    elif not ecn_marked and qdelay_us >= target_qdelay_us:
        nscc_branch = "fair_increase"
        inc_bytes = fi * mtu_bytes * newly_acked_bytes
        nscc_cwnd = max(float(mtu_bytes), init_cwnd + (inc_bytes + eta) / init_cwnd)
    elif ecn_marked and qdelay_us >= target_qdelay_us:
        nscc_branch = "multiplicative_decrease"
        md_factor = max(1.0 - gamma * (1.0 - target_qdelay_us / qdelay_us), max_md_factor)
        dec_bytes = fd * newly_acked_bytes
        cwnd_after_md = init_cwnd * md_factor
        nscc_cwnd = max(
            float(mtu_bytes),
            cwnd_after_md + eta / cwnd_after_md - (dec_bytes * cwnd_after_md / bdp_bytes),
        )
    else:
        # spcl/pcm/algorithms/nscc.c:110-118: get_signal(SIG_NUM_ECN) && q_delay < TARGET_Q_DELAY -> NOOP, switch path
        nscc_branch = "ecn_low_delay_noop_switch_path"
        nscc_cwnd = float(init_cwnd)

    # SMaRTT (`spcl/pcm/algorithms/smartt.c:62-82`) reaction on the same signal
    trtt_us = base_rtt_us + target_qdelay_us
    x_gain = 0.15
    y_gain = 1.0
    w_gain = 0.8
    z_gain = 0.5
    smartt_cwnd = float(init_cwnd)
    if packet_trimmed_or_nack:
        smartt_branch = "quick_adapt"
        smartt_cwnd = float(max(mtu_bytes, achieved_bytes_in_qa_window))
    elif not ecn_marked and rtt_sample_us < trtt_us:
        smartt_branch = "case1_rtt_fair_increase"
        per_ack = min(
            ((trtt_us - rtt_sample_us) / rtt_sample_us) * y_gain * mtu_bytes * (mtu_bytes / smartt_cwnd),
            float(mtu_bytes),
        ) + (mtu_bytes / smartt_cwnd) * x_gain * mtu_bytes
        smartt_cwnd += acked_mtus * per_ack
    elif ecn_marked and rtt_sample_us > trtt_us:
        smartt_branch = "case2_hybrid_decrease"
        per_ack_dec = min(
            w_gain * ((rtt_sample_us - trtt_us) / rtt_sample_us) * mtu_bytes
            + (smartt_cwnd / bdp_bytes) * z_gain * mtu_bytes,
            float(mtu_bytes),
        )
        smartt_cwnd = max(float(mtu_bytes), smartt_cwnd - acked_mtus * per_ack_dec)
    elif ecn_marked and rtt_sample_us <= trtt_us:
        smartt_branch = "case3_gentle_window_decrease"
        per_ack_dec = max(float(mtu_bytes), (smartt_cwnd / bdp_bytes) * mtu_bytes * z_gain)
        smartt_cwnd = max(float(mtu_bytes), smartt_cwnd - per_ack_dec)
    else:
        smartt_branch = "case4_proportional_fairness_only"
        smartt_cwnd += acked_mtus * ((mtu_bytes / smartt_cwnd) * x_gain * mtu_bytes)

    # Swift (`spcl/pcm/algorithms/swift.c:8-47`) pure delay-based reaction (ignores ECN bit)
    fs_alpha = 4.0
    fs_beta = 1.0
    fs_range = 12.0
    hop_delay_us = 2.0
    fs_delay = min(fs_range, max(0.0, fs_alpha / math.sqrt(max(1.0, init_cwnd / float(mtu_bytes))) + fs_beta))
    swift_target_rtt_us = base_rtt_us + fs_delay + hop_delay_us
    swift_ai = 1.0
    swift_beta = 0.8
    swift_max_mdf = 0.5
    if packet_trimmed_or_nack:
        swift_cwnd = max(float(mtu_bytes), init_cwnd * (1.0 - swift_max_mdf))
    elif rtt_sample_us < swift_target_rtt_us:
        swift_cwnd = init_cwnd + (mtu_bytes * swift_ai * acked_mtus * mtu_bytes) / float(init_cwnd)
    else:
        mdf = swift_beta * ((rtt_sample_us - swift_target_rtt_us) / rtt_sample_us)
        swift_cwnd = max(float(mtu_bytes), init_cwnd * max(1.0 - mdf, 1.0 - swift_max_mdf))

    return {
        "case_id": str(case["case_id"]),
        "bdp_bytes": bdp_bytes,
        "cwnd_max_bytes": cwnd_max,
        "target_qdelay_us": round(target_qdelay_us, 2),
        "qa_thresh_us": round(qa_thresh_us, 2),
        "init_cwnd_bytes": init_cwnd,
        "measured_qdelay_us": round(qdelay_us, 2),
        "ecn_marked": ecn_marked,
        "packet_trimmed_or_nack": packet_trimmed_or_nack,
        "nscc_branch": nscc_branch,
        "nscc_next_cwnd_bytes": int(round(nscc_cwnd)),
        "smartt_branch": smartt_branch,
        "smartt_next_cwnd_bytes": int(round(smartt_cwnd)),
        "swift_target_rtt_us": round(swift_target_rtt_us, 2),
        "swift_next_cwnd_bytes": int(round(swift_cwnd)),
    }


def evaluate_multipath_spraying(case: dict[str, Any]) -> dict[str, Any]:
    """Compare Single-Path ECMP, Oblivious Spray, UET Selective Spray (`uet_cc_st.c`), and PCM REPS (`reps.c`)."""
    num_paths = _validate_positive_int("num_paths", case.get("num_paths"))
    if num_paths & (num_paths - 1) != 0:
        raise ValueError(f"num_paths must be a power of two for REPS PATH_MASK, got {num_paths}")
    congested_paths = _validate_non_negative_int("congested_paths", case.get("congested_paths"))
    failed_blackhole_paths = _validate_non_negative_int(
        "failed_blackhole_paths", case.get("failed_blackhole_paths", 0)
    )
    if congested_paths + failed_blackhole_paths >= num_paths:
        raise ValueError("congested_paths + failed_blackhole_paths must be < num_paths")

    total_packets = _validate_positive_int("total_packets", case.get("total_packets"))
    mtu_bytes = _validate_positive_int("mtu_bytes", case.get("mtu_bytes", 4096))
    healthy_path_bw_gbps = _validate_positive_float("healthy_path_bw_gbps", case.get("healthy_path_bw_gbps", 50.0))
    congested_path_bw_gbps = _validate_positive_float(
        "congested_path_bw_gbps", case.get("congested_path_bw_gbps", 10.0)
    )
    rto_penalty_us = _validate_positive_float("rto_penalty_us", case.get("rto_penalty_us", 1000.0))

    healthy_paths = num_paths - congested_paths - failed_blackhole_paths
    total_payload_bytes = total_packets * mtu_bytes

    def _path_type(ev: int) -> str:
        if ev < failed_blackhole_paths:
            return "bad_rto"
        if ev < failed_blackhole_paths + congested_paths:
            return "congested_ecn"
        return "good"

    # 1. UET_CC_OBLIVIOUS_SPRAY (uet_cc_st.c:268-276): cycles through all num_paths unconditionally
    oblivious_counts = [0] * num_paths
    for pkt_idx in range(total_packets):
        ev = pkt_idx % num_paths
        oblivious_counts[ev] += 1

    # 2. UET_CC_SELECTIVE_SPRAY (uet_cc_st.c:110-128, 225-287):
    # Each EV has 2-bit state: GOOD (0), BAD (1 on RTO), SKIP (2 on ECN/NACK).
    # When select_entropy encounters SKIP, it skips it once and resets SKIP -> GOOD; BAD stays skipped.
    selective_state = ["GOOD"] * num_paths
    selective_counts = [0] * num_paths
    next_ev = 0
    for _ in range(total_packets):
        while True:
            ev = next_ev
            next_ev = (next_ev + 1) % num_paths
            st = selective_state[ev]
            if st == "GOOD":
                break
            if st == "SKIP":
                selective_state[ev] = "GOOD"
        selective_counts[ev] += 1
        ptype = _path_type(ev)
        if ptype == "bad_rto":
            selective_state[ev] = "BAD"
        elif ptype == "congested_ecn":
            selective_state[ev] = "SKIP"
        else:
            selective_state[ev] = "GOOD"

    # 3. PCM REPS (`spcl/pcm/algorithms/reps.c:13-59`):
    # Circular EV cache of size EVC_SIZE=num_paths. Initial explore_counter=num_paths.
    # Non-ECN ACKs push their EV into circular cache; ECN/RTO packets are discarded and NOT cached.
    evc_size = num_paths
    evc_mask = evc_size - 1
    evc = [0] * evc_size
    evc_bitmap = [0] * evc_size
    head_idx = 0
    num_valid_evs = 0
    explore_counter = num_paths
    ev_seed = 0
    reps_counts = [0] * num_paths

    for _ in range(total_packets):
        if num_valid_evs == 0 or explore_counter > 0:
            packet_ev = (ev_seed * 1103515245 + 12345) & evc_mask
            ev_seed = (ev_seed + 1) & evc_mask
            packet_ev = ev_seed  # deterministic uniform exploration step across paths
            if explore_counter > 0:
                explore_counter -= 1
        else:
            ev_cache_idx = (head_idx - num_valid_evs) & evc_mask
            packet_ev = evc[ev_cache_idx]
            evc_bitmap[ev_cache_idx] = 0
            num_valid_evs -= 1

        reps_counts[packet_ev] += 1
        # Immediate feedback model (1 BDP steady-state pipeline): non-ECN ACK caches EV
        ptype = _path_type(packet_ev)
        if ptype == "good":
            evc[head_idx] = packet_ev
            if not evc_bitmap[head_idx]:
                num_valid_evs += 1
                evc_bitmap[head_idx] = 1
            head_idx = (head_idx + 1) & evc_mask

    def _completion_time_us(counts: list[int]) -> tuple[float, int, int]:
        bad_pkts = sum(counts[i] for i in range(failed_blackhole_paths))
        cong_pkts = sum(counts[i] for i in range(failed_blackhole_paths, failed_blackhole_paths + congested_paths))
        max_time_us = 0.0
        for i, cnt in enumerate(counts):
            if cnt == 0:
                continue
            ptype = _path_type(i)
            bytes_on_path = cnt * mtu_bytes
            if ptype == "bad_rto":
                # First packet triggers RTO + retransmission on healthy path
                t_us = rto_penalty_us + (bytes_on_path * 8.0) / (healthy_path_bw_gbps * 1e3)
            elif ptype == "congested_ecn":
                t_us = (bytes_on_path * 8.0) / (congested_path_bw_gbps * 1e3)
            else:
                t_us = (bytes_on_path * 8.0) / (healthy_path_bw_gbps * 1e3)
            if t_us > max_time_us:
                max_time_us = t_us
        return max_time_us, cong_pkts, bad_pkts

    obliv_time_us, obliv_cong_pkts, obliv_bad_pkts = _completion_time_us(oblivious_counts)
    sel_time_us, sel_cong_pkts, sel_bad_pkts = _completion_time_us(selective_counts)
    reps_time_us, reps_cong_pkts, reps_bad_pkts = _completion_time_us(reps_counts)

    obliv_eff_gbps = (total_payload_bytes * 8.0) / (obliv_time_us * 1e3)
    sel_eff_gbps = (total_payload_bytes * 8.0) / (sel_time_us * 1e3)
    reps_eff_gbps = (total_payload_bytes * 8.0) / (reps_time_us * 1e3)

    return {
        "case_id": str(case["case_id"]),
        "num_paths": num_paths,
        "healthy_paths": healthy_paths,
        "congested_paths": congested_paths,
        "failed_blackhole_paths": failed_blackhole_paths,
        "total_packets": total_packets,
        "oblivious_congested_pkts": obliv_cong_pkts,
        "oblivious_bad_pkts": obliv_bad_pkts,
        "oblivious_completion_us": round(obliv_time_us, 2),
        "oblivious_effective_bw_gbps": round(obliv_eff_gbps, 2),
        "selective_congested_pkts": sel_cong_pkts,
        "selective_bad_pkts": sel_bad_pkts,
        "selective_completion_us": round(sel_time_us, 2),
        "selective_effective_bw_gbps": round(sel_eff_gbps, 2),
        "reps_congested_pkts": reps_cong_pkts,
        "reps_bad_pkts": reps_bad_pkts,
        "reps_completion_us": round(reps_time_us, 2),
        "reps_effective_bw_gbps": round(reps_eff_gbps, 2),
        "reps_speedup_vs_oblivious_x": round(obliv_time_us / reps_time_us, 2),
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise ValueError("Fixture schema_version must be 1")

    header_cases = fixture.get("header_efficiency_cases")
    cc_cases = fixture.get("congestion_control_cases")
    spray_cases = fixture.get("multipath_spraying_cases")
    if not isinstance(header_cases, list) or not header_cases:
        raise ValueError("header_efficiency_cases must be a non-empty list")
    if not isinstance(cc_cases, list) or not cc_cases:
        raise ValueError("congestion_control_cases must be a non-empty list")
    if not isinstance(spray_cases, list) or not spray_cases:
        raise ValueError("multipath_spraying_cases must be a non-empty list")

    return {
        "schema_version": 1,
        "experiment_id": "uec_fabric_congestion",
        "evidence_type": "own-offline",
        "header_efficiency": [evaluate_header_efficiency(c) for c in header_cases],
        "congestion_control": [evaluate_nscc_smartt_swift_reaction(c) for c in cc_cases],
        "multipath_spraying": [evaluate_multipath_spraying(c) for c in spray_cases],
    }


def _to_csv_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in results["header_efficiency"]:
        rows.append(
            {
                "section": "header_efficiency",
                "case_id": item["case_id"],
                "primary_metric": item["uet_goodput_gbps"],
                "secondary_metric": item["rocev2_goodput_gbps"],
                "delta_or_ratio": item["goodput_gain_pct_vs_rocev2"],
                "note": f"saved_bytes={item['wire_bytes_saved_vs_rocev2']}",
            }
        )
    for item in results["congestion_control"]:
        rows.append(
            {
                "section": "congestion_control",
                "case_id": item["case_id"],
                "primary_metric": item["nscc_next_cwnd_bytes"],
                "secondary_metric": item["swift_next_cwnd_bytes"],
                "delta_or_ratio": item["smartt_next_cwnd_bytes"],
                "note": item["nscc_branch"],
            }
        )
    for item in results["multipath_spraying"]:
        rows.append(
            {
                "section": "multipath_spraying",
                "case_id": item["case_id"],
                "primary_metric": item["reps_effective_bw_gbps"],
                "secondary_metric": item["oblivious_effective_bw_gbps"],
                "delta_or_ratio": item["reps_speedup_vs_oblivious_x"],
                "note": f"sel_bw={item['selective_effective_bw_gbps']}",
            }
        )
    return rows


def _format_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = ["section", "case_id", "primary_metric", "secondary_metric", "delta_or_ratio", "note"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="Verify committed outputs match without writing")
    args = parser.parse_args(argv)

    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_fixture(fixture)
    json_text = json.dumps(results, indent=2, sort_keys=True) + "\n"
    csv_text = _format_csv(_to_csv_rows(results))

    if args.check:
        if not args.json_out.exists() or not args.csv_out.exists():
            raise SystemExit("Committed results.json or results.csv missing")
        if args.json_out.read_text(encoding="utf-8") != json_text:
            raise SystemExit("results.json does not match deterministic evaluation")
        if args.csv_out.read_text(encoding="utf-8") != csv_text:
            raise SystemExit("results.csv does not match deterministic evaluation")
        print("own-offline uec_fabric_congestion outputs verified")
        return 0

    args.json_out.write_text(json_text, encoding="utf-8")
    args.csv_out.write_text(csv_text, encoding="utf-8")
    print("own-offline uec_fabric_congestion outputs generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
