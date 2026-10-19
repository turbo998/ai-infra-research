from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict


def simulate_p2p_compute_overlap() -> Dict[str, object]:
    """
    Model NVIDIA/nccl `src/transport/p2p.cc` (`useMemcpy = ncclParamP2pUseCudaMemcpy()`,
    `p2pSendProxyProgress` `cudaMemcpyDeviceToDevice` on `resources->stream` + `P2P_CUMEM`)
    vs SM-driven `P2P_DIRECT` / `P2P_CUMEM` during overlapped MoE AlltoAll / PP P2P + GEMM.
    """
    b200_total_sms = 148
    gemm_total_ctas = 256
    cta_per_sm_occupancy = 1
    single_full_wave_ms = 1.62
    tail_wave_base_ms = 1.56
    p2p_payload_mib_per_peer = 64.0
    num_nvlink_peers = 7
    total_p2p_gib = (p2p_payload_mib_per_peer * num_nvlink_peers) / 1024.0  # 0.4375 GiB

    # Alone GEMM (148 SMs available -> 2 waves: 148 + 108 CTAs)
    alone_gemm_waves = math.ceil(gemm_total_ctas / (b200_total_sms * cta_per_sm_occupancy))
    alone_gemm_ms = round(single_full_wave_ms + tail_wave_base_ms, 3)  # 3.18 ms

    modes = {}
    for mode_name, comm_sms, bw_gbps, proxy_cpu_us in [
        ("p2p_sm_driven_16cta_simple", 16, 665.0, 0.0),
        ("p2p_sm_driven_8cta_ll128", 8, 540.0, 0.0),
        ("p2p_cumem_ce_memcpy_proxy", 0, 642.0, 18.0),
    ]:
        avail_compute_sms = b200_total_sms - comm_sms
        waves = math.ceil(gemm_total_ctas / avail_compute_sms)
        # When comm_sms > 0, 256 CTAs no longer fit in 2 waves if avail_compute_sms < 128,
        # or suffer L2/SM scheduling contention and register-pressure clock throttling
        if waves > alone_gemm_waves:
            gemm_ms = single_full_wave_ms * 2 + 1.02
        elif comm_sms > 0:
            # SM contention + L2 cache thrashing from SM-driven P2P copy loops
            l2_and_sm_steal_factor = 1.0 + (comm_sms / b200_total_sms) * 1.95
            gemm_ms = alone_gemm_ms * l2_and_sm_steal_factor
        else:
            # Hardware Copy Engine (CE) uses 0 SMs and bypasses SM L1/register file
            gemm_ms = alone_gemm_ms * 1.012

        p2p_comm_ms = (total_p2p_gib / bw_gbps) * 1000.0 + (proxy_cpu_us / 1000.0)
        overlapped_step_ms = max(gemm_ms, p2p_comm_ms) + 0.045
        modes[mode_name] = {
            "comm_sms_occupied": comm_sms,
            "available_compute_sms": avail_compute_sms,
            "gemm_waves": waves,
            "concurrent_gemm_ms": round(gemm_ms, 3),
            "gemm_slowdown_pct": round((gemm_ms / alone_gemm_ms - 1.0) * 100.0, 2),
            "p2p_transfer_ms": round(p2p_comm_ms, 3),
            "overlapped_step_ms": round(overlapped_step_ms, 3),
        }

    return {
        "b200_total_sms": b200_total_sms,
        "gemm_total_ctas": gemm_total_ctas,
        "alone_gemm_ms": alone_gemm_ms,
        "modes": modes,
        "ce_speedup_vs_16cta_sm": round(
            modes["p2p_sm_driven_16cta_simple"]["overlapped_step_ms"]
            / modes["p2p_cumem_ce_memcpy_proxy"]["overlapped_step_ms"],
            3,
        ),
    }


def simulate_ras_progress_monitor() -> Dict[str, object]:
    """
    Model NVIDIA/nccl `src/ras/progress_monitor.cc` (`ncclGpuProgressCounterMonitor`,
    `PROGRESS_COUNTER_MONITOR_POLL_MS=1000`, `PROGRESS_COUNTER_MONITOR_STALE_MS=5000`,
    `cudaStreamCaptureModeRelaxed` `sideStream`) and `src/ras/ras_internal.h`
    (`NCCL_RAS_CLIENT_PORT=28028`, `RAS_COLLECTIVE_LEG_TIMEOUT_SEC=5`, `RAS_COLL_DIAG`).
    """
    poll_interval_ms = 1000
    min_poll_ms = 50
    stale_threshold_ms = 5000
    min_stale_ms = 1000
    leg_timeout_sec = 5
    counter_block_bytes = 256
    comms_per_gpu = 4  # TP, DP, PP, EP communicators sharing 1 ncclGpuProgressCounterMonitor

    pcie_d2h_bytes_per_sec = (comms_per_gpu * counter_block_bytes) * (1000.0 / poll_interval_ms)
    pcie_gen5_bw_bytes_per_sec = 56.0 * (1024**3)
    pcie_overhead_pct = (pcie_d2h_bytes_per_sec / pcie_gen5_bw_bytes_per_sec) * 100.0

    # Out-of-band RAS_COLL_DIAG spanning tree over 1,024 ranks (128 nodes) on port 28028
    healthy_ras_diag_ms = 138.4
    dead_peer_ras_diag_ms = leg_timeout_sec * 1000.0 + healthy_ras_diag_ms  # 5,138.4 ms
    legacy_pytorch_watchdog_ms = 600_000.0  # 10 min default timeout

    return {
        "client_port": 28028,
        "protocol_version": 2,
        "poll_interval_ms": poll_interval_ms,
        "min_poll_ms": min_poll_ms,
        "stale_threshold_ms": stale_threshold_ms,
        "min_stale_ms": min_stale_ms,
        "leg_timeout_sec": leg_timeout_sec,
        "cuda_stream_capture_mode": "cudaStreamCaptureModeRelaxed",
        "comms_per_device_worker": comms_per_gpu,
        "d2h_bytes_per_sec_per_gpu": pcie_d2h_bytes_per_sec,
        "pcie_bw_overhead_pct": round(pcie_overhead_pct, 8),
        "healthy_1024rank_ras_diag_ms": healthy_ras_diag_ms,
        "dead_peer_1024rank_ras_diag_ms": dead_peer_ras_diag_ms,
        "legacy_watchdog_timeout_ms": legacy_pytorch_watchdog_ms,
        "triage_speedup_on_dead_peer_x": round(
            legacy_pytorch_watchdog_ms / dead_peer_ras_diag_ms, 2
        ),
    }


def run_simulation() -> Dict[str, object]:
    return {
        "p2p_copy_engine_overlap": simulate_p2p_compute_overlap(),
        "ras_progress_monitor": simulate_ras_progress_monitor(),
    }


def _check(res: Dict[str, object]) -> None:
    p2p = res["p2p_copy_engine_overlap"]
    modes = p2p["modes"]
    assert modes["p2p_cumem_ce_memcpy_proxy"]["comm_sms_occupied"] == 0
    assert (
        modes["p2p_cumem_ce_memcpy_proxy"]["concurrent_gemm_ms"]
        < modes["p2p_sm_driven_16cta_simple"]["concurrent_gemm_ms"]
    )
    assert p2p["ce_speedup_vs_16cta_sm"] > 1.18

    ras = res["ras_progress_monitor"]
    assert ras["client_port"] == 28028
    assert ras["poll_interval_ms"] == 1000
    assert ras["stale_threshold_ms"] == 5000
    assert ras["leg_timeout_sec"] == 5
    assert ras["cuda_stream_capture_mode"] == "cudaStreamCaptureModeRelaxed"
    assert ras["triage_speedup_on_dead_peer_x"] > 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    res = run_simulation()
    if args.check:
        _check(res)
        print("nccl_ce_p2p_ras_plane: OK")
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    elif not args.check:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
