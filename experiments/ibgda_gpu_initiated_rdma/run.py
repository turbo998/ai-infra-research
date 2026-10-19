#!/usr/bin/env python3
"""Deterministic simulator for GPU-Initiated RDMA (IBGDA / NCCL GIN GDAKI) in DeepEP V2.5.

Models:
1. Host-CPU Proxy RDMA vs IBGDA / NCCL_GIN_TYPE_GDAKI (csrc/kernels/comm/context.cpp)
   latency & message-rate (IOPS) across MoE Decode small payloads (128 B .. 64 KiB).
2. QP Resource Sharing Modes (deep_ep/include/deep_ep/comm/barrier.cuh get_qp_mode):
   NCCL_GIN_RESOURCE_SHARING_CTA (kNumSMs <= kNumAvailableQPs, lock-free per-SM QP)
   vs NCCL_GIN_RESOURCE_SHARING_GPU (kNumSMs > kNumAvailableQPs, cross-SM spinlock).
3. Doorbell Aggregation (ncclGinOptFlagsAggregateRequests) + Programmatic Dependent
   Launch (cudaTriggerProgrammaticLaunchCompletion) + System-Scope Fence correctness
   when mixing NVLink and RDMA (ptx::fence_acq_rel_sys).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RESULTS_PATH = Path(__file__).with_name("results.json")


def simulate_message_rate_curve() -> list[dict[str, Any]]:
    """Compare CPU Proxy Thread RDMA vs IBGDA (No-Batch) vs IBGDA + Doorbell Aggregation."""
    payload_bytes_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # 400 Gbps RoCEv2 / InfiniBand NDR400 per NIC (50 GB/s wire, ~46.5 GB/s goodput ceiling)
    line_rate_gbps = 46.5
    rows: list[dict[str, Any]] = []

    for payload_bytes in payload_bytes_list:
        wire_us = (payload_bytes / (line_rate_gbps * 1e9)) * 1e6

        # 1. Host CPU Proxy Thread (GPU -> Host ring -> ibv_post_send -> PCIe UAR -> CQE -> GPU flag)
        cpu_proxy_ctrl_us = 5.40
        cpu_proxy_iops_cap_m = 2.15  # 2.15M msgs/s per NIC proxy thread ceiling
        cpu_proxy_lat_us = round(cpu_proxy_ctrl_us + 1.15 + wire_us, 3)
        cpu_proxy_bw_gbps = round(
            min(line_rate_gbps, (cpu_proxy_iops_cap_m * 1e6 * payload_bytes) / 1e9), 3
        )

        # 2. IBGDA / GDAKI Naive (1 MMIO 64-bit UAR Doorbell per WQE, CTA-exclusive QP)
        ibgda_single_ctrl_us = 0.42
        ibgda_single_iops_cap_m = 14.80  # Limited by PCIe posted MMIO Write-Combining credits
        ibgda_single_lat_us = round(ibgda_single_ctrl_us + 0.68 + wire_us, 3)
        ibgda_single_bw_gbps = round(
            min(line_rate_gbps, (ibgda_single_iops_cap_m * 1e6 * payload_bytes) / 1e9), 3
        )

        # 3. DeepEP V2.5 GDAKI + ncclGinOptFlagsAggregateRequests (batched WQE + 1 doorbell per 8 puts)
        ibgda_agg_ctrl_us = 0.14
        ibgda_agg_iops_cap_m = 36.40  # BlueFlame / WQE batching amortizes UAR MMIO doorbell
        ibgda_agg_lat_us = round(ibgda_agg_ctrl_us + 0.68 + wire_us, 3)
        ibgda_agg_bw_gbps = round(
            min(line_rate_gbps, (ibgda_agg_iops_cap_m * 1e6 * payload_bytes) / 1e9), 3
        )

        rows.append(
            {
                "payload_bytes": payload_bytes,
                "cpu_proxy_lat_us": cpu_proxy_lat_us,
                "ibgda_single_lat_us": ibgda_single_lat_us,
                "ibgda_agg_lat_us": ibgda_agg_lat_us,
                "cpu_proxy_bw_gbps": cpu_proxy_bw_gbps,
                "ibgda_single_bw_gbps": ibgda_single_bw_gbps,
                "ibgda_agg_bw_gbps": ibgda_agg_bw_gbps,
                "latency_speedup_vs_proxy": round(cpu_proxy_lat_us / ibgda_agg_lat_us, 2),
                "bw_gain_vs_proxy_x": round(ibgda_agg_bw_gbps / max(cpu_proxy_bw_gbps, 1e-6), 2),
            }
        )
    return rows


def simulate_qp_sharing_contention() -> list[dict[str, Any]]:
    """Simulate DeepEP get_qp_mode (CTA-exclusive vs GPU-wide shared QP spinlock) at kNumSMs=24."""
    num_sms = 24
    channels_per_sm = 4  # 4 dispatch warps per SM = 96 total channels
    total_channels = num_sms * channels_per_sm
    # kWithNotifyWarps = True reserves QP 0 for notify warp on SM 0
    allocated_qps_list = [2, 5, 9, 13, 17, 25, 33, 49, 97]
    rows: list[dict[str, Any]] = []

    for k_num_qps in allocated_qps_list:
        k_available_qps = k_num_qps - 1  # QP 0 reserved for notify warp
        is_cta_exclusive = num_sms <= k_available_qps
        sharing_mode = (
            "NCCL_GIN_RESOURCE_SHARING_CTA"
            if is_cta_exclusive
            else "NCCL_GIN_RESOURCE_SHARING_GPU"
        )
        contenders_per_qp = total_channels / k_available_qps

        if is_cta_exclusive:
            # Lock-free within SM (warp-serialized only across channels on the same SM)
            wqe_lock_stall_us = round(0.18 * (contenders_per_qp - 1.0), 3)
        else:
            # Device-wide L2 atomic spinlock across SMs + L2 cacheline ping-pong on WQ PI
            cross_sm_factor = num_sms / k_available_qps
            wqe_lock_stall_us = round(
                0.85 * contenders_per_qp * math.sqrt(cross_sm_factor), 3
            )

        # NIC ICM (Interconnect Context Memory) cache thrashing grows slightly past 32 QPs/rank
        nic_icm_miss_us = round(max(0.0, (k_num_qps - 25) * 0.045), 3)
        base_dispatch_us = 38.40
        total_dispatch_us = round(base_dispatch_us + wqe_lock_stall_us + nic_icm_miss_us, 3)

        rows.append(
            {
                "num_allocated_qps": k_num_qps,
                "num_available_data_qps": k_available_qps,
                "num_sms": num_sms,
                "sharing_mode": sharing_mode,
                "contenders_per_qp": round(contenders_per_qp, 2),
                "wqe_lock_stall_us": wqe_lock_stall_us,
                "nic_icm_miss_us": nic_icm_miss_us,
                "total_dispatch_us": total_dispatch_us,
            }
        )
    return rows


def simulate_moe_decode_step_breakdown() -> dict[str, Any]:
    """Compare 60-layer MoE Decode (EP=64, BS=32/rank, top-8, hidden=7168 FP8) end-to-end."""
    num_moe_layers = 60
    # Per-layer Dispatch + Combine latency in microseconds
    schemes = [
        {
            "scheme_id": "nccl_host_proxy_ring",
            "label": "Host-Proxy RDMA (CPU Proxy Thread + Bounce Buffer)",
            "dispatch_us": 268.0,
            "combine_us": 284.0,
            "epilogue_launch_tail_us": 14.5,
            "mixed_nvl_rdma_race_risk": "none (serialized on host)",
        },
        {
            "scheme_id": "ibgda_gpu_shared_qp8",
            "label": "IBGDA / GDAKI (kNumQPs=9, SHARING_GPU Spinlock, No Batching)",
            "dispatch_us": 112.4,
            "combine_us": 119.6,
            "epilogue_launch_tail_us": 12.0,
            "mixed_nvl_rdma_race_risk": "high without fence_acq_rel_sys",
        },
        {
            "scheme_id": "deepep_v25_gdaki_cta_pdl",
            "label": "DeepEP V2.5 GDAKI (kNumQPs=25 CTA-Exclusive + AggregateRequests + PDL)",
            "dispatch_us": 44.6,
            "combine_us": 48.2,
            "epilogue_launch_tail_us": 1.8,  # Overlapped via cudaTriggerProgrammaticLaunchCompletion
            "mixed_nvl_rdma_race_risk": "eliminated via ptx::fence_acq_rel_sys()",
        },
    ]

    compute_per_layer_us = 142.0  # MLA Attention + Router + Grouped Expert GEMM
    results = []
    baseline_step_ms = 0.0

    for idx, s in enumerate(schemes):
        comm_layer_us = s["dispatch_us"] + s["combine_us"] + s["epilogue_launch_tail_us"]
        step_ms = round(num_moe_layers * (compute_per_layer_us + comm_layer_us) / 1000.0, 3)
        comm_ms = round(num_moe_layers * comm_layer_us / 1000.0, 3)
        if idx == 0:
            baseline_step_ms = step_ms
        results.append(
            {
                **s,
                "comm_per_layer_us": round(comm_layer_us, 2),
                "total_60l_comm_ms": comm_ms,
                "total_60l_step_tpot_ms": step_ms,
                "comm_share_pct": round((comm_ms / step_ms) * 100.0, 2),
                "comm_speedup_vs_proxy": round(
                    (schemes[0]["dispatch_us"] + schemes[0]["combine_us"] + schemes[0]["epilogue_launch_tail_us"])
                    / comm_layer_us,
                    2,
                ),
                "e2e_tpot_speedup_vs_proxy": round(baseline_step_ms / step_ms, 2),
            }
        )

    return {
        "workload": "671B-MoE-60L-EP64-Decode-BS32-FP8",
        "compute_per_layer_us": compute_per_layer_us,
        "schemes": results,
    }


def build_results() -> dict[str, Any]:
    msg_curve = simulate_message_rate_curve()
    qp_curve = simulate_qp_sharing_contention()
    moe_breakdown = simulate_moe_decode_step_breakdown()

    # Key anchors for verification and 7-card reporting
    p512 = next(r for r in msg_curve if r["payload_bytes"] == 512)
    qp_opt = next(r for r in qp_curve if r["num_allocated_qps"] == 25)
    qp_contended = next(r for r in qp_curve if r["num_allocated_qps"] == 5)
    s_proxy = moe_breakdown["schemes"][0]
    s_opt = moe_breakdown["schemes"][2]

    return {
        "experiment_id": "ibgda_gpu_initiated_rdma",
        "status": "own-offline",
        "source_contracts": [
            "deepseek-ai/DeepEP@b5a33eefda39485f438dc764695b71ba7b9249b6:csrc/kernels/comm/context.cpp",
            "deepseek-ai/DeepEP@b5a33eefda39485f438dc764695b71ba7b9249b6:deep_ep/include/deep_ep/comm/handle.cuh",
            "deepseek-ai/DeepEP@b5a33eefda39485f438dc764695b71ba7b9249b6:deep_ep/include/deep_ep/comm/barrier.cuh",
            "deepseek-ai/DeepEP@b5a33eefda39485f438dc764695b71ba7b9249b6:deep_ep/include/deep_ep/impls/ep/dispatch.cuh",
        ],
        "summary_metrics": {
            "payload_512b_proxy_lat_us": p512["cpu_proxy_lat_us"],
            "payload_512b_gdaki_agg_lat_us": p512["ibgda_agg_lat_us"],
            "payload_512b_lat_speedup_x": p512["latency_speedup_vs_proxy"],
            "payload_512b_bw_gain_x": p512["bw_gain_vs_proxy_x"],
            "qp5_gpu_sharing_dispatch_us": qp_contended["total_dispatch_us"],
            "qp25_cta_exclusive_dispatch_us": qp_opt["total_dispatch_us"],
            "qp25_vs_qp5_speedup_x": round(
                qp_contended["total_dispatch_us"] / qp_opt["total_dispatch_us"], 2
            ),
            "moe_60l_proxy_comm_ms": s_proxy["total_60l_comm_ms"],
            "moe_60l_gdaki_comm_ms": s_opt["total_60l_comm_ms"],
            "moe_60l_comm_speedup_x": s_opt["comm_speedup_vs_proxy"],
            "moe_60l_proxy_tpot_ms": s_proxy["total_60l_step_tpot_ms"],
            "moe_60l_gdaki_tpot_ms": s_opt["total_60l_step_tpot_ms"],
            "moe_60l_e2e_speedup_x": s_opt["e2e_tpot_speedup_vs_proxy"],
        },
        "message_rate_curve": msg_curve,
        "qp_sharing_curve": qp_curve,
        "moe_decode_step_breakdown": moe_breakdown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against sealed results.json")
    args = parser.parse_args()

    computed = build_results()
    if args.check:
        if not RESULTS_PATH.exists():
            raise SystemExit(f"Missing {RESULTS_PATH}")
        expected = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        if computed != expected:
            raise SystemExit("ibgda_gpu_initiated_rdma: results mismatch")
        print("ibgda_gpu_initiated_rdma: OK")
        return 0

    RESULTS_PATH.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
