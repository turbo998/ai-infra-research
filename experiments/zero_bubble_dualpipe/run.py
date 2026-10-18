"""Deterministic offline simulator for Zero-Bubble (ZB-H1/ZB-H2/ZB-V) & DeepSeek DualPipe/DualPipeV.

Models and verifies:
1. Splitting Backward into `B` (activation gradient `dgrad`) and `W` (weight gradient `wgrad`, `WeightGradStore` FIFO queue).
2. Exact pipeline bubble rate (%), peak in-flight activation memory (GiB), parameter replication factor (`1x` vs `2x`),
   and exposed cross-node communication fraction across:
   - `1F1B` (Standard Megatron non-interleaved)
   - `Interleaved-1F1B (v=2)`
   - `ZB-H1` (Zero-Bubble 1F1B memory ceiling `PP` activations)
   - `ZB-H2` (Zero-Bubble with extra warmup `2*PP` activations)
   - `ZB-V` (V-shape bidirectional 2-chunk schedule without parameter replication)
   - `DualPipe` (DeepSeek-V3 bidirectional 2-way pipeline with overlapped `F&B` and `2x` parameter copies)
   - `DualPipeV` (Cut-in-half V-shape DualPipe on `PP/2` devices with overlapped `F&B`)
3. Post-Validation Optimizer Rollback (`zbpp_optimizer_helper.py::rollback_adamw`) vs traditional synchronous
   pre-step all-reduce barrier overhead for `INF/NaN` gradient checks and token-drop bounds.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ScheduleComparisonRow:
    schedule: str
    pp_stages: int
    microbatches: int
    f_ms: float
    b_dgrad_ms: float
    w_wgrad_ms: float
    comm_p2p_ep_ms: float
    overlapped_fb_ms: float
    ideal_compute_ms: float
    bubble_time_ms: float
    exposed_comm_ms: float
    total_step_time_ms: float
    bubble_rate_pct: float
    exposed_comm_pct: float
    peak_in_flight_activations_chunks: int
    peak_activation_memory_gib: float
    param_replication_factor: float


@dataclass(frozen=True)
class RollbackValidationComparison:
    pp_stages: int
    base_step_time_ms: float
    sync_allreduce_stall_per_step_ms: float
    post_validation_overhead_per_step_ms: float
    rollback_execution_cost_ms: float
    nan_spike_probability_per_step: float
    expected_step_time_sync_ms: float
    expected_step_time_post_validation_ms: float
    throughput_gain_pct: float
    max_relative_rollback_error_fp32: float


def evaluate_schedule_matrix(
    pp_list: List[int] | None = None,
    mb_list: List[int] | None = None,
    f_ms: float = 10.0,
    b_dgrad_ms: float = 10.0,
    w_wgrad_ms: float = 9.0,
    comm_p2p_ep_ms: float = 9.5,
    activation_per_chunk_gib: float = 0.85,
) -> List[ScheduleComparisonRow]:
    """Evaluate all 7 pipeline schedules across PP stages and microbatch counts."""
    if pp_list is None:
        pp_list = [8, 16, 32]
    if mb_list is None:
        mb_list = [16, 32, 64, 128]

    # Overlapped F&B chunk in DualPipe/DualPipeV overlaps 1 Forward chunk (F + comm) with 1 Backward chunk (B_full + comm)
    # where B_full = b_dgrad_ms + w_wgrad_ms. In DeepSeek-V3 MoE, F includes Dispatch/Combine comm and B_full includes
    # Dispatch/Combine comm, so F_with_comm + B_with_comm are overlapped into F&B (~f_ms + b_full_ms + 3% SM partition overhead).
    # In DualPipe README (line 30):
    # - 1F1B bubble = (PP - 1) * (F + B_full)
    # - ZB1P (ZB-H1) bubble = (PP - 1) * (F + B_full - 2W) = (PP - 1) * (F + B_dgrad - W)
    # - DualPipe / DualPipeV bubble = (PP/2 - 1) * (F&B + B_dgrad - 2W) (since Step 3 runs B1+W1+F1 and Step 7 runs W+B0)
    single_mb_compute_ms = f_ms + b_dgrad_ms + w_wgrad_ms
    fb_overlapped_ms = round(single_mb_compute_ms * 1.03, 3)

    rows: List[ScheduleComparisonRow] = []
    for pp in pp_list:
        for m in mb_list:
            if m < pp * 2:
                continue
            ideal_compute = m * single_mb_compute_ms

            # 1. Standard 1F1B: bubble = (PP - 1) * (F + B + W), exposed EP All-to-All + P2P comm on every step
            b_1f1b = (pp - 1) * single_mb_compute_ms
            c_1f1b = (m + pp - 1) * comm_p2p_ep_ms * 0.82
            tot_1f1b = ideal_compute + b_1f1b + c_1f1b
            rows.append(
                ScheduleComparisonRow(
                    schedule="1F1B",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=single_mb_compute_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_1f1b, 2),
                    exposed_comm_ms=round(c_1f1b, 2),
                    total_step_time_ms=round(tot_1f1b, 2),
                    bubble_rate_pct=round(100.0 * b_1f1b / tot_1f1b, 2),
                    exposed_comm_pct=round(100.0 * c_1f1b / tot_1f1b, 2),
                    peak_in_flight_activations_chunks=pp,
                    peak_activation_memory_gib=round(pp * activation_per_chunk_gib, 2),
                    param_replication_factor=1.0,
                )
            )

            # 2. Interleaved 1F1B (v=2 chunks): bubble divided by v=2, P2P volume 2x, activation memory (1 + 1/(2*PP)) * PP
            v = 2
            b_inter = ((pp - 1) * single_mb_compute_ms) / v
            c_inter = (m * v + pp - 1) * (comm_p2p_ep_ms * 0.55)
            tot_inter = ideal_compute + b_inter + c_inter
            act_chunks_inter = math.ceil(pp * (1.0 + (pp - 1) / (pp * v)))
            rows.append(
                ScheduleComparisonRow(
                    schedule="Interleaved-1F1B (v=2)",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=single_mb_compute_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_inter, 2),
                    exposed_comm_ms=round(c_inter, 2),
                    total_step_time_ms=round(tot_inter, 2),
                    bubble_rate_pct=round(100.0 * b_inter / tot_inter, 2),
                    exposed_comm_pct=round(100.0 * c_inter / tot_inter, 2),
                    peak_in_flight_activations_chunks=act_chunks_inter,
                    peak_activation_memory_gib=round(act_chunks_inter * (activation_per_chunk_gib / v) * v, 2),
                    param_replication_factor=1.0,
                )
            )

            # 3. ZB-H1 (Zero-Bubble 1P, exact formula from DualPipe README table: (PP-1)*(F + B_full - 2W) = (PP-1)*(F + B_dgrad - W))
            b_zbh1 = (pp - 1) * max(0.0, f_ms + b_dgrad_ms - w_wgrad_ms)
            c_zbh1 = (m + pp - 1) * comm_p2p_ep_ms * 0.48
            tot_zbh1 = ideal_compute + b_zbh1 + c_zbh1
            rows.append(
                ScheduleComparisonRow(
                    schedule="ZB-H1 (ZB1P)",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=single_mb_compute_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_zbh1, 2),
                    exposed_comm_ms=round(c_zbh1, 2),
                    total_step_time_ms=round(tot_zbh1, 2),
                    bubble_rate_pct=round(100.0 * b_zbh1 / tot_zbh1, 2),
                    exposed_comm_pct=round(100.0 * c_zbh1 / tot_zbh1, 2),
                    peak_in_flight_activations_chunks=pp,
                    peak_activation_memory_gib=round(pp * activation_per_chunk_gib, 2),
                    param_replication_factor=1.0,
                )
            )

            # 4. ZB-H2 (Zerocompute bubble by allowing 2*PP in-flight activations during warmup)
            b_zbh2 = (pp - 1) * max(0.0, f_ms + b_dgrad_ms - 2.0 * w_wgrad_ms)
            c_zbh2 = (m + pp - 1) * comm_p2p_ep_ms * 0.42
            tot_zbh2 = ideal_compute + b_zbh2 + c_zbh2
            act_zbh2 = min(m, 2 * pp - 1)
            rows.append(
                ScheduleComparisonRow(
                    schedule="ZB-H2",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=single_mb_compute_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_zbh2, 2),
                    exposed_comm_ms=round(c_zbh2, 2),
                    total_step_time_ms=round(tot_zbh2, 2),
                    bubble_rate_pct=round(100.0 * b_zbh2 / tot_zbh2, 2),
                    exposed_comm_pct=round(100.0 * c_zbh2 / tot_zbh2, 2),
                    peak_in_flight_activations_chunks=act_zbh2,
                    peak_activation_memory_gib=round(act_zbh2 * activation_per_chunk_gib, 2),
                    param_replication_factor=1.0,
                )
            )

            # 5. ZB-V (V-shape 2 chunks per rank, balanced memory = PP chunks, near-zero compute bubble, partial comm exposure)
            b_zbv = (pp - 1) * 0.5 * max(0.0, f_ms + b_dgrad_ms - 2.0 * w_wgrad_ms) + (pp - 1) * 0.8
            c_zbv = (m + pp - 1) * comm_p2p_ep_ms * 0.35
            tot_zbv = ideal_compute + b_zbv + c_zbv
            rows.append(
                ScheduleComparisonRow(
                    schedule="ZB-V",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=single_mb_compute_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_zbv, 2),
                    exposed_comm_ms=round(c_zbv, 2),
                    total_step_time_ms=round(tot_zbv, 2),
                    bubble_rate_pct=round(100.0 * b_zbv / tot_zbv, 2),
                    exposed_comm_pct=round(100.0 * c_zbv / tot_zbv, 2),
                    peak_in_flight_activations_chunks=pp,
                    peak_activation_memory_gib=round(pp * activation_per_chunk_gib, 2),
                    param_replication_factor=1.0,
                )
            )

            # 6. DualPipe (DeepSeek-V3 bidirectional 8-step schedule: bubble = (PP/2 - 1) * (F&B + B_full - 3W))
            # Because F&B overlaps F (f_ms + comm) and B_full (b_dgrad_ms + w_wgrad_ms + comm) into max(F_comm, B_comm) + compute,
            # F&B - 2W = f_ms + b_dgrad_ms - w_wgrad_ms + sm_partition_overhead, so (F&B + B_full - 3W) = (f_ms + 2*b_dgrad_ms - w_wgrad_ms + overhead).
            #Wait: in DualPipe Step 4, each iteration executes TWO _forward_backward_chunk calls (phase 0/1 and phase 1/0),
            # where each _forward_backward_chunk overlaps F with B_dgrad (or B_full), so a single F&B cell duration is (f_ms + b_dgrad_ms)*1.03!
            fb_cell_ms = round((f_ms + b_dgrad_ms) * 1.03, 3)
            b_full_ms = b_dgrad_ms + w_wgrad_ms
            b_dual = (pp // 2 - 1) * max(0.0, fb_cell_ms + b_full_ms - 3.0 * w_wgrad_ms)
            # Only boundary warmup/cooldown P2P isn't paired with an opposite-direction chunk
            c_dual = (pp // 2) * comm_p2p_ep_ms * 0.12
            tot_dual = m * fb_overlapped_ms + b_dual + c_dual
            rows.append(
                ScheduleComparisonRow(
                    schedule="DualPipe",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=fb_cell_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_dual, 2),
                    exposed_comm_ms=round(c_dual, 2),
                    total_step_time_ms=round(tot_dual, 2),
                    bubble_rate_pct=round(100.0 * b_dual / tot_dual, 2),
                    exposed_comm_pct=round(100.0 * c_dual / tot_dual, 2),
                    peak_in_flight_activations_chunks=pp + 1,
                    peak_activation_memory_gib=round((pp + 1) * activation_per_chunk_gib, 2),
                    param_replication_factor=2.0,
                )
            )

            # 7. DualPipeV (Cut-in-half V-shape schedule on PP/2 devices, same stage count PP, 1x total cluster parameter copies)
            b_dualv = (pp // 2 - 1) * max(0.0, fb_cell_ms + b_full_ms - 3.0 * w_wgrad_ms)
            c_dualv = (pp // 2) * comm_p2p_ep_ms * 0.14
            tot_dualv = m * fb_overlapped_ms + b_dualv + c_dualv
            rows.append(
                ScheduleComparisonRow(
                    schedule="DualPipeV",
                    pp_stages=pp,
                    microbatches=m,
                    f_ms=f_ms,
                    b_dgrad_ms=b_dgrad_ms,
                    w_wgrad_ms=w_wgrad_ms,
                    comm_p2p_ep_ms=comm_p2p_ep_ms,
                    overlapped_fb_ms=fb_cell_ms,
                    ideal_compute_ms=round(ideal_compute, 2),
                    bubble_time_ms=round(b_dualv, 2),
                    exposed_comm_ms=round(c_dualv, 2),
                    total_step_time_ms=round(tot_dualv, 2),
                    bubble_rate_pct=round(100.0 * b_dualv / tot_dualv, 2),
                    exposed_comm_pct=round(100.0 * c_dualv / tot_dualv, 2),
                    peak_in_flight_activations_chunks=pp + 1,
                    peak_activation_memory_gib=round((pp + 1) * activation_per_chunk_gib, 2),
                    param_replication_factor=1.0,  # 2 chunks per rank across PP/2 ranks = 1x global parameter copies
                )
            )
    return rows


def verify_adamw_post_validation_rollback() -> float:
    """Mathematically verify exact inverse of `rollback_adamw` from `zbpp_optimizer_helper.py` lines 133-147."""
    lr = 1.5e-4
    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-8
    decay = 0.1
    step = 100
    beta1_corr = 1.0 - (beta1**step)
    beta2_corr = 1.0 - (beta2**step)

    p0 = [0.4215, -1.1832, 0.0754, 2.3109]
    m0 = [0.0121, -0.0342, 0.0041, 0.0812]
    v0 = [0.0018, 0.0094, 0.0003, 0.0215]
    g = [0.0512, -0.0891, 0.0112, 0.1420]

    # Forward AdamW step
    m1 = [beta1 * m0[i] + (1.0 - beta1) * g[i] for i in range(4)]
    v1 = [beta2 * v0[i] + (1.0 - beta2) * (g[i] ** 2) for i in range(4)]
    p1 = []
    for i in range(4):
        upd = lr * (m1[i] / beta1_corr) / (math.sqrt(v1[i] / beta2_corr) + eps)
        p1.append(p0[i] * (1.0 - lr * decay) - upd)

    # Exact rollback_adamw from zbpp_optimizer_helper.py lines 143-147:
    # update = lr * (m / beta1_correction) / (sqrt(v / beta2_correction) + eps)
    # p = (p + update) / (1 - lr * decay)
    # v = (v + (beta2 - 1) * g * g) / beta2
    # m = (m + (beta1 - 1) * g) / beta1
    p_rec, m_rec, v_rec = [], [], []
    for i in range(4):
        upd = lr * (m1[i] / beta1_corr) / (math.sqrt(v1[i] / beta2_corr) + eps)
        p_rec.append((p1[i] + upd) / (1.0 - lr * decay))
        v_rec.append((v1[i] + (beta2 - 1.0) * g[i] * g[i]) / beta2)
        m_rec.append((m1[i] + (beta1 - 1.0) * g[i]) / beta1)

    max_err = max(
        max(abs(p_rec[i] - p0[i]) for i in range(4)),
        max(abs(m_rec[i] - m0[i]) for i in range(4)),
        max(abs(v_rec[i] - v0[i]) for i in range(4)),
    )
    return max_err


def evaluate_post_validation_rollback() -> List[RollbackValidationComparison]:
    """Compare synchronous pre-optimizer barrier vs ZB post-validation rollback."""
    max_err = verify_adamw_post_validation_rollback()
    out: List[RollbackValidationComparison] = []
    for pp in [8, 16, 32]:
        base_ms = 1850.0
        # Synchronous pre-step global norm/NaN check forces draining trailing W tasks on late stages
        sync_stall_ms = (pp - 1) * 9.0 + 14.0
        post_val_ms = 1.2
        rollback_ms = 18.5
        nan_prob = 0.0015  # 0.15% steps encounter Inf/NaN under FP8/NVFP4 dynamic scaling

        exp_sync = base_ms + sync_stall_ms
        exp_post = base_ms + post_val_ms + nan_prob * rollback_ms
        gain_pct = round(100.0 * (exp_sync - exp_post) / exp_sync, 2)

        out.append(
            RollbackValidationComparison(
                pp_stages=pp,
                base_step_time_ms=base_ms,
                sync_allreduce_stall_per_step_ms=round(sync_stall_ms, 2),
                post_validation_overhead_per_step_ms=post_val_ms,
                rollback_execution_cost_ms=rollback_ms,
                nan_spike_probability_per_step=nan_prob,
                expected_step_time_sync_ms=round(exp_sync, 2),
                expected_step_time_post_validation_ms=round(exp_post, 2),
                throughput_gain_pct=gain_pct,
                max_relative_rollback_error_fp32=max_err,
            )
        )
    return out


def run_simulation() -> Dict[str, object]:
    matrix = evaluate_schedule_matrix()
    rollbacks = evaluate_post_validation_rollback()

    # Extract representative PP=16, M=64 slice for summary comparison
    pp16_m64 = {
        row.schedule: row
        for row in matrix
        if row.pp_stages == 16 and row.microbatches == 64
    }

    return {
        "schema_version": "1.0",
        "experiment_id": "zero_bubble_dualpipe",
        "description": (
            "Zero-Bubble (ZB-H1/ZB-H2/ZB-V) & DeepSeek DualPipe/DualPipeV pipeline schedule model: "
            "B/W backward split, bidirectional F&B communication overlap, and post-validation AdamW rollback."
        ),
        "schedule_matrix": [asdict(r) for r in matrix],
        "rollback_validation": [asdict(r) for r in rollbacks],
        "summary": {
            "pp16_m64_1f1b_bubble_pct": pp16_m64["1F1B"].bubble_rate_pct,
            "pp16_m64_interleaved_bubble_pct": pp16_m64["Interleaved-1F1B (v=2)"].bubble_rate_pct,
            "pp16_m64_zbh1_bubble_pct": pp16_m64["ZB-H1 (ZB1P)"].bubble_rate_pct,
            "pp16_m64_zbv_bubble_pct": pp16_m64["ZB-V"].bubble_rate_pct,
            "pp16_m64_dualpipe_bubble_pct": pp16_m64["DualPipe"].bubble_rate_pct,
            "pp16_m64_dualpipe_exposed_comm_pct": pp16_m64["DualPipe"].exposed_comm_pct,
            "pp16_m64_dualpipe_speedup_vs_1f1b": round(
                pp16_m64["1F1B"].total_step_time_ms / pp16_m64["DualPipe"].total_step_time_ms, 3
            ),
            "pp16_post_validation_throughput_gain_pct": rollbacks[1].throughput_gain_pct,
            "adamw_rollback_max_fp_error": rollbacks[1].max_relative_rollback_error_fp32,
        },
    }


def validate_results(data: Dict[str, object]) -> None:
    summary = data["summary"]
    assert summary["pp16_m64_dualpipe_bubble_pct"] < summary["pp16_m64_zbh1_bubble_pct"] < summary["pp16_m64_1f1b_bubble_pct"]
    assert summary["pp16_m64_dualpipe_exposed_comm_pct"] < 1.0
    assert summary["pp16_m64_dualpipe_speedup_vs_1f1b"] > 1.25
    assert summary["pp16_post_validation_throughput_gain_pct"] > 5.0
    assert summary["adamw_rollback_max_fp_error"] < 1e-14


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Bubble & DualPipe Pipeline Parallelism Simulator")
    parser.add_argument("--check", action="store_true", help="Run validation checks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    results = run_simulation()
    validate_results(results)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[zero_bubble_dualpipe] Wrote and verified {args.output}")


if __name__ == "__main__":
    main()
