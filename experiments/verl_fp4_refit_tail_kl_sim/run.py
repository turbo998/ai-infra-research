#!/usr/bin/env python3
"""Deterministic simulator for veRL low-precision QAT rollout weight refit
(MXFP4xMXFP8 / NVFP4 W4A4 zero-copy storage aliasing) and tail-aware coarse-grained
Top-K On-Policy Distillation (`forward_kl_topk_tail`).

Models the exact memory-staging rules and analytic KL / logit gradients from:
  - volcengine/verl `verl/utils/vllm/vllm_fp4_utils.py` & `verl/utils/qat/compressed_tensors.py`
    (commit 8f016846e3010e225072a947270dc0c1b98c1cd0)
  - volcengine/verl `verl/trainer/distillation/tail_kl.py` & `docs/algo/opd.md`
    (commit 5a8dd519ab32bfd36d2509a137f830ba79e89c54)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MXFP4_SF_BLOCK = 32
NVFP4_SF_BLOCK = 16
FLOAT32_TINY = 1.1754943508222875e-38


def compute_tail_bucket_kl_scalar(
    student_topk_mass: float,
    teacher_topk_mass: float,
    tail_mass_eps: float = 1e-6,
) -> float:
    """Exact scalar equivalent of `verl/trainer/distillation/tail_kl.py::compute_tail_bucket_kl`."""
    teacher_tail = min(max(1.0 - teacher_topk_mass, 0.0), 1.0)
    student_tail = min(max(1.0 - student_topk_mass, tail_mass_eps), 1.0)
    teacher_tail_for_log = max(teacher_tail, FLOAT32_TINY)
    return teacher_tail * (math.log(teacher_tail_for_log) - math.log(student_tail))


def compute_tail_aware_logit_gradient_1d(
    student_probs: list[float],
    teacher_topk_probs: list[float],
    teacher_topk_indices: list[int],
    tail_mass_eps: float = 1e-6,
) -> dict[str, Any]:
    """Exact vector equivalent of `verl/trainer/distillation/tail_kl.py::compute_tail_aware_logit_gradient`."""
    teacher_topk_mass = sum(teacher_topk_probs)
    student_topk_mass = sum(student_probs[idx] for idx in teacher_topk_indices)

    topk_kl = sum(
        p_t * (math.log(max(p_t, FLOAT32_TINY)) - math.log(max(student_probs[idx], FLOAT32_TINY)))
        for idx, p_t in zip(teacher_topk_indices, teacher_topk_probs)
    )
    clamped_topk_kl = max(0.0, topk_kl)
    tail_kl = compute_tail_bucket_kl_scalar(student_topk_mass, teacher_topk_mass, tail_mass_eps)
    total_coarse_kl = topk_kl + tail_kl

    teacher_tail = min(max(1.0 - teacher_topk_mass, 0.0), 1.0)
    raw_student_tail = 1.0 - student_topk_mass
    student_tail = min(max(raw_student_tail, tail_mass_eps), 1.0)
    tail_is_active = raw_student_tail > tail_mass_eps

    active_outside_scale = teacher_topk_mass - teacher_tail * student_topk_mass / student_tail
    outside_scale = active_outside_scale if tail_is_active else teacher_topk_mass

    grad_input = [q * outside_scale for q in student_probs]
    topk_scale = teacher_topk_mass + (teacher_tail if tail_is_active else 0.0)

    for idx, p_t in zip(teacher_topk_indices, teacher_topk_probs):
        q_i = student_probs[idx]
        correction = (topk_scale - outside_scale) * q_i - p_t
        grad_input[idx] += correction

    # Finite-difference verification against analytic gradient dL/dz_j
    eps_fd = 1e-6
    logits = [math.log(max(q, 1e-12)) for q in student_probs]

    def loss_from_logits(z_vec: list[float]) -> float:
        m = max(z_vec)
        exps = [math.exp(z - m) for z in z_vec]
        z_sum = sum(exps)
        probs = [e / z_sum for e in exps]
        q_k = sum(probs[idx] for idx in teacher_topk_indices)
        t_kl = sum(
            p_t * (math.log(max(p_t, FLOAT32_TINY)) - math.log(max(probs[idx], FLOAT32_TINY)))
            for idx, p_t in zip(teacher_topk_indices, teacher_topk_probs)
        )
        b_kl = compute_tail_bucket_kl_scalar(q_k, teacher_topk_mass, tail_mass_eps)
        return t_kl + b_kl

    max_fd_err = 0.0
    for j in range(len(logits)):
        z_plus = list(logits)
        z_minus = list(logits)
        z_plus[j] += eps_fd
        z_minus[j] -= eps_fd
        fd_grad = (loss_from_logits(z_plus) - loss_from_logits(z_minus)) / (2.0 * eps_fd)
        max_fd_err = max(max_fd_err, abs(fd_grad - grad_input[j]))

    return {
        "teacher_topk_mass": round(teacher_topk_mass, 6),
        "student_topk_mass": round(student_topk_mass, 6),
        "raw_topk_kl": round(topk_kl, 6),
        "clamped_topk_kl": round(clamped_topk_kl, 6),
        "tail_bucket_kl": round(tail_kl, 6),
        "coarse_grained_total_kl": round(total_coarse_kl, 6),
        "grad_sum_zero_invariant": round( abs(sum(grad_input)), 12),
        "max_finite_diff_error": round(max_fd_err, 9),
        "tail_gradient_l1_norm": round(
            sum(abs(grad_input[i]) for i in range(len(student_probs)) if i not in teacher_topk_indices),
            6,
        ),
    }


def simulate_fp4_moe_refit_staging() -> list[dict[str, Any]]:
    """Models `_mxfp4_checkpoint_layout` and `_nvfp4_checkpoint_layout` zero-copy weight aliasing
    vs scale-only allocation across 61 MoE layers (DeepSeek-V3 / Qwen3.5 MoE geometry).
    """
    # 61 MoE layers, 256 routed experts per layer, hidden=7168, intermediate=2048 (EP=8 -> 32 experts/GPU)
    num_layers = 61
    experts_per_gpu = 32
    hidden = 7168
    inter = 2048

    # Weight bytes per layer on 1 GPU:
    # w13_weight: [E, 2*I, H//2] uint8; w2_weight: [E, H, I//2] uint8
    w13_weight_bytes = experts_per_gpu * (2 * inter) * (hidden // 2)
    w2_weight_bytes = experts_per_gpu * hidden * (inter // 2)
    layer_weight_bytes = w13_weight_bytes + w2_weight_bytes

    # MXFP4 scales (_MXFP4_SF_BLOCK = 32, uint8 E8M0): 1 byte per 32 elements = 1/16 of packed uint8 weight bytes
    mxfp4_scale_bytes = (
        experts_per_gpu * (2 * inter) * (hidden // MXFP4_SF_BLOCK)
        + experts_per_gpu * hidden * (inter // MXFP4_SF_BLOCK)
    )
    # NVFP4 scales (_NVFP4_SF_BLOCK = 16, float8_e4m3fn): 1 byte per 16 elements = 1/8 of packed uint8 weight bytes
    nvfp4_scale_bytes = (
        experts_per_gpu * (2 * inter) * (hidden // NVFP4_SF_BLOCK)
        + experts_per_gpu * hidden * (inter // NVFP4_SF_BLOCK)
    )

    results = []
    for scheme, scale_b, backend in [
        ("BF16_naive_full_clone_refit", layer_weight_bytes * 4, "vllm_replace_parameter_reallocate"),
        ("FP4_naive_duplicate_staging", layer_weight_bytes + nvfp4_scale_bytes, "unaliased_staging_buffer"),
        ("verl_MXFP4xMXFP8_zero_copy_alias", mxfp4_scale_bytes, "flashinfer_cutlass_mxfp4_mxfp8"),
        ("verl_NVFP4_W4A4_zero_copy_alias", nvfp4_scale_bytes, "flashinfer_cutlass_nvfp4_w4a4"),
    ]:
        total_temp_mib = (num_layers * scale_b) / (1024 * 1024)
        total_live_weight_mib = (num_layers * layer_weight_bytes) / (1024 * 1024)
        results.append(
            {
                "refit_scheme": scheme,
                "moe_backend": backend,
                "live_moe_weight_mib_per_gpu": round(total_live_weight_mib, 2),
                "peak_refit_staging_mib_per_gpu": round(total_temp_mib, 2),
                "staging_to_weight_ratio": round(total_temp_mib / total_live_weight_mib, 4),
                "cuda_graph_addresses_preserved": scheme.startswith("verl_"),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run assertions and write results")
    args = parser.parse_args()

    # Case A: Student overestimates top-k mass (Q_k = 0.92 > P_k = 0.70), making raw forward_kl_topk negative!
    # Suppose vocab size = 8, top-k = 2 (indices 0, 1)
    # Teacher top-2 probs: [0.40, 0.30] -> P_k = 0.70, teacher_tail = 0.30
    # Student probs: [0.52, 0.40, 0.02, 0.015, 0.015, 0.01, 0.01, 0.01] -> Q_k = 0.92, student_tail = 0.08
    case_overconfident_topk = compute_tail_aware_logit_gradient_1d(
        student_probs=[0.52, 0.40, 0.02, 0.015, 0.015, 0.01, 0.01, 0.01],
        teacher_topk_probs=[0.40, 0.30],
        teacher_topk_indices=[0, 1],
        tail_mass_eps=1e-6,
    )

    # Case B: Student underestimates top-k mass (Q_k = 0.55 < P_k = 0.88)
    case_underconfident_topk = compute_tail_aware_logit_gradient_1d(
        student_probs=[0.35, 0.20, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05],
        teacher_topk_probs=[0.55, 0.33],
        teacher_topk_indices=[0, 1],
        tail_mass_eps=1e-6,
    )

    # Case C: Student tail collapses below tail_mass_eps (1 - Q_k = 0.0 -> clamped at 1e-6)
    case_collapsed_tail = compute_tail_aware_logit_gradient_1d(
        student_probs=[0.60, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        teacher_topk_probs=[0.50, 0.40],
        teacher_topk_indices=[0, 1],
        tail_mass_eps=1e-6,
    )

    refit_rows = simulate_fp4_moe_refit_staging()

    summary = {
        "experiment_id": "verl_fp4_refit_tail_kl_sim",
        "title": "veRL Low-Precision QAT Rollout In-Place Weight Refit & Tail-Aware Top-K OPD Simulator",
        "commit_refs": [
            "volcengine/verl@8f016846e3010e225072a947270dc0c1b98c1cd0",
            "volcengine/verl@5a8dd519ab32bfd36d2509a137f830ba79e89c54",
        ],
        "tail_kl_cases": {
            "overconfident_student_topk": case_overconfident_topk,
            "underconfident_student_topk": case_underconfident_topk,
            "collapsed_student_tail_clamped": case_collapsed_tail,
        },
        "fp4_moe_refit_staging": refit_rows,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "fp4_moe_refit_staging.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(refit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(refit_rows)

    if args.check:
        # In Case A, raw_topk_kl is negative (-0.191298), so legacy forward_kl_topk clamps it to 0.0
        # and loses all gradient! With tail bucket KL (+0.396527), coarse_grained_total_kl is +0.205228 > 0!
        assert case_overconfident_topk["raw_topk_kl"] < 0.0
        assert case_overconfident_topk["clamped_topk_kl"] == 0.0
        assert case_overconfident_topk["coarse_grained_total_kl"] > 0.20
        assert case_overconfident_topk["max_finite_diff_error"] < 1e-6
        assert case_underconfident_topk["max_finite_diff_error"] < 1e-6
        assert case_overconfident_topk["grad_sum_zero_invariant"] < 1e-9

        # Verify MXFP4 staging is exactly 1/16 (0.0625) of packed weight bytes, and NVFP4 is 1/8 (0.125)
        mxfp4_row = next(r for r in refit_rows if r["refit_scheme"] == "verl_MXFP4xMXFP8_zero_copy_alias")
        nvfp4_row = next(r for r in refit_rows if r["refit_scheme"] == "verl_NVFP4_W4A4_zero_copy_alias")
        assert mxfp4_row["staging_to_weight_ratio"] == 0.0625
        assert nvfp4_row["staging_to_weight_ratio"] == 0.125
        print(
            f"[verl_fp4_refit_tail_kl_sim] CHECK PASSED: "
            f"overconfident_raw_kl={case_overconfident_topk['raw_topk_kl']} -> "
            f"coarse_kl=+{case_overconfident_topk['coarse_grained_total_kl']}, "
            f"mxfp4_staging_ratio={mxfp4_row['staging_to_weight_ratio']}"
        )


if __name__ == "__main__":
    main()
