"""Phase 86 Track 3: FA4 1CTA MLA & hd256, veRL Truncated Score Centering, Megatron Live-Row MoE, Mooncake EFA THP & XPU dma-buf, NIXL GPUNetIO 2-SGE, Dynamo Tiered Prefix & TE Fused MLA KV RoPE MXFP8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_fa4_1cta_mla_and_te_fused_kv_rope() -> dict[str, Any]:
    """Model FA4 PR #2938 1CTA MLA forward/backward + ptxas -O2 and TE PR #3672 fused MLA KV RoPE MXFP8."""
    # FA4 PR #2938: ptxas -O2 eliminates 4064 B/thread local memory spill in dQdQvGemmKernel (3.0-4.2x speedup)
    # 3-stage TMEM dV accumulator ring at 64 heads (3 x 128 + 32 + 32 = 448 <= 512 TMEM cols): 18.32 -> 15.29 ms
    # Sparse recompute-P training step at 24 heads: 30.2 ms (2CTA load-P) -> 22.5 ms (1CTA kb64 recompute-P)
    dq_dqv_spill_bytes_default = 4064
    dq_dqv_spill_bytes_o2 = 0
    bwd_64h_2stage_dv_ms = 18.32
    bwd_64h_3stage_dv_ms = 15.29
    train_24h_2cta_load_p_ms = 30.2
    train_24h_1cta_recompute_p_ms = 22.5
    kb64_decode_hbm_tbps = 6.9
    gb300_wave_crossover_tokens = 80  # 2 x 76 = 152 SMs (1 wave on GB300)

    # TE PR #3672: nvte_fused_mla_kv_rope_mxfp8 at [4096, 1, 128 heads] on VR200
    te_unfused_split_rope_us = 60.8
    te_unfused_quant_us = 75.6
    te_unfused_total_us = round(te_unfused_split_rope_us + te_unfused_quant_us, 1)
    te_fused_total_us = 100.1

    return {
        "dq_dqv_spill_bytes_default": dq_dqv_spill_bytes_default,
        "dq_dqv_spill_bytes_o2": dq_dqv_spill_bytes_o2,
        "bwd_64h_3stage_tmem_cols": 3 * 128 + 32 + 32,
        "bwd_64h_dv_pipeline_speedup": round(
            bwd_64h_2stage_dv_ms / bwd_64h_3stage_dv_ms, 3
        ),
        "train_24h_recompute_p_speedup": round(
            train_24h_2cta_load_p_ms / train_24h_1cta_recompute_p_ms, 3
        ),
        "kb64_decode_hbm_tbps": kb64_decode_hbm_tbps,
        "gb300_wave_crossover_tokens": gb300_wave_crossover_tokens,
        "te_unfused_total_us": te_unfused_total_us,
        "te_fused_total_us": te_fused_total_us,
        "te_fused_speedup": round(te_unfused_total_us / te_fused_total_us, 3),
    }


def simulate_verl_truncated_score_centering_and_mcore_moe() -> dict[str, Any]:
    """Model veRL PR #08b8387 score_centering_top_p and Megatron-LM PR #8389/#8385."""
    # veRL score_centering_support_correction over truncated support S (k=4 head, 3 tokens in support)
    sampler_log_probs = [-0.5, -1.2, -1.2039728, float("-inf")]
    train_log_probs = [-0.8, -1.1, -1.5, -0.3]  # slot 3 is outside S (-inf in sampler)

    in_support = [math.isfinite(lp) for lp in sampler_log_probs]
    q = [math.exp(lp) if fin else 0.0 for lp, fin in zip(sampler_log_probs, in_support)]
    sampler_head_mass = sum(q)
    train_support_mass = sum(
        math.exp(tlp) for tlp, fin in zip(train_log_probs, in_support) if fin
    )
    log_support_mass = math.log(train_support_mass)
    p_s = [
        math.exp(tlp - log_support_mass) if fin else 0.0
        for tlp, fin in zip(train_log_probs, in_support)
    ]
    # Under identity weight_fn(r) = r, weighted = q * (p^S / q) = p^S, so residual = p^S - 1 * p^S = 0
    # Under clipped IS weight_fn(r) = min(r, 1.2), verify exact support normalization sum(p_s) == 1.0
    weighted = [
        qi * min(psi / qi, 1.2) if fin else 0.0
        for qi, psi, fin in zip(q, p_s, in_support)
    ]
    weighted_sum = sum(weighted)
    residual = [
        wi - weighted_sum * psi if fin else 0.0
        for wi, psi, fin in zip(weighted, p_s, in_support)
    ]

    # Megatron-LM PR #8385: Live-row bounded MXFP8 MoE inference
    max_rows = 16384
    live_rows = 1280
    scale_col_blocks = math.ceil((2688 // 32) / 4)
    worst_case_scale_bytes = (max_rows // 128) * scale_col_blocks * 512
    live_scale_bytes = (live_rows // 128) * scale_col_blocks * 512

    return {
        "sampler_head_mass": round(sampler_head_mass, 4),
        "renormalized_train_support_sum": round(sum(p_s), 6),
        "residual_sum_over_support": round(sum(residual), 8),
        "mcore_moe_max_rows": max_rows,
        "mcore_moe_live_rows": live_rows,
        "mcore_moe_work_reduction_x": round(max_rows / live_rows, 2),
        "worst_case_scale_bytes": worst_case_scale_bytes,
        "live_scale_bytes": live_scale_bytes,
    }


def simulate_mooncake_nixl_dynamo_fabric() -> dict[str, Any]:
    """Model Mooncake PR #4564 EFA 2MB THP PTE budget & PR #4378 XPU dma-buf, NIXL PR #2172 2-SGE WRITE, Dynamo PR #16501."""
    # Mooncake PR #4564: EFA max PTE entries = 22M (22 * 1024 * 1024)
    max_pte_entries = 22 * 1024 * 1024
    max_single_nic_4kb_gib = round(
        (max_pte_entries * 4096) / (1024**3), 1
    )  # 88.0 GiB
    max_single_nic_2mb_gib = round(
        (max_pte_entries * 2 * 1024 * 1024) / (1024**3), 1
    )  # 45056.0 GiB (44 TiB)

    # NIXL PR #2172: GPUNetIO two-SGE WRITE pairing for 4096 x 4 KiB descriptors (16 MiB)
    total_descriptors = 4096
    # Final descriptor kept separate for arrival-marker ordering -> 4095 pairable -> 2047 pairs + 1 odd + 1 final = 2049 WQEs
    paired_wqes = (total_descriptors - 1) // 2 + ((total_descriptors - 1) % 2) + 1
    main_median_gbps = 10.396867
    candidate_median_gbps = 12.696834
    main_tx_p99_us = 1613
    candidate_tx_p99_us = 1325

    return {
        "efa_max_pte_entries": max_pte_entries,
        "efa_single_nic_4kb_cap_gib": max_single_nic_4kb_gib,
        "efa_single_nic_2mb_thp_cap_gib": max_single_nic_2mb_gib,
        "nixl_4096_desc_wqes_before": total_descriptors,
        "nixl_4096_desc_wqes_after": paired_wqes,
        "nixl_main_median_gbps": main_median_gbps,
        "nixl_candidate_median_gbps": candidate_median_gbps,
        "nixl_throughput_gain_pct": round(
            (candidate_median_gbps - main_median_gbps) / main_median_gbps * 100.0, 2
        ),
        "nixl_main_tx_p99_us": main_tx_p99_us,
        "nixl_candidate_tx_p99_us": candidate_tx_p99_us,
        "dynamo_frontier_key_tuple": ["WorkerWithDpRank", "StorageTier"],
    }


def run_all() -> dict[str, Any]:
    fa4_te = simulate_fa4_1cta_mla_and_te_fused_kv_rope()
    verl_mcore = simulate_verl_truncated_score_centering_and_mcore_moe()
    fabric = simulate_mooncake_nixl_dynamo_fabric()
    return {
        "experiment": "fa4_verl_mcore_nixl_phase86",
        "phase": 86,
        "priority": 95,
        "fa4_1cta_mla_and_te_kv_rope": fa4_te,
        "verl_score_centering_and_mcore_moe": verl_mcore,
        "mooncake_nixl_dynamo_fabric": fabric,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = run_all()
    summary_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not summary_path.exists():
            raise SystemExit("summary.json does not exist")
        if summary_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("summary.json is out of date")
        return 0

    summary_path.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
