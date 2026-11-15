#!/usr/bin/env python3
"""Round 278: LLM Serving Goodput (TTFT + TPOT Joint SLO) & Preemption (Recompute Thrash vs Watermark Admission + Async Pinned Swap) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_serving_goodput_preemption_tiers() -> dict[str, Any]:
    """Evaluate 4 LLM serving admission control & KV preemption architectures under burst traffic."""
    tiers = [
        {
            "tier_id": "tier1_unbounded_max_num_seqs_256_plus_recompute_preemption_thrash",
            "name": "Tier 1: 盲目调大 max_num_seqs=256 + 默认 Recompute 抢占 (KV 满载后丢弃已生成词从头重算，TPOT 飙至 68ms，Goodput 仅 14.0%)",
            "joint_slo_goodput_compliance_pct": 14.0,
            "effective_goodput_qps": 1.2,
            "kv_preemption_recompute_waste_pct": 38.5,
            "verdict": "只看总 Token/s 表面繁荣！256 条序列半途撑爆 KV Cache 触发连环丢弃重算，且每步耗时 68ms 突破 50ms 红线，86% 请求违约 ✕",
        },
        {
            "tier_id": "tier2_unbounded_admission_with_synchronous_cpu_swap_preemption",
            "name": "Tier 2: 开启 Swap 内存换出免除重算，但仍无 KV 水位准入控制 (频繁换入换出打断 Decode，Goodput 42.0%)",
            "joint_slo_goodput_compliance_pct": 42.0,
            "effective_goodput_qps": 3.5,
            "kv_preemption_recompute_waste_pct": 8.2,
            "verdict": "虽然把被抢占序列的 KV 拷到了主存，但因为没预留生成增长水位，显存顶满后频繁触发 PCIe 换入换出抖动 ⚠️",
        },
        {
            "tier_id": "tier3_static_conservative_max_num_seqs_32_hard_cap",
            "name": "Tier 3: 一刀切保守写死 max_num_seqs=32 (虽消除了抢占与 TPOT 超标，但短请求并发受限导致入口排队 TTFT 违约，64.0%)",
            "joint_slo_goodput_compliance_pct": 64.0,
            "effective_goodput_qps": 5.4,
            "kv_preemption_recompute_waste_pct": 0.0,
            "verdict": "在长请求时不爆显存，但在短请求时白白闲置一半显存，导致新请求在队列里干等超过 1000ms TTFT 红线 ✅",
        },
        {
            "tier_id": "tier4_kv_watermark_admission_gate_82pct_plus_tpot_token_cap_and_async_pinned_swap",
            "name": "Tier 4: 82% KV 增长水位准入闸门 + 动态 TPOT 预算限流 + 极端长尾异步 Pinned DRAM 换出 (Goodput 高达 96.5%!)",
            "joint_slo_goodput_compliance_pct": 96.5,
            "effective_goodput_qps": 8.6,
            "kv_preemption_recompute_waste_pct": 0.3,
            "verdict": "当 KV 占用达 82% 时暂停准入新 Prefill，预留 18% 空间保在途请求零抢占跑完，有效达标吞吐 (Goodput) 暴涨 7.17 倍 🏆",
        },
    ]

    return {
        "experiment_id": "llm_serving_goodput_ttft_tpot_slo_admission_control_preemption_selection",
        "round": 278,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "ttft_slo_ms": 1000.0,
            "tpot_slo_ms": 50.0,
            "goodput_qps_gain_x": round(8.6 / 1.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_serving_goodput_preemption_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
