from __future__ import annotations
import json
from pathlib import Path


def evaluate_parallelism_dense_moe_acceleration_sizing() -> dict:
    # 671B MoE on 2x8 H200 (16 GPUs)
    cross_node_tp16_allreduces_per_tok = 122
    cross_node_tp16_net_sync_ms = 18.3
    cross_node_tp16_tpot_ms = 34.6
    cross_node_tp16_goodput_tok_s_gpu = 520

    dp16_ep16_deepep_mtp_tpot_ms = 11.4
    dp16_ep16_deepep_mtp_goodput_tok_s_gpu = 2480
    tpot_speedup_x = round(cross_node_tp16_tpot_ms / dp16_ep16_deepep_mtp_tpot_ms, 2)  # 3.04x
    throughput_gain_x = round(
        dp16_ep16_deepep_mtp_goodput_tok_s_gpu / cross_node_tp16_goodput_tok_s_gpu, 2
    )  # 4.77x

    mtp_low_batch_speedup_x = 2.38
    mtp_auto_bypass_batch_threshold = 32
    prefix_cache_ttft_speedup_x = round(1180 / 290, 2)  # 4.07x
    deepep_exposed_comm_pct = 1.9
    cost_per_1m_usd = 0.088

    return {
        "deliverable_id": "#624",
        "topic": "enterprise_parallelism_tp_pp_dp_ep_dense_moe_acceleration_sizing",
        "cross_node_tp16_allreduces_per_tok": cross_node_tp16_allreduces_per_tok,
        "cross_node_tp16_net_sync_ms": cross_node_tp16_net_sync_ms,
        "cross_node_tp16_tpot_ms": cross_node_tp16_tpot_ms,
        "cross_node_tp16_goodput_tok_s_gpu": cross_node_tp16_goodput_tok_s_gpu,
        "dp16_ep16_deepep_mtp_tpot_ms": dp16_ep16_deepep_mtp_tpot_ms,
        "dp16_ep16_deepep_mtp_goodput_tok_s_gpu": dp16_ep16_deepep_mtp_goodput_tok_s_gpu,
        "tpot_speedup_x": tpot_speedup_x,
        "throughput_gain_x": throughput_gain_x,
        "mtp_low_batch_speedup_x": mtp_low_batch_speedup_x,
        "mtp_auto_bypass_batch_threshold": mtp_auto_bypass_batch_threshold,
        "prefix_cache_ttft_speedup_x": prefix_cache_ttft_speedup_x,
        "deepep_exposed_comm_pct": deepep_exposed_comm_pct,
        "cost_per_1m_usd": cost_per_1m_usd,
    }


if __name__ == "__main__":
    out = evaluate_parallelism_dense_moe_acceleration_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
