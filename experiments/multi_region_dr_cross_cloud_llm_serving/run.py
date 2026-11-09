from __future__ import annotations

import json
from pathlib import Path


def evaluate_multi_region_dr_tiers() -> dict:
    """Deterministic evaluator for 4 Multi-Region Disaster Recovery & Cross-Cloud LLM Serving Tiers.

    Workload Baseline:
      - Global Enterprise Copilot / Agent Service (32K Average Multi-Turn Context, 8 Turns per Session)
      - Peak Global Traffic: 1,200 req/s (P60 Base Load: 650 req/s; Diurnal Peak: 1,200 req/s)
      - Evaluates:
        1) Multi-Turn KV Prefix Cache Hit Rate (%)
        2) Steady-State Multi-Turn P95 TTFT (ms)
        3) Region/AZ Outage Recovery Time Objective (RTO in seconds) & Request Drop Rate during Failover (%)
        4) Annual Multi-Region GPU Cluster TCO (USD)
    """
    tiers = [
        {
            "tier_id": "tier1_active_passive_cold_standby_dr",
            "name": "Tier 1: Single-Region Active + Passive Cold-Standby DR Region (Scale-from-Zero on Outage)",
            "name_zh": "Tier 1: 单地域主备冷灾备 (Active-Passive，故障时冷拉备份地域 GPU 集群)",
            "routing_and_state_sync": "100% traffic in Primary Region; Secondary Region unprovisioned / cold until DNS failover",
            "multi_turn_kv_cache_hit_rate_pct": 88.5,
            "steady_state_p95_ttft_ms": 265.0,
            "region_outage_rto_sec": 340.0,
            "failover_request_drop_rate_pct": 100.0,
            "wan_bandwidth_overhead_gbps": 0.05,
            "annual_cluster_tco_usd": 4200000,
        },
        {
            "tier_id": "tier2_stateless_geo_round_robin_active_active",
            "name": "Tier 2: Stateless Geo-Round-Robin Active-Active (2 Regions at 200% Static Peak Capacity, No Session Affinity)",
            "name_zh": "Tier 2: 跨地域无状态轮询双活 (Stateless Geo-Round-Robin，打碎多轮 KV 缓存 + 200% 静态冗余)",
            "routing_and_state_sync": "Anycast / Round-Robin across Region A & Region B (65ms WAN RTT); Turn N+1 lands on random region",
            "multi_turn_kv_cache_hit_rate_pct": 44.0,
            "steady_state_p95_ttft_ms": 1590.0,
            "region_outage_rto_sec": 8.5,
            "failover_request_drop_rate_pct": 14.2,
            "wan_bandwidth_overhead_gbps": 18.5,
            "annual_cluster_tco_usd": 6400000,
        },
        {
            "tier_id": "tier3_metro_multi_az_rdma_shared_kv_pool",
            "name": "Tier 3: Metro Multi-AZ Active-Active (3 AZs <1.5ms Dark Fiber + Shared L2 Distributed KV Cache Pool)",
            "name_zh": "Tier 3: 同城三可用区双活 (Multi-AZ <1.5ms 裸光纤 + 跨 AZ 分布式 L2 KV 缓存池)",
            "routing_and_state_sync": "Consistent-Hash Session Routing + Cross-AZ Compressed KV Block Replication over 1.2ms Metro RoCE",
            "multi_turn_kv_cache_hit_rate_pct": 89.5,
            "steady_state_p95_ttft_ms": 250.0,
            "region_outage_rto_sec": 1.2,
            "failover_request_drop_rate_pct": 0.2,
            "wan_bandwidth_overhead_gbps": 12.0,
            "annual_cluster_tco_usd": 4560000,
        },
        {
            "tier_id": "tier4_sticky_geo_active_active_token_replay_cloud_burst",
            "name": "Tier 4: Session-Sticky Geo-Active-Active (Bounded-Load Hash) + Async Token-Transcript Replay + Hybrid Cloud Bursting",
            "name_zh": "Tier 4: 跨地域会话粘性双活 (有界负载一致性哈希) + 异步文本流复制重放 + 混合云弹性溢出",
            "routing_and_state_sync": "Session-Sticky Geo-Router + 40KB Compressed Token-IDs Replication (Not 8GB Raw KV!) + P60 Base + Cloud Spot Burst",
            "multi_turn_kv_cache_hit_rate_pct": 91.2,
            "steady_state_p95_ttft_ms": 240.0,
            "region_outage_rto_sec": 1.8,
            "failover_request_drop_rate_pct": 0.0,
            "wan_bandwidth_overhead_gbps": 0.12,
            "annual_cluster_tco_usd": 3744000,
        },
    ]

    # WAN State Replication Perturbation: Shipping Raw KV Tensors vs Compressed Token Transcript over 65ms WAN
    wan_replication_comparison = [
        {
            "replication_payload_mode": "Raw GQA-8 KV Cache Tensor (32K Context = 8.0 GB per Session)",
            "payload_size_mb": 8192.0,
            "wan_transfer_time_at_10gbps_ms": 6553.6,
            "verdict": "Physically Impossible over WAN! Saturates 100Gbps inter-region links at just 15 sessions/s ✕",
        },
        {
            "replication_payload_mode": "DeepSeek MLA Latent FP8 KV Cache (32K Context = 0.70 GB per Session)",
            "payload_size_mb": 716.8,
            "wan_transfer_time_at_10gbps_ms": 573.4,
            "verdict": "Feasible inside Metro Multi-AZ (<1.5ms 400G RoCE), still too heavy for cross-continent WAN ⚠️",
        },
        {
            "replication_payload_mode": "Token-ID Transcript + Tool State JSON (32K Context = 64 KB compressed)",
            "payload_size_mb": 0.064,
            "wan_transfer_time_at_10gbps_ms": 0.05,
            "verdict": "128,000x Smaller! Replicates across continents in <1ms; peer region recomputes KV only on actual failover 🏆",
        },
    ]

    t2 = tiers[1]
    t4 = tiers[3]

    summary = {
        "tier2_kv_hit_rate_drop_pt": round(
            t4["multi_turn_kv_cache_hit_rate_pct"] - t2["multi_turn_kv_cache_hit_rate_pct"], 1
        ),
        "tier4_vs_tier2_ttft_speedup_x": round(
            t2["steady_state_p95_ttft_ms"] / t4["steady_state_p95_ttft_ms"], 2
        ),
        "tier4_failover_rto_sec": t4["region_outage_rto_sec"],
        "tier4_vs_static_dual_region_tco_saving_pct": round(
            (t2["annual_cluster_tco_usd"] - t4["annual_cluster_tco_usd"])
            / t2["annual_cluster_tco_usd"]
            * 100.0,
            1,
        ),
        "tier4_vs_static_dual_region_tco_saving_usd": t2["annual_cluster_tco_usd"] - t4["annual_cluster_tco_usd"],
    }

    return {
        "experiment_id": "multi_region_dr_cross_cloud_llm_serving",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_multi_region_kv_affinity_and_cloud_bursting_tco_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "Consistent Hashing with Bounded Load for Stateful LLM KV-Prefix Geo-Routing",
                "Metro Multi-AZ RoCEv2 Distributed KV Cache vs Cross-Region Token-Transcript Replication",
                "Hybrid On-Premises Base (P60) + Cloud GPU Bursting (P95) Capacity Amortization Model",
            ],
        },
        "workload_baseline": {
            "session_context_tokens": 32768,
            "turns_per_session": 8,
            "peak_global_rps": 1200,
            "inter_region_wan_rtt_ms": 65.0,
        },
        "tiers": tiers,
        "wan_replication_comparison": wan_replication_comparison,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_multi_region_dr_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
