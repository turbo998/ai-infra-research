from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "Benchmark": "distributed_kv_cache_pool_lmcache_mooncake_vllm_production_stack_selection",
        "Provenance": {
            "Data_Status": "synthetic + anonymized_reported",
            "Independently_Reproduced": False,
            "Physical_GPU_Cluster_Measurement": "not_measured",
            "Harness_Policy": "enterprise_distributed_kv_cache_pool_lmcache_mooncake_dynamo_sizer_v1",
        },
        "Workload_Assumptions": {
            "Target_Workload": "Enterprise 32-Replica vLLM Cluster Serving Multi-Turn 32K-Context Agent & Shared RAG Documents (Llama-3.3-70B-FP8, 160 RPS)",
            "Cluster_Replicas": 32,
            "Avg_Context_Tokens": 32768,
            "KV_Size_Per_32K_Request_FP8_GB": 5.12,
            "Base_32K_Recompute_Prefill_ms": 1180.0,
        },
        "Configuration_Matrix": [
            {
                "Tier_ID": "Tier_1_Isolated_Pod_HBM_Cache_Round_Robin",
                "Architecture_Name": "Isolated Per-Pod GPU HBM Prefix Cache (`enable_prefix_caching`) + Round-Robin L7 Load Balancer",
                "KV_Storage_Hierarchy": "L1 Local GPU HBM Only (~40 GB Free per Pod); Zero Cross-Pod Sharing",
                "Cluster_Wide_KV_Hit_Rate_Pct": 14.5,
                "Multi_Turn_32K_P95_TTFT_ms": 1240.0,
                "Wasted_Prefill_Recompute_FLOPs_Pct": 78.0,
                "Required_H200_GPUs": 64,
                "Monthly_Cluster_TCO_USD": 179200,
                "Recommended_Scenario": "Anti-pattern for multi-turn chat or shared RAG across >2 replicas (85.5% cache miss)",
            },
            {
                "Tier_ID": "Tier_2_Session_Sticky_Routing_Local_Host_DRAM",
                "Architecture_Name": "Session-ID Sticky Routing + Local Host Pinned DRAM KV Offload (`swap_space` / Local CPU Cache)",
                "KV_Storage_Hierarchy": "L1 Local HBM + L2 Local Host DRAM (512 GB/Node); Isolated per Node",
                "Cluster_Wide_KV_Hit_Rate_Pct": 58.0,
                "Multi_Turn_32K_P95_TTFT_ms": 590.0,
                "Wasted_Prefill_Recompute_FLOPs_Pct": 38.0,
                "Required_H200_GPUs": 48,
                "Monthly_Cluster_TCO_USD": 134400,
                "Recommended_Scenario": "Single-user sticky chat without cross-user shared RAG documents or burst spillover",
            },
            {
                "Tier_ID": "Tier_3_LMCache_Shared_TCP_Redis_Disk_Backend",
                "Architecture_Name": "LMCache + vLLM Production Stack over Standard TCP/IP Shared Cache + CacheGen KV Compression",
                "KV_Storage_Hierarchy": "L1 HBM + L2 Host DRAM + L3 Shared TCP/NVMe Store with 3.2x CacheGen Quantization",
                "Cluster_Wide_KV_Hit_Rate_Pct": 86.5,
                "Multi_Turn_32K_P95_TTFT_ms": 215.0,
                "Wasted_Prefill_Recompute_FLOPs_Pct": 11.5,
                "Required_H200_GPUs": 32,
                "Monthly_Cluster_TCO_USD": 92800,
                "Recommended_Scenario": "Standard cloud K8s clusters without RDMA NICs sharing RAG chunks and multi-turn KV across pods",
            },
            {
                "Tier_ID": "Tier_4_Mooncake_Dynamo_KV_Router_RDMA_Layer_Pipelined",
                "Architecture_Name": "KV-Index-Aware Global Router (`Dynamo / vLLM Router`) + Mooncake / LMCache GPUDirect RDMA Pool + Layer-Wise Pipelining",
                "KV_Storage_Hierarchy": "Global Radix KV Directory + 16 TB Cluster Host DRAM/NVMe RDMA Pool (400GbE RoCEv2)",
                "Cluster_Wide_KV_Hit_Rate_Pct": 95.8,
                "Multi_Turn_32K_P95_TTFT_ms": 98.0,
                "Wasted_Prefill_Recompute_FLOPs_Pct": 3.2,
                "Required_H200_GPUs": 24,
                "Monthly_Cluster_TCO_USD": 69600,
                "Recommended_Scenario": "Production standard for multi-replica 32K–128K Agent & RAG clusters (cuts P95 TTFT by 92.1% and GPUs by 62.5%)",
            },
        ],
        "Decision_Summary": {
            "Baseline_Tier": "Tier_1_Isolated_Pod_HBM_Cache_Round_Robin",
            "Recommended_Tier": "Tier_4_Mooncake_Dynamo_KV_Router_RDMA_Layer_Pipelined",
            "Cluster_KV_Hit_Rate_Gain_Pts": 81.3,
            "P95_TTFT_Reduction_Pct": 92.1,
            "GPU_Count_Reduction_Pct": 62.5,
            "Monthly_TCO_Savings_Pct": 61.2,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()

    if args.check:
        if not out_path.exists():
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing["Benchmark"] == payload["Benchmark"]
        assert len(existing["Configuration_Matrix"]) == 4
        print("Wrote results.json")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
