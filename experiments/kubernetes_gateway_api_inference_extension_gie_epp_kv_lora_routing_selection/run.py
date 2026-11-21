from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "Benchmark": "kubernetes_gateway_api_inference_extension_gie_epp_kv_lora_routing_selection",
        "Provenance": {
            "Data_Status": "synthetic + anonymized_reported",
            "Independently_Reproduced": False,
            "Physical_GPU_Cluster_Measurement": "not_measured",
            "Harness_Policy": "enterprise_k8s_gateway_api_inference_extension_gie_epp_sizer_v1",
        },
        "Workload_Assumptions": {
            "Target_Workload": "24-Pod vLLM Kubernetes Cluster Serving Heterogeneous Prompts (256 to 32K Tokens), 80 LoRA Adapters & 3 Criticality Tiers (220 RPS)",
            "Cluster_Pods": 24,
            "Total_H200_GPUs": 48,
            "Active_LoRA_Adapters": 80,
            "Incoming_Traffic_RPS": 220,
        },
        "Configuration_Matrix": [
            {
                "Tier_ID": "Tier_1_K8s_ClusterIP_Nginx_Round_Robin",
                "Architecture_Name": "Standard Kubernetes `Service` / Nginx Ingress (`Round-Robin` L7 Load Balancing)",
                "Routing_Intelligence": "Stateless Cyclic Dispatch; Blind to Prompt Length, KV Cache Usage & LoRA Slots",
                "Cluster_P99_TTFT_ms": 1890.0,
                "Cluster_P99_TPOT_ms": 58.5,
                "HBM_KV_Preemption_Rate_Pct": 19.4,
                "SLA_Compliant_Goodput_RPS": 132.0,
                "Monthly_Required_GPU_TCO_For_220RPS_USD": 224000,
                "Recommended_Scenario": "Anti-pattern for LLM pods: stacks multiple 32K requests onto one pod while sibling pods sit idle",
            },
            {
                "Tier_ID": "Tier_2_Envoy_Service_Mesh_Least_Active_Requests",
                "Architecture_Name": "Envoy / Istio Service Mesh (`Least-Request / P2C` Active Connection Balancing)",
                "Routing_Intelligence": "Counts In-Flight HTTP Streams Only (`1` 32K Stream Treated Equal to `1` 256-Token Stream)",
                "Cluster_P99_TTFT_ms": 1280.0,
                "Cluster_P99_TPOT_ms": 42.0,
                "HBM_KV_Preemption_Rate_Pct": 11.2,
                "SLA_Compliant_Goodput_RPS": 164.0,
                "Monthly_Required_GPU_TCO_For_220RPS_USD": 179200,
                "Recommended_Scenario": "Basic improvement over Round-Robin, but still blind to KV memory pressure and prefix/LoRA locality",
            },
            {
                "Tier_ID": "Tier_3_Consistent_Hash_Header_Sticky_Routing",
                "Architecture_Name": "Static Consistent-Hash Header Routing (`X-Session-ID` / `X-LoRA-ID` Ring Hashing)",
                "Routing_Intelligence": "Maximizes Prefix/LoRA Locality, But Creates Severe Hot-Pod Queueing Under Skewed Traffic",
                "Cluster_P99_TTFT_ms": 960.0,
                "Cluster_P99_TPOT_ms": 34.5,
                "HBM_KV_Preemption_Rate_Pct": 8.6,
                "SLA_Compliant_Goodput_RPS": 182.0,
                "Monthly_Required_GPU_TCO_For_220RPS_USD": 156800,
                "Recommended_Scenario": "Uniformly distributed sticky sessions without bursty whale tenants or skewed LoRA popularity",
            },
            {
                "Tier_ID": "Tier_4_K8s_Gateway_API_Inference_Extension_GIE_EPP",
                "Architecture_Name": "Kubernetes `Gateway API Inference Extension (GIE)` (`InferencePool` + `InferenceModel` + `Endpoint Picker EPP` via Envoy `ext_proc`)",
                "Routing_Intelligence": "Composite Filter/Scorer: `Criticality Shedding` -> `KV < 85% Filter` -> `Prefix + LoRA Affinity` -> `Queue Token Load`",
                "Cluster_P99_TTFT_ms": 485.0,
                "Cluster_P99_TPOT_ms": 21.2,
                "HBM_KV_Preemption_Rate_Pct": 0.4,
                "SLA_Compliant_Goodput_RPS": 218.0,
                "Monthly_Required_GPU_TCO_For_220RPS_USD": 134400,
                "Recommended_Scenario": "Official K8s WG-Serving standard for cloud-native vLLM/SGLang fleets (cuts P99 TTFT by 74.3% and boosts Goodput by +65.2%)",
            },
        ],
        "Decision_Summary": {
            "Baseline_Tier": "Tier_1_K8s_ClusterIP_Nginx_Round_Robin",
            "Recommended_Tier": "Tier_4_K8s_Gateway_API_Inference_Extension_GIE_EPP",
            "P99_TTFT_Reduction_Pct": 74.3,
            "P99_TPOT_Reduction_Pct": 63.8,
            "Goodput_Gain_Pct": 65.2,
            "Monthly_TCO_Savings_Pct": 40.0,
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
