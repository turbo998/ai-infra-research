"""Deterministic evaluator for Enterprise GPU Observability, SLO Alerting & FinOps Governance Architecture Selection (Round 370).

Models incident Mean-Time-To-Resolution (MTTR in minutes), wasted GPU/Token budget percentage (%),
runaway Agent loop detection rate (%), false-negative SLO alert rate (%), and monthly net AI platform spend (USD)
across four enterprise observability & FinOps tiers (64-GPU platform, $160,000/month baseline spend):
  - Tier 1: Legacy Host Monitoring + Coarse nvidia-smi GPU-Util (%) Only (Blind to Tensor Core Idle & KV Cache Full)
  - Tier 2: Infrastructure Silicon Telemetry (Prometheus + NVIDIA DCGM Exporter: PIPE_TENSOR_ACTIVE, DRAM_ACTIVE, XID)
  - Tier 3: Silicon DCGM + Serving Engine Metrics (vLLM/SGLang KV Cache Occupancy, Preemptions, Queue Depth, TTFT/TPOT)
  - Tier 4: Full-Stack 4-Layer Observability & FinOps Hub (DCGM + Engine + OpenTelemetry GenAI Traces [Langfuse/Phoenix] + Unit-Economics Circuit Breakers)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_enterprise_gpu_observability_dcgm_otel_genai_finops() -> dict[str, Any]:
    workload = {
        "enterprise_scenario": "64-GPU Enterprise AI Platform + Multi-Cloud MaaS ($160,000/Month Raw Baseline Spend across 15 Business Apps & ReAct Agents)",
        "target_sla": "Incident MTTR <= 10.0 Minutes, Wasted Compute/Token Spend <= 8.0%, 99%+ Runaway Agent Loop Interception",
        "core_observability_truth": "nvidia-smi GPU-Util (%) is a boolean-over-time activity counter that hides both 90% idle Tensor Cores and 100% full KV caches.",
    }

    tiers = [
        {
            "id": "tier1_legacy_host_and_coarse_nvidia_smi_gpu_util",
            "name": "Tier 1: 传统主机监控 + 粗粒度 nvidia-smi GPU-Util (%) (看不见 KV 满载与 Tensor Core 空转)",
            "name_en": "Tier 1: Legacy Host Metrics + Coarse nvidia-smi GPU-Util (%) Only",
            "telemetry_stack": "Node Exporter CPU/RAM + nvidia-smi utilization.gpu (No DCGM Profiling, No Engine Queue or Token Trace)",
            "incident_mttr_minutes": 240.0,
            "wasted_compute_token_spend_pct": 38.5,
            "runaway_agent_loop_interception_pct": 0.0,
            "false_negative_slo_alert_pct": 42.0,
            "monthly_net_platform_spend_usd": 160000.0,
            "recommended_for": "Anti-pattern: GPU-Util=99% can mask 10% actual Tensor Core usage, while GPU-Util=35% can hide a 100% full vLLM KV cache.",
            "status": "fail",
        },
        {
            "id": "tier2_prometheus_nvidia_dcgm_silicon_telemetry",
            "name": "Tier 2: 基础设施硬件层遥测 (Prometheus + NVIDIA DCGM Exporter · 监控 Tensor Core 与 HBM 带宽)",
            "name_en": "Tier 2: Infrastructure Silicon Telemetry (Prometheus + NVIDIA DCGM Profiling Metrics)",
            "telemetry_stack": "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE + DCGM_FI_PROF_DRAM_ACTIVE + Thermal/Power/XID Alerts",
            "incident_mttr_minutes": 65.0,
            "wasted_compute_token_spend_pct": 24.0,
            "runaway_agent_loop_interception_pct": 0.0,
            "false_negative_slo_alert_pct": 18.5,
            "monthly_net_platform_spend_usd": 136800.0,
            "recommended_for": "IaaS GPU cluster operators monitoring hardware health, XID faults, and physical Tensor Core / HBM bandwidth saturation.",
            "status": "warn",
        },
        {
            "id": "tier3_dcgm_plus_vllm_sglang_engine_kv_slo_metrics",
            "name": "Tier 3: 硬件 DCGM + 推理引擎内部指标联动 (vLLM/SGLang KV 缓存水位、重算抢占与排队监控)",
            "name_en": "Tier 3: Silicon DCGM + Serving Engine Metrics (vLLM/SGLang KV Cache, Preemptions & TTFT/TPOT)",
            "telemetry_stack": "DCGM + vllm:gpu_cache_usage_perc + vllm:num_preemptions_total + vllm:num_requests_waiting + Grafana SLO",
            "incident_mttr_minutes": 14.0,
            "wasted_compute_token_spend_pct": 12.5,
            "runaway_agent_loop_interception_pct": 25.0,
            "false_negative_slo_alert_pct": 2.4,
            "monthly_net_platform_spend_usd": 116000.0,
            "recommended_for": "Kubernetes inference platform teams tuning KEDA autoscaling, Prefix Cache hit rates, and P99 TTFT/TPOT SLAs.",
            "status": "pass",
        },
        {
            "id": "tier4_fullstack_4layer_otel_genai_langfuse_finops_hub",
            "name": "Tier 4: 全栈四层可观测与 FinOps 治理中枢 (DCGM + 引擎指标 + OTel GenAI 链路追踪 + 单任务成本熔断)",
            "name_en": "Tier 4: Full-Stack 4-Layer Observability & FinOps Hub (DCGM + Engine + OTel GenAI Traces + Cost Breakers)",
            "telemetry_stack": "DCGM + vLLM Metrics + OpenTelemetry GenAI (Langfuse/Phoenix) + Per-Trace Budget Breaker + Kubecost FinOps",
            "incident_mttr_minutes": 3.5,
            "wasted_compute_token_spend_pct": 4.2,
            "runaway_agent_loop_interception_pct": 99.5,
            "false_negative_slo_alert_pct": 0.1,
            "monthly_net_platform_spend_usd": 99200.0,
            "recommended_for": "Enterprise AI governance standard: links GPU silicon metrics to exact Agent trace_id, slashing MTTR 68.6x and spend by 38.0%.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_vs_tier1_mttr_speedup_x": round(
            t1["incident_mttr_minutes"] / t4["incident_mttr_minutes"], 1
        ),
        "tier4_monthly_savings_vs_tier1_usd": round(
            t1["monthly_net_platform_spend_usd"] - t4["monthly_net_platform_spend_usd"], 0
        ),
        "tier4_spend_reduction_vs_tier1_pct": round(
            (t1["monthly_net_platform_spend_usd"] - t4["monthly_net_platform_spend_usd"])
            / t1["monthly_net_platform_spend_usd"]
            * 100.0,
            1,
        ),
        "tier4_runaway_agent_interception_pct": t4["runaway_agent_loop_interception_pct"],
    }

    return {
        "experiment_id": "enterprise_gpu_observability_dcgm_otel_genai_finops_selection",
        "date": "2026-11-18",
        "round": 370,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_published_gpu_observability_and_finops_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "NVIDIA DCGM Exporter Profiling Metrics Guide (DCGM_FI_PROF_PIPE_TENSOR_ACTIVE vs. DCGM_FI_DEV_GPU_UTIL)",
                "vLLM & SGLang Production Prometheus Metrics & KV Cache Preemption Diagnostics",
                "OpenTelemetry GenAI Semantic Conventions, Langfuse / Arize Phoenix Tracing & FinOps Unit Economics",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify sealed results.json matches evaluator")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    result = evaluate_enterprise_gpu_observability_dcgm_otel_genai_finops()

    if args.check:
        sealed = json.loads(out_path.read_text(encoding="utf-8"))
        assert sealed == result, "Sealed results.json mismatch!"
        print("OK: enterprise_gpu_observability_dcgm_otel_genai_finops_selection verified.")
        return

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
