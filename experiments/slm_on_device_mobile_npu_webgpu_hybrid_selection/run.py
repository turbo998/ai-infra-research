"""Deterministic evaluator for Small Language Model (SLM 1B-8B) On-Device Mobile NPU, WebGPU & Cloud-Edge Hybrid Routing Selection (Round 63).

Models median Time-To-First-Token (TTFT ms), sustained Decode speed after 5-minute thermal soak (tok/s),
device active power draw (Watts), complex + routine blended task accuracy (%),
and daily cloud GPU serving bill across 100M daily queries (USD/day) for four architectures:
  - Tier 1: 100% Cloud 70B API Fallback (High accuracy 91.2%, but 380ms network TTFT & $42,000/day cloud bill)
  - Tier 2: On-Device Mobile CPU/GPU GGUF Q4_K_M / Browser WebGPU (8.5W thermal throttling -> 14.2 tok/s & 70.8% accuracy)
  - Tier 3: Pure On-Device Mobile NPU Static Graph + SpinQuant W4A8 (ExecuTorch/QNN/ANE -> 54.0 tok/s @ 2.4W, 71.5% accuracy)
  - Tier 4: Confidence-Gated Cloud-Edge Hybrid Cascade (75% Local 3B NPU @ $0 + 25% Hard Queries Routed to Cloud 70B)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_slm_on_device_npu_hybrid_matrix() -> dict[str, Any]:
    workload = {
        "user_and_traffic_scale": "10 Million Daily Active Users generating 100 Million Queries/Day (75% routine UI/summarize/rewrite/local-RAG, 25% complex reasoning/code)",
        "client_hardware_target": "Flagship Mobile SoC / Thin-and-Light AI PC (68-136 GB/s LPDDR5X, 45 TOPS INT8/W4A8 NPU, 3.5W sustained thermal envelope)",
        "model_pair": "On-Device 3.8B SLM (SpinQuant W4A8) + Cloud 70B FP8 Teacher Fallback",
    }

    tiers = [
        {
            "id": "tier1_pure_cloud_70b_api_serving",
            "name": "Tier 1: 100% 纯云端 70B API 推理 (全量上云 -> 网络延迟 380ms 且年烧 $1,533万 云算力)",
            "name_en": "Tier 1: 100% Pure Cloud 70B API Serving (380ms Network TTFT & $42,000/Day Cloud Bill)",
            "execution_stack": "All 100M daily queries sent over cellular/Wi-Fi to cloud 70B FP8 cluster",
            "thermal_and_privacy_behavior": "0W device compute heat, but zero offline availability, PII leaves device, and huge cloud GPU OPEX",
            "median_ttft_ms": 380.0,
            "sustained_decode_tok_per_s": 42.0,
            "device_power_draw_watts": 0.8,
            "blended_task_accuracy_pct": 91.2,
            "daily_cloud_cost_usd": 42000.0,
            "recommended_for": "Low-volume B2B apps where client devices lack NPUs and cloud cost is negligible.",
            "status": "fail",
        },
        {
            "id": "tier2_on_device_cpu_gpu_gguf_q4km_webgpu",
            "name": "Tier 2: 端侧通用 CPU/GPU (llama.cpp GGUF Q4_K_M / WebGPU -> 8.5W 发热降频 + 复杂题崩盘)",
            "name_en": "Tier 2: On-Device Mobile CPU/GPU GGUF Q4_K_M / WebGPU (8.5W Thermal Throttling)",
            "execution_stack": "W4A16 dynamic dequantization on mobile GPU shaders / WebGPU / ARM NEON CPU",
            "thermal_and_privacy_behavior": "8.5W power draw hits 44C skin temp limit in 90s, throttling Decode to 14.2 tok/s & draining battery",
            "median_ttft_ms": 245.0,
            "sustained_decode_tok_per_s": 14.2,
            "device_power_draw_watts": 8.5,
            "blended_task_accuracy_pct": 70.8,
            "daily_cloud_cost_usd": 0.0,
            "recommended_for": "Zero-install browser WebGPU demos (WebLLM) and desktop laptops with active fan cooling.",
            "status": "fail",
        },
        {
            "id": "tier3_pure_on_device_npu_spinquant_w4a8",
            "name": "Tier 3: 纯端侧专用 NPU 静态图 + SpinQuant W4A8 旋转量化 (ExecuTorch/QNN/CoreML -> 2.4W 极速省电)",
            "name_en": "Tier 3: Pure On-Device NPU Static Graph + SpinQuant W4A8 (54.0 tok/s @ 2.4W)",
            "execution_stack": "Learned Hadamard rotation (SpinQuant) + W4A8 integer NPU static graph (Qualcomm QNN / Apple ANE)",
            "thermal_and_privacy_behavior": "2.4W ultra-cool sustained execution (54.0 tok/s, zero throttling), but 3.8B SLM fails on 25% hard reasoning queries",
            "median_ttft_ms": 65.0,
            "sustained_decode_tok_per_s": 54.0,
            "device_power_draw_watts": 2.4,
            "blended_task_accuracy_pct": 71.5,
            "daily_cloud_cost_usd": 0.0,
            "recommended_for": "100% offline/air-gapped mobile tasks (UI action routing, notification summary, local keyboard rewrite).",
            "status": "pass",
        },
        {
            "id": "tier4_confidence_gated_cloud_edge_hybrid_cascade",
            "name": "Tier 4: 端云协同置信度路由 (75% 端侧 3.8B NPU 零成本秒回 + 25% 高熵难题无缝上云 70B)",
            "name_en": "Tier 4: Confidence-Gated Cloud-Edge Hybrid Cascade (75% Local NPU + 25% Cloud 70B)",
            "execution_stack": "On-device SpinQuant W4A8 NPU executes first-token margin/entropy gate -> escalates top 25% hard queries to Cloud 70B",
            "thermal_and_privacy_behavior": "2.4W cool NPU + 75% private offline execution + 89.8% flagship accuracy at -74.8% cloud cost",
            "median_ttft_ms": 72.0,
            "sustained_decode_tok_per_s": 51.5,
            "device_power_draw_watts": 2.1,
            "blended_task_accuracy_pct": 89.8,
            "daily_cloud_cost_usd": 10600.0,
            "recommended_for": "Consumer & enterprise AI apps (Apple Intelligence / Copilot+ PC paradigm) needing flagship IQ, 72ms TTFT, and -74.8% cloud spend.",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier3_vs_tier2_npu_energy_efficiency_ratio": round(
            (t3["sustained_decode_tok_per_s"] / t3["device_power_draw_watts"])
            / (t2["sustained_decode_tok_per_s"] / t2["device_power_draw_watts"]),
            2,
        ),
        "tier4_vs_tier1_ttft_speedup_ratio": round(
            t1["median_ttft_ms"] / t4["median_ttft_ms"], 2
        ),
        "tier4_vs_tier1_cloud_cost_reduction_pct": round(
            (1.0 - t4["daily_cloud_cost_usd"] / t1["daily_cloud_cost_usd"]) * 100.0, 1
        ),
        "tier4_annual_cloud_savings_usd": round(
            (t1["daily_cloud_cost_usd"] - t4["daily_cloud_cost_usd"]) * 365.0, 0
        ),
    }

    return {
        "experiment_id": "slm_on_device_mobile_npu_webgpu_hybrid_selection",
        "date": "2026-11-12",
        "round": 63,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "SpinQuant (LLM Quantization with Learned Rotations for W4A8 Mobile NPU Deployment)",
                "PyTorch ExecuTorch / Qualcomm QNN Hexagon NPU & Apple CoreML ANE SLM Benchmarks",
                "Hybrid LLM / RouteLLM Confidence-Gated Edge-Cloud Cascade Cost-Quality Trade-off Studies",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_slm_on_device_npu_hybrid_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
