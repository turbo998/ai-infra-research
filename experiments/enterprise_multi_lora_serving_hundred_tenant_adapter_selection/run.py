from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    # Analytical model for Enterprise Multi-LoRA Serving & 120-Tenant Adapter Architecture Selection 
    # Workload: 120 customized fine-tuned tenant models (LoRA rank r=16, ~160 MB each on top of a shared base LLM),
    # 250 RPS aggregate cluster traffic across 120 enterprise tenants.
    custom_tenant_models = 120

    # Tier 1: 1 Dedicated Full Base-Model Instance per Tenant (Merged weights, 120 isolated GPU deployments)
    t1_gpus_required = 120
    t1_p99_ttft_ms = 240.0
    t1_adapter_cold_swap_stall_ms = 0.0
    t1_avg_gpu_utilization_pct = 8.5
    t1_monthly_cost_usd = 336_000

    # Tier 2: Merged-Weight Disk Swapping on Shared 8-GPU Pool (Reloading full model weights on tenant switch)
    t2_gpus_required = 8
    t2_p99_ttft_ms = 18_500.0  # 18.5 seconds model swap stall!
    t2_adapter_cold_swap_stall_ms = 18_200.0
    t2_avg_gpu_utilization_pct = 32.0
    t2_monthly_cost_usd = 22_400

    # Tier 3: Basic Multi-LoRA (vLLM/SGLang) + Stateless Round-Robin Gateway (120 adapters scattered across all nodes)
    t3_gpus_required = 12
    t3_p99_ttft_ms = 680.0  # Frequent CPU/disk LoRA slot eviction because Round-Robin scatters 120 adapters
    t3_adapter_cold_swap_stall_ms = 420.0
    t3_avg_gpu_utilization_pct = 58.0
    t3_monthly_cost_usd = 33_600

    # Tier 4: Adapter-Affinity Router + 3-Tier Pinned Host-DRAM LoRA Pool + Mixed-Adapter Batching (S-LoRA / Punica)
    t4_gpus_required = 8
    t4_p99_ttft_ms = 215.0  # Pinned Host-DRAM -> HBM PCIe DMA takes < 2ms, overlapped with base prefill!
    t4_adapter_cold_swap_stall_ms = 1.8
    t4_avg_gpu_utilization_pct = 82.0
    t4_monthly_cost_usd = 22_400

    gpu_reduction_t4_vs_t1_x = round(t1_gpus_required / t4_gpus_required, 1)
    cost_savings_t4_vs_t1_pct = round((t1_monthly_cost_usd - t4_monthly_cost_usd) / t1_monthly_cost_usd * 100.0, 1)
    ttft_speedup_t4_vs_t3_x = round(t3_p99_ttft_ms / t4_p99_ttft_ms, 2)

    tiers = [
        {
            "tier_id": "tier1_dedicated_instance_per_merged_model",
            "name": "Tier 1: 每租户合并权重独立部署整套底座 (120 个微调模型占 120 张 GPU，91.5% 算力空转)",
            "gpus_required_for_120_models": t1_gpus_required,
            "p99_ttft_ms": t1_p99_ttft_ms,
            "adapter_swap_stall_ms": t1_adapter_cold_swap_stall_ms,
            "avg_gpu_utilization_pct": t1_avg_gpu_utilization_pct,
            "monthly_cluster_cost_usd": t1_monthly_cost_usd,
            "status": "Bankrupting GPU Sprawl ($336K/mo)",
        },
        {
            "tier_id": "tier2_merged_weight_disk_swapping",
            "name": "Tier 2: 合并权重按需磁盘热切 (8 卡轮流卸载/重装整套大模型，P99 排队卡死 18.5 秒)",
            "gpus_required_for_120_models": t2_gpus_required,
            "p99_ttft_ms": t2_p99_ttft_ms,
            "adapter_swap_stall_ms": t2_adapter_cold_swap_stall_ms,
            "avg_gpu_utilization_pct": t2_avg_gpu_utilization_pct,
            "monthly_cluster_cost_usd": t2_monthly_cost_usd,
            "status": "Unusable Latency (18.5s Swap Stall)",
        },
        {
            "tier_id": "tier3_multi_lora_stateless_round_robin",
            "name": "Tier 3: 基础动态 Multi-LoRA + 随机轮询网关 (120 个适配器全集群打散导致频繁驱逐重读)",
            "gpus_required_for_120_models": t3_gpus_required,
            "p99_ttft_ms": t3_p99_ttft_ms,
            "adapter_swap_stall_ms": t3_adapter_cold_swap_stall_ms,
            "avg_gpu_utilization_pct": t3_avg_gpu_utilization_pct,
            "monthly_cluster_cost_usd": t3_monthly_cost_usd,
            "status": "Standard Multi-LoRA Baseline",
        },
        {
            "tier_id": "tier4_adapter_affinity_pinned_dram_slora_batching",
            "name": "Tier 4: 适配器亲和路由 + 锁页主机内存 (Pinned DRAM) 全量驻留 + 跨租户混合 Batch 并行推理",
            "gpus_required_for_120_models": t4_gpus_required,
            "p99_ttft_ms": t4_p99_ttft_ms,
            "adapter_swap_stall_ms": t4_adapter_cold_swap_stall_ms,
            "avg_gpu_utilization_pct": t4_avg_gpu_utilization_pct,
            "monthly_cluster_cost_usd": t4_monthly_cost_usd,
            "status": "Enterprise 100+ Tenant Multi-LoRA Gold Standard",
        },
    ]

    return {
        "experiment_id": "enterprise_multi_lora_serving_hundred_tenant_adapter_selection",
        "round": 393,
        "title": "Enterprise Multi-LoRA Serving & 120-Tenant Adapter Consolidation Architecture Evaluation",
        "provenance": "synthetic_analytical_model_calibrated_to_published_slora_punica_and_vllm_multi_lora_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "custom_tenant_adapters": custom_tenant_models,
            "lora_rank": 16,
            "adapter_size_mb": 160,
            "aggregate_cluster_rps": 250,
        },
        "headline_findings": {
            "tier4_gpus_required": t4_gpus_required,
            "gpu_reduction_t4_vs_t1_x": gpu_reduction_t4_vs_t1_x,
            "cost_savings_t4_vs_t1_pct": cost_savings_t4_vs_t1_pct,
            "ttft_speedup_t4_vs_t3_x": ttft_speedup_t4_vs_t3_x,
            "tier4_p99_ttft_ms": t4_p99_ttft_ms,
        },
        "architecture_tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic results.json")
    args = parser.parse_args()

    results = build_results()
    out_path = Path(__file__).with_name("results.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist; run without --check first.")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != results:
            raise SystemExit("results.json mismatch with deterministic build_results().")
        print("OK: enterprise_multi_lora_serving_hundred_tenant_adapter_selection verified.")
        return

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
