from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "dedicated_merged_model_per_tenant",
            "name": "Anti-Pattern 1: Dedicated Merged-Weight 70B Instance per Business Unit (64 Adapters = 64 Pods)",
            "active_adapters": 64,
            "base_weight_hbm_gb_total": 4480.0,
            "adapter_switch_latency_ms": 0.0,
            "p99_ttft_ms": 340.0,
            "p99_tpot_ms": 14.0,
            "required_h200_gpus": 128,
            "annual_tco_usd": 3430000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "naive_serial_disk_lora_swap",
            "name": "Anti-Pattern 2: Shared Base with Blocking Disk/Network LoRA Reload on Request",
            "active_adapters": 64,
            "base_weight_hbm_gb_total": 140.0,
            "adapter_switch_latency_ms": 1320.0,
            "p99_ttft_ms": 1650.0,
            "p99_tpot_ms": 24.5,
            "required_h200_gpus": 8,
            "annual_tco_usd": 214000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "vllm_sglang_slora_paged_multi_lora_pool",
            "name": "Optimal: vLLM / SGLang S-LoRA Shared 70B Base + HBM/Pinned-DRAM Paged Adapter Pool",
            "active_adapters": 64,
            "base_weight_hbm_gb_total": 140.0,
            "adapter_switch_latency_ms": 2.8,
            "p99_ttft_ms": 355.0,
            "p99_tpot_ms": 14.9,
            "required_h200_gpus": 8,
            "annual_tco_usd": 214000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_multi_lora_serving_slora_punica_shared_base_sizing_selection",
        "title": "Enterprise Multi-LoRA Serving (64+ Fine-Tuned Adapters on 1 Shared 70B Base Pool via vLLM / SGLang S-LoRA) Sizer (#517)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "gpu_saving_pct_vs_dedicated_merged_pods": round((1.0 - 8 / 128) * 100, 1),
            "annual_tco_saved_usd": 3430000 - 214000,
            "ttft_speedup_vs_disk_swap": round(1650.0 / 355.0, 2),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
