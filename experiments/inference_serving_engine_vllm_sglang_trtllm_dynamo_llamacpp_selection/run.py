from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "inference_serving_engine_vllm_sglang_trtllm_dynamo_llamacpp_selection",
        "title": "LLM Inference Serving Engine Selection: vLLM V1 vs. SGLang vs. TensorRT-LLM vs. NVIDIA Dynamo vs. Ollama/llama.cpp",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_nodes_8x_h200": 4,
            "total_gpus": 32,
            "workload_mix": "Multi-Turn Agent + JSON Tool Calling + DeepSeek/Llama Models",
        },
        "tiers": [
            {
                "tier_id": "tier1_ollama_llamacpp_deployed_for_datacenter_high_concurrency",
                "name": "Tier 1: Misusing Ollama / llama.cpp GGUF for High-Concurrency Datacenter Serving (64+ Concurrent Users)",
                "sla_compliant_goodput_tps_per_gpu": 290,
                "multi_turn_prefix_cache_hit_rate_pct": 15.0,
                "p99_tpot_ms": 68.5,
                "cost_per_1m_tokens_usd": 2.95,
                "verdict": "Anti-Pattern for Datacenter: Great for local laptop/edge (B<=4), chokes under 64-user continuous batching",
            },
            {
                "tier_id": "tier2_vllm_v1_unified_engine_general_enterprise_maas_standard",
                "name": "Tier 2: vLLM V1 Unified Engine (Zero-Overhead Scheduler + FlashInfer + Piece-wise CUDA Graph)",
                "sla_compliant_goodput_tps_per_gpu": 1960,
                "multi_turn_prefix_cache_hit_rate_pct": 76.4,
                "p99_tpot_ms": 14.8,
                "cost_per_1m_tokens_usd": 0.44,
                "verdict": "Best general-purpose enterprise MaaS default: Day-0 support for 100+ models across NVIDIA, AMD & TPU",
            },
            {
                "tier_id": "tier3_standalone_tensorrt_llm_aot_compiled_engines",
                "name": "Tier 3: Standalone TensorRT-LLM (AOT Compiled C++ Engines, Peak Single-Node FP8/NVFP4 Kernel Speed)",
                "sla_compliant_goodput_tps_per_gpu": 2120,
                "multi_turn_prefix_cache_hit_rate_pct": 74.0,
                "p99_tpot_ms": 12.6,
                "cost_per_1m_tokens_usd": 0.41,
                "verdict": "Lowest raw TPOT (12.6ms) on fixed models, but requires per-GPU-SKU engine compilation & higher ops effort",
            },
            {
                "tier_id": "tier4_dynamo_kv_router_nixl_orchestrating_sglang_vllm_trtllm_workers",
                "name": "Tier 4: NVIDIA Dynamo (KV-Aware Router + NIXL RDMA PD Split) + SGLang (Radix/MoE) & vLLM V1 Workers",
                "sla_compliant_goodput_tps_per_gpu": 2490,
                "multi_turn_prefix_cache_hit_rate_pct": 92.8,
                "p99_tpot_ms": 11.9,
                "cost_per_1m_tokens_usd": 0.34,
                "verdict": "Recommended: 2,490 Goodput tok/s/GPU, 92.8% Radix KV hit rate, 11.9ms TPOT & $0.34/1M (-88.5% vs Tier 1)",
            },
        ],
        "headline_findings": {
            "goodput_tps_per_gpu": 2490,
            "radix_kv_hit_rate_pct": 92.8,
            "p99_tpot_ms": 11.9,
            "cost_per_1m_usd": 0.34,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
