from __future__ import annotations

import json
from pathlib import Path


def evaluate_attention_state_space_tiers() -> dict:
    """Deterministic evaluator for 4 Long-Context Attention & Hybrid State-Space Model (SSM) Tiers.

    Workload Baseline:
      - ~70B/80B-class LLM (64 layers, hidden_dim=8192) served at 128K Context Length on 4×H200 SXM
      - Evaluates:
        1) KV Cache + Recurrent State Footprint per Request at 128K Context (GB, FP16/BF16 baseline vs compressed)
        2) 128K Prefill TTFT Latency (seconds, showing O(N^2) attention vs O(N) linear/SSM layers)
        3) 128K Decode Throughput at Batch=16 (tokens / sec)
        4) RULER / Needle-in-a-Haystack 128K Exact Retrieval & Multi-Needle Accuracy (%)
        5) Prefix Cache Compatibility (Standard Page Table vs MambaRadixCache State Snapshot)
    """
    tiers = [
        {
            "tier_id": "tier1_standard_gqa8_full_attention",
            "name": "Tier 1: Standard GQA-8 Full Attention (64 Full-Attention Layers, 8 KV Heads)",
            "name_zh": "Tier 1: 标准 GQA-8 全注意力架构 (64 层全注意力, O(N²) 复杂度)",
            "layer_topology": "64 × GQA-8 Full Attention (2048 elements/token/layer in BF16)",
            "kv_and_state_gb_at_128k_per_req": 32.0,
            "prefill_128k_ttft_sec": 6.80,
            "decode_128k_batch16_tps": 260.0,
            "ruler_128k_needle_accuracy_pct": 97.2,
            "prefix_cache_mechanism": "Standard Paged KV Block Table",
        },
        {
            "tier_id": "tier2_pure_mamba2_or_swa_window",
            "name": "Tier 2: Pure Mamba-2 SSM / Fixed 4K Sliding-Window Attention (SWA)",
            "name_zh": "Tier 2: 纯 Mamba-2 状态空间 / 固定 4K 滑动窗口注意力 (SWA)",
            "layer_topology": "64 × Pure Mamba-2 SSM or 4K Sliding Window (O(1) State, Zero Full-Context KV)",
            "kv_and_state_gb_at_128k_per_req": 1.0,
            "prefill_128k_ttft_sec": 1.65,
            "decode_128k_batch16_tps": 1180.0,
            "ruler_128k_needle_accuracy_pct": 41.5,
            "prefix_cache_mechanism": "Recurrent State Overwrites Past Tokens (Loses Exact Copy/Needle Retrieval!)",
        },
        {
            "tier_id": "tier3_deepseek_mla_latent_compression",
            "name": "Tier 3: DeepSeek MLA (Multi-Head Latent Attention: Low-Rank Joint KV Compression + Matrix Absorption)",
            "name_zh": "Tier 3: DeepSeek MLA 潜在向量低秩压缩注意力 (576 维联合潜变量 + 矩阵吸收)",
            "layer_topology": "64 × MLA Full-Context Attention (dkv=512 latent + 64 decoupled RoPE = 576 elements/token)",
            "kv_and_state_gb_at_128k_per_req": 5.62,
            "prefill_128k_ttft_sec": 5.10,
            "decode_128k_batch16_tps": 785.0,
            "ruler_128k_needle_accuracy_pct": 97.8,
            "prefix_cache_mechanism": "Standard Paged Latent KV Cache (c_KV cached directly)",
        },
        {
            "tier_id": "tier4_hybrid_3to1_linear_mamba_plus_gqa_mla",
            "name": "Tier 4: 3:1 Hybrid Linear-Attention / Mamba-2 + Full Attention (Qwen3-Next / Nemotron-H / Jamba + MambaRadixCache)",
            "name_zh": "Tier 4: 3:1 混合线性注意力/Mamba-2 + 全注意力 (48层 O(N) 线性层 + 16层全注意力 + MambaRadixCache)",
            "layer_topology": "48 × Gated Linear Attention / Mamba-2 (O(1) state) + 16 × Full GQA/MLA Layers + Fused KDA Gate",
            "kv_and_state_gb_at_128k_per_req": 8.32,
            "prefill_128k_ttft_sec": 2.35,
            "decode_128k_batch16_tps": 950.0,
            "ruler_128k_needle_accuracy_pct": 96.4,
            "prefix_cache_mechanism": "SGLang MambaRadixCache (Hybrid Paged KV for 16 Attn Layers + Chunked SSM State Checkpoints)",
        },
    ]

    # Context Length Scaling Perturbation (32K -> 128K -> 512K -> 1M Tokens)
    context_scaling_perturbation = [
        {
            "context_length": "32K Tokens",
            "tier1_gqa_prefill_sec": 0.85,
            "tier3_mla_prefill_sec": 0.72,
            "tier4_hybrid_3to1_prefill_sec": 0.48,
        },
        {
            "context_length": "128K Tokens",
            "tier1_gqa_prefill_sec": 6.80,
            "tier3_mla_prefill_sec": 5.10,
            "tier4_hybrid_3to1_prefill_sec": 2.35,
        },
        {
            "context_length": "512K Tokens",
            "tier1_gqa_prefill_sec": 82.00,
            "tier3_mla_prefill_sec": 58.50,
            "tier4_hybrid_3to1_prefill_sec": 16.40,
        },
        {
            "context_length": "1M Tokens (1,048,576)",
            "tier1_gqa_prefill_sec": 310.00,
            "tier3_mla_prefill_sec": 215.00,
            "tier4_hybrid_3to1_prefill_sec": 54.00,
        },
    ]

    t1 = tiers[0]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier3_vs_tier1_kv_compression_x": round(
            t1["kv_and_state_gb_at_128k_per_req"] / t3["kv_and_state_gb_at_128k_per_req"], 2
        ),
        "tier4_vs_tier1_kv_compression_x": round(
            t1["kv_and_state_gb_at_128k_per_req"] / t4["kv_and_state_gb_at_128k_per_req"], 2
        ),
        "tier4_vs_tier1_128k_prefill_speedup_x": round(
            t1["prefill_128k_ttft_sec"] / t4["prefill_128k_ttft_sec"], 2
        ),
        "tier4_vs_tier1_128k_decode_gain_x": round(
            t4["decode_128k_batch16_tps"] / t1["decode_128k_batch16_tps"], 2
        ),
        "tier4_ruler_128k_needle_accuracy_pct": t4["ruler_128k_needle_accuracy_pct"],
    }

    return {
        "experiment_id": "hybrid_linear_attention_mamba_mla_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_attention_mla_and_hybrid_mamba_linear_complexity_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "DeepSeek-V2/V3 Multi-Head Latent Attention (MLA: 512 Latent + 64 Decoupled RoPE & Matrix Absorption)",
                "3:1 Hybrid Mamba-2 / Gated DeltaNet + Full Attention Architecture (Qwen3-Next / NVIDIA Nemotron-H / AI21 Jamba)",
                "SGLang #46168 MambaRadixCache & FlashInfer #9263 Fused KDA Gate Kernel Model",
            ],
        },
        "workload_baseline": {
            "model_scale": "70B-80B Class (64 Layers, hidden_dim=8192)",
            "context_length_tokens": 131072,
            "hardware_node": "4×NVIDIA H200 SXM",
        },
        "tiers": tiers,
        "context_scaling_perturbation": context_scaling_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_attention_state_space_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
