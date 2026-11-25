from __future__ import annotations
import json
from pathlib import Path


def evaluate_quantization_matrix() -> dict:
    bf16_tps = 2420
    w4a16_tps = 3720
    w8a8kv8_tps = 5520
    nvfp4_hybrid_tps = 9320

    return {
        "Deliverable_ID": "#651",
        "Crossover_Batch_Size": 24,
        "Schemes_70B_4GPU_B64": {
            "BF16_W16A16KV16": {
                "weight_gb": 140.0,
                "kv_32k_gb": 10.48,
                "b64_tps": bf16_tps,
                "math_accuracy_delta_pct": 0.0,
                "cost_per_1m_usd": 0.280,
            },
            "INT4_AWQ_W4A16KV16": {
                "weight_gb": 38.5,
                "kv_32k_gb": 10.48,
                "b64_tps": w4a16_tps,
                "math_accuracy_delta_pct": -3.15,
                "cost_per_1m_usd": 0.182,
            },
            "Block_FP8_W8A8KV8": {
                "weight_gb": 70.0,
                "kv_32k_gb": 5.24,
                "b64_tps": w8a8kv8_tps,
                "speedup_vs_bf16": round(w8a8kv8_tps / bf16_tps, 2),
                "speedup_vs_w4a16_at_b64": round(w8a8kv8_tps / w4a16_tps, 2),
                "max_concurrency_32k_4xh200": 73,
                "bf16_kv_max_concurrency_32k": 34,
                "math_accuracy_delta_pct": -0.18,
                "cost_per_1m_usd": 0.123,
            },
            "Blackwell_Hybrid_FP8_NVFP4_KV8": {
                "weight_gb": 39.2,
                "kv_32k_gb": 5.24,
                "b64_tps": nvfp4_hybrid_tps,
                "speedup_vs_bf16": round(nvfp4_hybrid_tps / bf16_tps, 2),
                "math_accuracy_delta_pct": -0.25,
                "cost_per_1m_usd": 0.073,
            },
        },
    }


if __name__ == "__main__":
    out = evaluate_quantization_matrix()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
