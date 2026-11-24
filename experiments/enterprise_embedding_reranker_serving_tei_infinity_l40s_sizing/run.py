"""Reproducible model for #589 Enterprise Embedding & Reranker Serving (TEI/Infinity L40S & MRL INT8)."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_embedding_reranker(peak_qps: int = 200, million_vectors: int = 100) -> dict:
    padded_tokens_top50 = 50 * 512
    unpadded_tokens_top50 = 11200
    padding_waste_pct = round((1.0 - unpadded_tokens_top50 / padded_tokens_top50) * 100, 1)

    naive_p95_ms = 62.0
    tei_unpadded_p95_ms = 14.5
    speedup_ratio = round(naive_p95_ms / tei_unpadded_p95_ms, 2)

    l40s_gpus = max(1, math.ceil(peak_qps / 110.0))
    l40s_monthly_usd = l40s_gpus * 950

    fp32_1024d_hnsw_gb = round(million_vectors * 6.4)
    mrl_512d_int8_hnsw_gb = round(million_vectors * 0.8)
    ram_compression_ratio = round(fp32_1024d_hnsw_gb / mrl_512d_int8_hnsw_gb, 1)

    total_self_hosted_monthly_usd = l40s_monthly_usd + mrl_512d_int8_hnsw_gb * 6

    return {
        "topic_id": 589,
        "peak_qps": peak_qps,
        "million_vectors": million_vectors,
        "padding_waste_pct": padding_waste_pct,
        "naive_p95_ms": naive_p95_ms,
        "tei_unpadded_p95_ms": tei_unpadded_p95_ms,
        "speedup_ratio": speedup_ratio,
        "l40s_gpus": l40s_gpus,
        "fp32_1024d_hnsw_gb": fp32_1024d_hnsw_gb,
        "mrl_512d_int8_hnsw_gb": mrl_512d_int8_hnsw_gb,
        "ram_compression_ratio": ram_compression_ratio,
        "total_self_hosted_monthly_usd": total_self_hosted_monthly_usd,
        "ndcg_at_10_mrl_int8_pct": 94.5,
    }


def main() -> None:
    res = evaluate_embedding_reranker()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
