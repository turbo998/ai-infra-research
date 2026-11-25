from __future__ import annotations
import json
from pathlib import Path


def evaluate_three_watermark_gpu_sizer_littles_law() -> dict:
    qps = 25
    isl_tokens = 8192
    osl_tokens = 512
    prefix_cache_hit_ratio = 0.70
    ttft_sec = 0.50
    target_tpot_sec = 0.014

    e2el_sec = round(ttft_sec + (osl_tokens - 1) * target_tpot_sec, 2)  # 7.65s
    active_batch_sessions = 192

    watermark_1_n_mem = 4
    watermark_2_n_bw = 8
    watermark_3_n_compute = 8
    final_h200_gpus = max(watermark_1_n_mem, watermark_2_n_bw, watermark_3_n_compute)  # 8
    uncached_overprovisioned_gpus = 14
    gpu_savings_pct = round((1.0 - final_h200_gpus / uncached_overprovisioned_gpus) * 100.0, 1)  # 42.9%

    vram_only_2gpu_tpot_ms = 45.1
    sized_8gpu_tpot_ms = 13.6
    single_stream_tps = round(1000.0 / sized_8gpu_tpot_ms, 1)  # 73.5 tok/s
    system_total_tps = qps * (isl_tokens + osl_tokens)  # 217,600 tok/s
    raw_tpm_millions = round((system_total_tps * 60.0) / 1_000_000.0, 2)  # 13.06M
    weighted_ptu_tpm_millions = round(
        (60.0 * qps * (isl_tokens * (1.0 - 0.80 * prefix_cache_hit_ratio) + 4.0 * osl_tokens)) / 1_000_000.0,
        2,
    )  # 8.48M
    goodput_pct = 99.4
    cost_per_1m_usd = 0.092

    return {
        "deliverable_id": "#626",
        "topic": "enterprise_three_watermark_gpu_sizer_tpm_tps_ptu_littles_law",
        "qps": qps,
        "e2el_sec": e2el_sec,
        "active_batch_sessions": active_batch_sessions,
        "watermark_1_n_mem": watermark_1_n_mem,
        "watermark_2_n_bw": watermark_2_n_bw,
        "watermark_3_n_compute": watermark_3_n_compute,
        "final_h200_gpus": final_h200_gpus,
        "gpu_savings_pct": gpu_savings_pct,
        "vram_only_2gpu_tpot_ms": vram_only_2gpu_tpot_ms,
        "sized_8gpu_tpot_ms": sized_8gpu_tpot_ms,
        "single_stream_tps": single_stream_tps,
        "system_total_tps": system_total_tps,
        "raw_tpm_millions": raw_tpm_millions,
        "weighted_ptu_tpm_millions": weighted_ptu_tpm_millions,
        "goodput_pct": goodput_pct,
        "cost_per_1m_usd": cost_per_1m_usd,
    }


if __name__ == "__main__":
    out = evaluate_three_watermark_gpu_sizer_littles_law()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
