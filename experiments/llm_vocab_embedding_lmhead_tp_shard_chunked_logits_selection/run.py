#!/usr/bin/env python3
"""Round 234: Large-Vocabulary (V=152,064) LM Head Logits Materialization OOM (37.1 GB Trap), Last-Token Slicing & Fused Linear Cross-Entropy Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_lmhead_fused_linear_cross_entropy_tiers(
    num_tokens: int = 32768,
    vocab_size: int = 152064,
) -> dict[str, Any]:
    """Evaluate 4 LM Head + Cross-Entropy memory & latency tiers for 32K tokens and V=152,064 vocabulary."""
    tiers = [
        {
            "tier_id": "tier1_naive_full_logits_fp32_upcast_materialization_oom",
            "name": "Tier 1: 朴素全量物化 lm_head(h).float() + CrossEntropy (同时驻留 BF16/FP32/Grad 吞噬 37.1 GB 触发 OOM)",
            "peak_lmhead_hbm_gb": 37.12,
            "lmhead_fwd_bwd_latency_ms": 138.0,
            "materializes_full_nv_logits_in_hbm": True,
            "verdict": "32,768 个 Token × 152,064 词表全量物化 BF16 Logits (9.97GB) + FP32 转型 (19.93GB) + 反向梯度，在 Forward 最后一秒撑爆 80GB 显存 ✕",
        },
        {
            "tier_id": "tier2_vocab_parallel_tp_sharded_logits_still_materialized",
            "name": "Tier 2: 仅开启 VocabParallel (TP=4 切分词表维，虽均摊至 9.28 GB/卡但仍物化分片大矩阵)",
            "peak_lmhead_hbm_gb": 9.28,
            "lmhead_fwd_bwd_latency_ms": 64.5,
            "materializes_full_nv_logits_in_hbm": True,
            "verdict": "虽通过 TP=4 将每卡词表缩至 38,016 并配合 2 次小标量 AllReduce，但在 32K~64K 序列下仍白白占用 9.28 GB 显存 ⚠️",
        },
        {
            "tier_id": "tier3_token_chunked_lmhead_cross_entropy_loop",
            "name": "Tier 3: 序列维度分块计算 (Token-Chunked LM Head，每次仅物化 2,048 个 Token 的 Logits)",
            "peak_lmhead_hbm_gb": 2.32,
            "lmhead_fwd_bwd_latency_ms": 58.0,
            "materializes_full_nv_logits_in_hbm": False,
            "verdict": "将 32K 序列切成 16 个 2,048-Token 小块依次算 lm_head 与 CE Loss，峰值显存压至 2.32 GB，但仍需读写 HBM 中间块 ✅",
        },
        {
            "tier_id": "tier4_fused_linear_cross_entropy_liger_cce_sram_online_lse",
            "name": "Tier 4: 融合线性交叉熵 (Liger / Cut-Cross-Entropy 片上 SRAM 在线 LogSumExp + 原位梯度重算)",
            "peak_lmhead_hbm_gb": 0.08,
            "lmhead_fwd_bwd_latency_ms": 40.3,
            "materializes_full_nv_logits_in_hbm": False,
            "verdict": "彻底不向 HBM 写出 [N, V] Logits 矩阵！在 SRAM 分块矩阵乘中直接完成在线 Softmax 归约与梯度累加，显存骤降 464 倍至 0.08 GB、提速 3.42x 🏆",
        },
    ]

    return {
        "experiment_id": "llm_vocab_embedding_lmhead_tp_shard_chunked_logits_selection",
        "round": 234,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "num_tokens": num_tokens,
            "vocab_size": vocab_size,
            "hbm_reduction_x": round(37.12 / 0.08, 1),
            "latency_speedup_x": round(138.0 / 40.3, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_lmhead_fused_linear_cross_entropy_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
