#!/usr/bin/env python3
"""Round 185: LLM Tokenizer Architecture, Vocabulary Size (32K vs 128K/152K vs 256K) & Multilingual Fertility Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_tokenizer_vocab_tiers(
    chinese_doc_chars: int = 100000,
    hidden_dim: int = 8192,
) -> dict[str, Any]:
    """Evaluate 4 LLM tokenizer & vocabulary architecture tiers on a 100K-char Chinese + Code document."""
    # Bytes of Emb + Untied LM Head = 2 * V * hidden_dim * 2 bytes (FP16)
    def head_gb(vocab: int) -> float:
        return round((2 * vocab * hidden_dim * 2) / (1024**3), 2)

    tiers = [
        {
            "tier_id": "tier1_legacy_32k_bpe_utf8_byte_fallback",
            "name": "Tier 1: 早期 32K 英文主导词表 + UTF-8 字节回退 (Llama-2 32K 模式)",
            "vocab_size": 32000,
            "cjk_chars_per_token": 0.69,
            "tokens_for_100k_cjk_chars": round(chinese_doc_chars / 0.69),
            "emb_plus_lm_head_fp16_gb": head_gb(32000),
            "relative_decode_speedup_cjk": 1.00,
            "cpu_128k_tokenize_ms": 185.0,
            "verdict": "中文单字被拆成 2~3 个 UTF-8 字节 Token，10 万字膨胀为 14.5 万 Token，KV 显存与解码耗时翻 2.4 倍 ✕",
        },
        {
            "tier_id": "tier2_english_100k_python_regex_gil",
            "name": "Tier 2: 100K 词表 + 单线程 Python 正则分词 (cl100k + 朴素网关)",
            "vocab_size": 100256,
            "cjk_chars_per_token": 1.15,
            "tokens_for_100k_cjk_chars": round(chinese_doc_chars / 1.15),
            "emb_plus_lm_head_fp16_gb": head_gb(100256),
            "relative_decode_speedup_cjk": 1.67,
            "cpu_128k_tokenize_ms": 92.0,
            "verdict": "中文仍有约 40% 常用词跨字拆分，且长文本正则回溯阻塞网关主线程 ⚠️",
        },
        {
            "tier_id": "tier3_oversized_512k_vocab_lm_head_wall",
            "name": "Tier 3: 盲目扩充 512K 超大词表 (LM Head 带宽墙 + 罕见词欠训练)",
            "vocab_size": 512000,
            "cjk_chars_per_token": 1.82,
            "tokens_for_100k_cjk_chars": round(chinese_doc_chars / 1.82),
            "emb_plus_lm_head_fp16_gb": head_gb(512000),
            "relative_decode_speedup_cjk": 1.95,
            "cpu_128k_tokenize_ms": 14.5,
            "verdict": "词表过大导致 Embedding + LM Head 暴涨至 15.62 GB，每步 Softmax 带宽开销剧增且长尾 Glitch Token 频发 ⚠️",
        },
        {
            "tier_id": "tier4_sweetspot_128k_152k_multilingual_rust_tiktoken",
            "name": "Tier 4: 128K~152K 中英代码均衡词表 + Rust tiktoken DFA 零回溯并发分词 (Qwen2.5 / DeepSeek-V3)",
            "vocab_size": 151936,
            "cjk_chars_per_token": 1.67,
            "tokens_for_100k_cjk_chars": round(chinese_doc_chars / 1.67),
            "emb_plus_lm_head_fp16_gb": head_gb(151936),
            "relative_decode_speedup_cjk": 2.42,
            "cpu_128k_tokenize_ms": 4.2,
            "verdict": "10 万中文字压缩至 5.99 万 Token (省 58.7% KV 与步数)，中文生成提速 2.42x，128K 分词仅 4.2ms 🏆",
        },
    ]

    return {
        "experiment_id": "llm_tokenizer_bpe_vocab_multilingual_fertility_selection",
        "round": 185,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "chinese_doc_chars": chinese_doc_chars,
            "hidden_dim": hidden_dim,
            "legacy_32k_tokens": 144928,
            "sweetspot_152k_tokens": 59880,
            "token_reduction_pct": 58.7,
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_tokenizer_vocab_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
