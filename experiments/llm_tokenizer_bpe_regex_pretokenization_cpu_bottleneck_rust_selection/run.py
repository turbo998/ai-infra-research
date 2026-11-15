from __future__ import annotations
import json
from pathlib import Path


def evaluate_tokenizer_detokenizer_overhead() -> dict:
    # Workload: 128K-token (512 KB JSON/code) prompt tokenization + BS=256 streaming detokenization (4,096 output tokens)
    tiers = [
        {
            "tier_id": "tier1_single_thread_regex_bpe_plus_on2_full_history_decode",
            "name": "Single-Thread Regex Pre-Tokenization + O(N^2) Full-History decode(all_ids) Per Step",
            "prompt_128k_tokenize_ms": 640.0,
            "bs256_step_detokenize_ms": 4.80,
            "utf8_cjk_fidelity_pct": 100.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "512KB unspaced JSON triggers regex backtracking (640ms TTFT penalty), and re-decoding all 4,096 tokens every step at BS=256 stalls CPU 4.80ms/step!"
        },
        {
            "tier_id": "tier2_naive_single_token_decode_utf8_corruption",
            "name": "Naive Single-Token decode([token_id]) Without UTF-8 Byte-Fallback Buffer",
            "prompt_128k_tokenize_ms": 185.0,
            "bs256_step_detokenize_ms": 0.12,
            "utf8_cjk_fidelity_pct": 81.5,
            "action": "Send to a human",
            "status": "fail",
            "note": "Fast O(1) per step (0.12ms), but multi-byte UTF-8 CJK/Emoji split across 2 byte-level BPE tokens emit replacement chars (81.5% fidelity)!"
        },
        {
            "tier_id": "tier3_hf_fast_rust_tokenizer_in_main_scheduler_loop",
            "name": "HF Rust Fast Tokenizer in Main Engine Process + Sliding-Window Detokenizer",
            "prompt_128k_tokenize_ms": 42.0,
            "bs256_step_detokenize_ms": 0.65,
            "utf8_cjk_fidelity_pct": 100.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Preserves 100% UTF-8 fidelity and O(1) detokenization, but running 42ms Rust BPE inside the main engine process still blocks GPU scheduling."
        },
        {
            "tier_id": "tier4_disaggregated_zmq_rust_chunk_pool_plus_o1_utf8_window_detok",
            "name": "Disaggregated ZMQ Rust Chunked Tokenizer Pool + O(1) 6-Token UTF-8 Window Detokenizer",
            "prompt_128k_tokenize_ms": 11.5,
            "bs256_step_detokenize_ms": 0.08,
            "utf8_cjk_fidelity_pct": 100.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Splits 512KB prompts into 32KB newline/punctuation chunks across Rust worker pool (11.5ms, 55.7x faster) + O(1) 6-token UTF-8 state machine (0.08ms/step)!"
        },
    ]
    return {
        "experiment_id": "llm_tokenizer_bpe_regex_pretokenization_cpu_bottleneck_rust_selection",
        "round": 291,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "prompt_tokens": 131072,
            "prompt_bytes_kb": 512,
            "decode_batch_size": 256,
            "output_seq_len": 4096
        },
        "tiers": tiers,
        "summary": {
            "tokenize_speedup_x": round(640.0 / 11.5, 1),
            "detokenize_speedup_x": round(4.80 / 0.08, 1),
            "recommended_detok_ms": 0.08,
            "recommended_tier": "tier4_disaggregated_zmq_rust_chunk_pool_plus_o1_utf8_window_detok"
        }
    }


if __name__ == "__main__":
    data = evaluate_tokenizer_detokenizer_overhead()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
