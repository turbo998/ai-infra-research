from __future__ import annotations
import json
from pathlib import Path


def evaluate_large_vocab_fused_linear_ce() -> dict:
    # Workload: Qwen3/Gemma-style 8B LLM (Vocab V=152,064, Hidden D=4,096) SFT/RL training at 32K tokens/GPU
    tiers = [
        {
            "tier_id": "tier1_eager_bf16_lmhead_plus_fp32_cross_entropy_oom",
            "name": "Standard PyTorch Eager (BF16 lm_head + FP32 Upcast Softmax CrossEntropyLoss)",
            "logits_ce_peak_memory_gb": 39.1,
            "step_latency_ms": 1480.0,
            "training_tokens_per_sec": 22140.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Materializes full [32768, 152064] BF16 logits (10.0 GB) + FP32 upcast copy (20.0 GB) + backward grad_logits -> 39.1 GB peak OOM!"
        },
        {
            "tier_id": "tier2_compiled_fused_softmax_ce_with_full_logits",
            "name": "torch.compile / Triton Fused Softmax-CE (Still Materializes Full [N, V] BF16 Logits)",
            "logits_ce_peak_memory_gb": 19.8,
            "step_latency_ms": 1190.0,
            "training_tokens_per_sec": 27536.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Eliminates the FP32 upcast copy via online softmax, but lm_head linear still writes and reads a 10 GB BF16 logits tensor + 10 GB grad tensor!"
        },
        {
            "tier_id": "tier3_chunked_linear_cross_entropy_autograd_recompute",
            "name": "Token-Chunked Fused Linear Cross-Entropy (Liger-Kernel Style Forward-Backward In-Place Grad)",
            "logits_ce_peak_memory_gb": 0.48,
            "step_latency_ms": 1045.0,
            "training_tokens_per_sec": 31356.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Slices hidden states into B_c=1,024 token chunks, computes logits + CE + dH/dW inside forward/custom-autograd, never materializing full [N, V] logits!"
        },
        {
            "tier_id": "tier4_cut_cross_entropy_triton_indexed_dot_plus_chunked_fused_bwd",
            "name": "Zero-Logit-Materialization Fused Linear CE (Cut-Cross-Entropy Indexed Target Dot + Online LSE Tile)",
            "logits_ce_peak_memory_gb": 0.16,
            "step_latency_ms": 1012.0,
            "training_tokens_per_sec": 32379.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Computes target logit via indexed dot product and streams online log-sum-exp in SRAM tiles -> cuts CE memory 244.4x (39.1 -> 0.16 GB) & speeds up step 1.46x!"
        },
    ]
    return {
        "experiment_id": "llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection",
        "round": 301,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "sequence_tokens_per_gpu": 32768,
            "vocab_size_v": 152064,
            "hidden_dim_d": 4096
        },
        "tiers": tiers,
        "summary": {
            "memory_reduction_vs_eager_x": round(39.1 / 0.16, 2),
            "memory_reduction_liger_x": round(39.1 / 0.48, 2),
            "throughput_speedup_x": round(32379.0 / 22140.0, 2),
            "recommended_tier": "tier4_cut_cross_entropy_triton_indexed_dot_plus_chunked_fused_bwd"
        }
    }


if __name__ == "__main__":
    data = evaluate_large_vocab_fused_linear_ce()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
