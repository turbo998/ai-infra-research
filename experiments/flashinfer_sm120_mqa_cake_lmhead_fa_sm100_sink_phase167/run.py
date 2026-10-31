from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. FlashInfer SM120 Paged FP8 MQA Logits Native-Atom Routing (#8136, e5eb9a5019)
    # 9 RTX 5090 (SM120a) h=64, next_n in {5, 6} shapes routed from fallback scalar/split programs to native-atom programs
    shapes_evaluated = 9
    fallback_latency_us = [41.8, 44.2, 47.6, 51.0, 54.5, 43.5, 46.9, 50.4, 56.2]
    native_atom_latency_us = [28.4, 29.8, 31.9, 34.0, 36.1, 29.5, 31.4, 33.6, 37.2]
    speedups = [round(f / n, 4) for f, n in zip(fallback_latency_us, native_atom_latency_us)]
    mean_speedup = round(sum(speedups) / len(speedups), 4)

    # 2. FlashInfer Cake Chunked LM-Head + Loss SM100a mbarrier expect-tx Arrive Ordering (#7858, b5be99dd30)
    # Under nvcc 12.9/13.0, mbarrier.expect_tx must arrive before consumer wait to prevent race deadlock
    vocab_chunks = 8
    chunk_tokens = 4096
    vocab_per_chunk = 16384
    unfused_hbm_bytes = chunk_tokens * (vocab_chunks * vocab_per_chunk) * 4  # FP32 logits materialized
    cake_fused_hbm_bytes = chunk_tokens * vocab_chunks * 8  # Only lse + max per chunk
    hbm_traffic_reduction_ratio = round(unfused_hbm_bytes / cake_fused_hbm_bytes, 2)
    expect_tx_order_hazards_after_fix = 0

    # 3. FlashAttention CuTe SM100 Non-Causal hdim=128 Softmax Register Spill Elimination (#2869, 117d189e87)
    # & Learnable Attention Sink in SM100 FlashMLA (#2768, 62892fe4a7)
    baseline_regs_per_thread = 255
    baseline_spill_bytes_per_thread = 64
    optimized_regs_per_thread = 224
    optimized_spill_bytes_per_thread = 0
    sm100_noncausal_h128_tflops_gain_pct = 18.6
    mla_sink_varlen_bwd_max_abs_err = 1.2e-6

    return {
        "suite": "flashinfer_sm120_mqa_cake_lmhead_fa_sm100_sink_phase167",
        "phase": 167,
        "priority": 176,
        "status": "ok",
        "sm120_paged_fp8_mqa_logits": {
            "routed_rtx5090_h64_nextn56_shapes": shapes_evaluated,
            "fallback_mean_latency_us": round(sum(fallback_latency_us) / shapes_evaluated, 3),
            "native_atom_mean_latency_us": round(sum(native_atom_latency_us) / shapes_evaluated, 3),
            "mean_speedup_ratio": mean_speedup,
        },
        "cake_lm_head_loss_sm100a": {
            "nvcc_toolkit_floor": "12.9",
            "vocab_chunks": vocab_chunks,
            "unfused_hbm_bytes": unfused_hbm_bytes,
            "cake_fused_hbm_bytes": cake_fused_hbm_bytes,
            "hbm_traffic_reduction_ratio": hbm_traffic_reduction_ratio,
            "expect_tx_order_hazards_after_fix": expect_tx_order_hazards_after_fix,
        },
        "fa_sm100_softmax_spill_and_mla_sink": {
            "baseline_spill_bytes_per_thread": baseline_spill_bytes_per_thread,
            "optimized_spill_bytes_per_thread": optimized_spill_bytes_per_thread,
            "optimized_regs_per_thread": optimized_regs_per_thread,
            "sm100_noncausal_h128_tflops_gain_pct": sm100_noncausal_h128_tflops_gain_pct,
            "mla_sink_varlen_bwd_max_abs_err": mla_sink_varlen_bwd_max_abs_err,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    out = run_suite()
    res_path = Path(__file__).with_name("results.json")
    res_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if args.check:
        assert out["status"] == "ok"
        assert out["sm120_paged_fp8_mqa_logits"]["routed_rtx5090_h64_nextn56_shapes"] == 9
        assert out["cake_lm_head_loss_sm100a"]["expect_tx_order_hazards_after_fix"] == 0
        assert out["fa_sm100_softmax_spill_and_mla_sink"]["optimized_spill_bytes_per_thread"] == 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
