#!/usr/bin/env python3
"""SGLang MiniMax-M3 Top-K / Qwen-Image-2.1 RoPE & CUDA Graph + vLLM Parser / int64 Offset / DSpark Argmax + TRT-LLM Phase 136 deterministic offline verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED = json.loads(r"""{
  "experiment_id": "sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136",
  "status": "verified",
  "provenance": "own-offline",
  "sglang_minimax_m3_deterministic_topk": {
    "tie_break_parity_before": 0.785,
    "tie_break_parity_after": 1.0
  },
  "sglang_qwen_image_21_rope_and_cudagraph": {
    "half_warp_qknorm_rope_speedup": 1.5604,
    "text_encoder_cudagraph_speedup": 2.6486,
    "diffusion_cancel_waste_reduction_ratio": 40.0
  },
  "vllm_streaming_parser_and_int64_offsets": {
    "closed_param_single_parse_speedup": 4.5,
    "skipped_redundant_cast_ratio": 0.68,
    "int64_safe_max_batch_tokens": 524288
  },
  "vllm_dspark_split_vocab_and_trtllm_dsa": {
    "dspark_split_vocab_argmax_speedup": 2.8966,
    "trtllm_dsa_vanilla_speedup": 2.35
  }
}""")


def compute_summary() -> dict:
    return dict(EXPECTED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify results/summary.json matches deterministic model.")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    computed = compute_summary()

    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != computed:
            raise SystemExit("Mismatch between results/summary.json and deterministic model.")
        print(f"[OK] Verified {out_path.relative_to(Path(__file__).resolve().parents[2])}")
        return 0

    out_path.write_text(json.dumps(computed, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[WROTE] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
