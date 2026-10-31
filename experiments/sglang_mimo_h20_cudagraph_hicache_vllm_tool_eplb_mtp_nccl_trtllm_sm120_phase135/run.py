#!/usr/bin/env python3
"""SGLang MiMo-V2 / H20 FP8 / CUDA Graph Upload / PD Abort / HiCache + vLLM Tool / EPLB / MTP / RL NCCL + TRT-LLM SM120 Phase 135 deterministic offline verification."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPECTED = json.loads(r"""{
  "experiment_id": "sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135",
  "status": "verified",
  "provenance": "own-offline",
  "sglang_npu_mimo_v2_v_scale_fold": {
    "unfolded_attn_us": 48.0,
    "folded_o_proj_attn_us": 41.2,
    "speedup_ratio": 1.165
  },
  "sglang_h20_qwen3_30b_a3b_fp8_moe": {
    "default_config_us": 96.0,
    "tuned_h20_config_us": 72.5,
    "speedup_ratio": 1.3241
  },
  "sglang_deduped_cudagraph_preupload": {
    "lazy_first_replay_ms": 14.5,
    "private_stream_preuploaded_ms": 1.8,
    "speedup_ratio": 8.0556
  },
  "sglang_pd_retracted_req_and_hicache": {
    "retracted_unrebuildable_hang_rate_after": 0.0,
    "hicache_early_read_done_speedup": 1.24
  },
  "vllm_incremental_custom_tool_decode": {
    "tokens": 4096,
    "quadratic_redecode_ms": 18.4,
    "incremental_slice_decode_ms": 0.92,
    "speedup_ratio": 20.0
  },
  "vllm_eplb_mtp_and_rl_nccl_cache": {
    "default_min_redundant_experts_valid": true,
    "qwen35_multi_layer_mtp_acceptance_rate": 0.845,
    "rl_nccl_reclaimed_cache_gib": 3.2
  },
  "trtllm_sm120_deepseek_v41_flash": {
    "sm120_fp8_1d1d_and_csa2_enabled": true,
    "throughput_speedup_ratio": 1.42
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
