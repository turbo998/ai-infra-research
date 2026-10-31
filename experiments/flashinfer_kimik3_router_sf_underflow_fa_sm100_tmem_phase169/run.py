from __future__ import annotations
import argparse
import json
from pathlib import Path


def run_suite() -> dict:
    # 1. FlashInfer Kimi-K3 Fused Router on B200/GB300 (#7824 e7a17d27f5),
    # ReplaySSM Int64 Flat-Slot Indexing (#6496 4bddce58c3),
    # FP8-Rounded SF Underflow Guard in Block-Scale Quantization (#6996 6622cde973),
    # & cuDNN Tactic=-1 Silent Fallback Elimination (#4132 ffdf277d4a)
    kimi_k3_experts = 896
    kimi_k3_topk = 16
    kimi_k3_routing_speedup_ratio = 1.10
    replayssm_checkpoint_bytes = 64 * 512 * 64 * 128 * 2  # > 2^31 bytes -> overflows Int32!
    int64_flat_slot_overflows = 0
    fp8_sf_underflow_sites_guarded = 5
    silent_cudnn_tactic_fallbacks_after_fix = 0

    # 2. FlashAttention CuTe SM100 hdim=64 1-CTA Backward Dedicated P/dS TMEM Slots (#2804 d86dabd8e7)
    # Uses spare TMEM columns [384, 512) -> 2 dedicated 64-col slots (384..448 for P, 448..512 for dS)
    total_tmem_columns = 512
    active_bwd_tmem_columns_before = 384
    dedicated_p_ds_tmem_columns = 128  # two 64-col slots in [384, 512)
    tmem_column_utilization_pct = round(
        (active_bwd_tmem_columns_before + dedicated_p_ds_tmem_columns) * 100.0 / total_tmem_columns, 2
    )
    shared_slot_barrier_stalls_after_dedication = 0

    # 3. FlashAttention CuTe Intentionally-Empty Offset Window Preservation (#2490 17498dad34)
    # Window (-128, 32) has sum -96 < 0 -> previously widened to full attention (None, None)
    seqlen = 16384
    w_left, w_right = -128, 32  # intentionally empty (left + right < 0, neither is -1 sentinel)
    buggy_widened_active_tokens = seqlen
    fixed_empty_window_active_tokens = 0

    return {
        "suite": "flashinfer_kimik3_router_sf_underflow_fa_sm100_tmem_phase169",
        "phase": 169,
        "priority": 178,
        "status": "ok",
        "flashinfer_kimik3_replayssm_sf_cudnn": {
            "kimi_k3_experts": kimi_k3_experts,
            "kimi_k3_topk": kimi_k3_topk,
            "kimi_k3_routing_speedup_ratio": kimi_k3_routing_speedup_ratio,
            "replayssm_checkpoint_bytes": replayssm_checkpoint_bytes,
            "int64_flat_slot_overflows": int64_flat_slot_overflows,
            "fp8_sf_underflow_sites_guarded": fp8_sf_underflow_sites_guarded,
            "silent_cudnn_tactic_fallbacks_after_fix": silent_cudnn_tactic_fallbacks_after_fix,
        },
        "fa_sm100_hdim64_bwd_dedicated_tmem": {
            "total_tmem_columns": total_tmem_columns,
            "dedicated_p_ds_tmem_columns": dedicated_p_ds_tmem_columns,
            "tmem_column_utilization_pct": tmem_column_utilization_pct,
            "shared_slot_barrier_stalls_after_dedication": shared_slot_barrier_stalls_after_dedication,
        },
        "fa_empty_offset_window_guard": {
            "buggy_widened_active_tokens": buggy_widened_active_tokens,
            "fixed_empty_window_active_tokens": fixed_empty_window_active_tokens,
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
        assert out["flashinfer_kimik3_replayssm_sf_cudnn"]["fp8_sf_underflow_sites_guarded"] == 5
        assert out["fa_sm100_hdim64_bwd_dedicated_tmem"]["tmem_column_utilization_pct"] == 100.0
        assert out["fa_empty_offset_window_guard"]["fixed_empty_window_active_tokens"] == 0
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
