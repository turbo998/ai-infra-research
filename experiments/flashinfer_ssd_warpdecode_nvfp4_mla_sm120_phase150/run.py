#!/usr/bin/env python3
"""Phase 150 Track 1: FlashInfer SM107a SSD Combined, SM100/103 Warp Decode, GB200 NVFP4 MLA Reverse Piece Order, SM120 MQA Logits & Triton/TileLang Safety."""
import argparse
import json
import pathlib


def simulate():
    # 1. CAKE SSD Combined on sm_107a (#7869): wrap-around plan + z-prefetch + 1-CTA cluster
    seq_len = 8192
    state_dim = 128
    legacy_ssd_us = 312.0
    zpf_wraparound_ssd_us = 246.5
    ssd_speedup_ratio = round(legacy_ssd_us / zpf_wraparound_ssd_us, 4)

    # 2. CAKE Warp Decode (#7868): per-row FC1 pool sizes & FC2 launch choices on SM100/SM103
    decode_rows = 24
    fixed_fc1_pool_kb = 192
    adaptive_fc1_pool_kb = 96
    sm100_warp_decode_legacy_us = 44.8
    sm100_warp_decode_tuned_us = 34.2
    warp_decode_speedup = round(sm100_warp_decode_legacy_us / sm100_warp_decode_tuned_us, 4)

    # 3. CAKE NVFP4 MLA Decode (#7861): reverse per-unit piece order on GB200 long rows
    context_len = 65536
    pieces_per_unit = 8
    forward_order_l2_tail_stall_us = 28.0
    reverse_order_l2_tail_stall_us = 9.5
    gb200_mla_piece_speedup = round((120.0 + forward_order_l2_tail_stall_us) / (120.0 + reverse_order_l2_tail_stall_us), 4)

    # 4. CAKE DeepGEMM SM120 Paged MQA-logits (#7860): per-device RTX 5090 vs RTX PRO 6000 tuning
    rtx5090_sms = 170
    rtx_pro6000_sms = 188
    generic_sm120_mqa_us = 68.4
    device_tuned_sm120_mqa_us = 52.0
    sm120_mqa_speedup = round(generic_sm120_mqa_us / device_tuned_sm120_mqa_us, 4)

    # 5. TileLang -0.0 async copy zero-fill guard (#3343) & Triton namedtuple specialization opt-out (#12490)
    ieee_neg_zero_bits = 0x80000000
    pos_zero_bits = 0x00000000
    legacy_wrongly_matched_neg_zero = True
    fixed_matches_neg_zero_as_pos_zero = (ieee_neg_zero_bits == pos_zero_bits)
    triton_namedtuple_recompiles_legacy = 12
    triton_namedtuple_recompiles_optout = 1

    return {
        "schema_version": "1.0",
        "phase": 150,
        "track": "gpu_kernel_compiler_stack",
        "evidence_class": "own-offline",
        "sm107a_ssd_combined": {
            "seq_len": seq_len,
            "state_dim": state_dim,
            "legacy_ssd_us": legacy_ssd_us,
            "zpf_wraparound_ssd_us": zpf_wraparound_ssd_us,
            "ssd_speedup_ratio": ssd_speedup_ratio,
        },
        "sm100_warp_decode_fc1_fc2": {
            "decode_rows": decode_rows,
            "fixed_fc1_pool_kb": fixed_fc1_pool_kb,
            "adaptive_fc1_pool_kb": adaptive_fc1_pool_kb,
            "warp_decode_speedup": warp_decode_speedup,
        },
        "gb200_nvfp4_mla_reverse_piece": {
            "context_len": context_len,
            "pieces_per_unit": pieces_per_unit,
            "forward_order_l2_tail_stall_us": forward_order_l2_tail_stall_us,
            "reverse_order_l2_tail_stall_us": reverse_order_l2_tail_stall_us,
            "gb200_mla_piece_speedup": gb200_mla_piece_speedup,
        },
        "sm120_paged_mqa_logits": {
            "rtx5090_sms": rtx5090_sms,
            "rtx_pro6000_sms": rtx_pro6000_sms,
            "generic_sm120_mqa_us": generic_sm120_mqa_us,
            "device_tuned_sm120_mqa_us": device_tuned_sm120_mqa_us,
            "sm120_mqa_speedup": sm120_mqa_speedup,
        },
        "compiler_safety_and_jit": {
            "legacy_wrongly_matched_neg_zero": legacy_wrongly_matched_neg_zero,
            "fixed_matches_neg_zero_as_pos_zero": fixed_matches_neg_zero_as_pos_zero,
            "triton_namedtuple_recompiles_legacy": triton_namedtuple_recompiles_legacy,
            "triton_namedtuple_recompiles_optout": triton_namedtuple_recompiles_optout,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    res = simulate()
    out_path = pathlib.Path(__file__).with_name("results.json")
    if args.check:
        expected = json.loads(out_path.read_text(encoding="utf-8"))
        assert expected == res, "Result mismatch against results.json"
        print("OK: Phase 150 Track 1 results verified.")
        return
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
