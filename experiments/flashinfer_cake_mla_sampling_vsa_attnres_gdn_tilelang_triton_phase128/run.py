from __future__ import annotations

import json
from pathlib import Path


def run_experiment() -> dict:
    # 1. FlashInfer Cake NVFP4 MLA decode setup-time first-item load (949a9f2d53ca)
    nvfp4_mla_cases = [
        {"gpu": "B200", "shape": "2req_256_300k_64h_6q_2split", "speedup_x": 1.026},
        {"gpu": "B200", "shape": "3req_1000_8192_5000k_8h_6q", "speedup_x": 1.017},
        {"gpu": "B200", "shape": "4req_64_65_4096_777k_64h_1q", "speedup_x": 1.012},
        {"gpu": "GB200", "shape": "2req_256_300k_64h_6q_2split", "speedup_x": 1.029},
        {"gpu": "GB200", "shape": "3req_1000_8192_5000k_8h_6q", "speedup_x": 1.007},
        {"gpu": "GB200", "shape": "4req_64_65_4096_777k_64h_1q", "speedup_x": 1.015},
        {"gpu": "GB300", "shape": "3req_1000_8192_5000k_8h_6q", "speedup_x": 1.010},
    ]

    # 2. FlashInfer Cake sampling round 17 whole-CTA tails & compact-list streams (3812cddafdb7)
    sampling_round17 = {
        "stage1_builds_before": 108,
        "stage1_builds_after": 135,
        "total_builds_before": 123,
        "total_builds_after": 150,
        "compact_list_entries_per_cta": 8192,
        "sm12x_smem_opt_in_bytes": 101376,
        "rtx_pro_6000_max_speedup_x": 1.413,
        "rtx_pro_6000_min_routed_speedup_x": 1.307,
        "b200_max_speedup_x": 1.125,
        "h100_max_speedup_x": 1.121,
        "gb300_max_speedup_x": 1.119,
        "r200_max_speedup_x": 1.091,
    }

    # 3. FlashInfer Cake VSA SM90 position-selected Sage-FP8 GQA static program (279f9a082fb2)
    vsa_sm90_pin = {
        "pin_min_cta_loop_iters": 14,
        "routed_static_fp8_gqa_rows": 20,
        "total_static_fp8_gqa_rows": 48,
        "main_loop_insts_before": 675,
        "main_loop_insts_after": 633,
        "main_loop_inst_delta": -42,
        "routed_geomean_speedup_proc_a_x": 1.0287,
        "routed_geomean_speedup_proc_b_x": 1.0284,
        "max_routed_speedup_x": 1.0361,
        "registers_used": 168,
        "register_spills": 0,
    }

    # 4. FlashInfer Cake Kimi-K3 AttnRes parked-accumulator pair programs (d0ac3f0b1649)
    kimi_k3_attn_res_park = {
        "consumer_regs_before": 232,
        "consumer_regs_after": 200,
        "producer_regs_before": 40,
        "producer_regs_after": 104,
        "tmem_cols_per_stream_before": 256,
        "tmem_cols_per_stream_after": 512,
        "register_spills_after": 0,
        "b200_max_speedup_pct": 9.9,
        "gb300_max_speedup_pct": 9.9,
        "b300_max_speedup_pct": 7.9,
    }

    # 5. FlashInfer Cake Kimi-K3 FP8 projection GEMM twins & SM103a 352-wide q_proj (21e02addf9f6)
    kimi_k3_fp8_proj = {
        "gb300_tp1_kv_b_m256_speedup_x": 1.042,
        "gb300_tp8_f_b_m4096_speedup_x": 1.038,
        "gb300_tp1_q_proj_m512_speedup_x": 1.027,
        "gb300_tp1_f_b_m4096_speedup_x": 1.014,
        "b200_tp1_f_b_m4096_speedup_x": 1.014,
    }

    # 6. FlashInfer Cake GDN training prefetch adjoint walk + no-sync prepare + tape reuse (b6089e61ae3f, 025f5acf562f)
    gdn_train_kcp = {
        "dynamic_smem_bytes_before": 82944,
        "dynamic_smem_bytes_after": 83968,
        "gb300_cp4_zigzag_step_speedup_x": 1.04,
        "gb300_cp8_contig_step_speedup_x": 1.03,
        "prepare_sync_calls_before": 24,
        "prepare_sync_calls_after": 0,
        "prepare_idle_ms_before": 52.0,
        "prepare_queued_ms_before": 147.0,
        "prepare_ms_after": 5.8,
        "prepare_queued_speedup_x": round(147.0 / 5.8, 2),
        "bwd_peak_mib_kept_tapes": 9251,
        "bwd_peak_mib_auto_tapes": 6927,
        "bwd_peak_saved_mib": 9251 - 6927,
        "workspace_mib_reported_before": 3885,
        "workspace_mib_bound_after": 2894,
    }

    # 7. FlashInfer MoE EP Prims-TS split layer & DeepSeek FP8 sender-side dispatch + Cake W4A16 CUDA 13.0 gate (9eb51e64a872, a2b15151fe59)
    # Sending 7168-dim token row: BF16 = 14336 B vs DeepSeek FP8 (7168 B values + 56 * 4 B FP32 scales = 7392 B)
    bf16_dispatch_bytes_per_token = 7168 * 2
    ds_fp8_dispatch_bytes_per_token = 7168 * 1 + (7168 // 128) * 4
    moe_ep_and_w4a16 = {
        "bf16_dispatch_bytes_per_token": bf16_dispatch_bytes_per_token,
        "deepseek_fp8_dispatch_bytes_per_token": ds_fp8_dispatch_bytes_per_token,
        "dispatch_payload_reduction_pct": round(
            (1.0 - ds_fp8_dispatch_bytes_per_token / bf16_dispatch_bytes_per_token) * 100.0, 2
        ),
        "cake_w4a16_min_cuda_major": 13,
        "cake_w4a16_min_cuda_minor": 0,
    }

    # 8. TileLang T.Parallel coalesced_width IntImm coercion & Triton AMD GFX1250-strict block32 scaled_upcast (714cbf150d90, 8afd743d38de)
    compiler_contracts = {
        "tilelang_raw_int_coalesced_width_aborts_after": 0,
        "triton_gfx1250_strict_block32_lane_stride": 16,
        "triton_gfx1250_strict_hw_scaled_upcast_enabled": True,
    }

    summary = {
        "phase": 128,
        "track": "Track A: GPU/HBM/Kernel Stack",
        "nvfp4_mla_cases": nvfp4_mla_cases,
        "nvfp4_mla_max_speedup_x": max(c["speedup_x"] for c in nvfp4_mla_cases),
        "sampling_round17": sampling_round17,
        "vsa_sm90_pin": vsa_sm90_pin,
        "kimi_k3_attn_res_park": kimi_k3_attn_res_park,
        "kimi_k3_fp8_proj": kimi_k3_fp8_proj,
        "gdn_train_kcp": gdn_train_kcp,
        "moe_ep_and_w4a16": moe_ep_and_w4a16,
        "compiler_contracts": compiler_contracts,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    summary = run_experiment()
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[Phase 128 Track A] sampling RTX PRO 6000 {summary['sampling_round17']['rtx_pro_6000_max_speedup_x']}x, "
        f"GDN prepare {summary['gdn_train_kcp']['prepare_queued_speedup_x']}x ({summary['gdn_train_kcp']['bwd_peak_saved_mib']} MiB saved)"
    )


if __name__ == "__main__":
    main()
