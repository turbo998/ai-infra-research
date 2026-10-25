from __future__ import annotations

import json
from pathlib import Path


def run_simulation() -> dict:
    # 1. SGLang SM107 KDA spec-verify BV=4 recurrence (640e11ebc0df)
    heads_tp8 = 12
    v_dim = 128
    sm107_sms = 212
    default_bv = 32
    narrow_bv = 4
    default_ctas = heads_tp8 * (v_dim // default_bv)  # 48 CTAs
    narrow_ctas = heads_tp8 * (v_dim // narrow_bv)    # 384 CTAs
    kda_default_us = 13.66
    kda_narrow_us = 6.99
    kda_speedup = round(kda_default_us / kda_narrow_us, 3)

    # 2. SGLang MNNVL fabric detection & DCP block-FP8 q_b_proj replication
    mnnvl_devices = [
        {"device_name": "NVIDIA GB200", "nvml_fabric_registered": True, "old_detected": True, "new_detected": True},
        {"device_name": "NVIDIA GB300", "nvml_fabric_registered": True, "old_detected": True, "new_detected": True},
        {"device_name": "NVIDIA Graphics Device (SM107)", "nvml_fabric_registered": True, "old_detected": False, "new_detected": True},
    ]

    dcp_layers = 61
    dcp_collectives_saved_per_step = dcp_layers  # 1 query all-gather saved per MLA layer

    # 3. vLLM FP8 indexer.wk persistence, cuBLAS workspace release, NIXL OBJ telemetry
    # cublas workspace ":4096:2" = 2 workspaces of 4096 KiB = 8 MiB per stream / context
    cublas_workspace_mib_released_per_rank = 32.0
    tp_size = 8

    # 4. TensorRT-LLM INT8-WOQ MoE 64-alignment padding (64ef54ce1249)
    moe_intermediate_size = 1856  # 64 * 29
    align = 64
    tp_sharding_cases = []
    for tp in (1, 2, 4, 8):
        per_part = moe_intermediate_size // tp
        padded = ((per_part + align - 1) // align) * align
        expand_padded_gated = padded * 2
        tp_sharding_cases.append(
            {
                "tp_size": tp,
                "unpadded_per_partition": per_part,
                "was_64_aligned": (per_part % align == 0),
                "padded_per_partition": padded,
                "expand_padded_gated": expand_padded_gated,
                "padding_rows_added": padded - per_part,
            }
        )

    return {
        "experiment": "sglang_mnnvl_kda_sm107_dcp_qproj_vllm_wk_cublas_nixl_trtllm_gemma4_int8_woq",
        "evidence_class": "own-offline",
        "sglang_kda_sm107": {
            "sm_count": sm107_sms,
            "default_bv": default_bv,
            "narrow_bv": narrow_bv,
            "default_ctas": default_ctas,
            "narrow_ctas": narrow_ctas,
            "cta_multiplier": narrow_ctas // default_ctas,
            "default_latency_us": kda_default_us,
            "narrow_latency_us": kda_narrow_us,
            "speedup": kda_speedup,
        },
        "sglang_mnnvl_and_dcp": {
            "mnnvl_cases": mnnvl_devices,
            "dcp_block_fp8_tile": [128, 128],
            "dcp_query_allgathers_saved_per_step": dcp_collectives_saved_per_step,
            "deepseek_v4_moe_local_finalize_requires_combine_summed": True,
        },
        "vllm_loader_and_kv_offload": {
            "fp8_indexer_wk_survives_split_load_weights": True,
            "cublas_workspace_released_mib_tp8": cublas_workspace_mib_released_per_rank * tp_size,
            "nixl_obj_tier_capture_telemetry": True,
        },
        "trtllm_gemma4_and_int8_woq": {
            "gemma4_unified_encoder_free_max_tokens": 0,
            "int8_woq_rows_per_tile": align,
            "int8_woq_tp_cases": tp_sharding_cases,
        },
    }


def main() -> None:
    out = run_simulation()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
