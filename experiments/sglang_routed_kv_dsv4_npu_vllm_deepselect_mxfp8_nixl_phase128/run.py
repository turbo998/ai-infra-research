from __future__ import annotations

import json
from pathlib import Path


def run_experiment() -> dict:
    # 1. SGLang 5-commit declarative unified KV pool stack & routed cache views
    # (5433093d83b8, 298b56359664, f80cb3a2fd0f, 240862923e8d, 146378629d3e)
    sglang_routed_kv = {
        "unified_pool_entrypoint": "build_unified_pools",
        "lifetime_ranks": {"StateLifetime": 1, "FullLifetime": 2, "WindowLifetime": 3},
        "pd_transfer_owner_regions_per_subpool": 1,
        "legacy_per_layer_registration_eliminated": True,
        "routed_spec_modes_supported": ["EAGLE", "EAGLE3", "DSPARK", "NGRAM"],
        "shared_kv_storage_consumers": ["disagg_prefill", "disagg_decode", "beam_search_coordinator", "beam_fork"],
    }

    # 2. SGLang Ascend NPU DeepSeek-V4 decode host-sync elimination (f59cc7e45fd0)
    sglang_dsv4_npu_nosync = {
        "spec_verify_boolean_index_sync_ms_before": 3.9,
        "non_spec_alloc_boolean_index_sync_ms_before": 32.5,
        "decode_hook_d2h_sync_ms_before": 31.4,
        "host_sync_ops_per_decode_step_before": 3,
        "host_sync_ops_per_decode_step_after": 0,
        "host_sync_wait_ms_after": 0.0,
        "device_compaction_pipeline": "cumsum -> masked_fill -> scatter",
    }

    # 3. vLLM DSA prefill indexer BF16 DeepSelect ranking (c105af429241)
    vllm_dsa_deepselect_rows = [
        {"rows": 128, "base_us": 66.5, "new_us": 13.6, "speedup_x": 4.9},
        {"rows": 2048, "base_us": 414.4, "new_us": 110.1, "speedup_x": 3.8},
        {"rows": 4096, "base_us": 777.2, "new_us": 210.0, "speedup_x": 3.7},
        {"rows": 8192, "base_us": 1503.9, "new_us": 410.7, "speedup_x": 3.7},
    ]
    vllm_dsa_e2e_gb200 = {
        "step_ms_before": 429.8,
        "step_ms_after": 401.5,
        "step_reduction_pct": 6.6,
        "indexer_ms_before": 64.5,
        "indexer_ms_after": 30.9,
        "indexer_speedup_x": round(64.5 / 30.9, 3),
        "upcast_plus_topk_ms_before": 39.4,
        "upcast_plus_topk_ms_after": 5.7,
        "upcast_plus_topk_speedup_x": round(39.4 / 5.7, 3),
    }

    # 4. vLLM NixlConnector transfer_mode push/pull selection (066d9799347c)
    vllm_nixl_transfer_mode = {
        "default_mode": "pull",
        "resolved_pull_class": "NixlPullConnector",
        "resolved_push_class": "NixlPushConnector",
        "invalid_mode_raises_value_error": True,
    }

    # 5. vLLM ROCm MiniMax-M3.1 fused Gemma RMSNorm + MXFP8 activation quant (1cb31be70284)
    vllm_rocm_minimax_mxfp8 = {
        "norm_plus_quant_min_speedup_x": 1.41,
        "norm_plus_quant_max_speedup_x": 2.06,
        "norm_quant_gemm_min_speedup_x": 1.07,
        "norm_quant_gemm_max_speedup_x": 1.19,
        "layers_dropping_quant_launch": 59,
        "total_decoder_layers": 60,
        "decode_gpu_kernel_time_saved_pct": 1.9,
        "agentx_p90_c16_max_speedup_x": 1.045,
    }

    summary = {
        "phase": 128,
        "track": "Track B: Serving & KV Disaggregation",
        "sglang_routed_kv": sglang_routed_kv,
        "sglang_dsv4_npu_nosync": sglang_dsv4_npu_nosync,
        "vllm_dsa_deepselect_rows": vllm_dsa_deepselect_rows,
        "vllm_dsa_e2e_gb200": vllm_dsa_e2e_gb200,
        "vllm_nixl_transfer_mode": vllm_nixl_transfer_mode,
        "vllm_rocm_minimax_mxfp8": vllm_rocm_minimax_mxfp8,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    summary = run_experiment()
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[Phase 128 Track B] vLLM DeepSelect {summary['vllm_dsa_deepselect_rows'][0]['speedup_x']}x max "
        f"(upcast+topk {summary['vllm_dsa_e2e_gb200']['upcast_plus_topk_speedup_x']}x), "
        f"ROCm MXFP8 norm+quant {summary['vllm_rocm_minimax_mxfp8']['norm_plus_quant_max_speedup_x']}x"
    )


if __name__ == "__main__":
    main()
