"""Phase 124 Track B: SGLang ComfyUI SGLDiffusion / StaticKVStorage / Routed Draft KV Views / Disagg Transfer Pins, vLLM P2P Retained Source Pins / SWA Lazy KV Offload / DSA Decode Indexer Memory / NixlPushConnector / ROCm MXFP8 Fusion & TensorRT-LLM 64-Bit AttnRes Stride / Eager GDN Pre-Alloc / GLM-5.3-Flash KPool Verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. SGLang ComfyUI SGLDiffusion fail-fast kwargs & CUDA IPC worker status (`4517a5ff1b`, `b1c6945659`)
    sglang_comfyui_diffusion = {
        "fail_fast_commit": "4517a5ff1b1615e86db3c233591b64234385c4f8",
        "worker_status_commit": "b1c694565934818e878df27b89c417ebcaabca3f",
        "rejected_silent_drop_kwargs": [
            "attention_mask",
            "additional_t_cond",
            "siglip_feats",
            "ref_contexts",
            "control_Scale_list",
            "control_context_scale",
        ],
        "flux_controlnet_h100_1024x1024_25steps": {
            "native_comfyui_s": 13.1,
            "sgl_diffusion_cuda_ipc_s": 4.7,
            "speedup_x": round(13.1 / 4.7, 4),
            "required_comfyui_flag": "--disable-cuda-malloc",
        },
        "wan22_ti2v_5b_h100_1280x704x121": {
            "native_comfyui_s": 132.0,
            "sgl_diffusion_s": 39.4,
            "speedup_x": round(132.0 / 39.4, 4),
        },
    }

    # 2. SGLang StaticKVStorage, IdSpaceKind, routed KV views, disagg transfer pins & ROCm window guard
    # (`146378629d`, `53592ea5ff`, `ca8184e13a`, `240862923e`, `eb79ecca11`, `14fea75124`)
    sglang_kv_storage_and_disagg = {
        "static_kv_storage_commit": "146378629db66987615693c584e90fb808e99e57",
        "id_space_kind_commit": "53592ea5ff0451ad54c2e13fb3b859a8d8d677fa",
        "drop_kv_pool_fallback_commit": "ca8184e13ae406eb06410e77e712a985bd702fc8",
        "routed_kv_views_commit": "240862923ef93c7e0c2921e77c976ebda683e38a",
        "disagg_transfer_pins_commit": "eb79ecca11da6c278eb5df8430b5ba23b444da13",
        "rocm_fused_kv_guard_commit": "14fea75124da9d72b3e302a833696f73e2524c28",
        "unified_storage_protocol": "StaticKVStorage (kv_storage_of)",
        "allocation_id_spaces": ["TOKEN_SLOT", "PAGE_BLOCK", "MHA_WINDOW_SLOT", "LINEAR_STATE_SLOT"],
        "disagg_pin_counter": "disagg_transfers_in_flight",
        "compaction_blocked_when_in_flight_gt_0": True,
        "rocm_fused_set_kv_buffer_allowed_when_window_translated": False,
    }

    # 3. vLLM P2P KV Offload Retained Source Pins (`a1cfd2c1b5`)
    vllm_p2p_retire = {
        "commit": "a1cfd2c1b5c1c970744e5e0a8fd7bd3ba7cf459d",
        "cancel_mode_on_teardown": "wait",
        "retained_list_attr": "_retained",
        "retained_hold_cap_s": 30.0,
        "prevents_dma_source_uaf": True,
        "prevents_reconnect_peer_id_collision": True,
        "lifecycle_states": ["active_inflight", "cancel_wait_requested", "retained_pinned", "retired_unpinned"],
    }

    # 4. vLLM SimpleKVOffload SWA Lazy Target Blocks & DSA Decode Indexer Logits Memory (`77f58e5304`, `317081fb6d`)
    # For DSA decode indexer logits: max_num_seqs = 256, max_model_len = 65536, fp32 (4 bytes)
    # 256 * 65536 * 4 = 67,108,864 bytes = 64 MiB per layer invocation or buffer
    dsa_max_seqs = 256
    dsa_max_seq_len = 65536
    dsa_fp32_logits_bytes = dsa_max_seqs * dsa_max_seq_len * 4
    dsa_bf16_logits_bytes = dsa_max_seqs * dsa_max_seq_len * 2

    vllm_swa_and_dsa = {
        "swa_offload_commit": "77f58e5304799bb9f16ec6353263d93192b8502c",
        "dsa_indexer_commit": "317081fb6dc8b125a3134711a7750707bfd55371",
        "swa_lazy_target_groups_before": "prefix_cacheable_groups_only",
        "swa_lazy_target_groups_after": "all_kv_cache_groups_including_SlidingWindowMLASpec",
        "hybrid_48_full_16_swa_lazy_blocks_ratio": round((48 + 16) / 48, 4),
        "dsa_fp32_decode_logits_mib_at_256x64k": round(dsa_fp32_logits_bytes / (1024 * 1024), 2),
        "dsa_bf16_decode_logits_mib_at_256x64k": round(dsa_bf16_logits_bytes / (1024 * 1024), 2),
        "dsa_fp32_condition": "topk_backend != 'deep_select' or dcp_world_size != 1",
    }

    # 5. vLLM NixlPushConnector, ROCm MiniMax-M3.1 MXFP8 Norm Fusion & SpecDecode Adaptive Padding (`066d979934`, `1cb31be702`, `5118d990fb`)
    vllm_nixl_rocm_spec = {
        "nixl_push_commit": "066d979934c6eb52cf375bf453f0655880b857c5",
        "rocm_mxfp8_commit": "1cb31be70276eb50bb264c93aa11583881893d72",
        "spec_decode_padding_commit": "5118d990fb67306808cc169b900ef0fc83b0f56d",
        "nixl_transfer_modes": {"pull": "NixlConnector", "push": "NixlPushConnector"},
        "rocm_fused_pass": "MinimaxM3RMSNormQuantFusionPass -> RocmDotScaledMxfp8LinearKernel",
        "mxfp8_block_size": 32,
        "spec_decode_adaptive_draft_max_query_len": None,
    }

    # 6. TensorRT-LLM Kimi-K3 64-bit AttnRes Stride, Eager GDN Pre-Allocation & GLM-5.3-Flash KPool (`e2b1544160`, `99466ea9ad`, `2fe91cdfc9`)
    # In e2b1544160: block_stride_r = num_tokens * hidden_dim; max source index = N - 2.
    # For H = 7168, N = 9 -> max source = 7.
    # 32-bit signed INT_MAX = 2,147,483,647.
    # Threshold num_tokens where 7 * T * 7168 > 2,147,483,647:
    # 2147483647 // (7 * 7168) + 1 = 42799 + 1 = 42800.
    int32_max = (1 << 31) - 1
    kimi_h = 7168
    kimi_n = 9
    kimi_max_source = kimi_n - 2
    overflow_token_threshold = (int32_max // (kimi_max_source * kimi_h)) + 1
    offset_at_42800 = kimi_max_source * 42800 * kimi_h
    int32_wrapped_at_42800 = offset_at_42800 - (1 << 32)

    trtllm_verification = {
        "kimi_attn_res_commit": "e2b1544160681e6bb379a3ab05da7c53852d36b2",
        "eager_gdn_commit": "99466ea9ad4d11936fc627792dc214cf5a106077",
        "glm_kpool_commit": "2fe91cdfc9b232c7a8773fd3b68a6ab1f49b8320",
        "kimi_hidden_dim": kimi_h,
        "kimi_max_n": kimi_n,
        "int32_max": int32_max,
        "int32_overflow_token_threshold": overflow_token_threshold,
        "offset_elements_at_42800": offset_at_42800,
        "int32_wrapped_negative_offset_at_42800": int32_wrapped_at_42800,
        "promoted_stride_type": "long long (int64_t)",
        "eager_gdn_torch_cat_eliminated": True,
        "glm53_flash_4xb300_ep1_mtp3_concurrency1_gain_pct": 6.0,
    }

    return {
        "phase": 124,
        "track": "B",
        "question_id": "q-sglang-comfyui-kv-storage-routed-views-vllm-p2p-retain-swa-dsa-nixl-rocm-trtllm-phase124",
        "sglang_comfyui_diffusion": sglang_comfyui_diffusion,
        "sglang_kv_storage_and_disagg": sglang_kv_storage_and_disagg,
        "vllm_p2p_retire": vllm_p2p_retire,
        "vllm_swa_and_dsa": vllm_swa_and_dsa,
        "vllm_nixl_rocm_spec": vllm_nixl_rocm_spec,
        "trtllm_verification": trtllm_verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic summary.json")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).with_name("summary.json")

    if args.check:
        if not out_path.exists():
            raise SystemExit("summary.json missing; run without --check first")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json mismatch")
        print("Phase 124 Track B check passed.")
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
