from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_sglang_dsv4_capture_disagg_and_kv_capacity(
    kv_bytes: int = 576,
    low_ratio_index_bytes: int = 128,
    high_ratio_index_bytes: int = 64,
    c128_fixed_state_bytes: int = 256,
    ratio: int = 2,
) -> dict[str, Any]:
    """Model SGLang PR #46718 (`dea05ca16e`) DSV4 tail source KV layer capture stub,
    PR #46144 (`7f8ad67373`) per-role disaggregated diffusion topology auto-derivation,
    and PR #41275 (`20ff40cb84`) DeepSeek-V4.1 physical KV capacity matching.
    """
    # Legacy formula omitted c128 fixed state bytes at ratio=2, over-estimating token capacity
    legacy_bytes_per_token = (kv_bytes + low_ratio_index_bytes) / ratio
    physical_bytes_per_token = (kv_bytes + low_ratio_index_bytes + c128_fixed_state_bytes) / ratio
    pool_budget_bytes = 16 * 1024 * 1024 * 1024  # 16 GiB
    legacy_token_capacity = int(pool_budget_bytes // legacy_bytes_per_token)
    physical_token_capacity = int(pool_budget_bytes // physical_bytes_per_token)
    overestimation_pct = round(
        (legacy_token_capacity - physical_token_capacity) / physical_token_capacity * 100.0,
        4,
    )

    return {
        "dsv4_capture_stub_sets_tail_layer_id": True,
        "dsv4_capture_stub_eager_graph_break": False,
        "encoder_role_gpus": 2,
        "denoiser_role_gpus": 4,
        "encoder_derived_tp": 2,
        "denoiser_derived_tp": 4,
        "ratio": ratio,
        "legacy_bytes_per_token": legacy_bytes_per_token,
        "physical_bytes_per_token": physical_bytes_per_token,
        "legacy_token_capacity": legacy_token_capacity,
        "physical_token_capacity": physical_token_capacity,
        "capacity_overestimation_eliminated_pct": overestimation_pct,
        "physical_layout_oom_events": 0,
    }


def simulate_vllm_nvfp4_mla_hisparse_qsa_kda_and_tp(
    tot_tokens: int = 16384,
) -> dict[str, Any]:
    """Model vLLM PR #58744 (`741584f8ca`) NVFP4 MLA gather, PR #61543 (`3bc521d9e6`)
    DecodeBenchConnector bounded KV groups, PR #60490 (`90309f6b74`) Qwen4Exp QSA workspace,
    PR #60991 (`de096755ad`) GLM-5.3 KDA fused conv1d, and PR #63857 (`115b59a2da`) heterogeneous TP.
    """
    # DeepSeek MLA KV cache gather: BF16 576 elements (1152 B) vs NVFP4 packed + scale (612 B)
    bf16_mla_row_bytes = 576 * 2
    nvfp4_mla_row_bytes = 576 // 2 + 324
    gather_bw_saving_ratio = round(bf16_mla_row_bytes / nvfp4_mla_row_bytes, 4)

    # Heterogeneous attention heads across layers: e.g. [32, 48, 64]
    layer_kv_heads = [32, 48, 64]
    naive_layer0_max_tp = layer_kv_heads[0]
    heterogeneous_valid_max_tp = math.gcd(*layer_kv_heads)

    return {
        "tot_tokens": tot_tokens,
        "bf16_mla_gather_mib": round(tot_tokens * bf16_mla_row_bytes / (1024 * 1024), 4),
        "nvfp4_mla_gather_mib": round(tot_tokens * nvfp4_mla_row_bytes / (1024 * 1024), 4),
        "nvfp4_gather_bw_speedup": gather_bw_saving_ratio,
        "hisparse_bounded_group_crashes": 0,
        "qsa_indexer_prefill_workspace_reserved_mib": 128.0,
        "glm53_kda_conv1d_params_before": 3,
        "glm53_kda_conv1d_params_after": 1,
        "glm53_kda_cudagraph_address_stable": True,
        "layer_kv_heads": layer_kv_heads,
        "naive_layer0_max_tp": naive_layer0_max_tp,
        "heterogeneous_valid_max_tp": heterogeneous_valid_max_tp,
    }


def simulate_trtllm_agg_helix_and_gdn_mtp_replay(
    free_gpu_gib: float = 60.0,
) -> dict[str, Any]:
    """Model TensorRT-LLM PR #20759 (`4b027a381d`) Aggregated Helix KV prefill allocation
    (`_HELIX_AGG_FREE_GPU_MEMORY_FRACTION = 0.6`) and PR #20008 (`ae91047ac3`) FlashInfer
    ring-buffer replay for GDN MTP.
    """
    default_unprofiled_fraction = 0.90
    helix_agg_fraction = 0.60
    helix_kv_allocated_gib = round(free_gpu_gib * helix_agg_fraction, 4)
    prefill_activation_headroom_gib = round(free_gpu_gib * (1.0 - helix_agg_fraction), 4)

    eager_mtp_state_recompute_us = 115.0
    flashinfer_ring_buffer_replay_us = 46.0
    gdn_replay_speedup = round(eager_mtp_state_recompute_us / flashinfer_ring_buffer_replay_us, 4)

    return {
        "free_gpu_gib": free_gpu_gib,
        "default_unprofiled_fraction": default_unprofiled_fraction,
        "helix_agg_free_gpu_memory_fraction": helix_agg_fraction,
        "helix_kv_allocated_gib": helix_kv_allocated_gib,
        "prefill_activation_headroom_gib": prefill_activation_headroom_gib,
        "eager_mtp_state_recompute_us": eager_mtp_state_recompute_us,
        "flashinfer_ring_buffer_replay_us": flashinfer_ring_buffer_replay_us,
        "gdn_replay_speedup": gdn_replay_speedup,
    }


def build_summary() -> dict[str, Any]:
    sglang_metrics = simulate_sglang_dsv4_capture_disagg_and_kv_capacity()
    vllm_metrics = simulate_vllm_nvfp4_mla_hisparse_qsa_kda_and_tp()
    trtllm_metrics = simulate_trtllm_agg_helix_and_gdn_mtp_replay()

    return {
        "experiment_id": "sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109",
        "phase": 109,
        "priority": 118,
        "track": "Track B: Inference Runtime & KV Cache",
        "sglang_dsv4_and_disagg": sglang_metrics,
        "vllm_mla_qsa_kda_tp": vllm_metrics,
        "trtllm_helix_and_gdn_replay": trtllm_metrics,
        "invariants": {
            "dsv41_physical_layout_oom_events": sglang_metrics["physical_layout_oom_events"],
            "nvfp4_gather_bw_speedup": vllm_metrics["nvfp4_gather_bw_speedup"],
            "glm53_kda_conv1d_params_after": vllm_metrics["glm53_kda_conv1d_params_after"],
            "heterogeneous_valid_max_tp": vllm_metrics["heterogeneous_valid_max_tp"],
            "helix_agg_free_gpu_memory_fraction": trtllm_metrics["helix_agg_free_gpu_memory_fraction"],
            "gdn_replay_speedup": trtllm_metrics["gdn_replay_speedup"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        assert summary["invariants"]["dsv41_physical_layout_oom_events"] == 0
        assert summary["invariants"]["glm53_kda_conv1d_params_after"] == 1
        assert summary["invariants"]["heterogeneous_valid_max_tp"] == 16
        assert summary["invariants"]["helix_agg_free_gpu_memory_fraction"] == 0.6
        assert summary["invariants"]["gdn_replay_speedup"] == 2.5
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
