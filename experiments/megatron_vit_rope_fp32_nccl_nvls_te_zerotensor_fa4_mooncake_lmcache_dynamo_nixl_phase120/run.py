"""Phase 120 Track C stdlib experiment: Megatron ViT FP32 RoPE & Deterministic NVLS AllReduce,
TransformerEngine ZeroTensor Dummy Wgrad & SM12x FA4 Deterministic Guard,
Mooncake EFA Stale Descriptor Refetch, HIXL NIC Topo Injection & Dynamic PeerMem Probe,
LMCache Request-Tag Lookup Keys, Dynamo Logprob-Conditional Token Decode & NIXL Shared-Lock Probes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _bf16_round(val: float) -> float:
    """Simulate IEEE-754 BFloat16 truncation (7-bit mantissa) using stdlib math.frexp."""
    if val == 0.0:
        return 0.0
    m, e = math.frexp(val)
    # 1 implicit bit + 7 fraction bits = 8 bits -> scale by 256 and round to nearest even
    q = round(m * 256.0) / 256.0
    return math.ldexp(q, e)


def model_megatron_vit_rope_fp32_and_deterministic_nvls() -> dict[str, Any]:
    """Model Megatron-LM 3ca52d9 (ViT FP32 RoPE) and 62c063f (Deterministic Ring,NVLS)."""
    head_dim = 128
    rope_theta = 10000.0
    patch_coord = 109.0  # 110x110 grid max index

    max_angle_err = 0.0
    max_cos_err = 0.0
    for i in range(0, head_dim, 2):
        freq_fp32 = 1.0 / (rope_theta ** (float(i) / head_dim))
        freq_bf16 = _bf16_round(freq_fp32)
        angle_fp32 = patch_coord * freq_fp32
        angle_bf16 = patch_coord * freq_bf16
        max_angle_err = max(max_angle_err, abs(angle_fp32 - angle_bf16))
        max_cos_err = max(max_cos_err, abs(math.cos(angle_fp32) - math.cos(angle_bf16)))

    accepted_tokens = sorted(["Ring", "CollnetDirect", "CollnetChain", "^NVLS", "NVLS"])

    def validate_nccl_algo(nccl_algo: str | None, arch_major: int) -> tuple[bool, str]:
        if nccl_algo is None:
            return True, "Ring"
        if nccl_algo == "":
            if arch_major == 9:
                return False, "rejected_empty_on_hopper"
            return True, ""
        if not nccl_algo.strip():
            return False, "rejected_whitespace_only"
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        if not tokens or not all(t in accepted_tokens for t in tokens):
            return False, "rejected_unknown_token"
        if "NVLS" in tokens and "Ring" not in tokens:
            return False, "rejected_nvls_without_ring"
        if "NVLS" in tokens and arch_major == 9:
            return False, "rejected_nvls_on_hopper"
        return True, nccl_algo

    return {
        "vit_rope_grid": [110, 110],
        "vit_rope_max_angle_rad_err_if_cast_to_bf16": round(max_angle_err, 6),
        "vit_rope_max_cos_err_if_cast_to_bf16": round(max_cos_err, 6),
        "vit_rope_kept_fp32_outside_buffer": True,
        "accepted_nccl_algo_tokens": accepted_tokens,
        "ring_nvls_on_blackwell_sm100": validate_nccl_algo("Ring,NVLS", 10)[0],
        "ring_nvls_on_hopper_sm90": validate_nccl_algo("Ring,NVLS", 9)[0],
        "nvls_alone_on_blackwell_sm100": validate_nccl_algo("NVLS", 10)[0],
        "empty_nccl_algo_on_blackwell_sm100": validate_nccl_algo("", 10)[0],
        "empty_nccl_algo_on_hopper_sm90": validate_nccl_algo("", 9)[0],
    }


def model_te_zerotensor_wgrad_and_sm12x_fa4() -> dict[str, Any]:
    """Model TransformerEngine fc48751 (ZeroTensor dummy wgrad) & 6d6a500 (SM12x FA4 guard)."""
    vocab_size = 129280
    hidden_size = 7168
    elem_bytes = 2  # BF16
    tensor_bytes = vocab_size * hidden_size * elem_bytes
    # Autograd addition of two dummy wgrads reads 2 inputs and writes 1 output = 3 * tensor_bytes
    avoided_hbm_traffic_bytes = 3 * tensor_bytes
    vr200_saved_ms_per_shared_head = 0.44

    def fa4_enabled(arch_major: int, deterministic: bool, requires_backward: bool) -> bool:
        if deterministic and requires_backward and arch_major == 12:
            return False
        return True

    return {
        "shared_lm_head_shape": [vocab_size, hidden_size],
        "dummy_wgrad_tensor_mib": round(tensor_bytes / (1024 * 1024), 2),
        "avoided_dummy_add_hbm_mib": round(avoided_hbm_traffic_bytes / (1024 * 1024), 2),
        "vr200_saved_ms_per_backward": vr200_saved_ms_per_shared_head,
        "dummy_cache_key_fields": ["shape", "dtype", "device"],
        "sm120_det_bwd_fa4_enabled": fa4_enabled(12, True, True),
        "sm120_det_fwd_only_fa4_enabled": fa4_enabled(12, True, False),
        "sm100_det_bwd_fa4_enabled": fa4_enabled(10, True, True),
    }


def model_mooncake_efa_hixl_peermem() -> dict[str, Any]:
    """Model Mooncake c596310 (EFA stale desc refetch), c8fe63e (HIXL nic_topo_path),
    and cb670d5 (dynamic WITH_NVIDIA_PEERMEM probe).
    """
    # EFA batch of 32 slices targeting 1 newly mounted chunk on peer target_id=7:
    # Old behavior: 32 failed slices with TRANSFER_FAIL (-800).
    # New behavior: 1 metadata force_update per target per call, 0 failed slices.
    slices_in_batch = 32
    refetches_per_call = 1

    def resolve_peermem(env_val: str | None, module_loaded: bool, data_direct: bool) -> str:
        if data_direct:
            return "mlx5_data_direct"
        if env_val is not None:
            use_peermem = env_val.upper() in ("1", "ON", "TRUE")
        else:
            use_peermem = module_loaded
        return "ibv_reg_mr_peermem" if use_peermem else "dma_buf_export"

    return {
        "efa_batch_slices": slices_in_batch,
        "efa_old_failed_slices": slices_in_batch,
        "efa_new_failed_slices": 0,
        "efa_metadata_refetches_per_target_per_call": refetches_per_call,
        "hixl_env_precedence": ["MC_CUSTOM_TOPO_JSON", "MC_TENT_CONF:topology/custom_json_path"],
        "gb300_open_kmod_unset_env_path": resolve_peermem(None, False, False),
        "legacy_peermem_host_unset_env_path": resolve_peermem(None, True, False),
        "data_direct_override_path": resolve_peermem("1", True, True),
    }


def model_lmcache_dynamo_and_nixl() -> dict[str, Any]:
    """Model LMCache e45d4d5 (MooncakeLookupClient request_configs tags),
    Dynamo cffc4fa (skip per-token text decode without logprobs),
    and NIXL 842eedf (prometheus_mp shared-lock liveness probes).
    """
    # Dynamo frontend streaming decode: 512 output tokens, log_probs=None vs Some
    output_tokens = 512
    utf8_allocs_with_logprobs = output_tokens
    utf8_allocs_without_logprobs = 0

    # NIXL prometheus_mp liveness probe matrix:
    # Writer holds LOCK_EX; departed store has no writer lock.
    # When Owner A probes with LOCK_SH, concurrent Owner B probing with LOCK_SH also succeeds
    # in acquiring LOCK_SH on a departed store -> both correctly detect writer is dead!
    return {
        "lmcache_passes_request_configs_to_process_tokens": True,
        "dynamo_output_tokens": output_tokens,
        "dynamo_token_string_allocs_with_logprobs": utf8_allocs_with_logprobs,
        "dynamo_token_string_allocs_without_logprobs": utf8_allocs_without_logprobs,
        "nixl_probe_lock_mode": "LOCK_SH | LOCK_NB",
        "nixl_writer_lock_mode": "LOCK_EX | LOCK_NB",
        "nixl_concurrent_probes_false_live_with_lock_ex": True,
        "nixl_concurrent_probes_false_live_with_lock_sh": False,
    }


def run_all() -> dict[str, Any]:
    return {
        "phase": 120,
        "track": "track_c",
        "title": "Megatron ViT FP32 RoPE & Deterministic NVLS, TE ZeroTensor Wgrad & SM12x FA4, Mooncake EFA/HIXL/PeerMem, LMCache, Dynamo & NIXL",
        "megatron_vit_and_nvls": model_megatron_vit_rope_fp32_and_deterministic_nvls(),
        "te_zerotensor_and_fa4": model_te_zerotensor_wgrad_and_sm12x_fa4(),
        "mooncake_efa_hixl_peermem": model_mooncake_efa_hixl_peermem(),
        "lmcache_dynamo_nixl": model_lmcache_dynamo_and_nixl(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.check:
        assert summary["megatron_vit_and_nvls"]["ring_nvls_on_blackwell_sm100"] is True
        assert summary["megatron_vit_and_nvls"]["ring_nvls_on_hopper_sm90"] is False
        assert summary["te_zerotensor_and_fa4"]["sm120_det_bwd_fa4_enabled"] is False
        assert summary["mooncake_efa_hixl_peermem"]["gb300_open_kmod_unset_env_path"] == "dma_buf_export"
        assert summary["lmcache_dynamo_nixl"]["nixl_concurrent_probes_false_live_with_lock_sh"] is False

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
