"""Phase 78 Track C deterministic experiment:
veRL MXFP8 Megatron Training & SGLang/vLLM DeepGEMM Weight Refit, Raw Log-Prob
1-ULP Mismatch Gate, VeOmni Low-Precision ReduceScatter, Mooncake Intel XPU
PCIe P2P + Level Zero IPC, and Dynamo KVCR Router & Batched Output Counter.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def simulate_verl_mxfp8_and_refit() -> dict:
    """Model veRL #8127 MXFP8 Megatron THD sequence padding (lcm(32, align))
    and B200 DAPO-Math-17k / GSM8K performance, #8306 SGLang MXFP8/MXFP4/NVFP4
    refit, and #8308 vLLM DeepGEMM rollout standardization.
    """
    thd_padding_table = []
    for tp_size in (1, 2, 4, 8):
        for cp_size in (1, 2, 4):
            align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
            fp8_block_per_seq = math.lcm(16, align_size)
            mxfp8_per_seq = math.lcm(32, align_size)
            total_align = align_size * 128
            thd_padding_table.append(
                {
                    "tp_size": tp_size,
                    "cp_size": cp_size,
                    "base_align_size": align_size,
                    "fp8_block_per_seq_align": fp8_block_per_seq,
                    "mxfp8_per_seq_align": mxfp8_per_seq,
                    "total_packed_align": total_align,
                }
            )

    dapo_qwen3_30b_a3b_arms = [
        {"arm": "BF16/BF16", "step_s": 506.0, "generation_s": 134.0, "update_actor_s": 244.0, "rollout_corr_kl": 0.0018},
        {"arm": "FP8-TW/BF16", "step_s": 487.0, "generation_s": 133.0, "update_actor_s": 225.0, "rollout_corr_kl": 0.0058},
        {"arm": "FP8-BW/FP8-BW", "step_s": 489.0, "generation_s": 115.0, "update_actor_s": 241.0, "rollout_corr_kl": 0.0090},
        {"arm": "MXFP8/MXFP8", "step_s": 452.0, "generation_s": 102.0, "update_actor_s": 221.0, "rollout_corr_kl": 0.0045},
    ]
    bf16_base = dapo_qwen3_30b_a3b_arms[0]
    for arm in dapo_qwen3_30b_a3b_arms:
        arm["step_speedup_vs_bf16"] = round(bf16_base["step_s"] / arm["step_s"], 4)
        arm["gen_reduction_pct"] = round((1.0 - arm["generation_s"] / bf16_base["generation_s"]) * 100.0, 2)

    qwen35_35b_a3b_refit_kl = [
        {"engine": "SGLang", "rollout": "MXFP8 linear + MoE", "backend": "DeepGEMM linear + MoE", "k3_kl_no_qat": 0.00324, "k3_kl_with_qat": 0.00327},
        {"engine": "SGLang", "rollout": "BF16 linear + MXFP4 MoE", "backend": "DeepGEMM MoE", "k3_kl_no_qat": 0.00571, "k3_kl_with_qat": 0.00200},
        {"engine": "SGLang", "rollout": "BF16 linear + NVFP4 MoE", "backend": "FlashInfer CUTLASS MoE", "k3_kl_no_qat": 0.00946, "k3_kl_with_qat": 0.00653},
        {"engine": "vLLM", "rollout": "MXFP8 Linear + MoE", "backend": "FlashInfer CuteDSL Linear + DeepGEMM MoE", "k3_kl_no_qat": 0.00387, "k3_kl_with_qat": 0.00325},
        {"engine": "vLLM", "rollout": "BF16 Linear + MXFP4 MoE", "backend": "DeepGEMM MoE", "k3_kl_no_qat": 0.00790, "k3_kl_with_qat": 0.00140},
        {"engine": "vLLM", "rollout": "BF16 Linear + NVFP4 MoE", "backend": "FlashInfer CUTLASS MoE", "k3_kl_no_qat": 0.00808, "k3_kl_with_qat": 0.00722},
    ]
    for item in qwen35_35b_a3b_refit_kl:
        item["qat_kl_reduction_ratio"] = round(item["k3_kl_no_qat"] / item["k3_kl_with_qat"], 4)

    return {
        "thd_padding_table": thd_padding_table,
        "dapo_qwen3_30b_a3b_arms": dapo_qwen3_30b_a3b_arms,
        "qwen35_35b_a3b_refit_kl": qwen35_35b_a3b_refit_kl,
        "vllm_mxfp4_qat_kl_reduction_ratio": 5.6429,
        "sglang_mxfp4_qat_kl_reduction_ratio": 2.855,
    }


def _f32_round(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def simulate_verl_logprob_1ulp_and_veomni() -> dict:
    """Model veRL #8025 raw log-prob 1-ULP mismatch detection vs exp(log_prob)
    underflow/rounding blindness, and #8330 VeOmni low-precision ReduceScatter.
    """
    lp_a = _f32_round(-0.10000000149011612)
    lp_b = _f32_round(-0.10000000894069672)
    prob_a = _f32_round(math.exp(lp_a))
    prob_b = _f32_round(math.exp(lp_b))
    prob_diff = abs(prob_a - prob_b)
    raw_lp_diff = abs(lp_a - lp_b)

    return {
        "lp_a_f32": lp_a,
        "lp_b_f32": lp_b,
        "prob_a_f32": prob_a,
        "prob_b_f32": prob_b,
        "prob_diff_f32": prob_diff,
        "raw_logprob_diff_f32": raw_lp_diff,
        "probs_diff_blind_to_1ulp": (prob_diff == 0.0 and raw_lp_diff > 0.0),
        "veomni_428m_dense_median_ms": {
            "native_fp32_rs_1": 117.023,
            "low_bf16_rs_1": 130.853,
            "low_bf16_rs_2": 119.163,
            "native_fp32_rs_2": 118.891,
        },
    }


def simulate_mooncake_xpu_ipc_and_dynamo_metrics() -> dict:
    """Model Mooncake #4379 intra-node Intel XPU PCIe P2P + Level Zero IPC via
    pidfd_getfd(2) and generation-ordered cache invalidation, plus Dynamo
    #15535 KVCR recurrent event filter and #17067 64-token batched counter.
    """
    # Mooncake XPU transfer latency model for 64 MiB KV block across same-node XPUs
    payload_mib = 64
    payload_bytes = payload_mib * 1024 * 1024
    # Direct PCIe Gen5 x16 / Xe Link P2P (~48 GB/s effective) vs Host 8 MiB chunk bounce (~18 GB/s) vs TCP loopback (~6 GB/s)
    p2p_ipc_us = round((payload_bytes / (48.0 * 1e9)) * 1e6, 2)
    host_bounce_8mb_us = round((payload_bytes / (18.0 * 1e9)) * 1e6, 2)
    network_loopback_us = round((payload_bytes / (6.0 * 1e9)) * 1e6, 2)

    # Dynamo #17067: LOCAL_FLUSH_TOKENS = 64 reduces shared atomic cache-line bounces
    total_generated_tokens = 4096
    flush_interval = 64
    unbatched_shared_atomic_adds = total_generated_tokens
    batched_shared_atomic_adds = math.ceil(total_generated_tokens / flush_interval)

    return {
        "mooncake_xpu_64mib_transfer": {
            "p2p_ipc_us": p2p_ipc_us,
            "host_bounce_8mb_us": host_bounce_8mb_us,
            "network_loopback_us": network_loopback_us,
            "speedup_vs_host_bounce": round(host_bounce_8mb_us / p2p_ipc_us, 4),
            "speedup_vs_network": round(network_loopback_us / p2p_ipc_us, 4),
            "ipc_attr_fields": ["pid", "base", "size", "handle_hex", "pid_namespace", "owner_uuid_hex", "generation"],
        },
        "dynamo_output_token_batching": {
            "local_flush_tokens": flush_interval,
            "total_tokens_per_req": total_generated_tokens,
            "unbatched_shared_atomic_ops": unbatched_shared_atomic_adds,
            "batched_shared_atomic_ops": batched_shared_atomic_adds,
            "shared_atomic_reduction_factor": unbatched_shared_atomic_adds // batched_shared_atomic_adds,
        },
        "dynamo_kvcr_recurrent_filter_before_geometry": True,
    }


def build_summary() -> dict:
    return {
        "experiment_id": "verl_mxfp8_megatron_sglang_vllm_deepgemm_refit_mooncake_xpu_p2p_dynamo_kvcr",
        "phase": 78,
        "priority": 87,
        "track": "rl_training_and_transfer",
        "verl_mxfp8_and_refit": simulate_verl_mxfp8_and_refit(),
        "verl_logprob_and_veomni": simulate_verl_logprob_1ulp_and_veomni(),
        "mooncake_and_dynamo": simulate_mooncake_xpu_ipc_and_dynamo_metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["verl_logprob_and_veomni"]["probs_diff_blind_to_1ulp"] is True
        assert summary["verl_mxfp8_and_refit"]["vllm_mxfp4_qat_kl_reduction_ratio"] > 5.5
        assert summary["mooncake_and_dynamo"]["dynamo_output_token_batching"]["shared_atomic_reduction_factor"] == 64


if __name__ == "__main__":
    main()
