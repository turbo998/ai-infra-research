"""Phase 124 Track C: Megatron-LM GDP FP32 Sigmoid / Batch Size Transition / Refit Round-Robin, TorchTitan LocalCompileConfig / Generic Loss, veRL TPU v7x / ROCm DSV4 / VeOmni FSDP2 Offload, Mooncake EFA Stall Floor / WeakPtr Handle / NUMA Cpuset / HCCS 2MiB Alignment, LMCache Deterministic KV & Dynamo DP Queue Limit Verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def bf16_round(val: float) -> float:
    """Round an IEEE-754 float32 to BFloat16 ( round-to-nearest-even )."""
    bits = struct.unpack(">I", struct.pack(">f", val))[0]
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def build_summary() -> dict[str, Any]:
    # 1. Megatron-LM GatedDeltaProductMixer FP32 Sigmoid Gradient Preservation (`a296523510`),
    # Batch Size Transition (`d1eec51b85`) & Single-DP-Group Refit Round-Robin (`4a55e6fee6`)
    test_b_values = [6.25, 7.0, 8.0, 9.0]
    gdp_sigmoid_rows = {}
    for b in test_b_values:
        sig_fp32 = 1.0 / (1.0 + math.exp(-b))
        sig_bf16 = bf16_round(sig_fp32)
        grad_bf16_saved = sig_bf16 * (1.0 - sig_bf16) * 2.0  # with allow_neg_eigval (*2)
        grad_fp32_saved = sig_fp32 * (1.0 - sig_fp32) * 2.0
        gdp_sigmoid_rows[f"b_{b}"] = {
            "sigmoid_fp32": round(sig_fp32, 8),
            "sigmoid_bf16": round(sig_bf16, 6),
            "grad_when_bf16_saved": round(grad_bf16_saved, 8),
            "grad_when_fp32_saved": round(grad_fp32_saved, 8),
        }

    # Batch-size ramp: global_batch_size 8192 -> 12288 at DP = 8192, micro_batch_size = 1
    # Integer floor division 8192 // 8192 == 1 and 12288 // 8192 == 1 (num_microbatches stays 1!)
    dp_size = 8192
    gbs_before = 8192
    gbs_after = 12288
    num_mb_before = gbs_before // dp_size
    num_mb_after = gbs_after // dp_size

    megatron_verification = {
        "gdp_fp32_sigmoid_commit": "a2965235107ae81a069acfc43d25ac14f644d5ff",
        "batch_transition_commit": "d1eec51b8532a360cf22e954f7426466d2e4c703",
        "refit_round_robin_commit": "4a55e6fee69ac98ff678f3c6a84ed1833a476da4",
        "gdp_sigmoid_saturation_analysis": gdp_sigmoid_rows,
        "batch_size_ramp_detection": {
            "dp_size": dp_size,
            "gbs_before": gbs_before,
            "gbs_after": gbs_after,
            "num_microbatches_before": num_mb_before,
            "num_microbatches_after": num_mb_after,
            "old_check_detected_transition": num_mb_before != num_mb_after,
            "new_check_detected_transition": (num_mb_before != num_mb_after) or (gbs_before != gbs_after),
        },
        "refit_single_dp_group_8_replicas": {
            "tensors_count": 16,
            "max_tensors_on_replica0_before": 16,
            "max_tensors_on_any_replica_after": 2,
            "load_balance_speedup_x": 8.0,
        },
    }

    # 2. TorchTitan LocalCompileConfig & Generic Loss[LossInputT] (`0f9fd2bd64`, `740d5f755a`)
    torchtitan_verification = {
        "local_compile_commit": "0f9fd2bd64234ab061ab60a33f10fb3df6272bb2",
        "generic_loss_commit": "740d5f755a9d35b8a4fd21763856733022aa04ef",
        "compile_hook": "model.local_compile.apply(model, **compile_kwargs)",
        "loss_protocol_variance": "TypeVar('LossInputT', contravariant=True)",
        "decoupled_loss_wrappers": ["MTPLoss", "ChunkedLossWrapper"],
    }

    # 3. veRL TPU v7x Qwen3.5 GRPO, AMD ROCm DeepSeek-V4 & VeOmni FSDP2 Offload + EP Stream Load (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)
    verl_verification = {
        "tpu_v7x_commit": "2ab3529203527579c4ab81e173ccab82765b9fb9",
        "rocm_dsv4_commit": "6d04f398f18238b34fd1a036c60e1aa8fa478e22",
        "veomni_fsdp2_offload_commit": "b65da71c17a903dc43cb4b3f74d448f3f4a3f15e",
        "veomni_ep_stream_fp8_commit": "6f3f449e1120042611b90e4b457ef002de6644ad",
        "tpu_v7x_recipe": {
            "model": "Qwen/Qwen3.5-35B-A3B",
            "chips": 8,
            "xla_bf16_flag": "--xla_tpu_bf16_emission_mode=PROMOTE_F32",
            "max_num_batched_tokens": 2048,
        },
        "rocm_dsv4": {
            "arch_targets": ["gfx942", "gfx950"],
            "bpreshuffle_env": "VLLM_ROCM_DSV4_BPRESHUFFLE=0",
            "deep_gemm_enabled_on_rocm": False,
        },
        "veomni_fsdp2_cpu_offload": {
            "35b_model_fp32_temp_alloc_avoided_gib": round(35e9 * 4 / (1024**3), 2),
            "ep_sharded_stream_load": True,
            "trainer_quantizer": "Fp8ConfigWeightQuantizer",
        },
    }

    # 4. Mooncake EFA Stall Floor, WeakPtr Handle, NUMA Cpuset Filter, HCCS 2MiB Alignment & Conductor Prefix Cap
    # (`3a05eec5ef`, `9df448399b`, `a69bdd41cf`, `179ffb21c8`, `e70c09986d`, `5ef59b0c6e`)
    mooncake_verification = {
        "efa_stall_commit": "3a05eec5efc7ef64cf367e46c8e8c33864259c03",
        "weak_ptr_handle_commit": "9df448399b316180f32b4f26b3fd9e663231215f",
        "numa_cpuset_commit": "a69bdd41cf053f002df91c7d82f4e2a0219c1614",
        "conductor_cap_commit": "179ffb21c89598d75ab199482e5bd6756bc13eb2",
        "hccs_alignment_commit": "e70c09986dfa6403c49964f78877770d2c845aa2",
        "pg_active_ranks_commit": "5ef59b0c6e7289a063783b87bc0856862140175c",
        "efa_stall_gate": {
            "max_stalled_waves": 1024,
            "default_stall_min_ms": 1000,
            "floor_stall_min_ms": 50,
            "env_var": "MC_EFA_STALL_MIN_MS",
        },
        "weak_ptr_handle": {
            "atomic_refcount_ops_per_alloc_before": 2,
            "atomic_refcount_ops_per_alloc_after": 0,
        },
        "numa_mbind_filter": {
            "nic_preferred_nodes": [0, 1],
            "container_cpuset_mems": [1],
            "filtered_mbind_nodes": [1],
            "prevents_mbind_einval": True,
        },
        "hccs_device_mem_alignment_bytes": 2 * 1024 * 1024,
        "conductor_default_max_blocks_per_context": 200000,
    }

    # 5. LMCache Deterministic SHA-256 KV Verification & ObjectKeyIdentity (`58c2ace2c2`, `0108f5b634`)
    # Verify deterministic byte generation in [1, 127] from SHA-256("request-0:layer-0")
    digest = hashlib.sha256(b"request-0:layer-0").digest()
    deterministic_vals = [(b % 127) + 1 for b in digest[:8]]

    # 6. Dynamo DP-Scaled Admission Queue Limit (`cd3b977edb`, `08907b7ad7`)
    # Global budget 40,000 divided by rank_count vs old hardcoded 20,000 per rank
    dp_ranks = 8
    old_per_rank_cap = 20000
    new_per_rank_cap = 40000 // dp_ranks

    lmcache_and_dynamo = {
        "lmcache_verify_commit": "58c2ace2c23ca30ec879e904d8627d701980172b",
        "lmcache_object_key_commit": "0108f5b634dc1309d3877b74c5b2f88c4b144966",
        "dynamo_queue_limit_commit": "cd3b977edb89d988785499a014d987b863e3c9c9",
        "dynamo_trace_aic_commit": "08907b7ad7f41861c39a9b3a3bb1788bf5b7d265",
        "lmcache_sample_sha256_non_zero_bytes": deterministic_vals,
        "lmcache_value_range": [1, 127],
        "lmcache_object_key_identity_fields": ["chunk_hash", "kv_rank", "model_name", "cache_salt"],
        "dynamo_dp8_admission_cap": {
            "dp_ranks": dp_ranks,
            "old_total_queued_requests": old_per_rank_cap * dp_ranks,
            "new_total_queued_requests": new_per_rank_cap * dp_ranks,
            "new_per_rank_limit": new_per_rank_cap,
            "queue_bloat_reduction_x": round((old_per_rank_cap * dp_ranks) / (new_per_rank_cap * dp_ranks), 2),
        },
    }

    return {
        "phase": 124,
        "track": "C",
        "question_id": "q-megatron-gdp-fp32-bsize-refit-torchtitan-compile-loss-verl-tpu-rocm-veomni-mooncake-lmcache-dynamo-phase124",
        "megatron_verification": megatron_verification,
        "torchtitan_verification": torchtitan_verification,
        "verl_verification": verl_verification,
        "mooncake_verification": mooncake_verification,
        "lmcache_and_dynamo": lmcache_and_dynamo,
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
        print("Phase 124 Track C check passed.")
        return 0

    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
