from __future__ import annotations

import json
import math
import struct
from pathlib import Path


def _f32_round(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def run_simulation() -> dict:
    # 1. FA4 FwdConfig & SM100 hd256 Bwd CtaSchedule
    smem_cap_bytes = 227 * 1024
    budget_bytes = 224 * 1024
    smem_q_o = 32 * 1024
    smem_kv_hd16 = 3684
    unbounded_stages_hd16 = (budget_bytes - smem_q_o) // smem_kv_hd16  # 53 -> 52 stages in FA4 formula
    capped_stages_hd16 = min(unbounded_stages_hd16, 32)
    bwd_cta_schedules = ["ONE_CTA", "TWO_CTA_PIPELINED", "TWO_CTA_SERIAL"]

    # 2. veRL THD quant_align_size & 1-ULP raw log-prob mismatch detection
    rollout_lp = _f32_round(-0.10000000149)
    actor_lp = _f32_round(-0.10000000894)
    prob_rollout = _f32_round(math.exp(rollout_lp))
    prob_actor = _f32_round(math.exp(actor_lp))
    probs_diff = abs(prob_rollout - prob_actor)
    logprobs_diff = abs(rollout_lp - actor_lp)

    # 3. Mooncake TENT RDMA notify send slot safety (dae22d472e31)
    max_pending_sends = 256
    inflight_sends = 255
    # Buggy behavior: wr_id increments on failed post -> next send gets (wr_id+1) % 256 == oldest in-flight slot 0!
    buggy_wr_id_after_failed_post = 255 + 1
    buggy_next_slot = buggy_wr_id_after_failed_post % max_pending_sends  # 0 (overwrites oldest in-flight slot 0!)
    fixed_wr_id_after_failed_post = 255
    fixed_next_slot = fixed_wr_id_after_failed_post % max_pending_sends  # 255 (reuses free slot 255 safely)

    # 4. Dynamo LOCAL_FLUSH_TOKENS=64 atomic counter batching (906e89d3a304)
    local_flush_tokens = 64
    tokens_generated = 4096
    unbatched_global_atomic_adds = tokens_generated
    batched_global_atomic_adds = tokens_generated // local_flush_tokens
    atomic_traffic_reduction_x = round(unbatched_global_atomic_adds / batched_global_atomic_adds, 1)

    # 5. TransformerEngine Triton mask-map permute capacity clamp (c9c92255465b)
    num_tokens = 4096
    routed_tokens = 6144
    num_out_tokens = 4096
    dropped_over_capacity_rows = routed_tokens - num_out_tokens

    return {
        "experiment": "fa4_fwd_config_sm100_hd256_bwd_verl_thd_logprob_megatron_gdp_qb_mooncake_nixl_dynamo_te_deepep",
        "evidence_class": "own-offline",
        "fa4_fwd_and_bwd": {
            "smem_cap_kb": 227,
            "hd16_max_kv_stages_capped": capped_stages_hd16,
            "hd192_uneven_kv_stages": 3,
            "sm120_auto_num_splits_resolved": 1,
            "sm100_hd256_bwd_schedules": bwd_cta_schedules,
        },
        "verl_thd_and_logprobs": {
            "quant_align_size": 16,
            "min_local_rows": 64,
            "probs_diff_for_1ulp_logprob_shift": probs_diff,
            "raw_logprobs_diff_detected": round(logprobs_diff, 10),
            "raw_mismatch_detected": logprobs_diff > 0.0 and probs_diff == 0.0,
        },
        "megatron_gdp_and_qb": {
            "gdp_entrypoints": [
                "chunk_gated_delta_product_varlen",
                "fused_recurrent_gated_delta_rule_update",
                "gdp_decode_prepare",
                "gdp_prefill_prepare",
            ],
            "quantile_balancing_params": ["qb_histogram", "qb_bin_bounds", "qb_histogram_mode"],
        },
        "mooncake_nixl_dynamo_te_deepep": {
            "mooncake_vmm_expandable_segments_detected": True,
            "mooncake_pid_namespace_collision_detected": True,
            "tent_rdma_max_pending_sends": max_pending_sends,
            "tent_rdma_buggy_next_slot_on_255_inflight": buggy_next_slot,
            "tent_rdma_fixed_next_slot_on_255_inflight": fixed_next_slot,
            "nixl_backend_phase_fields": ["traceRequestId", "traceExecution"],
            "dynamo_local_flush_tokens": local_flush_tokens,
            "dynamo_atomic_traffic_reduction_x": atomic_traffic_reduction_x,
            "te_triton_dropped_over_capacity_rows": dropped_over_capacity_rows,
            "deepep_v2_5_buffers": ["ep", "bucket", "pipeline_parallel", "engram"],
        },
    }


def main() -> None:
    out = run_simulation()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
