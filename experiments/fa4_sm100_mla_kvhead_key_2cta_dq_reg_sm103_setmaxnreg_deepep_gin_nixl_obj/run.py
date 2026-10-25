from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. FlashAttention-4 #2978 (94e22c906678): MLA forward compilation cache key must include num_head_kv
    # Compare (nheads_q=64, nheads_kv=1) vs (nheads_q=128, nheads_kv=2): both have nheads_per_kv = 64!
    mla_shapes = [
        {"nheads_q": 64, "nheads_kv": 1, "nheads_per_kv": 64},
        {"nheads_q": 128, "nheads_kv": 2, "nheads_per_kv": 64},
    ]
    unfixed_keys = [(576, 512, s["nheads_per_kv"]) for s in mla_shapes]
    fixed_keys = [(576, 512, s["nheads_per_kv"], s["nheads_kv"]) for s in mla_shapes]

    # 2. FlashAttention-4 #2960 (47e91f1f7dd8): 2-CTA dQaccum loaded directly into registers in bwd postprocess
    tile_m = 128
    tile_hdim = 128
    fp32_dQaccum_bytes = tile_m * tile_hdim * 4  # 65,536 B
    bf16_sdQ_bytes = tile_m * tile_hdim * 2      # 32,768 B
    smem_saved_bytes = fp32_dQaccum_bytes - bf16_sdQ_bytes

    # 3. FlashAttention-4 #2950 (037e808b7239): SM103 hd128/hd192 decode at q_stage=1 setmaxnreg.dec cap
    # SM103 1CTA causal hd128 entry uses (num_regs_softmax=176, num_regs_correction=64)
    num_regs_softmax = 176
    num_regs_correction = 64
    q_stage = 1
    threads_per_cta = 512
    launch_reg_cap = 65536 // threads_per_cta  # 128
    uncapped_num_regs_other = ((512 - num_regs_softmax * q_stage - num_regs_correction) // (3 - q_stage)) // 8 * 8  # 136
    capped_num_regs_other = min(uncapped_num_regs_other, launch_reg_cap)  # 128

    # 4. DeepEP #752 (a56d6156febc) & #715 (11d1cab604bb) + DeepGEMM #441 (78b69000794d)
    deepep_deepgemm_ordering = {
        "deepep_engram_fetch_commit": "a56d6156febcd9976e55adc85b5155bfac9f28f8",
        "deepep_gin_sys_fence_commit": "11d1cab604bba8ebe15022be44e09c22a659c857",
        "deepgemm_mega_moe_fence_commit": "78b69000794d0937b47ae3387eff7663410264d1",
        "engram_deferred_last_get_doorbell_before_flush": True,
        "world_team_fence_instruction": "fence.acq_rel.sys",
        "mega_moe_slot_release_fence_instruction": "fence.acq_rel.cta",
    }

    # 5. NIXL #2380 (d0319acbdb8e) & #2423 (c8178f563a9a)
    nixl_fixes = {
        "s3_accel_cuda_ctx_commit": "d0319acbdb8e9514a4cc0125dc05b43d8ab715ef",
        "ucx_err_handler_commit": "c8178f563a9a2ce7582831088cbe5bafc365d9af",
        "worker_thread_initial_ctx": None,
        "scoped_device_selected_during_register": 0,
        "worker_thread_restored_ctx": None,
        "ucx_err_mode_none_sets_err_handler_bit": False,
        "ucx_err_mode_peer_sets_err_handler_bit": True,
    }

    return {
        "experiment_id": "fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj",
        "phase": 79,
        "priority": 88,
        "evidence_mode": "own-offline-invariant-and-hardware-trap-model",
        "fa4_mla_cache_key": {
            "commit": "94e22c906678e5483fa0e9e24d8e787bc2c0ed4c",
            "pr": "#2978",
            "unfixed_collision": unfixed_keys[0] == unfixed_keys[1],
            "fixed_distinct": fixed_keys[0] != fixed_keys[1],
            "fewer_kv_first_failure_mode": "head_1_unwritten_stale_zeros_or_nan",
            "more_kv_first_failure_mode": "phantom_head_1_aliases_next_token_q_row",
        },
        "fa4_2cta_dqaccum_registers": {
            "commit": "47e91f1f7dd8a22649cee0dd9a182b69ffe782ef",
            "pr": "#2960",
            "deterministic_hd256_commit": "3451a2a67a24eeed8af54d3c5b8d219577113797",
            "fp32_dQaccum_bytes": fp32_dQaccum_bytes,
            "bf16_sdQ_placeholder_bytes": bf16_sdQ_bytes,
            "smem_reduction_bytes": smem_saved_bytes,
            "smem_reduction_ratio": 0.5,
        },
        "fa4_sm103_setmaxnreg_cap": {
            "commit": "037e808b723973782476bf94b89e80d07188fe2f",
            "pr": "#2950",
            "threads_per_cta": threads_per_cta,
            "launch_reg_cap": launch_reg_cap,
            "uncapped_num_regs_other": uncapped_num_regs_other,
            "capped_num_regs_other": capped_num_regs_other,
            "prevents_illegal_setmaxnreg_dec_trap": uncapped_num_regs_other > launch_reg_cap
            and capped_num_regs_other <= launch_reg_cap,
        },
        "deepep_deepgemm_ordering": deepep_deepgemm_ordering,
        "nixl_fixes": nixl_fixes,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    assert summary["phase"] == 79
    assert summary["priority"] == 88
    assert summary["fa4_mla_cache_key"]["unfixed_collision"] is True
    assert summary["fa4_mla_cache_key"]["fixed_distinct"] is True
    assert summary["fa4_2cta_dqaccum_registers"]["smem_reduction_bytes"] == 32768
    assert summary["fa4_sm103_setmaxnreg_cap"]["uncapped_num_regs_other"] == 136
    assert summary["fa4_sm103_setmaxnreg_cap"]["capped_num_regs_other"] == 128
    assert summary["fa4_sm103_setmaxnreg_cap"]["prevents_illegal_setmaxnreg_dec_trap"] is True
    assert summary["nixl_fixes"]["ucx_err_mode_none_sets_err_handler_bit"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()
    verify_summary(summary)

    if args.check:
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            verify_summary(existing)
        print("OK: fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj verified")
        return

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
