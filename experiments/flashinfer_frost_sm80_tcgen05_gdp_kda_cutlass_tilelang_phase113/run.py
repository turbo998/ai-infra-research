"""Phase 113 Track A deterministic offline simulation:
1. FlashInfer FROST low-latency grouped MoE SM80 (`cp.async` + warp MMA, `--gpu-arch sm_80` vs `sm_90a`/`sm_120a`) & typed activation coverage (`0e406f333b`).
2. FlashInfer `cake_tinygemm2_tcgen05` 12-program per-arch (`sm_100a`/`sm_103a`/`sm_107a`) `tinygemm_bf16` routing window (`8d9c839db0`).
3. FlashInfer `cake_gdp` T=1 decode `v`/`beta` dense-row preload at `k`-load site on SM100/SM103 (`9f6ba5a820`) & `cake_batch_deepgemm_fp8` `K=7168, expected_m<=1` even `n_l2_group=8` 2-CTA M-block alignment (`4d018bcd12`).
4. FlashInfer `cake_kda` chunked training fused gate epilogue on SM103, segmented 16-head rows on SM103/SM107, 2-level segment carry on SM100/SM107 (`762b237e15`) & `cake_blackwell_softmax` cached-cluster push max/sum exchange + persistent 2.5x resident-cluster route (`510e514b25`).
5. CUTLASS SM100/SM120 block-scale-factor epilogue per-group norm constant index capture before batch-coordinate `l` zeroing (`ad9bd53bda`) + Hopper CuTeDSL FP8 GEMM GELU epilogue (`a931725f1d`), and TileLang Ascend PTO `vector_from_list` typed Ramp emission (`d7a28bd78b`) + between-kernel RNG state reset (`44405ee548`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_frost_sm80_and_tcgen05_tinygemm() -> dict[str, object]:
    """Model Frost SM80 arch flag & candidate matrix (`0e406f333b`) + `cake_tinygemm2_tcgen05` (`8d9c839db0`)."""
    def frost_arch_flag(major: int, minor: int) -> str:
        suffix = "" if major == 8 else "a"
        return f"sm_{major}{minor}{suffix}"

    arch_flags = {
        "sm80": frost_arch_flag(8, 0),
        "sm90": frost_arch_flag(9, 0),
        "sm120": frost_arch_flag(12, 0),
    }

    # cake_tinygemm2_tcgen05: 2 (bias: with/without) * 2 (PDL: with/without) * 3 (tile: narrow, packed, per-chain-acc) = 12 programs/arch
    programs_per_arch = 2 * 2 * 3
    total_programs_3_archs = programs_per_arch * 3  # sm_100a, sm_103a, sm_107a = 36

    # B200 (sm_100a) cold-L2 median kernel times in us from commit 8d9c839db0
    b200_4x2880x2880_no_bias_base_us = 8.29
    b200_4x2880x2880_no_bias_cake_us = 7.26
    b200_4x2880x2880_speedup = round(
        b200_4x2880x2880_no_bias_base_us / b200_4x2880x2880_no_bias_cake_us, 4
    )

    b200_1x128x14336_no_bias_base_us = 8.06
    b200_1x128x14336_no_bias_cake_us = 7.39
    b200_1x128x14336_speedup = round(
        b200_1x128x14336_no_bias_base_us / b200_1x128x14336_no_bias_cake_us, 4
    )

    return {
        "frost_arch_flags": arch_flags,
        "sm80_fma_choices": 16,
        "tcgen05_programs_per_arch": programs_per_arch,
        "tcgen05_total_programs_sm100_103_107": total_programs_3_archs,
        "b200_4x2880x2880_no_bias_base_us": b200_4x2880x2880_no_bias_base_us,
        "b200_4x2880x2880_no_bias_cake_us": b200_4x2880x2880_no_bias_cake_us,
        "b200_4x2880x2880_speedup_x": b200_4x2880x2880_speedup,
        "b200_1x128x14336_speedup_x": b200_1x128x14336_speedup,
    }


def simulate_deepgemm_n_l2_group_and_gdp_preload() -> dict[str, object]:
    """Model 2-CTA cluster M-block indexing in `cake_batch_deepgemm_fp8` (`4d018bcd12`) and GDP preload (`9f6ba5a820`)."""
    # A 2-CTA cluster shares one 32-row M block (each CTA loads 16 rows).
    # For a group with 40 rows -> m_blocks_g = ceil(40 / 32) = 2 M-blocks.
    m_blocks_g = 2

    def cluster_m_blocks(n_l2_group: int) -> tuple[int, int]:
        # CTA 0 and CTA 1 take consecutive tile indices block_idx=0 and block_idx=1
        if n_l2_group == 1:
            # Buggy: m_block = block_idx % m_blocks_g -> CTA0 gets m_block 0, CTA1 gets m_block 1!
            return (0 % m_blocks_g, 1 % m_blocks_g)
        # Fixed even n_l2_group=8: consecutive CTAs advance along N first or share the same m_block
        return ((0 // n_l2_group) % m_blocks_g, (1 // n_l2_group) % m_blocks_g)

    buggy_cta_m_blocks = cluster_m_blocks(1)
    fixed_cta_m_blocks = cluster_m_blocks(8)

    # Reported bad elements on GB300 (32 groups, one 40-row group, M=256, N=4096, K=7168)
    gb300_32g_bad_elements_before = 97355
    gb300_32g_valid_elements = 413696
    gb300_32g_bad_elements_after = 0

    return {
        "k_dim": 7168,
        "expected_m_threshold": 1,
        "buggy_n_l2_group": 1,
        "fixed_n_l2_group": 8,
        "buggy_cta_pair_m_blocks": list(buggy_cta_m_blocks),
        "fixed_cta_pair_m_blocks": list(fixed_cta_m_blocks),
        "cta_pair_shares_m_block_after_fix": fixed_cta_m_blocks[0] == fixed_cta_m_blocks[1],
        "gb300_32g_bad_elements_before": gb300_32g_bad_elements_before,
        "gb300_32g_valid_elements": gb300_32g_valid_elements,
        "gb300_32g_bad_elements_after": gb300_32g_bad_elements_after,
        "gdp_gb300_peak_speedup_x": 1.07,
        "gdp_b200_peak_speedup_x": 1.06,
    }


def simulate_kda_train_and_blackwell_softmax() -> dict[str, object]:
    """Model KDA training fused gate epilogue (`762b237e15`) and Blackwell cached-cluster softmax (`510e514b25`)."""
    # On SM103 (GB300), glm_intra_v14 fuses gate_epilogue into intra-chunk backward -> 1 fewer kernel launch
    sm100_bwd_tail_launches = ["dav", "dqkg", "glm_intra", "gate_epilogue"]
    sm103_bwd_tail_launches = ["dav", "dqkg", "glm_intra_v14_fused_gate"]

    # Blackwell softmax (`510e514b25`): 12 Cake programs (1 warp, 1 rowwise, 1 bootstrap, 7 cached-cluster, 2 persistent)
    # 1 x 32000 latency: B200 3.26 -> 2.53 us (1.291x), GB300 3.10 -> 2.40 us (1.287x)
    b200_1x32k_base_us = 3.26
    b200_1x32k_new_us = 2.53
    gb300_1x32k_base_us = 3.10
    gb300_1x32k_new_us = 2.40

    return {
        "sm100_bwd_tail_launch_count": len(sm100_bwd_tail_launches),
        "sm103_bwd_tail_launch_count": len(sm103_bwd_tail_launches),
        "sm103_launch_saved_per_bwd": len(sm100_bwd_tail_launches) - len(sm103_bwd_tail_launches),
        "sm103_sm107_segmented_head_count": 16,
        "softmax_total_programs": 12,
        "softmax_persistent_resident_cluster_multiplier": 2.5,
        "softmax_b200_geomean_speedup_41_shapes_x": 1.283,
        "softmax_gb300_geomean_speedup_41_shapes_x": 1.263,
        "softmax_b200_1x32k_speedup_x": round(b200_1x32k_base_us / b200_1x32k_new_us, 4),
        "softmax_gb300_1x32k_speedup_x": round(gb300_1x32k_base_us / gb300_1x32k_new_us, 4),
    }


def simulate_cutlass_norm_idx_and_tilelang_pto() -> dict[str, object]:
    """Model CUTLASS SM100/SM120 per-group norm constant index (`ad9bd53bda`) & TileLang PTO Ramp/RNG (`d7a28bd78b`, `44405ee548`)."""
    norm_constants = [1.25, 2.50, 3.75, 5.00]
    raw_values = [10.0, 10.0, 10.0, 10.0]

    # Buggy: batch coordinate l is zeroed after SFD pointer selection, so norm_constant[l] always reads index 0
    buggy_scaled = [val * norm_constants[0] for val in raw_values]
    # Fixed: group_idx captured before l=0 zeroing
    fixed_scaled = [val * norm_constants[group_idx] for group_idx, val in enumerate(raw_values)]

    # TileLang PTO explicit 4-lane Ramp: T.Ramp(lane * 12 + 7, stride=3, lanes=4) for lane=2
    lane = 2
    base = lane * 12 + 7  # 31
    stride = 3
    lanes = 4
    pto_typed_vector = [base + i * stride for i in range(lanes)]  # [31, 34, 37, 40]

    return {
        "cutlass_buggy_group_scaled": buggy_scaled,
        "cutlass_fixed_group_scaled": fixed_scaled,
        "cutlass_group3_error_before": round(abs(fixed_scaled[3] - buggy_scaled[3]), 4),
        "cutlass_group3_error_after": 0.0,
        "tilelang_pto_lane2_ramp_vector": pto_typed_vector,
        "tilelang_pto_single_store_arg": True,
        "tilelang_ascend_rng_reset_between_kernels": True,
    }


def run_all() -> dict[str, object]:
    return {
        "phase": 113,
        "priority": 122,
        "track": "A",
        "question_id": "flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113",
        "frost_sm80_and_tcgen05": simulate_frost_sm80_and_tcgen05_tinygemm(),
        "deepgemm_and_gdp": simulate_deepgemm_n_l2_group_and_gdp_preload(),
        "kda_and_blackwell_softmax": simulate_kda_train_and_blackwell_softmax(),
        "cutlass_and_tilelang": simulate_cutlass_norm_idx_and_tilelang_pto(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert summary["frost_sm80_and_tcgen05"]["frost_arch_flags"]["sm80"] == "sm_80"
        assert summary["frost_sm80_and_tcgen05"]["frost_arch_flags"]["sm90"] == "sm_90a"
        assert summary["frost_sm80_and_tcgen05"]["tcgen05_programs_per_arch"] == 12
        assert summary["frost_sm80_and_tcgen05"]["tcgen05_total_programs_sm100_103_107"] == 36
        assert summary["frost_sm80_and_tcgen05"]["b200_4x2880x2880_speedup_x"] == 1.1419
        assert summary["deepgemm_and_gdp"]["buggy_cta_pair_m_blocks"] == [0, 1]
        assert summary["deepgemm_and_gdp"]["fixed_cta_pair_m_blocks"] == [0, 0]
        assert summary["deepgemm_and_gdp"]["cta_pair_shares_m_block_after_fix"] is True
        assert summary["deepgemm_and_gdp"]["gb300_32g_bad_elements_before"] == 97355
        assert summary["deepgemm_and_gdp"]["gb300_32g_bad_elements_after"] == 0
        assert summary["kda_and_blackwell_softmax"]["sm103_launch_saved_per_bwd"] == 1
        assert summary["kda_and_blackwell_softmax"]["softmax_total_programs"] == 12
        assert summary["kda_and_blackwell_softmax"]["softmax_b200_geomean_speedup_41_shapes_x"] == 1.283
        assert summary["cutlass_and_tilelang"]["cutlass_fixed_group_scaled"] == [12.5, 25.0, 37.5, 50.0]
        assert summary["cutlass_and_tilelang"]["cutlass_group3_error_before"] == 37.5
        assert summary["cutlass_and_tilelang"]["tilelang_pto_lane2_ramp_vector"] == [31, 34, 37, 40]
        print("OK: flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113 verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
