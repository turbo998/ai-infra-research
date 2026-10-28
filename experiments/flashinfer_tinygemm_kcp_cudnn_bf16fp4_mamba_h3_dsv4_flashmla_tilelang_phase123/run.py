"""Deterministic offline models for Phase 123 Track A:
1. FlashInfer CAKE `tinygemm2` `tcgen05` `PACKED_M64_N8` (`(16, 8)` output features x batch rows,
   `101,376 B` dynamic SMEM, Rubin `sm_107a` 212 SMs) & `input_features_multiple=64` 8-row GEMM routing
   (PR #7735, commit 1d622fdb1b053fc532069914bc9a6b498bd5f722).
2. FlashInfer CAKE KDA KCP (`cake_kda_kcp`) backward send-pool aliasing (`SHARED_POOLS`: `b1_send->c1_send`,
   `b2_send->c2_send`) & lazy ring hop allocation (`LAZY_POOLS = ("ring_out", "ring_in")`)
   (PR #7714, commit e4c7e4a6dcad639e89ad915509ea60b075b928f0).
3. FlashInfer Rubin (`SM107`) AOT registration of `gen_cutlass_fused_moe_sm100_module` &
   `gen_gemm_sm100_module_cutlass_fp8` (eliminating `789 s` runtime JIT vs `1200 s` GSM8K deadline)
   (PR #6795, commit 4d0f564ed4e6b136c7f36025b35cfb80175b7ce2).
4. FlashInfer `_reuse_cudnn_prefill_preparation` (`_MAX_CUDNN_PREPARED_CACHE_ENTRIES = 4`) &
   `_rebind_prepared_prefill_state` for alternating prefill geometries on B200
   (PR #7720, commit 424946237f07097f9ac9a5d17dc1ee2806aff564).
5. FlashInfer CAKE Blackwell `bf16_fp4` (`cake_blackwell_bf16_fp4`) CuTe-layout `M=128` 2-CTA pair-tile
   (`768x2112x2048`) pre-cluster-barrier weight/scale load on GB300 & B200
   (PR #7732, commit e99087620fbc273b63920dd75e005096c2fac218).
6. FlashInfer CAKE Mamba2 `selective_state_update` (`sm_100a` 148-SM B200) batch-64 `64h/8g` z-gated
   `mtp_short_ring_heads64_groups8_t1_stp_z` dispatch
   (PR #7724, commit 6bac9852a9cc8a227b129ad55816c26d37f02855).
7. FlashInfer CAKE MiniMax-H3 (`cake_minimax_h3`) SM120 NVFP4 varlen attention warp-specialized
   `ATTENTION_WS_NODELTA_LEAD` / `ATTENTION_WS_NODELTA_TAIL` (`384` threads = 8 QK/softmax + 4 PV warps,
   `80,896 B` dynamic SMEM, `13 -> 15` programs) on 188-SM RTX PRO 6000
   (PR #7730, commit 4b948ed898e295dd1b9c49f35921b65defc842b3) + `cake_dsv4` sparse-MLA follow-up
   (PR #7729, commit 1531ff913edece38f993b5df6e72f243d10afc16).
8. DeepSeek FlashMLA SM100 sparse decode `head64` `NamedBarrier::arrive_and_wait(128, wg0_sync)`
   before `rowwise_max_buf[idx_in_warpgroup] = li` to eliminate cross-lane (`idx ^ 64`) WAR race
   (PR #241, commit 395cb3afbb331fef7b1b21e9340861b9436b42d8).
9. TileLang compiler & lowering fixes: `scalar_read_coeff` separation from 16-bit `DataType::lanes()`
   (PR #3453, commit 0573808d0c0d9d80e904104843155f6f495df371), `CUtensorMap` `desc_name_map`
   declaration verification (PR #3484, commit 10866657c0f7b96aca1ee162c792866de4323941),
   CUDA `AtomicAddx4` `float*` constraint (PR #3369, commit fab9a39f46e5a2a5fa10028193df636491e2bc10),
   `gemm_sp` `K % micro_size_k == 0` guard (PR #3367, commit fca340e5c2ce6263555ae5fdffe5c6fa90f0735b),
   and ROCm hipify cleanup (PR #3486, commit d86e5d473742977cb0047d3d15267e24d703bad6).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUMMARY_PATH = Path(__file__).with_name("summary.json")


def model_tinygemm2_packed_m64_n8_and_kcp_pools() -> dict[str, Any]:
    """Model FlashInfer tinygemm2 PACKED_M64_N8 (#7735) and KDA KCP pool sharing (#7714)."""
    # 1. tinygemm2 PACKED_M64_N8 on Rubin sm_107a (212 SMs)
    rubin_sm_count = 212
    packed_m64_n8_tile = (16, 8)  # (output_features_per_cta, batch_rows)
    packed_m64_n8_smem_bytes = 101376
    input_features_multiple = 64

    # Window for 8-row unbiased GEMMs routed to PACKED_M64_N8 vs M64_N8_2CTA / M128_N8_2CTA:
    # (N == 1536 and K in [2048, 3072]) or (N == 2048 and K in [2048, 2560]) or (N == 4096 and K in [3072, 6144])
    def select_8row_program(arch: str, m: int, n: int, k: int, has_bias: bool) -> str:
        if (
            arch == "sm_107a"
            and m == 8
            and not has_bias
            and k % input_features_multiple == 0
            and (
                (n == 1536 and 2048 <= k <= 3072)
                or (n == 2048 and 2048 <= k <= 2560)
                or (n == 4096 and 3072 <= k <= 6144)
            )
        ):
            return "PACKED_M64_N8"
        if n % 128 == 0:
            return "M128_N8_2CTA"
        return "M64_N8_2CTA"

    # CTA count for (M=8, N=4096, K=4096):
    # M128_N8_2CTA launches 2 * (4096 / 128) = 64 CTAs (32 2-CTA clusters);
    # PACKED_M64_N8 launches 4096 / 16 = 256 single-CTA blocks across 212 SMs without cluster sync overhead.
    ctas_m128_2cta_4096 = 2 * (4096 // 128)
    ctas_packed_m64_n8_4096 = 4096 // packed_m64_n8_tile[0]
    r200_8x4096x4096_before_us = 6.24
    r200_8x4096x4096_after_us = 5.98
    r200_8x4096x4096_speedup = round(r200_8x4096x4096_before_us / r200_8x4096x4096_after_us, 4)

    probe_shapes = [
        ("sm_107a", 8, 1536, 2048, False),
        ("sm_107a", 8, 2048, 2560, False),
        ("sm_107a", 8, 4096, 4096, False),
        ("sm_107a", 8, 4096, 4096, True),   # biased -> stays on 2CTA
        ("sm_107a", 8, 4096, 2048, False),  # outside K window -> stays on 2CTA
        ("sm_100a", 8, 4096, 4096, False),  # Blackwell B200 -> stays on 2CTA
    ]
    routed_programs = [select_8row_program(*s) for s in probe_shapes]

    # 2. cake_kda_kcp pool sharing & lazy ring allocation (#7714)
    all_pool_names = (
        "c1_send",
        "c1_recv",
        "c2_send",
        "c2_recv",
        "b1_send",
        "b1_recv",
        "b2_send",
        "b2_recv",
        "ring_out",
        "ring_in",
    )
    shared_pools = {"b1_send": "c1_send", "b2_send": "c2_send"}
    lazy_pools = ("ring_out", "ring_in")

    # Distinct underlying storage buffers allocated per (device, dtype) pool entry:
    buffers_before = len(all_pool_names)  # 10 distinct storages allocated eagerly
    buffers_after_all_to_all = len(all_pool_names) - len(shared_pools) - len(lazy_pools)  # 6 storages
    buffers_after_ring = len(all_pool_names) - len(shared_pools)  # 8 storages

    # Example buffer footprint for B=2, H=32, C=64, D=128, V=128, BF16 (2 B):
    # Each send/recv/ring buffer holds state chunk [B, H, D, V] = 2 * 32 * 128 * 128 * 2 = 2,097,152 B (2.0 MiB)
    per_buffer_mib = (2 * 32 * 128 * 128 * 2) / (1024.0 * 1024.0)
    saved_mib_all_to_all = round((buffers_before - buffers_after_all_to_all) * per_buffer_mib, 2)
    saved_mib_ring = round((buffers_before - buffers_after_ring) * per_buffer_mib, 2)

    return {
        "rubin_sm_count": rubin_sm_count,
        "packed_m64_n8_tile": list(packed_m64_n8_tile),
        "packed_m64_n8_smem_bytes": packed_m64_n8_smem_bytes,
        "ctas_m128_2cta_4096": ctas_m128_2cta_4096,
        "ctas_packed_m64_n8_4096": ctas_packed_m64_n8_4096,
        "r200_8x4096x4096_before_us": r200_8x4096x4096_before_us,
        "r200_8x4096x4096_after_us": r200_8x4096x4096_after_us,
        "r200_8x4096x4096_speedup": r200_8x4096x4096_speedup,
        "rerouted_8row_cases_count": 53,
        "rerouted_speedup_range": [1.01, 1.44],
        "routed_programs": routed_programs,
        "kcp_buffers_before": buffers_before,
        "kcp_buffers_after_all_to_all": buffers_after_all_to_all,
        "kcp_buffers_after_ring": buffers_after_ring,
        "kcp_saved_buffers_all_to_all": buffers_before - buffers_after_all_to_all,
        "kcp_saved_buffers_ring": buffers_before - buffers_after_ring,
        "kcp_saved_mib_all_to_all": saved_mib_all_to_all,
        "kcp_saved_mib_ring": saved_mib_ring,
    }


def model_rubin_aot_and_cudnn_prefill_cache() -> dict[str, Any]:
    """Model Rubin SM107 AOT registration (#6795) and cuDNN prefill preparation cache (#7720)."""
    # 1. Rubin SM107 AOT registration (#6795)
    # Without AOT registration for has_sm107, vLLM DeepSeek-R1 FP8 GSM8K triggers a 789s runtime JIT
    # inside a 1200s startup timeout, leaving insufficient time for server init + eval.
    gsm8k_timeout_s = 1200.0
    rubin_runtime_jit_s = 789.0
    server_init_and_eval_s = 465.0
    total_time_without_aot_s = rubin_runtime_jit_s + server_init_and_eval_s
    total_time_with_aot_s = server_init_and_eval_s
    timed_out_without_aot = total_time_without_aot_s > gsm8k_timeout_s
    passes_with_aot = total_time_with_aot_s <= gsm8k_timeout_s
    startup_speedup = round(total_time_without_aot_s / total_time_with_aot_s, 4)

    # 2. _reuse_cudnn_prefill_preparation (#7720)
    max_cudnn_prepared_cache_entries = 4
    # B200 alternating geometry switch (first layer host time in microseconds):
    b1_first_layer_before_us = 121.69
    b1_first_layer_after_us = 74.89
    b1_saved_us = round(b1_first_layer_before_us - b1_first_layer_after_us, 2)
    b1_speedup = round(b1_first_layer_before_us / b1_first_layer_after_us, 4)

    b3_first_layer_before_us = 117.31
    b3_first_layer_after_us = 69.64
    b3_saved_us = round(b3_first_layer_before_us - b3_first_layer_after_us, 2)
    b3_speedup = round(b3_first_layer_before_us / b3_first_layer_after_us, 4)

    # Simulate LRU cache of capacity 4 across 8 alternating steps of 3 distinct prefill geometries:
    sequence_keys = ["geom_A", "geom_B", "geom_C", "geom_A", "geom_B", "geom_C", "geom_A", "geom_B"]
    cache_single: list[str] = []
    hits_single = 0
    cache_four: list[str] = []
    hits_four = 0
    for key in sequence_keys:
        if key in cache_single:
            hits_single += 1
        else:
            cache_single = [key]
        if key in cache_four:
            hits_four += 1
            cache_four.remove(key)
            cache_four.append(key)
        else:
            if len(cache_four) >= max_cudnn_prepared_cache_entries:
                cache_four.pop(0)
            cache_four.append(key)

    return {
        "gsm8k_timeout_s": gsm8k_timeout_s,
        "rubin_runtime_jit_s": rubin_runtime_jit_s,
        "total_time_without_aot_s": total_time_without_aot_s,
        "total_time_with_aot_s": total_time_with_aot_s,
        "timed_out_without_aot": timed_out_without_aot,
        "passes_with_aot": passes_with_aot,
        "startup_speedup": startup_speedup,
        "max_cudnn_prepared_cache_entries": max_cudnn_prepared_cache_entries,
        "b1_first_layer_before_us": b1_first_layer_before_us,
        "b1_first_layer_after_us": b1_first_layer_after_us,
        "b1_saved_us": b1_saved_us,
        "b1_speedup": b1_speedup,
        "b3_first_layer_before_us": b3_first_layer_before_us,
        "b3_first_layer_after_us": b3_first_layer_after_us,
        "b3_saved_us": b3_saved_us,
        "b3_speedup": b3_speedup,
        "alternating_hits_single_slot": hits_single,
        "alternating_hits_four_slots": hits_four,
    }


def model_bf16_fp4_mamba_minimax_h3_dsv4() -> dict[str, Any]:
    """Model Blackwell bf16_fp4 (#7732), Mamba SSU (#7724), MiniMax-H3 SM120 (#7730), and DSV4 (#7729)."""
    # 1. cake_blackwell_bf16_fp4 M=128 2-CTA pair-tile pre-cluster-barrier load (#7732)
    gb300_768x2112x2048_before_us = 9.264
    gb300_768x2112x2048_after_us = 9.149
    gb300_bf16_fp4_speedup = round(gb300_768x2112x2048_before_us / gb300_768x2112x2048_after_us, 4)
    b200_768x2112x2048_before_us = 10.101
    b200_768x2112x2048_after_us = 9.992
    b200_bf16_fp4_speedup = round(b200_768x2112x2048_before_us / b200_768x2112x2048_after_us, 4)

    # 2. cake_selective_state_update sm_100a batch-64 64h/8g z-gated (#7724)
    b200_ssu_z_before_us = 40.440
    b200_ssu_z_after_us = 38.368
    b200_ssu_z_speedup = round(b200_ssu_z_before_us / b200_ssu_z_after_us, 4)
    b200_ssu_z_softplus_before_us = 40.480
    b200_ssu_z_softplus_after_us = 38.472
    b200_ssu_z_softplus_speedup = round(b200_ssu_z_softplus_before_us / b200_ssu_z_softplus_after_us, 4)

    # 3. cake_minimax_h3 SM120 NVFP4 warp-specialized lead/tail (#7730)
    # Work heuristic: total_sq_work = sum(L_i^2) in [2^23, 24576^2] and weighted_mean_len <= 40960
    min_work_sq = 1 << 23  # 8,388,608 (e.g., 2 * 2048^2)
    max_work_sq = 24576 * 24576  # 603,979,776
    max_weighted_mean_len = 40960.0

    def use_ws_nodelta_sm120(lengths: list[int], compensated: bool) -> bool:
        if compensated or not lengths:
            return False
        work_sq = sum(x * x for x in lengths)
        if work_sq < min_work_sq or work_sq > max_work_sq:
            return False
        weighted_mean = sum((x * x) * x for x in lengths) / float(work_sq)
        return weighted_mean <= max_weighted_mean_len

    rtx6000_single_4096_before_us = 603.01
    rtx6000_single_4096_after_us = 579.25
    rtx6000_single_4096_speedup = round(rtx6000_single_4096_before_us / rtx6000_single_4096_after_us, 4)
    rtx6000_seg3_ragged_before_us = 688.10
    rtx6000_seg3_ragged_after_us = 671.70
    rtx6000_seg3_ragged_speedup = round(rtx6000_seg3_ragged_before_us / rtx6000_seg3_ragged_after_us, 4)

    # 4. cake_dsv4 sparse-MLA follow-up (#7729)
    dsv4_total_shapes = 197
    dsv4_gb300_faster = 89
    dsv4_b200_faster = 36

    return {
        "gb300_bf16_fp4_speedup": gb300_bf16_fp4_speedup,
        "b200_bf16_fp4_speedup": b200_bf16_fp4_speedup,
        "b200_ssu_z_speedup": b200_ssu_z_speedup,
        "b200_ssu_z_softplus_speedup": b200_ssu_z_softplus_speedup,
        "minimax_h3_ws_threads": 384,
        "minimax_h3_qk_warps": 8,
        "minimax_h3_pv_warps": 4,
        "minimax_h3_dynamic_smem_bytes": 80896,
        "minimax_h3_programs_before": 13,
        "minimax_h3_programs_after": 15,
        "ws_single_4096_selected": use_ws_nodelta_sm120([4096], False),
        "ws_short_1024_selected": use_ws_nodelta_sm120([1024], False),
        "ws_compensated_selected": use_ws_nodelta_sm120([4096], True),
        "ws_32768_selected": use_ws_nodelta_sm120([32768], False),
        "rtx6000_single_4096_speedup": rtx6000_single_4096_speedup,
        "rtx6000_seg3_ragged_speedup": rtx6000_seg3_ragged_speedup,
        "dsv4_total_shapes": dsv4_total_shapes,
        "dsv4_gb300_faster": dsv4_gb300_faster,
        "dsv4_b200_faster": dsv4_b200_faster,
        "dsv4_slower_shapes": 0,
    }


def model_flashmla_war_barrier_and_tilelang_guards() -> dict[str, Any]:
    """Model FlashMLA SM100 sparse decode head64 WAR barrier (#241) and TileLang guards (#3453, #3484, #3369, #3367)."""
    # 1. FlashMLA SM100 sparse decode head64 WAR race (#241)
    # Inwarpgroup cross-half reduction reads plan.rowwise_max_buf[idx_in_warpgroup ^ 64]
    # and then writes plan.rowwise_max_buf[idx_in_warpgroup] = li.
    # Without NamedBarrier::arrive_and_wait(128, wg0_sync) between the read and write,
    # warps 0..1 (lanes 0..63) can overwrite rowwise_max_buf[0..63] with `li` before warps 2..3 (lanes 64..127)
    # finish reading rowwise_max_buf[idx ^ 64] (which is 0..63).
    old_max_lane0 = 6.0
    old_max_lane64 = 8.0
    true_row_max = max(old_max_lane0, old_max_lane64)
    premature_li_lane0 = 128.0  # denominator sum li overwritten into rowwise_max_buf[0]

    # Without barrier when warp 0 runs ahead of warp 2:
    corrupted_max_lane64 = max(old_max_lane64, premature_li_lane0)
    # With NamedBarrier::arrive_and_wait(128, wg0_sync):
    synced_max_lane64 = max(old_max_lane64, old_max_lane0)
    war_error_without_barrier = round(abs(corrupted_max_lane64 - true_row_max), 4)
    war_error_with_barrier = round(abs(synced_max_lane64 - true_row_max), 4)

    # 2. TileLang scalar_read_coeff separation from 16-bit DataType::lanes() (#3453)
    # Previously coeff was multiplied into DataType::lanes() (uint16_t / int16_t, max 32767 or 65535).
    # For a buffer with stride coeff = 65536 and vec_lanes = 4, lanes = 262,144 overflows 16-bit storage.
    coeff = 65536
    vec_lanes = 4
    int16_max = 32767
    overflows_int16_lanes = (coeff * vec_lanes) > int16_max
    preserved_scalar_read_coeff = coeff
    preserved_vec_lanes = vec_lanes

    # 3. TileLang CUtensorMap desc_name_map verification (#3484), AtomicAddx4 float* (#3369), gemm_sp K (#3367)
    desc_calls = [
        {"var": "b_desc", "has_cutensormap_decl": True},
        {"var": "tmp_ptr", "has_cutensormap_decl": False},
    ]
    recorded_tma_descs = [d["var"] for d in desc_calls if d["has_cutensormap_decl"]]

    atomic_addx4_valid_dtypes = [
        dt for dt in ("float32", "float16", "bfloat16") if dt == "float32"
    ]
    gemm_sp_k_tests = [
        {"K": 64, "micro_size_k": 32, "valid": (64 % 32 == 0)},
        {"K": 48, "micro_size_k": 32, "valid": (48 % 32 == 0)},
    ]

    return {
        "true_row_max": true_row_max,
        "corrupted_max_lane64": corrupted_max_lane64,
        "synced_max_lane64": synced_max_lane64,
        "war_error_without_barrier": war_error_without_barrier,
        "war_error_with_barrier": war_error_with_barrier,
        "overflows_int16_lanes": overflows_int16_lanes,
        "preserved_scalar_read_coeff": preserved_scalar_read_coeff,
        "preserved_vec_lanes": preserved_vec_lanes,
        "recorded_tma_descs": recorded_tma_descs,
        "atomic_addx4_valid_dtypes": atomic_addx4_valid_dtypes,
        "gemm_sp_k_tests": gemm_sp_k_tests,
    }


def build_summary() -> dict[str, Any]:
    tinygemm_kcp = model_tinygemm2_packed_m64_n8_and_kcp_pools()
    aot_cudnn = model_rubin_aot_and_cudnn_prefill_cache()
    kernels = model_bf16_fp4_mamba_minimax_h3_dsv4()
    flashmla_tilelang = model_flashmla_war_barrier_and_tilelang_guards()
    return {
        "experiment": "flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123",
        "phase": "Phase 123",
        "priority": 132,
        "tinygemm2_and_kcp_pools": tinygemm_kcp,
        "rubin_aot_and_cudnn_prefill_cache": aot_cudnn,
        "bf16_fp4_mamba_minimax_h3_dsv4": kernels,
        "flashmla_war_and_tilelang_guards": flashmla_tilelang,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    tk = summary["tinygemm2_and_kcp_pools"]
    assert tk["packed_m64_n8_tile"] == [16, 8]
    assert tk["packed_m64_n8_smem_bytes"] == 101376
    assert tk["r200_8x4096x4096_speedup"] == 1.0435
    assert tk["routed_programs"][:3] == ["PACKED_M64_N8", "PACKED_M64_N8", "PACKED_M64_N8"]
    assert tk["routed_programs"][3:] == ["M128_N8_2CTA", "M128_N8_2CTA", "M128_N8_2CTA"]
    assert tk["kcp_saved_buffers_all_to_all"] == 4
    assert tk["kcp_saved_buffers_ring"] == 2
    assert tk["kcp_saved_mib_all_to_all"] == 8.0

    ac = summary["rubin_aot_and_cudnn_prefill_cache"]
    assert ac["timed_out_without_aot"] is True
    assert ac["passes_with_aot"] is True
    assert ac["max_cudnn_prepared_cache_entries"] == 4
    assert ac["b1_saved_us"] == 46.80
    assert ac["b3_saved_us"] == 47.67
    assert ac["alternating_hits_single_slot"] == 0
    assert ac["alternating_hits_four_slots"] == 5

    kr = summary["bf16_fp4_mamba_minimax_h3_dsv4"]
    assert kr["gb300_bf16_fp4_speedup"] == 1.0126
    assert kr["b200_ssu_z_speedup"] == 1.0540
    assert kr["minimax_h3_programs_after"] == 15
    assert kr["ws_single_4096_selected"] is True
    assert kr["ws_short_1024_selected"] is False
    assert kr["ws_compensated_selected"] is False
    assert kr["ws_32768_selected"] is False
    assert kr["dsv4_slower_shapes"] == 0

    ft = summary["flashmla_war_and_tilelang_guards"]
    assert ft["war_error_without_barrier"] == 120.0
    assert ft["war_error_with_barrier"] == 0.0
    assert ft["overflows_int16_lanes"] is True
    assert ft["preserved_scalar_read_coeff"] == 65536
    assert ft["recorded_tma_descs"] == ["b_desc"]
    assert ft["atomic_addx4_valid_dtypes"] == ["float32"]
    assert ft["gemm_sp_k_tests"][0]["valid"] is True
    assert ft["gemm_sp_k_tests"][1]["valid"] is False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic output against summary.json")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    if args.check:
        existing = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch with committed summary.json"
        print("PASS: flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123 (--check)")
        return

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
