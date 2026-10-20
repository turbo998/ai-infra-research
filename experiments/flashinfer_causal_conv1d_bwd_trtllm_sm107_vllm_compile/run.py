from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FALLBACK_PROGRAMS: dict[str, str] = {
    "sm_100a": "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
    "sm_103a": "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
    "sm_107a": "cake_causal_conv1d_train_a93b636beb1d573942bf",
}
BIG_CHANNEL_PROGRAM = "cake_causal_conv1d_train_90351ecca30604957251"
BIG_CHANNEL_THRESHOLDS: dict[str, int] = {
    "sm_100a": 512,
    "sm_103a": 512,
    "sm_107a": 1024,
}
FALLBACK_CTAS_PER_SM: dict[str, int] = {
    "sm_100a": 1,
    "sm_103a": 1,
    "sm_107a": 2,
}
BIG_CHANNEL_CTAS_PER_SM = 2
SM_COUNTS: dict[str, int] = {
    "sm_100a": 148,
    "sm_103a": 160,
    "sm_107a": 144,
}


def select_conv1d_bwd_program(arch: str, channels: int) -> dict[str, Any]:
    if channels <= 0 or channels % 64 != 0:
        raise ValueError(f"C must be a positive multiple of 64, got {channels}")
    threshold = BIG_CHANNEL_THRESHOLDS[arch]
    is_big = channels >= threshold
    program = BIG_CHANNEL_PROGRAM if is_big else FALLBACK_PROGRAMS[arch]
    ctas_per_sm = BIG_CHANNEL_CTAS_PER_SM if is_big else FALLBACK_CTAS_PER_SM[arch]
    owner_ctas_per_sm = max(FALLBACK_CTAS_PER_SM[arch], BIG_CHANNEL_CTAS_PER_SM)
    sm_count = SM_COUNTS[arch]
    n_ch_tiles = max(1, math.ceil(channels / (128 * 8)))
    program_rows = math.ceil((ctas_per_sm * sm_count) / n_ch_tiles)
    workspace_owner_rows = math.ceil((owner_ctas_per_sm * sm_count) / n_ch_tiles)
    program_partial_bytes = program_rows * channels * 4 * 4
    public_workspace_bytes = workspace_owner_rows * channels * 4 * 4
    return {
        "arch": arch,
        "channels": channels,
        "selected_program": program,
        "routed_to_big_channel": is_big,
        "program_ctas_per_sm": ctas_per_sm,
        "workspace_owner_ctas_per_sm": owner_ctas_per_sm,
        "program_partial_rows": program_rows,
        "workspace_owner_rows": workspace_owner_rows,
        "program_partial_bytes": program_partial_bytes,
        "public_workspace_bytes": public_workspace_bytes,
    }


def evaluate_conv1d_bwd_benchmarks() -> dict[str, Any]:
    gb300_rows = [
        (239718, 512, 155.0, 145.3, 1.0671),
        (245028, 512, 156.4, 150.4, 1.0403),
        (249866, 512, 159.0, 150.6, 1.0557),
        (250340, 512, 160.5, 149.4, 1.0741),
        (258574, 512, 163.1, 154.3, 1.0570),
        (259042, 512, 164.6, 157.5, 1.0448),
        (259584, 512, 164.3, 155.7, 1.0550),
        (261393, 512, 167.3, 155.8, 1.0735),
        (250340, 128, 59.0, 59.0, 1.0000),
        (250340, 256, 94.8, 94.8, 1.0000),
        (250340, 1024, 294.0, 272.4, 1.0790),
        (250340, 2048, 584.0, 545.4, 1.0709),
        (250340, 4096, 1142.3, 1052.4, 1.0854),
        (250340, 8192, 2264.3, 2060.1, 1.0991),
    ]
    b200_rows = [
        (239718, 512, 171.0, 157.4, 1.0868),
        (245028, 512, 172.6, 159.2, 1.0839),
        (249866, 512, 177.8, 165.8, 1.0728),
        (250340, 512, 177.5, 162.9, 1.0895),
        (258574, 512, 181.6, 166.6, 1.0900),
        (259042, 512, 181.7, 168.4, 1.0790),
        (259584, 512, 183.3, 171.2, 1.0702),
        (261393, 512, 185.2, 167.9, 1.1033),
        (250340, 128, 63.3, 63.3, 1.0000),
        (250340, 256, 102.1, 102.1, 1.0000),
        (250340, 1024, 330.6, 294.5, 1.1226),
        (250340, 2048, 658.2, 611.0, 1.0772),
        (250340, 4096, 1289.5, 1179.7, 1.0931),
        (250340, 8192, 2570.5, 2315.7, 1.1100),
    ]
    r200_rows = [
        (239718, 512, 97.9, 97.9, 1.0000),
        (245028, 512, 99.4, 99.4, 1.0000),
        (249866, 512, 101.8, 101.8, 1.0000),
        (250340, 512, 101.3, 101.3, 1.0000),
        (258574, 512, 104.0, 104.0, 1.0000),
        (259042, 512, 104.6, 104.6, 1.0000),
        (259584, 512, 105.3, 105.3, 1.0000),
        (261393, 512, 105.3, 105.3, 1.0000),
        (250340, 128, 40.4, 40.4, 1.0000),
        (250340, 256, 58.5, 58.5, 1.0000),
        (250340, 1024, 190.3, 178.0, 1.0694),
        (250340, 2048, 428.3, 356.4, 1.2017),
        (250340, 4096, 836.0, 672.3, 1.2435),
        (250340, 8192, 1652.2, 1313.3, 1.2581),
    ]

    def summarize(arch: str, rows: list[tuple[int, int, float, float, float]]) -> dict[str, Any]:
        threshold = BIG_CHANNEL_THRESHOLDS[arch]
        all_ratios = [ratio for _, _, _, _, ratio in rows]
        routed_ratios = [ratio for _, c, _, _, ratio in rows if c >= threshold]
        geomean_all = math.exp(sum(math.log(r) for r in all_ratios) / len(all_ratios))
        geomean_routed = math.exp(
            sum(math.log(r) for r in routed_ratios) / len(routed_ratios)
        )
        c8192 = next(r for r in rows if r[1] == 8192)
        sel_8192 = select_conv1d_bwd_program(arch, 8192)
        bwd_logical_bytes_8192 = (
            6 * c8192[0] * c8192[1]
            + 32 * c8192[1]
            + 32 * c8192[1] * sel_8192["program_partial_rows"]
        )
        return {
            "arch": arch,
            "threshold_channels": threshold,
            "num_shapes_total": len(rows),
            "num_shapes_routed": len(routed_ratios),
            "geomean_speedup_all": round(geomean_all, 4),
            "geomean_speedup_routed": round(geomean_routed, 4),
            "max_speedup_routed": round(max(routed_ratios), 4),
            "c8192_prev_us": c8192[2],
            "c8192_new_us": c8192[3],
            "c8192_speedup": c8192[4],
            "c8192_bwd_logical_bytes": bwd_logical_bytes_8192,
        }

    return {
        "gb300_sm103a": summarize("sm_103a", gb300_rows),
        "b200_sm100a": summarize("sm_100a", b200_rows),
        "r200_sm107a": summarize("sm_107a", r200_rows),
        "route_checks": {
            "sm103a_c256": select_conv1d_bwd_program("sm_103a", 256),
            "sm103a_c512": select_conv1d_bwd_program("sm_103a", 512),
            "sm107a_c512": select_conv1d_bwd_program("sm_107a", 512),
            "sm107a_c1024": select_conv1d_bwd_program("sm_107a", 1024),
        },
    }


def evaluate_trtllm_sm107_dspark_and_vllm_compile() -> dict[str, Any]:
    dsv4_dspark_arch_by_sm = {100: "sm_100", 103: "sm_103"}
    dspark_rmsnorm_rope_sms = frozenset((100, 103, 107))
    sm_matrix: list[dict[str, Any]] = []
    for sm in (90, 100, 103, 107, 120):
        sm_matrix.append(
            {
                "sm_version": sm,
                "legacy_rmsnorm_rope_supported": sm in dsv4_dspark_arch_by_sm,
                "decoupled_rmsnorm_rope_supported": sm in dspark_rmsnorm_rope_sms,
                "dsv4_dspark_attention_arch": dsv4_dspark_arch_by_sm.get(sm),
            }
        )

    ernie_lm_compiled_modules = 2
    siglip_encoder_layers = 27
    vllm_compile_cases = {
        "compile_mm_encoder_false": {
            "compile_mm_encoder": False,
            "compiled_lm_modules": ernie_lm_compiled_modules,
            "compiled_vision_layers": 0,
            "total_compiled_modules": ernie_lm_compiled_modules,
        },
        "compile_mm_encoder_true": {
            "compile_mm_encoder": True,
            "compiled_lm_modules": ernie_lm_compiled_modules,
            "compiled_vision_layers": siglip_encoder_layers,
            "total_compiled_modules": ernie_lm_compiled_modules + siglip_encoder_layers,
            "dynamic_arg_dims": {
                "hidden_states": 1,
                "cu_seqlens": 0,
                "rotary_pos_emb": 0,
            },
        },
    }
    return {
        "trtllm_dspark_sm_matrix": sm_matrix,
        "vllm_paddleocr_vl_compile": vllm_compile_cases,
    }


def build_summary() -> dict[str, Any]:
    conv1d = evaluate_conv1d_bwd_benchmarks()
    trtllm_vllm = evaluate_trtllm_sm107_dspark_and_vllm_compile()
    return {
        "experiment": "flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile",
        "evidence_class": "own-offline",
        "flashinfer_cake_causal_conv1d_bwd": conv1d,
        "trtllm_sm107_dspark_and_vllm_mm_compile": trtllm_vllm,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    conv = summary["flashinfer_cake_causal_conv1d_bwd"]
    assert conv["gb300_sm103a"]["num_shapes_routed"] == 12
    assert conv["gb300_sm103a"]["c8192_speedup"] == 1.0991
    assert conv["b200_sm100a"]["max_speedup_routed"] == 1.1226
    assert conv["r200_sm107a"]["num_shapes_routed"] == 4
    assert conv["r200_sm107a"]["geomean_speedup_routed"] == 1.1908
    assert conv["r200_sm107a"]["c8192_speedup"] == 1.2581
    assert not conv["route_checks"]["sm103a_c256"]["routed_to_big_channel"]
    assert conv["route_checks"]["sm103a_c512"]["routed_to_big_channel"]
    assert not conv["route_checks"]["sm107a_c512"]["routed_to_big_channel"]
    assert conv["route_checks"]["sm107a_c1024"]["routed_to_big_channel"]
    assert (
        conv["route_checks"]["sm103a_c256"]["public_workspace_bytes"]
        == 2 * conv["route_checks"]["sm103a_c256"]["program_partial_bytes"]
    )

    tv = summary["trtllm_sm107_dspark_and_vllm_mm_compile"]
    sm107_row = next(
        r for r in tv["trtllm_dspark_sm_matrix"] if r["sm_version"] == 107
    )
    assert not sm107_row["legacy_rmsnorm_rope_supported"]
    assert sm107_row["decoupled_rmsnorm_rope_supported"]
    assert sm107_row["dsv4_dspark_attention_arch"] is None
    assert (
        tv["vllm_paddleocr_vl_compile"]["compile_mm_encoder_false"][
            "total_compiled_modules"
        ]
        == 2
    )
    assert (
        tv["vllm_paddleocr_vl_compile"]["compile_mm_encoder_true"][
            "total_compiled_modules"
        ]
        == 29
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    summary = build_summary()
    verify_summary(summary)

    out_paths = [base_dir / "summary.json", base_dir / "results" / "summary.json"]
    if args.check:
        for p in out_paths:
            existing = json.loads(p.read_text(encoding="utf-8"))
            verify_summary(existing)
            assert existing == summary, f"Mismatch in {p}"
        print("flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile: check passed")
        return

    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rendered, encoding="utf-8")
    print("flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile: generated summary.json")


if __name__ == "__main__":
    main()
