"""Deterministic model for FlashInfer Cake GDP Nemotron 4 decode/MTP, NVFP4 4/6 slot-mapping append, and Cake causal conv1d backward routing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def l2norm(vec: list[float], eps: float = 1e-6) -> list[float]:
    norm_sq = sum(v * v for v in vec)
    inv = 1.0 / math.sqrt(norm_sq + eps)
    return [v * inv for v in vec]


def simulate_cake_gdp_step(
    state: list[list[float]],
    q: list[float],
    k_updates: list[list[float]],
    v_updates: list[list[float]],
    a_log: float,
    a_t: float,
    dt_bias: float,
    b_updates: list[float],
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    disable_state_update: bool = False,
) -> tuple[list[float], list[list[float]], dict[str, object]]:
    """Simulate one token step of FlashInfer PR #6883 Cake Gated DeltaProduct (GDP) recurrence.

    Recurrence over state S in R^{V x K}:
      alpha_t = exp(-exp(A_log) * softplus(a_t + dt_bias))
      beta_{t,j} = sigmoid(b_{t,j})
      S <- alpha_t * S
      for j in 0..n-1:
        k_hat = l2norm(k_{t,j}) if use_qk_l2norm else k_{t,j}
        u = S @ k_hat
        S <- S + (beta_{t,j} * (v_{t,j} - u)) outer k_hat
      q_hat = l2norm(q_t) if use_qk_l2norm else q_t
      out_t = scale * (S @ q_hat)
    """
    v_dim = len(state)
    k_dim = len(state[0])
    num_householder = len(k_updates)
    if scale is None:
        scale = k_dim ** -0.5

    alpha_t = math.exp(-math.exp(a_log) * softplus(a_t + dt_bias))
    working = [[alpha_t * val for val in row] for row in state]
    betas: list[float] = []
    residual_norms: list[float] = []

    for j in range(num_householder):
        beta_tj = sigmoid(b_updates[j])
        betas.append(round(beta_tj, 6))
        k_hat = l2norm(k_updates[j]) if use_qk_l2norm else list(k_updates[j])
        v_tj = v_updates[j]
        u = [
            sum(working[row_idx][col_idx] * k_hat[col_idx] for col_idx in range(k_dim))
            for row_idx in range(v_dim)
        ]
        delta_v = [beta_tj * (v_tj[row_idx] - u[row_idx]) for row_idx in range(v_dim)]
        residual_norms.append(round(math.sqrt(sum(d * d for d in delta_v)), 6))
        for row_idx in range(v_dim):
            dv = delta_v[row_idx]
            row = working[row_idx]
            for col_idx in range(k_dim):
                row[col_idx] += dv * k_hat[col_idx]

    q_hat = l2norm(q) if use_qk_l2norm else list(q)
    out_t = [
        scale * sum(working[row_idx][col_idx] * q_hat[col_idx] for col_idx in range(k_dim))
        for row_idx in range(v_dim)
    ]
    next_state = [list(row) for row in (state if disable_state_update else working)]
    meta = {
        "num_householder": num_householder,
        "alpha_t": round(alpha_t, 6),
        "betas": betas,
        "residual_l2_per_stage": residual_norms,
        "state_updated": not disable_state_update,
    }
    return out_t, next_state, meta


def route_cake_gdp_module(
    arch: str,
    batch_size: int,
    heads: int,
    num_tokens: int,
    num_householder: int = 3,
    *,
    use_qk_l2norm: bool = True,
) -> dict[str, object]:
    """Select the Cake GDP kernel body and launch grid (FlashInfer PR #6883)."""
    if arch not in ("sm_100a", "sm_103a"):
        raise ValueError(f"Unsupported Cake GDP arch: {arch}")
    if not use_qk_l2norm or num_householder not in (1, 2, 3, 4):
        raise ValueError("Unsupported Householder count or l2norm flag")
    if not (1 <= num_tokens <= 16):
        raise ValueError("Cake GDP supports 1 <= T <= 16")

    state_heads = batch_size * heads
    is_static_t = num_householder == 3 and num_tokens in (1, 4, 8)
    program_kind = f"static_t{num_tokens}" if is_static_t else "runtime_t_up_to_16"

    if num_tokens == 1 and state_heads <= 160:
        body = "splitv"
        variant = "splitv8_tile8"
        ctas_per_state = 8
        grid_x = state_heads * ctas_per_state
    else:
        body = "lanerow"
        if not is_static_t:
            variant = "l4r4"
        elif num_tokens == 8 and state_heads >= 640:
            variant = "l8r2"
        else:
            variant = "l8r4"
        ctas_per_state = 1
        grid_x = state_heads

    return {
        "arch": arch,
        "batch_size": batch_size,
        "heads": heads,
        "state_heads": state_heads,
        "num_tokens": num_tokens,
        "num_householder": num_householder,
        "body": body,
        "variant": variant,
        "program_kind": program_kind,
        "ctas_per_state": ctas_per_state,
        "grid_x": grid_x,
    }


E2M1_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def quantize_fp8_e4m3_positive(val: float) -> float:
    """Approximate positive FP8 E4M3 quantization used by NVFP4 block scale factors."""
    if val <= 0.0:
        return 0.0
    max_e4m3 = 448.0
    clamped = min(val, max_e4m3)
    exp = math.floor(math.log2(clamped))
    step = 2.0 ** (exp - 3)
    return round(clamped / step) * step


def nearest_e2m1_mag(scaled_abs: float) -> float:
    best = E2M1_LEVELS[0]
    best_diff = abs(scaled_abs - best)
    for level in E2M1_LEVELS[1:]:
        diff = abs(scaled_abs - level)
        if diff < best_diff:
            best = level
            best_diff = diff
    return best


def evaluate_nvfp4_block_4over6(
    block16: list[float],
    global_scale: float = 1.0,
) -> dict[str, object]:
    """Compare standard NVFP4 max=6 scale vs 4/6 MSE candidate scale (FlashInfer PR #6013)."""
    if len(block16) != 16:
        raise ValueError("NVFP4 block size must be 16")
    amax = max(abs(x) for x in block16)
    sf6 = quantize_fp8_e4m3_positive(amax / (6.0 * global_scale))
    sf4 = quantize_fp8_e4m3_positive(sf6 * 1.5)

    def block_mse(sf: float) -> float:
        denom = sf * global_scale
        if denom <= 0.0:
            return sum(x * x for x in block16) / 16.0
        err_sq = 0.0
        for x in block16:
            mag = nearest_e2m1_mag(abs(x) / denom)
            recon = math.copysign(mag * denom, x)
            err_sq += (recon - x) ** 2
        return err_sq / 16.0

    mse_6 = block_mse(sf6)
    mse_4 = block_mse(sf4)
    use_4over6 = mse_4 < mse_6
    selected_sf = sf4 if use_4over6 else sf6
    selected_mse = mse_4 if use_4over6 else mse_6
    return {
        "amax": round(amax, 6),
        "sf6": round(sf6, 6),
        "sf4": round(sf4, 6),
        "mse_standard_6": round(mse_6, 8),
        "mse_candidate_4": round(mse_4, 8),
        "selected_policy": "4over6" if use_4over6 else "standard_6",
        "selected_sf": round(selected_sf, 6),
        "selected_mse": round(selected_mse, 8),
        "mse_reduction_pct": round(
            (mse_6 - selected_mse) / mse_6 * 100.0 if mse_6 > 0 else 0.0,
            2,
        ),
    }


def route_cake_causal_conv1d_bwd(arch: str, channels: int) -> dict[str, object]:
    """Route Cake causal conv1d backward schedule by architecture and channels (FlashInfer PR #6930)."""
    if arch == "sm_103a":
        if 512 <= channels < 768:
            return {
                "arch": arch,
                "channels": channels,
                "fold_overlap": True,
                "fast_roll": True,
                "roll_prefetch": False,
                "smem_total_bytes": 66560,
                "band": "512<=C<768_fold_overlap_fast_roll",
            }
        return {
            "arch": arch,
            "channels": channels,
            "fold_overlap": False,
            "fast_roll": False,
            "roll_prefetch": False,
            "smem_total_bytes": 49152,
            "band": "default_sm103a",
        }
    if arch == "sm_107a":
        if 512 <= channels < 1024:
            return {
                "arch": arch,
                "channels": channels,
                "fold_overlap": True,
                "fast_roll": True,
                "roll_prefetch": False,
                "smem_total_bytes": 66560,
                "band": "512<=C<1024_fold_overlap_fast_roll",
            }
        if channels >= 2048:
            return {
                "arch": arch,
                "channels": channels,
                "fold_overlap": False,
                "fast_roll": True,
                "roll_prefetch": True,
                "smem_total_bytes": 82944,
                "band": "C>=2048_fast_roll_prefetch",
            }
        return {
            "arch": arch,
            "channels": channels,
            "fold_overlap": False,
            "fast_roll": True,
            "roll_prefetch": False,
            "smem_total_bytes": 66560,
            "band": "1024<=C<2048_fast_roll",
        }
    raise ValueError(f"Unsupported Cake conv1d arch: {arch}")


def build_summary() -> dict[str, object]:
    # 1. Cake GDP single-head step with n=3 Householder updates (V=4, K=8 micro-slice)
    init_state = [[0.05 * ((r + 1) * (c + 1)) for c in range(8)] for r in range(4)]
    q = [0.4, -0.2, 0.5, 0.1, -0.3, 0.25, 0.6, -0.15]
    k_updates = [
        [0.5, 0.1, -0.4, 0.2, 0.3, -0.5, 0.2, 0.1],
        [-0.3, 0.6, 0.2, -0.1, 0.4, 0.2, -0.5, 0.3],
        [0.2, -0.4, 0.5, 0.3, -0.2, 0.1, 0.4, -0.6],
    ]
    v_updates = [
        [0.8, -0.4, 0.3, 0.5],
        [-0.2, 0.7, -0.5, 0.4],
        [0.6, 0.1, 0.9, -0.3],
    ]
    out_active, state_active, meta_active = simulate_cake_gdp_step(
        init_state,
        q,
        k_updates,
        v_updates,
        a_log=-1.2,
        a_t=0.35,
        dt_bias=-0.1,
        b_updates=[0.2, -0.1, 0.5],
        use_qk_l2norm=True,
        disable_state_update=False,
    )
    out_readonly, state_readonly, _ = simulate_cake_gdp_step(
        init_state,
        q,
        k_updates,
        v_updates,
        a_log=-1.2,
        a_t=0.35,
        dt_bias=-0.1,
        b_updates=[0.2, -0.1, 0.5],
        use_qk_l2norm=True,
        disable_state_update=True,
    )

    gdp_routes = [
        route_cake_gdp_module("sm_100a", batch_size=1, heads=80, num_tokens=1, num_householder=3),
        route_cake_gdp_module("sm_100a", batch_size=16, heads=20, num_tokens=1, num_householder=3),
        route_cake_gdp_module("sm_100a", batch_size=256, heads=10, num_tokens=8, num_householder=3),
        route_cake_gdp_module("sm_103a", batch_size=64, heads=24, num_tokens=6, num_householder=4),
    ]

    # 2. NVFP4 4/6 vs standard 6 block quantization + PR #6204 .cpu() removal benchmark
    # Mapping amax (~6.0) onto E2M1 level 4.0 (sf4 = 1.5 * sf6) places exact levels at 4.5 and 2.25,
    # eliminating the widest E2M1 gap between 4.0 and 6.0.
    outlier_clustered_block = [
        5.95, 4.48, -4.52, 4.45, -4.51, 2.24, -2.26, 4.49,
        -4.47, 2.23, -2.25, 4.53, -4.46, 2.27, -4.50, 0.74,
    ]
    uniform_heavy_block = [
        5.95, -3.98, 1.99, -1.01, 0.51, -3.97, 2.02, -1.02,
        3.99, -2.01, 1.00, -0.49, 4.01, -1.98, 0.99, -0.50,
    ]
    eval_clustered = evaluate_nvfp4_block_4over6(outlier_clustered_block, global_scale=1.0)
    eval_uniform = evaluate_nvfp4_block_4over6(uniform_heavy_block, global_scale=1.0)

    sm80_before_ms = 499.4
    sm80_after_ms = 451.7
    sm80_delta_ms = round(sm80_before_ms - sm80_after_ms, 1)
    sm80_speedup = round(sm80_before_ms / sm80_after_ms, 4)

    # 3. Cake causal conv1d backward routing on sm_103a and sm_107a
    conv1d_routes = [
        route_cake_causal_conv1d_bwd("sm_103a", 512),
        route_cake_causal_conv1d_bwd("sm_103a", 1024),
        route_cake_causal_conv1d_bwd("sm_107a", 512),
        route_cake_causal_conv1d_bwd("sm_107a", 1536),
        route_cake_causal_conv1d_bwd("sm_107a", 2048),
    ]

    return {
        "source_commits": [
            "flashinfer-ai/flashinfer@fc1790f4ec33a4df84ae9e6d1dcc9231fde7fd10",
            "flashinfer-ai/flashinfer@52fbbd090ca9c85b69c3a94420e360d27779b366",
            "flashinfer-ai/flashinfer@3821440929ee342fe9f939dbf5186c3e667a851c",
            "flashinfer-ai/flashinfer@7c866e85d3bb8095ebb247cdca42a56c2a4cab19",
        ],
        "cake_gdp": {
            "step_meta": meta_active,
            "output_values": [round(x, 6) for x in out_active],
            "readonly_matches_active_output": [round(x, 6) for x in out_readonly]
            == [round(x, 6) for x in out_active],
            "readonly_preserves_initial_state": state_readonly == init_state,
            "active_mutates_state": state_active != init_state,
            "routes": gdp_routes,
            "gb200_benchmark": {
                "shapes_won": 48,
                "shapes_total": 48,
                "min_speedup_x": 1.01,
                "max_speedup_x": 3.99,
                "peak_shape": "HV=10,B=256,T=8",
                "baseline_us": 301.4,
                "cake_gdp_us": 75.5,
            },
            "gb300_benchmark": {
                "shapes_won": 47,
                "shapes_total": 48,
                "min_speedup_x": 0.98,
                "max_speedup_x": 3.95,
                "peak_shape": "HV=10,B=256,T=8",
                "baseline_us": 296.6,
                "cake_gdp_us": 75.1,
            },
        },
        "nvfp4_slot_append": {
            "clustered_block_eval": eval_clustered,
            "uniform_block_eval": eval_uniform,
            "default_jit_module": "page",
            "four_over_six_jit_module": "page_nvfp4_4over6",
            "unchanged_default_sass_kernels": 51,
            "sm80_eager_prefill_sync_removal": {
                "model": "Qwen3.6-35B-A3B-NVFP4 (20 layers, 4096 tokens)",
                "before_mean_ms": sm80_before_ms,
                "after_mean_ms": sm80_after_ms,
                "saved_ms": sm80_delta_ms,
                "reduction_pct": round(sm80_delta_ms / sm80_before_ms * 100.0, 2),
                "speedup_x": sm80_speedup,
            },
        },
        "cake_causal_conv1d_bwd": {
            "routes": conv1d_routes,
            "gb300_c512_geomean_speedup_x": 1.0164,
            "gb300_all14_geomean_speedup_x": 1.0094,
            "r200_routed11_geomean_speedup_x": 1.0331,
            "r200_all14_geomean_speedup_x": 1.0264,
            "r200_peak_shape": {
                "shape": "B=1,S=250340,C=512,K=4",
                "before_us": 101.3,
                "after_us": 96.4,
                "speedup_x": 1.0513,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert summary["cake_gdp"]["readonly_matches_active_output"] is True
        assert summary["cake_gdp"]["readonly_preserves_initial_state"] is True
        assert summary["cake_gdp"]["active_mutates_state"] is True
        routes = summary["cake_gdp"]["routes"]
        assert routes[0]["body"] == "splitv" and routes[0]["grid_x"] == 640
        assert routes[1]["body"] == "lanerow" and routes[1]["variant"] == "l8r4"
        assert routes[2]["variant"] == "l8r2" and routes[2]["program_kind"] == "static_t8"
        assert routes[3]["variant"] == "l4r4" and routes[3]["program_kind"] == "runtime_t_up_to_16"
        assert summary["nvfp4_slot_append"]["clustered_block_eval"]["selected_policy"] == "4over6"
        assert summary["nvfp4_slot_append"]["uniform_block_eval"]["selected_policy"] == "standard_6"
        assert summary["nvfp4_slot_append"]["sm80_eager_prefill_sync_removal"]["saved_ms"] == 47.7
        bwd_routes = summary["cake_causal_conv1d_bwd"]["routes"]
        assert bwd_routes[0]["fold_overlap"] is True and bwd_routes[0]["smem_total_bytes"] == 66560
        assert bwd_routes[1]["fold_overlap"] is False
        assert bwd_routes[4]["roll_prefetch"] is True and bwd_routes[4]["smem_total_bytes"] == 82944

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
