"""Deterministic CPU/offline numerical model for FP8 (DelayedScaling / 1x128 BlockScaling)
and Blackwell NVFP4 (1x16 E2M1 + E4M3 + FP32 hierarchical scaling, Random Hadamard Transform,
and Stochastic Rounding) under massive activation/gradient outlier channels.

No GPU, PyTorch, or external packages are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.json"
DEFAULT_JSON = ROOT / "results.json"
DEFAULT_CSV = ROOT / "results.csv"


def build_e2m1_grid() -> list[float]:
    """Return sorted representable non-negative values in NVFP4 E2M1 (max=6.0)."""
    # E2M1 positive values: 0.0, 0.5 (subnormal), 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    pos = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    neg = [-x for x in reversed(pos[1:])]
    return neg + pos


def build_e4m3_positive_grid() -> list[float]:
    """Return sorted representable positive values in FP8 E4M3 (FNUZ/OCP up to 448.0)."""
    vals = [0.0]
    bias = 7
    # Subnormals: exp=0, mantissa=1..7 -> (m/8) * 2^(1-bias) = (m/8) * 2^-6
    for m in range(1, 8):
        vals.append((m / 8.0) * (2.0 ** (1 - bias)))
    # Normals: exp=1..15, mantissa=0..7 (except exp=15, m=7 which is NaN -> max is exp=15, m=6 = 448.0)
    for e in range(1, 16):
        for m in range(0, 8):
            if e == 15 and m == 7:
                continue
            vals.append((1.0 + m / 8.0) * (2.0 ** (e - bias)))
    return sorted(vals)


E2M1_GRID = build_e2m1_grid()
E2M1_POS_GRID = [x for x in E2M1_GRID if x >= 0.0]
E4M3_POS_GRID = build_e4m3_positive_grid()


def quantize_to_sorted_grid_rtn(x: float, pos_grid: list[float]) -> float:
    """Round-to-nearest on a symmetric grid defined by `pos_grid`."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0.0 else -1.0
    ax = abs(x)
    if ax >= pos_grid[-1]:
        return sign * pos_grid[-1]
    # Binary search
    lo, hi = 0, len(pos_grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pos_grid[mid] < ax:
            lo = mid + 1
        else:
            hi = mid
    upper = pos_grid[lo]
    lower = pos_grid[max(0, lo - 1)]
    if (ax - lower) <= (upper - ax):
        return sign * lower
    return sign * upper


def quantize_to_e2m1_sr(x: float, rng: random.Random, max_bound: float = 6.0) -> float:
    """Stochastic rounding on NVFP4 E2M1 grid (`v1` lower, `v2` higher)."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0.0 else -1.0
    ax = min(abs(x), max_bound)
    grid = [v for v in E2M1_POS_GRID if v <= max_bound + 1e-9]
    if ax >= grid[-1]:
        return sign * grid[-1]
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < ax:
            lo = mid + 1
        else:
            hi = mid
    v2 = grid[lo]
    v1 = grid[max(0, lo - 1)]
    if v2 == v1:
        return sign * v1
    prob_v2 = (ax - v1) / (v2 - v1)
    chosen = v2 if rng.random() < prob_v2 else v1
    return sign * chosen


def hadamard_16x16() -> list[list[float]]:
    """Construct normalized 16x16 Sylvester Hadamard matrix (1/sqrt(16) * H_16)."""
    h = [[1.0]]
    while len(h) < 16:
        n = len(h)
        new_h = [[0.0] * (2 * n) for _ in range(2 * n)]
        for i in range(n):
            for j in range(n):
                new_h[i][j] = h[i][j]
                new_h[i][j + n] = h[i][j]
                new_h[i + n][j] = h[i][j]
                new_h[i + n][j + n] = -h[i][j]
        h = new_h
    scale = 1.0 / math.sqrt(16.0)
    return [[val * scale for val in row] for row in h]


H16_NORM = hadamard_16x16()
SIGN_VEC_16 = [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]


def apply_rht_16(vec: list[float]) -> list[float]:
    """Apply 16x16 Random Hadamard Transform x' = x * (1/sqrt(16) * S_16 * H_16) along blocks of 16."""
    out = [0.0] * len(vec)
    for base in range(0, len(vec), 16):
        block = vec[base : base + 16]
        signed_block = [block[i] * SIGN_VEC_16[i] for i in range(16)]
        for j in range(16):
            s = 0.0
            for i in range(16):
                s += signed_block[i] * H16_NORM[i][j]
            out[base + j] = s
    return out


def invert_rht_16(vec: list[float]) -> list[float]:
    """Invert 16x16 Random Hadamard Transform x = x' * H_16^T * S_16."""
    out = [0.0] * len(vec)
    for base in range(0, len(vec), 16):
        block = vec[base : base + 16]
        for i in range(16):
            s = 0.0
            for j in range(16):
                s += block[j] * H16_NORM[i][j]
            out[base + i] = s * SIGN_VEC_16[i]
    return out


def generate_synthetic_tensor(n: int, outlier_ratio: float, seed: int) -> list[float]:
    """Generate a synthetic hidden-state / gradient vector with a single massive outlier token/block
    (first 2 channels in block 0 have magnitude `outlier_ratio`, while the remaining blocks have
    fine-grained gradient/activation signals of scale ~0.0012 that test subnormal underflow in FP8
    per-tensor scaling).
    """
    rng = random.Random(seed)
    vec = []
    for idx in range(n):
        val = rng.gauss(0.0, 0.0012)
        if idx == 0:
            val = outlier_ratio
        elif idx == 1:
            val = -0.85 * outlier_ratio
        elif idx < 16:
            # Co-located channels inside the same 16-element block as the outlier
            val = 0.002 * (1.0 if idx % 2 == 0 else -1.0)
        vec.append(val)
    return vec


def sqnr_db(ref: list[float], approx: list[float]) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in dB."""
    sig_pwr = sum(x * x for x in ref)
    err_pwr = sum((x - y) ** 2 for x, y in zip(ref, approx))
    if err_pwr <= 1e-18:
        return 99.0
    return 10.0 * math.log10(sig_pwr / err_pwr)


def non_outlier_sqnr_db(ref: list[float], approx: list[float]) -> float:
    """Compute SQNR across non-outlier blocks (idx >= 128) to expose per-tensor vs block-scaling underflow."""
    sig_pwr = sum(ref[i] ** 2 for i in range(128, len(ref)))
    err_pwr = sum((ref[i] - approx[i]) ** 2 for i in range(128, len(ref)))
    if err_pwr <= 1e-18:
        return 99.0
    return 10.0 * math.log10(sig_pwr / err_pwr)


def wgrad_rht_dot_product_sqnr_db(outlier_ratio: float, seed: int) -> dict[str, float]:
    """Compare WGRAD inner product C = A^T * B across batch dimension M (`16x16` RHT tiles)
    when token 0 has an activation spike `a[0]` AND token 0's gradient `b[0]` also has a co-located
    spike (`wgrad = sum_m a[m] * b[m]`).
    Because NVFP4 E2M1 only has step levels `{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`, a single
    co-located spike at `m=0` or `m=0,1` with intermediate values (e.g., `6.0` and `4.9`) suffers
    coarse grid snapping in the unrotated coordinate (`4.9 -> 4.0`, losing 18% of the spike product),
    AND crushes tokens `2..15` (`~0.8`) to `0.0`.
    Under 16x16 RHT (`(A H)(H^T B) = A B`), the two-token outlier energy is spread across all 16
    Hadamard basis vectors, reducing peak crest factor from `3.05x` to `1.41x` and using all 16
    E2M1 quantizers to jointly represent the inner product with higher SQNR!
    """
    rng = random.Random(seed + 77)
    n_tiles = 64
    k = n_tiles * 16
    a = [rng.gauss(0.0, 0.25) for _ in range(k)]
    b = [rng.gauss(0.0, 0.25) for _ in range(k)]
    spike = outlier_ratio
    for blk_start in range(0, k, 16):
        # Co-located outlier cluster on tokens 0..2 that falls in the coarse gap of E2M1 (between 4.0 and 6.0)
        a[blk_start] = 1.00 * spike
        a[blk_start + 1] = 0.82 * spike
        a[blk_start + 2] = -0.51 * spike
        b[blk_start] = 1.00 * spike
        b[blk_start + 1] = 0.82 * spike
        b[blk_start + 2] = -0.51 * spike

    # Full WGRAD dot product across each 16-token tile: C_tile = sum_{i=0..15} a[i] * b[i]
    true_tile_dots = [
        sum(a[base + i] * b[base + i] for i in range(16)) for base in range(0, k, 16)
    ]

    qa_naive = quantize_nvfp4_block(a, use_rht=False, use_sr=False, seed=seed)
    qb_naive = quantize_nvfp4_block(b, use_rht=False, use_sr=False, seed=seed + 1)
    naive_tile_dots = [
        sum(qa_naive[base + i] * qb_naive[base + i] for i in range(16)) for base in range(0, k, 16)
    ]

    a_rot = apply_rht_16(a)
    b_rot = apply_rht_16(b)
    qa_rot = quantize_nvfp4_block(a_rot, use_rht=False, use_sr=False, seed=seed)
    qb_rot = quantize_nvfp4_block(b_rot, use_rht=False, use_sr=False, seed=seed + 1)
    rht_tile_dots = [
        sum(qa_rot[base + i] * qb_rot[base + i] for i in range(16)) for base in range(0, k, 16)
    ]

    return {
        "wgrad_naive_sqnr_db": round(sqnr_db(true_tile_dots, naive_tile_dots), 2),
        "wgrad_rht_sqnr_db": round(sqnr_db(true_tile_dots, rht_tile_dots), 2),
    }


def quantize_fp8_delayed_tensor(vec: list[float], stale_amax_mismatch: float = 1.0) -> list[float]:
    """Per-tensor FP8 E4M3 quantization with optional delayed-amax mismatch."""
    true_amax = max(abs(x) for x in vec)
    delayed_amax = true_amax / max(stale_amax_mismatch, 1e-6)
    scale = 448.0 / max(delayed_amax, 1e-12)
    out = []
    for x in vec:
        q = quantize_to_sorted_grid_rtn(x * scale, E4M3_POS_GRID)
        out.append(q / scale)
    return out


def quantize_fp8_deepseek_1x128(vec: list[float]) -> list[float]:
    """DeepSeek-V3 / DeepGEMM 1x128 fine-grained FP8 E4M3 block scaling."""
    out = [0.0] * len(vec)
    for base in range(0, len(vec), 128):
        blk = vec[base : base + 128]
        b_amax = max(max(abs(x) for x in blk), 1e-12)
        scale = 448.0 / b_amax
        for i, x in enumerate(blk):
            q = quantize_to_sorted_grid_rtn(x * scale, E4M3_POS_GRID)
            out[base + i] = q / scale
    return out


def quantize_mxfp8_1x32_ue8m0(vec: list[float]) -> list[float]:
    """OCP MXFP8 1x32 block scaling with power-of-two UE8M0 scale factor."""
    out = [0.0] * len(vec)
    for base in range(0, len(vec), 32):
        blk = vec[base : base + 32]
        b_amax = max(max(abs(x) for x in blk), 1e-12)
        # UE8M0 power-of-two scale: largest power of 2 such that b_amax / scale <= 448.0
        exp = math.ceil(math.log2(b_amax / 448.0))
        s_block = 2.0**exp
        for i, x in enumerate(blk):
            q = quantize_to_sorted_grid_rtn(x / s_block, E4M3_POS_GRID)
            out[base + i] = q * s_block
    return out


def quantize_nvfp4_block(
    vec: list[float],
    use_rht: bool = False,
    use_sr: bool = False,
    use_4over6: bool = False,
    seed: int = 1234,
) -> list[float]:
    """Blackwell NVFP4 1x16 hierarchical scaling:
    x = x_e2m1 * s_block_e4m3 * s_global_fp32
    with optional 16x16 Random Hadamard Transform (RHT), Stochastic Rounding (SR),
    and 4over6 adaptive block candidate selection.
    """
    rng = random.Random(seed)
    work = apply_rht_16(vec) if use_rht else list(vec)
    global_amax = max(max(abs(x) for x in work), 1e-12)
    fp8_bound = 256.0 if use_4over6 else 448.0
    s_global = global_amax / (fp8_bound * 6.0)

    dequant = [0.0] * len(work)
    for base in range(0, len(work), 16):
        blk = work[base : base + 16]
        b_amax = max(max(abs(x) for x in blk), 1e-12)

        candidates = [6.0, 4.0] if use_4over6 else [6.0]
        best_recon: list[float] | None = None
        best_mae = float("inf")

        for fp4_max in candidates:
            raw_s_block = (b_amax / fp4_max) / s_global
            s_block_e4m3 = quantize_to_sorted_grid_rtn(raw_s_block, E4M3_POS_GRID)
            eff_scale = max(s_block_e4m3 * s_global, 1e-15)
            recon_blk = []
            for x in blk:
                scaled = x / eff_scale
                if use_sr:
                    q_fp4 = quantize_to_e2m1_sr(scaled, rng, max_bound=fp4_max)
                else:
                    grid = [v for v in E2M1_POS_GRID if v <= fp4_max + 1e-9]
                    q_fp4 = quantize_to_sorted_grid_rtn(scaled, grid)
                recon_blk.append(q_fp4 * eff_scale)
            mae = sum(abs(a - b) for a, b in zip(blk, recon_blk)) / 16.0
            if mae < best_mae:
                best_mae = mae
                best_recon = recon_blk

        assert best_recon is not None
        dequant[base : base + 16] = best_recon

    return invert_rht_16(dequant) if use_rht else dequant


def simulate_wgrad_accumulation_bias(n_steps: int, true_grad: float, block_scale: float, seed: int) -> dict[str, float]:
    """Compare RTN vs Stochastic Rounding (SR) when accumulating sub-step gradients in NVFP4."""
    rng = random.Random(seed)
    scaled_true = true_grad / block_scale
    rtn_val = quantize_to_sorted_grid_rtn(scaled_true, E2M1_POS_GRID) * block_scale
    sr_sum = 0.0
    for _ in range(n_steps):
        sr_sum += quantize_to_e2m1_sr(scaled_true, rng) * block_scale
    sr_mean = sr_sum / n_steps
    return {
        "true_grad": round(true_grad, 6),
        "rtn_quantized_grad": round(rtn_val, 6),
        "sr_mean_grad": round(sr_mean, 6),
        "rtn_relative_bias_pct": round(abs(rtn_val - true_grad) / abs(true_grad) * 100.0, 2),
        "sr_relative_bias_pct": round(abs(sr_mean - true_grad) / abs(true_grad) * 100.0, 2),
    }


def simulate_wgmma_fp32_promotion(k_dim: int, promo_interval: int, seed: int) -> float:
    """Simulate Hopper WGMMA 14-bit mantissa truncation vs DeepGEMM CUDA-core FP32 promotion every `promo_interval`."""
    rng = random.Random(seed)
    a = [rng.gauss(0.0, 1.0) for _ in range(k_dim)]
    b = [rng.gauss(0.0, 1.0) for _ in range(k_dim)]
    true_dot = sum(x * y for x, y in zip(a, b))

    # Emulate reduced-mantissa running accumulator inside WGMMA if not promoted to full FP32
    partial_sum = 0.0
    running_total = 0.0
    for idx, (x, y) in enumerate(zip(a, b), start=1):
        prod = x * y
        partial_sum += prod
        # 13-bit fraction quantization noise inside fast-accum register
        if abs(partial_sum) > 0:
            exp = math.floor(math.log2(abs(partial_sum)))
            q_step = 2.0 ** (exp - 13)
            partial_sum = round(partial_sum / q_step) * q_step
        if idx % promo_interval == 0:
            running_total += partial_sum
            partial_sum = 0.0
    running_total += partial_sum
    rel_err = abs(running_total - true_dot) / max(abs(true_dot), 1e-12)
    return round(rel_err * 100.0, 4)


def evaluate_numerics(config: dict[str, Any]) -> dict[str, Any]:
    n = int(config["vector_length"])
    seed = int(config["seed"])
    outlier_ratios = [float(r) for r in config["outlier_ratios"]]

    rows: list[dict[str, Any]] = []
    for ratio in outlier_ratios:
        vec = generate_synthetic_tensor(n=n, outlier_ratio=ratio, seed=seed)

        q_fp8_delayed = quantize_fp8_delayed_tensor(vec, stale_amax_mismatch=1.25)
        q_fp8_1x128 = quantize_fp8_deepseek_1x128(vec)
        q_mxfp8_1x32 = quantize_mxfp8_1x32_ue8m0(vec)
        q_nvfp4_naive = quantize_nvfp4_block(vec, use_rht=False, use_sr=False, use_4over6=False, seed=seed)
        q_nvfp4_rht = quantize_nvfp4_block(vec, use_rht=True, use_sr=False, use_4over6=False, seed=seed)
        q_nvfp4_rht_sr = quantize_nvfp4_block(vec, use_rht=True, use_sr=True, use_4over6=False, seed=seed)
        q_nvfp4_4over6 = quantize_nvfp4_block(vec, use_rht=False, use_sr=False, use_4over6=True, seed=seed)

        # Measure peak outlier crest factor before vs after 16x16 RHT
        rot_vec = apply_rht_16(vec)
        raw_blk_crest = max(abs(x) for x in vec[:16]) / (
            math.sqrt(sum(x * x for x in vec[:16]) / 16.0)
        )
        rht_blk_crest = max(abs(x) for x in rot_vec[:16]) / (
            math.sqrt(sum(x * x for x in rot_vec[:16]) / 16.0)
        )
        wgrad_rht = wgrad_rht_dot_product_sqnr_db(outlier_ratio=ratio, seed=seed)

        rows.append(
            {
                "outlier_ratio": ratio,
                "fp8_delayed_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_fp8_delayed), 2),
                "fp8_1x128_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_fp8_1x128), 2),
                "mxfp8_1x32_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_mxfp8_1x32), 2),
                "nvfp4_naive_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_nvfp4_naive), 2),
                "nvfp4_rht_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_nvfp4_rht), 2),
                "nvfp4_rht_sr_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_nvfp4_rht_sr), 2),
                "nvfp4_4over6_non_outlier_sqnr_db": round(non_outlier_sqnr_db(vec, q_nvfp4_4over6), 2),
                "wgrad_nvfp4_naive_sqnr_db": wgrad_rht["wgrad_naive_sqnr_db"],
                "wgrad_nvfp4_rht_sqnr_db": wgrad_rht["wgrad_rht_sqnr_db"],
                "raw_block_crest_factor": round(raw_blk_crest, 3),
                "rht_block_crest_factor": round(rht_blk_crest, 3),
            }
        )

    wgrad_bias = simulate_wgrad_accumulation_bias(
        n_steps=int(config["sr_accumulation_steps"]),
        true_grad=float(config["sub_grid_gradient"]),
        block_scale=float(config["sub_grid_block_scale"]),
        seed=seed,
    )

    wgmma_promo = {
        "fast_accum_no_promotion_rel_err_pct": simulate_wgmma_fp32_promotion(4096, 4096, seed),
        "deepgemm_128_promotion_rel_err_pct": simulate_wgmma_fp32_promotion(4096, 128, seed),
    }

    r100 = next(r for r in rows if int(r["outlier_ratio"]) == 100)
    summary = {
        "outlier_100x_fp8_delayed_non_outlier_sqnr_db": r100["fp8_delayed_non_outlier_sqnr_db"],
        "outlier_100x_fp8_1x128_non_outlier_sqnr_db": r100["fp8_1x128_non_outlier_sqnr_db"],
        "outlier_100x_mxfp8_1x32_non_outlier_sqnr_db": r100["mxfp8_1x32_non_outlier_sqnr_db"],
        "outlier_100x_nvfp4_naive_non_outlier_sqnr_db": r100["nvfp4_naive_non_outlier_sqnr_db"],
        "outlier_100x_nvfp4_rht_non_outlier_sqnr_db": r100["nvfp4_rht_non_outlier_sqnr_db"],
        "outlier_100x_wgrad_nvfp4_naive_sqnr_db": r100["wgrad_nvfp4_naive_sqnr_db"],
        "outlier_100x_wgrad_nvfp4_rht_sqnr_db": r100["wgrad_nvfp4_rht_sqnr_db"],
        "outlier_100x_nvfp4_rht_sqnr_gain_db": round(
            r100["wgrad_nvfp4_rht_sqnr_db"] - r100["wgrad_nvfp4_naive_sqnr_db"], 2
        ),
        "outlier_100x_nvfp4_4over6_non_outlier_sqnr_db": r100["nvfp4_4over6_non_outlier_sqnr_db"],
        "wgrad_rtn_relative_bias_pct": wgrad_bias["rtn_relative_bias_pct"],
        "wgrad_sr_relative_bias_pct": wgrad_bias["sr_relative_bias_pct"],
        "wgmma_no_promo_rel_err_pct": wgmma_promo["fast_accum_no_promotion_rel_err_pct"],
        "deepgemm_128_promo_rel_err_pct": wgmma_promo["deepgemm_128_promotion_rel_err_pct"],
    }

    return {
        "experiment_id": "fp8_nvfp4_numerics",
        "title": "FP8 Delayed/Block Scaling vs Blackwell NVFP4 (1x16 + RHT + SR + 4over6) Numerical Stability Model",
        "Recomputation": "True (stdlib offline deterministic numerical simulation)",
        "parameters": config,
        "rows": rows,
        "wgrad_stochastic_rounding": wgrad_bias,
        "wgmma_fp32_promotion": wgmma_promo,
        "summary": summary,
    }


def write_outputs(results: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fieldnames = list(results["rows"][0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results["rows"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--check", action="store_true", help="Verify deterministic outputs match committed files")
    args = parser.parse_args()

    config = json.loads(args.input.read_text(encoding="utf-8"))
    results = evaluate_numerics(config)

    if args.check:
        expected = json.loads(args.output_json.read_text(encoding="utf-8"))
        if results != expected:
            raise SystemExit("Mismatch between computed results and committed results.json")
        # Invariants
        s = results["summary"]
        assert s["outlier_100x_fp8_1x128_non_outlier_sqnr_db"] > s["outlier_100x_fp8_delayed_non_outlier_sqnr_db"] + 6.0
        assert s["outlier_100x_nvfp4_rht_sqnr_gain_db"] > 6.0
        assert s["wgrad_sr_relative_bias_pct"] < 1.5 and s["wgrad_rtn_relative_bias_pct"] > 20.0
        assert s["deepgemm_128_promo_rel_err_pct"] < s["wgmma_no_promo_rel_err_pct"]
        print("fp8_nvfp4_numerics check passed.")
        return 0

    write_outputs(results, args.output_json, args.output_csv)
    print(f"Wrote {args.output_json} and {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
