"""Original scalar hash-grid and LoD references; not a native/GPU reproduction."""

import argparse
import itertools
import json
import math
import platform
import statistics
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "3daa6e5b8b8e0fa17971a1a20acfbb3754494b38"
EPSILON = 1e-3
MAX_SAFE = 2**53 - 1
DEFAULT = dict(dimensions=3, levels=12, base_resolution=16, growth=2,
               log2_cap=19, features=2, parameter_bytes=2, batch=65536, lod=0.5)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def level_weights(levels, lod, mode):
    if not isinstance(levels, int) or isinstance(levels, bool) or not 1 <= levels <= 32:
        raise ValueError("levels must be an integer in [1,32]")
    if not finite(lod) or not 0 <= lod <= 1 or mode not in ("hard", "soft"):
        raise ValueError("lod must be finite in [0,1] and mode hard/soft")
    threshold = lod * levels + EPSILON
    return [float(level < threshold) if mode == "hard" else
            min(1.0, max(0.0, threshold - level)) for level in range(levels)]


def table_index(x, y, resolution, capacity):
    if resolution * resolution <= capacity:
        return (x + resolution * y) % capacity
    return ((x ^ ((y * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF) % capacity


def validate_grid(table, resolution, points):
    if (not isinstance(resolution, int) or isinstance(resolution, bool) or
            not 2 <= resolution <= 64 or not table or len(table) > 4096 or
            any(not finite(value) for value in table)):
        raise ValueError("finite table and integer resolution [2,64] required")
    if any(len(point) != 2 or any(not finite(v) or not 0 <= v <= 1 for v in point)
           for point in points):
        raise ValueError("finite two-dimensional points in [0,1] required")


def corner_terms(point, resolution, capacity):
    scale = resolution - 1
    position = [scale * value + 0.5 for value in point]
    cell = [math.floor(value) for value in position]
    fraction = [value - base for value, base in zip(position, cell)]
    terms = []
    for bx, by in itertools.product((0, 1), repeat=2):
        wx = fraction[0] if bx else 1 - fraction[0]
        wy = fraction[1] if by else 1 - fraction[1]
        index = table_index(cell[0] + bx, cell[1] + by, resolution, capacity)
        terms.append((index, wx * wy, scale * (1 if bx else -1) * wy,
                      scale * (1 if by else -1) * wx))
    return terms


def direct_encode(table, resolution, points):
    validate_grid(table, resolution, points)
    return [sum(table[index] * weight for index, weight, _, _ in
                corner_terms(point, resolution, len(table))) for point in points]


def expanded_encode(table, resolution, points):
    """Build the independent unfolded corner field once per batch."""
    validate_grid(table, resolution, points)
    field = [[table[table_index(x, y, resolution, len(table))]
              for x in range(resolution + 1)] for y in range(resolution + 1)]
    result = []
    for x, y in points:
        px, py = (resolution - 1) * x + 0.5, (resolution - 1) * y + 0.5
        ix, iy = math.floor(px), math.floor(py)
        u, v = px - ix, py - iy
        bottom = field[iy][ix] + u * (field[iy][ix + 1] - field[iy][ix])
        top = field[iy + 1][ix] + u * (field[iy + 1][ix + 1] - field[iy + 1][ix])
        result.append(bottom + v * (top - bottom))
    return result


def vjp(table, resolution, points, upstream, overwrite=False):
    validate_grid(table, resolution, points)
    if len(points) != len(upstream) or any(not finite(value) for value in upstream):
        raise ValueError("one finite upstream scalar per point required")
    parameter = [0.0] * len(table)
    coordinates = []
    for point, gradient in zip(points, upstream):
        coordinate = [0.0, 0.0]
        for index, weight, dx, dy in corner_terms(point, resolution, len(table)):
            contribution = gradient * weight
            if overwrite:
                parameter[index] = contribution
            else:
                parameter[index] += contribution
            coordinate[0] += gradient * table[index] * dx
            coordinate[1] += gradient * table[index] * dy
        coordinates.append(coordinate)
    return dict(parameter=parameter, coordinates=coordinates)


def estimate_budget(p):
    limits = dict(dimensions=(2, 3), levels=(1, 32), base_resolution=(2, 64),
                  growth=(1, 2), log2_cap=(1, 24), features=(1, 8),
                  parameter_bytes=(2, 4), batch=(1, MAX_SAFE))
    for key, (low, high) in limits.items():
        if (not isinstance(p[key], int) or isinstance(p[key], bool) or
                not low <= p[key] <= high):
            raise ValueError(f"{key} must be an integer in [{low},{high}]")
    if p["parameter_bytes"] not in (2, 4):
        raise ValueError("parameter_bytes must be 2 or 4")
    weights = level_weights(p["levels"], p["lod"], "soft")
    hard = level_weights(p["levels"], p["lod"], "hard")
    rows = []
    for level in range(p["levels"]):
        resolution = p["base_resolution"] * p["growth"]**level
        if resolution > 65536:
            raise ValueError("resolution exceeds this bounded exact calculator's 65536 limit")
        dense = resolution**p["dimensions"]
        aligned = ((min(dense, 2**31 - 1) + 7) // 8) * 8
        entries = min(aligned, 2**p["log2_cap"])
        rows.append(dict(level=level, resolution=resolution, dense_vertices=dense,
                         table_entries=entries, hard_weight=hard[level],
                         soft_weight=weights[level]))
    entries = sum(row["table_entries"] for row in rows)
    scalars = entries * p["features"]
    output = p["batch"] * p["levels"] * p["features"] * p["parameter_bytes"]
    result = dict(levels=rows, allocated_table_entries=entries, parameter_scalars=scalars,
                  parameter_bytes=scalars * p["parameter_bytes"],
                  ideal_dense_vertex_bytes=sum(row["dense_vertices"] for row in rows) *
                  p["features"] * p["parameter_bytes"],
                  output_feature_bytes=output, soft_unweighted_base_output_bytes=output,
                  hard_active_levels=int(sum(hard)),
                  table_entries_fit_uint32=entries <= 2**32 - 1,
                  parameter_scalars_fit_uint32=scalars <= 2**32 - 1,
                  peak_hbm_bytes=None, gpu_latency_ms=None, quality=None)
    integers = [value for value in result.values() if isinstance(value, int)]
    integers += [row[key] for row in rows for key in ("resolution", "dense_vertices", "table_entries")]
    if any(value > MAX_SAFE for value in integers):
        raise ValueError("budget exceeds exact JavaScript integer range")
    return result


def fixtures(batch):
    return [(0.01 + 0.98 * ((i * 17) % 997) / 996,
             0.01 + 0.98 * ((i * 31 + 7) % 991) / 990) for i in range(batch)]


def measure_latency():
    rows = []
    table = [math.sin(i * 0.3) for i in range(64)]
    for batch in (16, 64, 256, 1024):
        points = fixtures(batch)
        methods = (direct_encode, expanded_encode)
        samples = [[], []]
        for method in methods:
            for _ in range(2):
                method(table, 16, points)
        for repeat in range(7):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter_ns()
                methods[index](table, 16, points)
                samples[index].append((time.perf_counter_ns() - start) / 1e6)
        error = max(abs(a - b) for a, b in zip(
            direct_encode(table, 16, points), expanded_encode(table, 16, points)))
        rows.append(dict(batch=batch, resolution=16, table_entries=64,
                         primary_samples_ms=samples[0], baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]), maximum_error=error))
    data = dict(provenance="measured", scope="Original float64 Python CPU, forward only, validation included",
                primary="Direct hashed corner gather", baseline="Per-batch unfolded dense corner field",
                primary_includes="Four Python corner tuples with derivative coefficients; untuned reference",
                baseline_includes="Allocate and construct (resolution+1)^2 scalar field, then sample",
                hardware=platform.machine(), processor=platform.processor(), python=platform.python_version(),
                warmups=2, repeats=7, alternating_order=True, rows=rows,
                gpu_latency_ms=None, native_tiny_cuda_nn_measured=False)
    (HERE / "latency.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def simulate():
    table = [0.2, -0.4, 0.7, 1.1]
    points = [[0.23, 0.41], [0.71, 0.19]]
    upstream = [1.0, -0.7]
    outputs = direct_encode(table, 4, points)
    expanded = expanded_encode(table, 4, points)
    gradients = vjp(table, 4, points, upstream)
    loss = lambda values: sum(a * b for a, b in zip(direct_encode(values, 4, points), upstream))
    step = 1e-6
    fd = []
    for index in range(len(table)):
        plus, minus = table.copy(), table.copy()
        plus[index] += step
        minus[index] -= step
        fd.append((loss(plus) - loss(minus)) / (2 * step))
    coordinate_fd = []
    for i in range(len(points)):
        row = []
        for dimension in range(2):
            plus, minus = [p.copy() for p in points], [p.copy() for p in points]
            plus[i][dimension] += step
            minus[i][dimension] -= step
            objective = lambda ps: sum(a * b for a, b in zip(direct_encode(table, 4, ps), upstream))
            row.append((objective(plus) - objective(minus)) / (2 * step))
        coordinate_fd.append(row)
    single = vjp(table, 4, [points[0]], [1.0])
    overwritten = vjp(table, 4, [points[0]], [1.0], overwrite=True)
    lod = 0.5
    raw = [direct_encode(table, resolution, [points[0]])[0] for resolution in (4, 8, 16)]
    hard, soft = level_weights(3, lod, "hard"), level_weights(3, lod, "soft")
    weighted = [a * b for a, b in zip(raw, soft)]
    soft_fd = []
    for index in range(3):
        plus, minus = raw.copy(), raw.copy()
        plus[index] += step
        minus[index] -= step
        objective = lambda values: 0.5 * sum((a * b)**2 for a, b in zip(values, soft))
        soft_fd.append((objective(plus) - objective(minus)) / (2 * step))
    soft_gradient = [value * weight**2 for value, weight in zip(raw, soft)]
    hard_backward_wrong = [value * weight for value, weight in zip(raw, soft)]
    double_weight_wrong = [value * weight**3 for value, weight in zip(raw, soft)]
    variants = []
    for resolution, capacity in ((4, 16), (4, 4), (8, 8), (16, 32)):
        values = [math.sin(i * 0.7) for i in range(capacity)]
        ps = fixtures(17)
        a, b = direct_encode(values, resolution, ps), expanded_encode(values, resolution, ps)
        variants.append(dict(resolution=resolution, capacity=capacity,
                             dense_index_path=resolution**2 <= capacity,
                             maximum_error=max(abs(x - y) for x, y in zip(a, b))))
    if not (HERE / "latency.json").exists():
        raise FileNotFoundError("Run --measure once to seal original CPU timings")
    return dict(schema_version=1, round=105, provenance="synthetic",
                independently_reproduced=False, native_tiny_cuda_nn_measured=False,
                source=dict(pin=PIN, date="2026-09-29", capability="Generalized native hard/soft multilevel LoD",
                            license="BSD-3-Clause terms; GitHub license API NOASSERTION; LICENSE.txt inspected",
                            url=f"https://github.com/NVlabs/tiny-cuda-nn/blob/{PIN}/include/tiny-cuda-nn/encodings/multi_level_lod.h"),
                scope="2D scalar linear interpolation, coherent-prime low-bit hashing, bounded nonnegative coordinates; not float32 CUDA/JIT parity",
                collision=dict(table=table, point=points[0], resolution=4,
                               corner_terms=corner_terms(points[0], 4, 4),
                               output=outputs[0], parameter_gradient=single["parameter"],
                               wrong_overwrite_gradient=overwritten["parameter"],
                               overwrite_max_error=max(abs(a-b) for a,b in
                                                       zip(single["parameter"], overwritten["parameter"]))),
                gradient=dict(analytic=gradients, parameter_finite_difference=fd,
                              coordinate_finite_difference=coordinate_fd,
                              maximum_parameter_error=max(abs(a-b) for a,b in zip(fd, gradients["parameter"])),
                              maximum_coordinate_error=max(abs(a-b) for ra,rb in
                                                           zip(coordinate_fd,gradients["coordinates"]) for a,b in zip(ra,rb))),
                parity=dict(direct=outputs, unfolded=expanded,
                            maximum_error=max(abs(a-b) for a,b in zip(outputs,expanded))),
                lod=dict(fraction=lod, raw=raw, hard_weights=hard, soft_weights=soft,
                         soft_output=weighted, correct_gradient=soft_gradient,
                         wrong_missing_weight=hard_backward_wrong,
                         wrong_double_weight=double_weight_wrong,
                         finite_difference=soft_fd,
                         maximum_gradient_error=max(abs(a-b) for a,b in zip(soft_fd,soft_gradient)),
                         zero_fraction_soft=level_weights(3,0,"soft"),
                         zero_fraction_hard=level_weights(3,0,"hard"),
                         boundary_lod=(1-EPSILON)/4,
                         boundary_hard=level_weights(4,(1-EPSILON)/4,"hard"),
                         boundary_soft=level_weights(4,(1-EPSILON)/4,"soft"),
                         lod_derivative_claim=False),
                variants=variants, budget=dict(input=DEFAULT, **estimate_budget(DEFAULT)),
                latency=json.loads((HERE / "latency.json").read_text(encoding="utf-8")))


def outputs():
    from .render import render_outputs
    report = simulate()
    return {HERE / "report.json": json.dumps(report, indent=2) + "\n", **render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--measure", action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    print("Round 105 artifacts match." if args.check else "Wrote Round 105 artifacts.")


if __name__ == "__main__":
    main()
