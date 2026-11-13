"""Original batched 1D sparse-convolution coordinate maps and adjoint checks."""

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "263d6b47425ef843c82f997b12d8b714013d216c"
MAX_SAFE = 2**53 - 1
DEFAULT = dict(dimensions=3, batches=2, edge=128, kernel=3, active_input=200000,
               active_output=200000, valid_pairs=1800000, channels_in=64,
               channels_out=64, feature_bytes=2, index_bytes=4)


def integer(value, low=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= low


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CoordinateMap:
    inputs: tuple
    outputs: tuple
    pairs: tuple
    shape: int
    output_shape: int
    batches: int
    kernel: int
    stride: int
    padding: int
    dilation: int
    submanifold: bool


def validate_coordinates(coordinates, shape, batches):
    if not integer(shape, 1) or not integer(batches, 1):
        raise ValueError("positive integer shape and batches required")
    if any(len(point) != 2 or not integer(point[0]) or point[0] >= batches or
           not integer(point[1]) or point[1] >= shape for point in coordinates):
        raise ValueError("unique integer (batch, x) coordinates within shape required")
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate coordinates require an explicit coalescing policy")


def build_map(coordinates, shape, batches=1, kernel=3, stride=1, padding=1,
              dilation=1, submanifold=False):
    coordinates = tuple(tuple(point) for point in coordinates)
    validate_coordinates(coordinates, shape, batches)
    if (not all(integer(value, 1) for value in (kernel, stride, dilation)) or
            not integer(padding) or not isinstance(submanifold, bool)):
        raise ValueError("positive integer kernel/stride/dilation and nonnegative padding required")
    if submanifold and (stride != 1 or kernel % 2 != 1 or padding != dilation * (kernel//2)):
        raise ValueError("toy submanifold requires odd centered kernel and stride one")
    output_shape = shape if submanifold else (shape + 2*padding - dilation*(kernel-1) - 1)//stride + 1
    if output_shape <= 0:
        raise ValueError("output shape must remain positive")
    raw = []
    for row, (batch, x) in enumerate(coordinates):
        for offset in range(kernel):
            numerator = x + padding - dilation*offset
            if numerator % stride == 0:
                y = numerator // stride
                if 0 <= y < output_shape:
                    raw.append((offset, row, (batch, y)))
    outputs = coordinates if submanifold else tuple(sorted({point for _, _, point in raw}))
    output_rows = {point: row for row, point in enumerate(outputs)}
    pairs = tuple((offset, row, output_rows[point]) for offset, row, point in raw if point in output_rows)
    return CoordinateMap(coordinates, outputs, pairs, shape, output_shape, batches,
                         kernel, stride, padding, dilation, submanifold)


def validate_values(values, expected, name):
    if len(values) != expected or any(not finite(value) for value in values):
        raise ValueError(f"{name} requires {expected} finite scalars")


def apply_map(mapping, features, weights, coordinates=None):
    validate_values(features, len(mapping.inputs), "features")
    validate_values(weights, mapping.kernel, "weights")
    if coordinates is not None and tuple(tuple(point) for point in coordinates) != mapping.inputs:
        raise ValueError("cached map requires identical ordered input coordinates")
    result = [0.0] * len(mapping.outputs)
    for offset, row_in, row_out in mapping.pairs:
        result[row_out] += features[row_in] * weights[offset]
    return result


def dense_reference(mapping, features, weights):
    validate_values(features, len(mapping.inputs), "features")
    validate_values(weights, mapping.kernel, "weights")
    values = dict(zip(mapping.inputs, features))
    result = {}
    for batch in range(mapping.batches):
        for y in range(mapping.output_shape):
            result[(batch, y)] = sum(weights[offset] * values.get(
                (batch, y*mapping.stride - mapping.padding + offset*mapping.dilation), 0.0)
                for offset in range(mapping.kernel))
    return [result[point] for point in mapping.outputs]


def adjoint(mapping, upstream, weights, output_coordinates=None):
    validate_values(upstream, len(mapping.outputs), "upstream")
    validate_values(weights, mapping.kernel, "weights")
    if output_coordinates is not None and tuple(tuple(p) for p in output_coordinates) != mapping.outputs:
        raise ValueError("cached inverse requires identical ordered output coordinates")
    result = [0.0] * len(mapping.inputs)
    for offset, row_in, row_out in mapping.pairs:
        result[row_in] += upstream[row_out] * weights[offset]
    return result


def transposed_support(mapping, upstream, weights):
    """Generate ordinary transpose support independently, at the original shape."""
    validate_values(upstream, len(mapping.outputs), "upstream")
    validate_values(weights, mapping.kernel, "weights")
    result = {}
    for (batch, y), value in zip(mapping.outputs, upstream):
        for offset in range(mapping.kernel):
            x = y*mapping.stride - mapping.padding + offset*mapping.dilation
            if 0 <= x < mapping.shape:
                point = (batch, x)
                result[point] = result.get(point, 0.0) + value * weights[offset]
    return dict(coordinates=sorted(result), values=[result[p] for p in sorted(result)])


def estimate_budget(p):
    if not integer(p["dimensions"], 1) or p["dimensions"] > 3:
        raise ValueError("dimensions must be 1, 2 or 3")
    for key in ("batches", "edge", "kernel", "channels_in", "channels_out"):
        if not integer(p[key], 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("active_input", "active_output", "valid_pairs"):
        if not integer(p[key]):
            raise ValueError(f"{key} must be a nonnegative integer")
    if p["feature_bytes"] not in (2, 4) or p["index_bytes"] not in (4, 8):
        raise ValueError("feature_bytes 2/4 and index_bytes 4/8 required")
    volume = p["batches"] * p["edge"]**p["dimensions"]
    k = p["kernel"]**p["dimensions"]
    if max(p["active_input"], p["active_output"]) > volume:
        raise ValueError("active coordinates exceed declared batched spatial volume")
    if p["valid_pairs"] > k * min(p["active_input"], p["active_output"]):
        raise ValueError("valid pairs exceed kernel-volume per-row bound")
    result = dict(spatial_volume=volume, kernel_volume=k,
                  native_padded_pair_bytes=2*k*p["active_input"]*p["index_bytes"],
                  compact_valid_pair_id_bytes=2*p["valid_pairs"]*p["index_bytes"],
                  kernel_pair_count_bytes=k*p["index_bytes"],
                  input_coordinate_bytes=p["active_input"]*(p["dimensions"]+1)*p["index_bytes"],
                  output_coordinate_bytes=p["active_output"]*(p["dimensions"]+1)*p["index_bytes"],
                  input_feature_bytes=p["active_input"]*p["channels_in"]*p["feature_bytes"],
                  output_feature_bytes=p["active_output"]*p["channels_out"]*p["feature_bytes"],
                  source_selects_int64_hash_keys=volume >= 2**31 - 1,
                  active_row_ids_fit_signed_int32=max(p["active_input"],p["active_output"]) <= 2**31,
                  peak_hbm_bytes=None, gpu_latency_ms=None, model_quality=None)
    if any(not integer(value) or value > MAX_SAFE for value in result.values()
           if value is not None and not isinstance(value, bool)):
        raise ValueError("budget exceeds exact JavaScript integer range")
    return result


def fixtures(count):
    coordinates = [(0, (i*17) % (count*4)) for i in range(count)]
    return coordinates, [math.sin(i*0.2) for i in range(count)]


def measure_latency():
    rows = []
    for count in (16, 64, 256, 1024):
        coordinates, features = fixtures(count)
        mapping = build_map(coordinates, count*4, submanifold=True)
        weights = [0.2, 0.7, -0.3]
        methods = (
            lambda: apply_map(mapping, features, weights, coordinates),
            lambda: apply_map(build_map(coordinates, count*4, submanifold=True), features, weights, coordinates))
        samples = [[], []]
        for method in methods:
            for _ in range(2):
                method()
        for repeat in range(7):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start = time.perf_counter_ns()
                methods[index]()
                samples[index].append((time.perf_counter_ns()-start)/1e6)
        rows.append(dict(active_input=count, valid_pairs=len(mapping.pairs),
                         primary_samples_ms=samples[0], baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         maximum_error=max(abs(a-b) for a,b in zip(methods[0](),methods[1]()))))
    data = dict(provenance="measured", scope="Original Python CPU scalar submanifold map and application",
                primary="Reuse immutable map, check ordered coordinates, apply scalar weights",
                baseline="Build same map, check ordered coordinates, apply same scalar weights",
                primary_excludes="Initial map construction; reuse presumes unchanged operator and coordinates",
                hardware=platform.machine(), processor=platform.processor(), python=platform.python_version(),
                warmups=2, repeats=7, alternating_order=True, rows=rows,
                native_spconv_measured=False, gpu_latency_ms=None)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    coordinates = [(0,2),(0,3)]
    features, weights = [1.0,2.0], [1.0,1.0,1.0]
    regular = build_map(coordinates,8)
    subm = build_map(coordinates,8,submanifold=True)
    yr, ys = apply_map(regular,features,weights), apply_map(subm,features,weights)
    regular2 = build_map(regular.outputs,8)
    subm2 = build_map(subm.outputs,8,submanifold=True)
    yr2, ys2 = apply_map(regular2,yr,weights), apply_map(subm2,ys,weights)
    by_coordinate = dict(zip(regular2.outputs,yr2))
    zero = apply_map(subm,[1,1],[1,-1,1])
    mapping = build_map([(0,1),(0,4)],8,stride=2)
    x, w, u = [2.0,-1.0], [1.0,2.0,3.0], [0.5,-1.0,2.0]
    y = apply_map(mapping,x,w)
    dx = adjoint(mapping,u,w)
    fd = []
    h = 1e-6
    objective = lambda values: sum(a*b for a,b in zip(apply_map(mapping,values,w),u))
    for i in range(len(x)):
        plus,minus = x.copy(),x.copy()
        plus[i] += h
        minus[i] -= h
        fd.append((objective(plus)-objective(minus))/(2*h))
    permuted = list(reversed(mapping.inputs))
    permuted_features = list(reversed(x))
    fresh = build_map(permuted,8,stride=2)
    variants = []
    for stride,dilation,submanifold in ((1,1,False),(2,1,False),(1,2,False),(1,2,True)):
        m = build_map([(0,1),(0,4),(1,1),(1,6)],8,batches=2,
                      stride=stride,dilation=dilation,padding=dilation,submanifold=submanifold)
        values = [1.0,-2.0,3.0,0.0]
        a,b = apply_map(m,values,w),dense_reference(m,values,w)
        variants.append(dict(stride=stride,dilation=dilation,submanifold=submanifold,
                             active_output=len(m.outputs),valid_pairs=len(m.pairs),
                             maximum_error=max((abs(a-b) for a,b in zip(a,b)),default=0.0)))
    if not (HERE/"latency.json").exists():
        raise FileNotFoundError("Run --measure once to seal original CPU timings")
    return dict(schema_version=1,round=106,provenance="synthetic",independently_reproduced=False,
                native_spconv_measured=False,
                source=dict(pin=PIN,conv_last_change="2023-03-23",repository_pin_date="2024-12-15",
                            scope="Longstanding coordinate-map semantics, not a 2026 launch",
                            license="Apache-2.0",
                            url=f"https://github.com/traveller59/spconv/blob/{PIN}/spconv/pytorch/conv.py"),
                topology=dict(input_coordinates=coordinates,input_features=features,weights=weights,
                              regular_coordinates=regular.outputs,regular_output=yr,
                              submanifold_coordinates=subm.outputs,submanifold_output=ys,
                              second_regular_coordinates=regular2.outputs,second_regular_output=yr2,
                              second_submanifold_coordinates=subm2.outputs,second_submanifold_output=ys2,
                              second_regular_at_input=[by_coordinate[p] for p in subm.outputs],
                              zero_feature_output=zero,zero_feature_active_count=len(subm.outputs)),
                inverse=dict(input_coordinates=mapping.inputs,output_coordinates=mapping.outputs,
                             pairs=mapping.pairs,features=x,weights=w,forward=y,upstream=u,
                             cached_tied_weight_adjoint=dx,input_finite_difference=fd,
                             maximum_gradient_error=max(abs(a-b) for a,b in zip(dx,fd)),
                             dot_left=sum(a*b for a,b in zip(y,u)),
                             dot_right=sum(a*b for a,b in zip(x,dx)),
                             ordinary_transposed=transposed_support(mapping,u,w),
                             adjoint_of_forward=adjoint(mapping,y,w),
                             learned_inverse_not_necessarily_adjoint=True),
                cache=dict(permuted_coordinates=permuted,permuted_features=permuted_features,
                           fresh_output=apply_map(fresh,permuted_features,w),
                           stale_unchecked_output=apply_map(mapping,permuted_features,w),
                           maximum_stale_error=max(abs(a-b) for a,b in zip(
                               y,apply_map(mapping,permuted_features,w))),
                           guard="Exact ordered coordinates in original model; stronger than inspected source shape/count checks"),
                variants=variants,budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
                latency=json.loads((HERE/"latency.json").read_text(encoding="utf-8")))


def outputs():
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 106 artifacts match." if args.check else "Wrote Round 106 artifacts.")


if __name__ == "__main__":
    main()
