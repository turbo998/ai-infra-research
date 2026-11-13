"""Original finite-support Gaussian compositing and wide-offset arithmetic."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
SOURCE = "https://github.com/nerfstudio-project/gsplat/blob/81c13e668250fd5c3e78f1d87058b4ac071ebb31/gsplat/cuda/csrc/RasterizeToPixels3DGSSerialBatchFwd.cu"
MAX_SAFE = 9007199254740991
ALPHA_FLOOR = 1/255
MAX_ALPHA = 0.99
T_CUTOFF = 1e-4
BACKGROUND = (0.1,0.2,0.3)


def validate_color(color: tuple):
    if len(color) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in color):
        raise ValueError("toy RGB/background must be three finite values in [0,1]")


def composite(layers: list,background: tuple = BACKGROUND,cutoff: float | None = None,
              alpha_floor: float = 0.0,additive: bool = False,omit_background: bool = False) -> dict:
    validate_color(background)
    if not math.isfinite(alpha_floor) or not 0 <= alpha_floor <= 1:
        raise ValueError("alpha floor must lie in [0,1]")
    if cutoff is not None and (not math.isfinite(cutoff) or not 0 <= cutoff < 1):
        raise ValueError("cutoff must lie in [0,1) or be disabled")
    color,T,used = [0.0]*3,1.0,0
    stopped_before = None
    for alpha,rgb in layers:
        validate_color(rgb)
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("effective alpha must lie in [0,1]")
        if alpha < alpha_floor:
            continue
        following = T*(1-alpha)
        if cutoff is not None and following <= cutoff:
            stopped_before = T
            break
        for channel in range(3):
            color[channel] += rgb[channel]*alpha*(1.0 if additive else T)
        T = following
        used += 1
    if not omit_background:
        color = [value+T*bg for value,bg in zip(color,background)]
    return {"rgb":color,"transmittance":T,"render_alpha":1-T,"used_layers":used,
            "pre_exclusive_cutoff_transmittance":stopped_before}


def closed_composite(layers: list,background: tuple) -> list:
    """Independent product-form weights, with no mutable transmittance loop."""
    weights = [alpha*math.prod(1-a for a,_ in layers[:i]) for i,(alpha,_) in enumerate(layers)]
    background_weight = math.prod(1-a for a,_ in layers)
    return [math.fsum(weight*rgb[channel] for weight,(_,rgb) in zip(weights,layers))
            +background_weight*background[channel] for channel in range(3)]


def alpha_gradients(layers: list,background: tuple,upstream: tuple) -> list:
    result = []
    for i,(_,rgb) in enumerate(layers):
        before = math.prod(1-alpha for alpha,_ in layers[:i])
        behind = closed_composite(layers[i+1:],background)
        result.append(before*math.fsum(g*(c-tail) for g,c,tail in zip(upstream,rgb,behind)))
    return result


@dataclass(frozen=True)
class Gaussian:
    x: float
    y: float
    depth: float
    radius: float
    conic: tuple
    opacity: float
    color: tuple


def validate_scene(gaussians: list,width: int,height: int,tile: int,background: tuple):
    if any(type(n) is not int or not 1 <= n <= 2048 for n in (width,height,tile)):
        raise ValueError("original CPU scene dimensions/tile must be integers within 1..2048")
    validate_color(background)
    for g in gaussians:
        validate_color(g.color)
        if len(g.conic) != 3 or any(not math.isfinite(x) for x in
                                  (g.x,g.y,g.depth,g.radius,g.opacity,*g.conic)):
            raise ValueError("Gaussian scalar parameters must be finite")
        a,b,c = g.conic
        if g.depth < 0 or g.radius <= 0 or not 0 <= g.opacity <= 1 or a <= 0 or c <= 0 or a*c <= b*b:
            raise ValueError("nonnegative depth, positive support, opacity [0,1] and positive-definite conic required")


def weight(g: Gaussian,x: float,y: float) -> float | None:
    dx,dy = g.x-x,g.y-y
    if abs(dx) > g.radius or abs(dy) > g.radius:
        return None
    a,b,c = g.conic
    sigma = 0.5*(a*dx*dx+c*dy*dy)+b*dx*dy
    alpha = min(MAX_ALPHA,g.opacity*math.exp(-sigma))
    return None if sigma < 0 or alpha < ALPHA_FLOOR else alpha


def tile_bins(gaussians: list,width: int,height: int,tile: int,order: list) -> dict:
    bins = {}
    for idx in order:
        g = gaussians[idx]
        low_x,high_x = max(0,math.ceil(g.x-g.radius-0.5)),min(width-1,math.floor(g.x+g.radius-0.5))
        low_y,high_y = max(0,math.ceil(g.y-g.radius-0.5)),min(height-1,math.floor(g.y+g.radius-0.5))
        if low_x > high_x or low_y > high_y:
            continue
        for ty in range(low_y//tile,high_y//tile+1):
            for tx in range(low_x//tile,high_x//tile+1):
                bins.setdefault((tx,ty),[]).append(idx)
    return bins


def render(gaussians: list,width: int,height: int,tile: int = 4,use_tiles: bool = True,
           reverse_depth: bool = False,background: tuple = BACKGROUND) -> dict:
    validate_scene(gaussians,width,height,tile,background)
    order = sorted(range(len(gaussians)),key=lambda i:(gaussians[i].depth,i),reverse=reverse_depth)
    bins = tile_bins(gaussians,width,height,tile,order) if use_tiles else {}
    pixels,visits = [],0
    for y in range(height):
        for x in range(width):
            candidates = bins.get((x//tile,y//tile),[]) if use_tiles else order
            layers = []
            for idx in candidates:
                visits += 1
                alpha = weight(gaussians[idx],x+0.5,y+0.5)
                if alpha is not None:
                    layers.append((alpha,gaussians[idx].color))
            pixels.append(composite(layers,background)["rgb"])
    return {"pixels":pixels,"candidate_visits":visits,
            "tile_intersections":sum(len(ids) for ids in bins.values()) if use_tiles else None,
            "scope":"same declared finite square support; pixel centers; no early termination"}


def maximum_error(a: list,b: list) -> float:
    return max(abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra,rb))


def fixture(count: int,width: int = 16,height: int = 12) -> list:
    return [Gaussian((i*7)%width+0.25,(i*5)%height+0.75,1+i*0.001,1.25,
                     (0.75,0.05,1.25),0.2+0.05*(i%5),
                     ((i%3)/2,((i+1)%5)/4,((i+2)%7)/6)) for i in range(count)]


@dataclass(frozen=True)
class Budget:
    images: int = 1
    width: int = 1920
    height: int = 1080
    tile: int = 16
    gaussians_per_image: int = 1000000
    intersections_per_gaussian: int = 8


def estimate(b: Budget) -> dict:
    for name,value in asdict(b).items():
        minimum = 0 if name in ("gaussians_per_image","intersections_per_gaussian") else 1
        if type(value) is not int or not minimum <= value <= MAX_SAFE:
            raise ValueError(f"{name} outside exact nonnegative/positive integer range")
    tiles_x,tiles_y = (b.width+b.tile-1)//b.tile,(b.height+b.tile-1)//b.tile
    tiles = tiles_x*tiles_y
    if b.intersections_per_gaussian > tiles:
        raise ValueError("modeled per-Gaussian tile count cannot exceed tiles per image")
    entries = b.images*b.gaussians_per_image*b.intersections_per_gaussian
    ids = b.images*b.gaussians_per_image
    values = {"tiles_x":tiles_x,"tiles_y":tiles_y,"tiles_per_image":tiles,
              "intersection_entries":entries,"one_key_and_id_payload_bytes":12*entries,
              "two_sort_buffer_payload_bytes":24*entries,"tile_offset_int64_bytes":8*b.images*tiles,
              "tile_offset_int32_comparison_bytes":4*b.images*tiles}
    if any(value > MAX_SAFE for value in (*values.values(),ids)):
        raise ValueError("logical budget exceeds exact numerical range")
    tile_bits = (tiles-1).bit_length()
    image_bits = (b.images-1).bit_length()
    return {"input":asdict(b),**values,"offsets_exceed_signed_int32":entries > 2**31-1,
            "flattened_identity_fits_signed_int32":ids <= 2**31,
            "packed_image_tile_metadata_fits_32_bits":tile_bits+image_bits <= 32,
            "peak_memory_bytes":None,"gpu_latency_ms":None,"image_quality":None,
            "scope":"synthetic uniform intersection multiplicity; key64/id32 logical payload; remaining index limits still apply"}


def measure_latency():
    rows = []
    for count in (16,64,256,1024):
        scene = fixture(count)
        functions = {"tile_indexed":lambda:render(scene,16,12),
                     "dense_scan":lambda:render(scene,16,12,use_tiles=False)}
        if maximum_error(functions["tile_indexed"]()["pixels"],functions["dense_scan"]()["pixels"]) > 1e-12:
            raise ValueError("timed same-support compositing references disagree")
        for function in functions.values():
            for _ in range(2):
                function()
        samples = {key:[] for key in functions}
        for repeat in range(7):
            for key in list(functions) if repeat%2 == 0 else list(reversed(functions)):
                start = time.perf_counter_ns()
                functions[key]()
                samples[key].append(time.perf_counter_ns()-start)
        rows.append({"gaussians":count,"samples_ns":samples,
                     **{key+"_median_ms":statistics.median(values)/1e6 for key,values in samples.items()}})
    data = {"provenance":"measured","measured_at_utc":datetime.now(timezone.utc).isoformat(),
            "environment":{"os":platform.system(),"machine":platform.machine(),
                           "processor":platform.processor(),"python":platform.python_version()},
            "scope":"original CPU 16x12 RGB render including bin construction, validation and compositing; early termination disabled, same finite support; not gsplat/GPU",
            "timer":"perf_counter_ns","warmups_per_path":2,"repeats_per_path":7,
            "order":"alternating function order by repeat","rows":rows}
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def load_latency() -> dict:
    path = HERE/"latency.json"
    if not path.exists():
        raise ValueError("missing sealed timing file; explicitly run --measure")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["provenance"] != "measured" or data["repeats_per_path"] != 7:
        raise ValueError("invalid measured timing contract")
    if [row["gaussians"] for row in data["rows"]] != [16,64,256,1024]:
        raise ValueError("sealed timing tiers do not match")
    for row in data["rows"]:
        for key in ("tile_indexed","dense_scan"):
            samples = row["samples_ns"][key]
            if len(samples) != 7 or any(type(n) is not int or n <= 0 for n in samples):
                raise ValueError("samples must be seven positive nanosecond integers")
            if row[key+"_median_ms"] != statistics.median(samples)/1e6:
                raise ValueError("sealed median differs from samples")
    return data


def simulate() -> dict:
    layers = [(0.6,(1.0,0.0,0.0)),(0.5,(0.0,0.0,1.0))]
    correct = composite(layers)
    reverse = composite(list(reversed(layers)))
    additive = composite(layers,additive=True)
    no_background = composite(layers,omit_background=True)
    scene = fixture(12,9,7)
    tiled,dense = render(scene,9,7),render(scene,9,7,use_tiles=False)
    upstream = (0.7,-0.2,0.3)
    analytic,fd = alpha_gradients(layers,BACKGROUND,upstream),[]
    for i,(alpha,color) in enumerate(layers):
        plus,minus = layers[:],layers[:]
        plus[i],minus[i] = (alpha+1e-6,color),(alpha-1e-6,color)
        value = lambda data:math.fsum(a*b for a,b in zip(composite(data)["rgb"],upstream))
        fd.append((value(plus)-value(minus))/2e-6)
    opaque = [(0.98,(1.0,0,0)),(0.98,(0,1.0,0)),(0.98,(0,0,1.0))]
    exact_tail = composite(opaque,(0,0,0))
    cutoff = composite(opaque,(0,0,0),cutoff=T_CUTOFF)
    small = [(0.003,(0,1.0,0))]*500
    full_small,skipped_small = composite(small,(0,0,0)),composite(small,(0,0,0),alpha_floor=ALPHA_FLOOR)
    offsets = [0,2**31-8,2**31+12,2**31+52]
    return {
        "schema_version":1,"round":104,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,
        "constants":{"alpha_floor":ALPHA_FLOOR,"maximum_alpha":MAX_ALPHA,"transmittance_cutoff":T_CUTOFF},
        "layers":layers,"background":BACKGROUND,
        "compositing":{"correct":correct,"product_reference":closed_composite(layers,BACKGROUND),
                       "reversed_depth":reverse,"additive":additive,"omitted_background":no_background},
        "cases":[
            {"id":"tile-versus-same-support-dense","maximum_rgb_error":maximum_error(tiled["pixels"],dense["pixels"]),
             "intentional_approximation":False},
            *[{"id":name,"maximum_rgb_error":max(abs(a-b) for a,b in zip(data["rgb"],correct["rgb"])),
               "intentional_approximation":False} for name,data in
              (("reverse-depth",reverse),("additive-no-transmittance",additive),("omit-background",no_background))],
            {"id":"exclusive-transmittance-cutoff","maximum_rgb_error":max(abs(a-b) for a,b in zip(exact_tail["rgb"],cutoff["rgb"])),
             "intentional_approximation":True},
            {"id":"alpha-floor-many-small-layers","maximum_rgb_error":max(abs(a-b) for a,b in zip(full_small["rgb"],skipped_small["rgb"])),
             "intentional_approximation":True},
        ],
        "scene":{"width":9,"height":7,"tile":4,"gaussians":[asdict(g) for g in scene],
                 "dense_candidate_visits":dense["candidate_visits"],"tile_candidate_visits":tiled["candidate_visits"],
                 "tile_intersections":tiled["tile_intersections"]},
        "tile_variant_checks":[{"tile":size,"maximum_rgb_error":maximum_error(result["pixels"],dense["pixels"]),
                                "candidate_visits":result["candidate_visits"],"tile_intersections":result["tile_intersections"]}
                               for size in (1,2,4,8) for result in [render(scene,9,7,tile=size)]],
        "gradient":{"upstream":upstream,"effective_alpha_analytic":analytic,"effective_alpha_finite_difference":fd,
                    "maximum_error":max(abs(a-b) for a,b in zip(analytic,fd)),
                    "scope":"effective alphas away from clipping/cutoff and fixed order; not projection/covariance/sort derivatives"},
        "exclusive_cutoff":{"layers":opaque,"exact":exact_tail,"source_style_exclusive":cutoff,
                            "unit_rgb_error_bound_pre_T":cutoff["pre_exclusive_cutoff_transmittance"],
                            "note":"exclusive stop excludes current Gaussian, so hypothetical next_T does not bound omitted color"},
        "alpha_floor_accumulation":{"layers":500,"each_alpha":0.003,"full":full_small,"floor_policy":skipped_small},
        "wide_offsets":{"logical_offsets":offsets,"signed_int32_wrapped":[(v+2**31)%2**32-2**31 for v in offsets],
                        "large_arrays_allocated":False,"remaining_source_widths":{"flatten_ids":32,"tile_relative_last_ids":32}},
        "budget":estimate(Budget()),"latency":load_latency(),"image_quality":None,
        "notes":[
            "Original float64 CPU model, not bitwise parity with source float32 CUDA near thresholds.",
            "Finite square support is declared toy policy; no 3D projection, antialiasing or source ellipse-tile culling reproduction.",
            "Dense and tile references share exactly the same support, alpha cap/floor and depth order; early termination disabled.",
            "Source exclusive cutoff and per-alpha floor intentionally approximate a different untruncated compositing target.",
            "Unit RGB error bound does not generalize to unbounded/HDR feature channels.",
            "64-bit global intersection offsets do not widen every identity, tile-local counter or packed key field.",
        ],
    }


def outputs() -> dict[Path,str]:
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
    print("Round 104 artifacts match." if args.check else "Wrote Round 104 artifacts.")


if __name__ == "__main__":
    main()
