"""Original rational 1-D resize coordinates/interpolation, not native ONNX."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
MODES=("half_pixel","align_corners","asymmetric","pytorch_half_pixel")
DEFAULT={"input_side":224,"output_side":336,"channels":3,"element_bytes":4}


def extent(n,scale=None,size=None):
    if type(n) is not int or n<1 or (scale is None)==(size is None):
        raise ValueError("positive input width and exactly one of exact scale/size required")
    if size is not None:
        if type(size) is not int or size<1:
            raise ValueError("positive integer output size required")
        return size,Fraction(size,n)
    if type(scale) not in (int,Fraction) or scale<=0:
        raise ValueError("positive exact rational scale required")
    count=(n*scale).__floor__()
    if count<1:
        raise ValueError("original model excludes zero-length output")
    return count,Fraction(scale)


def coordinates(n,scale=None,size=None,mode="half_pixel",wrong_unrounded_corners=False):
    m,s=extent(n,scale,size)
    if mode not in MODES:
        raise ValueError("unsupported original coordinate mode")
    if mode=="align_corners":
        denominator=n*s-1 if wrong_unrounded_corners else m-1
        if denominator==0:
            return [Fraction()]*m
        return [Fraction(j*(n-1),denominator) for j in range(m)]
    if mode=="pytorch_half_pixel" and m==1:
        return [Fraction()]
    if mode in ("half_pixel","pytorch_half_pixel"):
        return [(Fraction(j)+Fraction(1,2))/s-Fraction(1,2) for j in range(m)]
    return [Fraction(j)/s for j in range(m)]


def interpolate(x,coords,nearest=None):
    if not x or any(type(value) not in (int,Fraction) for value in x):
        raise ValueError("nonempty exact rational data required")
    if nearest not in (None,"round_prefer_floor","round_prefer_ceil"):
        raise ValueError("original nearest mode unsupported")
    result=[]
    for raw in coords:
        p=min(max(Fraction(raw),0),len(x)-1)
        i=p.__floor__();t=p-i
        if nearest is None:
            result.append(Fraction(x[i])*(1-t)+Fraction(x[min(i+1,len(x)-1)])*t)
        else:
            choose_right=t>Fraction(1,2) or (t==Fraction(1,2) and nearest=="round_prefer_ceil")
            result.append(Fraction(x[min(i+int(choose_right),len(x)-1)]))
    return result


def hat_oracle(x,coords):
    result=[]
    for raw in coords:
        p=min(max(Fraction(raw),0),len(x)-1)
        result.append(sum((Fraction(value)*max(Fraction(),1-abs(p-i)) for i,value in enumerate(x)),Fraction()))
    return result


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(value) is not int or value<1 for value in p.values()) or p["element_bytes"] not in (1,2,4,8):
        raise ValueError("positive exact image shape and selected byte width required")
    n,m,c,b=p["input_side"],p["output_side"],p["channels"],p["element_bytes"]
    out={"selected_input_payload_bytes":n*n*c*b,"selected_output_payload_bytes":m*m*c*b,
         "selected_two_axis_float64_coordinate_bytes":2*m*8}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def float_paths(n):
    m=n*3//2
    data=[[float((j+c)%17) for j in range(n)] for c in range(8)]
    def plan():
        return [min(max((j+.5)/(m/n)-.5,0),n-1) for j in range(m)]
    def values(x,coords):
        out=[]
        for p in coords:
            i=math.floor(p);t=p-i
            out.append(x[i]*(1-t)+x[min(i+1,n-1)]*t)
        return out
    cached=plan()
    return lambda:[values(x,cached) for x in data],lambda:[values(x,plan()) for x in data]


def measure():
    rows=[]
    for n in (16,64,256,1024):
        functions=float_paths(n)
        assert functions[0]()==functions[1]()
        for _ in range(2):
            for f in functions:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"input_width":n,"output_width":n*3//2,"channels":8,"cached_samples_ms":samples[0],
                     "recompute_samples_ms":samples[1],"cached_median_ms":statistics.median(samples[0]),
                     "recompute_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python half-pixel linear samples across eight channels; cached grid versus recomputing grid per channel",
            "excludes":"initial cached-grid construction,2-D image pipeline,native ONNX/GPU,antialias,quality","rows":rows}


def build_report(latency):
    x=[1,2,3,4];scale=Fraction(3,5)
    fixed=coordinates(4,scale=scale,mode="align_corners")
    wrong=coordinates(4,scale=scale,mode="align_corners",wrong_unrounded_corners=True)
    allmodes={mode:[str(v) for v in interpolate([0,10,20],coordinates(3,size=5,mode=mode))] for mode in MODES}
    tiecoords=coordinates(2,size=3)
    return {"schema_version":1,"round":126,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"tree_date_utc":"2026-11-12T23:39:19Z","license":"Apache-2.0",
                      "resize_fix":"a95e28ea0cf151cedefba7f94f2f22e7bf7febec","fix_date_utc":"2026-09-30T16:00:35Z",
                      "chronology":"Sep30 align_corners integer-output-size fix,not new November interpolation theory"},
            "case":{"input":x,"scale":str(scale),"unrounded_output_length":str(4*scale),"integer_output_length":len(fixed),
                    "fixed_coordinates":[str(v) for v in fixed],"wrong_coordinates":[str(v) for v in wrong],
                    "fixed_output":[str(v) for v in interpolate(x,fixed)],"wrong_output":[str(v) for v in interpolate(x,wrong)],
                    "matching_coordinates":sum(a==b for a,b in zip(interpolate(x,fixed),interpolate(x,wrong))),
                    "singleton_half_pixel":[str(v) for v in interpolate(x,coordinates(4,size=1,mode="half_pixel"))],
                    "singleton_pytorch_half_pixel":[str(v) for v in interpolate(x,coordinates(4,size=1,mode="pytorch_half_pixel"))],
                    "sizes_3_to_5_outputs":allmodes,
                    "nearest_prefer_floor":[str(v) for v in interpolate([0,10],tiecoords,"round_prefer_floor")],
                    "nearest_prefer_ceil":[str(v) for v in interpolate([0,10],tiecoords,"round_prefer_ceil")],
                    "boundary":"clamp/edge replication,no antialias or ROI extrapolation"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_onnx_latency_ms":None,"vision_model_accuracy":None,"peak_hbm_bytes":None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    print("Round126 sealed artifacts match" if args.check else "Wrote Round126 original artifacts")


if __name__=="__main__":
    main()
