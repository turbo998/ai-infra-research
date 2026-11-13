"""Original exact-ratio integer QuantizeLinear contracts; not native ONNX execution."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
DEFAULT={"rows":1024,"columns":4096,"axis":1}
RANGES={"int8":(-128,127),"uint8":(0,255)}


def exact(value):
    if type(value) not in (int,Fraction):
        raise ValueError("original model requires exact integer or rational input")
    return Fraction(value)


def nearest_even(value):
    value=exact(value)
    whole,remainder=divmod(abs(value.numerator),value.denominator)
    if 2*remainder>value.denominator or (2*remainder==value.denominator and whole%2):
        whole+=1
    return -whole if value<0 else whole


def quantized_ratio(ratio,zero,dtype="int8"):
    ratio=exact(ratio)
    if dtype not in RANGES:
        raise ValueError("original integer scope is int8 or uint8")
    low,high=RANGES[dtype]
    if type(zero) is not int or not low<=zero<=high:
        raise ValueError("zero point must fit selected output dtype")
    rounded=nearest_even(ratio)
    if abs(rounded)>2**31-1:
        raise ValueError("original model restricts rounded quotient to signed-int32 magnitude")
    return max(low,min(high,rounded+zero))


def quantize(matrix,scales,zeros,axis=1,dtype="int8"):
    if not matrix or not matrix[0] or any(len(row)!=len(matrix[0]) for row in matrix):
        raise ValueError("nonempty rectangular rank-two matrix required")
    if type(axis) is not int or axis not in (-2,-1,0,1):
        raise ValueError("rank-two axis must be -2,-1,0 or 1")
    axis%=2
    width=(len(matrix),len(matrix[0]))[axis]
    scales=[exact(value) for value in scales]
    if len(scales)!=width or len(zeros)!=width or any(scale<=0 for scale in scales):
        raise ValueError("positive per-axis scales and zero points must match selected dimension")
    return [[quantized_ratio(exact(value)/scales[(r,c)[axis]],zeros[(r,c)[axis]],dtype)
             for c,value in enumerate(row)] for r,row in enumerate(matrix)]


def estimate_budget(p):
    if any(type(p[key]) is not int for key in DEFAULT):
        raise ValueError("exact integer dimensions/axis required")
    if not 1<=p["rows"]<=2**53-1 or not 1<=p["columns"]<=2**53-1 or p["axis"] not in (-2,-1,0,1):
        raise ValueError("positive bounded dimensions and rank-two axis required")
    selected=(p["rows"],p["columns"])[p["axis"]%2]
    out={"selected_float32_input_bytes":p["rows"]*p["columns"]*4,
         "selected_one_byte_output_bytes":p["rows"]*p["columns"],
         "selected_float32_scale_bytes":selected*4,
         "selected_one_byte_zero_point_bytes":selected}
    if any(value>2**53-1 for value in out.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return out


def flat_quantize(values,scales,zeros,cached):
    reciprocals=[1/scale for scale in scales] if cached else None
    return [quantized_ratio(value*reciprocals[i%len(scales)] if cached else value/scales[i%len(scales)],
                            zeros[i%len(scales)]) for i,value in enumerate(values)]


def measure():
    rows=[]
    scales=[Fraction(1,2),Fraction(1,4),Fraction(3,2),Fraction(1)]
    zeros=[-3,1,7,0]
    for count in (32,128,512,2048):
        values=[Fraction(i%21-10,4) for i in range(count)]
        functions=(lambda:flat_quantize(values,scales,zeros,True),
                   lambda:flat_quantize(values,scales,zeros,False))
        assert functions[0]()==functions[1]()
        for _ in range(2):
            for function in functions:
                function()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"elements":count,"cached_samples_ms":samples[0],"division_samples_ms":samples[1],
                     "cached_median_ms":statistics.median(samples[0]),
                     "division_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Fraction cached per-axis reciprocal multiplication versus per-element division, same exact integer outputs; reciprocal construction included",
            "excludes":"native float division precision, ONNX runtime, tensor layout conversion and GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    matrix=[[Fraction(1,2),Fraction(3,2)],[-Fraction(1,2),-Fraction(3,2)]]
    scales=[Fraction(1),Fraction(1,2)];zeros=[1,-1]
    result=quantize(matrix,scales,zeros)
    wrong_axis=quantize(matrix,scales,zeros,axis=0)
    wrong_round=[[nearest_even(value/scales[c]+zeros[c]) for c,value in enumerate(row)] for row in matrix]
    matches=lambda other:sum(a==b for rowa,rowb in zip(result,other) for a,b in zip(rowa,rowb))
    reconstruction=[[(result[r][c]-zeros[c])*scales[c] for c in range(2)] for r in range(2)]
    errors=[abs(matrix[r][c]-reconstruction[r][c]) for r in range(2) for c in range(2)]
    return {"schema_version":1,"round":121,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"tree_date_utc":"2026-11-12T23:39:19Z",
                      "license":"Apache-2.0","reference_last_change":"2026-08-29 float6 type support; integer law separately scoped, not a November algorithm launch"},
            "case":{"inputs":[[str(v) for v in row] for row in matrix],"scales":[str(v) for v in scales],
                    "zero_points":zeros,"axis":1,"dtype":"int8","correct":result,
                    "wrong_axis":wrong_axis,"wrong_zero_inside_rounding":wrong_round,
                    "elements":4,"correct_matches":matches(result),"wrong_axis_matches":matches(wrong_axis),
                    "wrong_rounding_matches":matches(wrong_round),
                    "reconstructed":[[str(v) for v in row] for row in reconstruction],
                    "max_exact_reconstruction_error":str(max(errors))},
            "boundary":{"half_ties":[str(Fraction(v,2)) for v in (-5,-3,-1,1,3,5)],
                        "rounded":[nearest_even(Fraction(v,2)) for v in (-5,-3,-1,1,3,5)],
                        "upper_saturation":quantized_ratio(Fraction(130),1),
                        "lower_saturation":quantized_ratio(Fraction(-140),1),
                        "odd_zero_point_counterexample":{"ratio":"1/2","zero_point":1,"round_then_add":1,"add_then_round":2}},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_quantize_latency_ms":None,"native_precision_compatibility":False,
            "fp6_fp8_int4_coverage":False,"peak_hbm_bytes":None,"model_quality":None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
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
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 121 artifacts match." if args.check else "Wrote Round 121 artifacts.")


if __name__=="__main__":
    main()
