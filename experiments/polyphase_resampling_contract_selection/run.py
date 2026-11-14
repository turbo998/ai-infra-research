"""Original exact centered-FIR rate conversion, not native SciPy execution."""

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
PIN="30149047c624e328459a4d66d1e5f8b047b7bde7"
DEFAULT={"input_samples":44100,"up":48000,"down":44100,"element_bytes":4}
FIR=[Fraction(v,9) for v in (1,2,3,2,1)]


def rate(up,down):
    if type(up) is not int or type(down) is not int or not 1<=up<=2**53-1 or not 1<=down<=2**53-1:
        raise ValueError("positive bounded exact integer rates required")
    g=math.gcd(up,down)
    return up//g,down//g


def budget(p):
    u,d=rate(p["up"],p["down"])
    if type(p["input_samples"]) is not int or not 1<=p["input_samples"]<=2**53-1 or type(p["element_bytes"]) is not int or p["element_bytes"] not in (1,2,4,8):
        raise ValueError("nonempty bounded input and selected 1/2/4/8 byte width required")
    count=(p["input_samples"]*u+d-1)//d
    result={"reduced_up":u,"reduced_down":d,"output_samples":count,
            "input_payload_bytes":p["input_samples"]*p["element_bytes"],
            "output_payload_bytes":count*p["element_bytes"]}
    if max(result.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return result


def validate(x,h,up,down):
    u,d=rate(up,down)
    if not x or not h or len(h)%2!=1 or list(h)!=list(reversed(h)):
        raise ValueError("original model requires nonempty input and odd symmetric exact FIR")
    if any(type(value) not in (int,Fraction) for value in [*x,*h]):
        raise ValueError("exact rational input and coefficients required")
    return u,d


def polyphase(x,h,up,down,phase_error=0,scale=True):
    u,d=validate(x,h,up,down)
    if u==d==1:
        return list(map(Fraction,x))
    count=(len(x)*u+d-1)//d
    half=len(h)//2
    result=[]
    for m in range(count):
        center=m*d+half+phase_error
        first=max(0,(center-len(h)+1+u-1)//u)
        last=min(len(x)-1,center//u)
        result.append(sum((Fraction(x[k])*h[center-k*u]*(u if scale else 1)
                           for k in range(first,last+1)),Fraction()))
    return result


def dense_oracle(x,h,up,down):
    u,d=validate(x,h,up,down)
    if u==d==1:
        return list(map(Fraction,x))
    inserted=[Fraction()] * ((len(x)-1)*u+1)
    for i,value in enumerate(x):
        inserted[i*u]=Fraction(value)
    convolved=[Fraction()] * (len(inserted)+len(h)-1)
    for i,a in enumerate(inserted):
        for j,b in enumerate(h):
            convolved[i+j]+=a*b*u
    output=(len(x)*u+d-1)//d
    return [convolved[m*d+len(h)//2] if m*d+len(h)//2<len(convolved) else Fraction()
            for m in range(output)]


def float_paths(n):
    x=[float((i%13)-6) for i in range(n)]
    h=[float(v) for v in FIR]
    def compact():
        out=[]
        for m in range((n*3+1)//2):
            center=2*m+2
            first=max(0,(center-4+2)//3);last=min(n-1,center//3)
            out.append(sum(x[k]*h[center-3*k]*3 for k in range(first,last+1)))
        return out
    def dense():
        z=[0.0]*((n-1)*3+1)
        for i,value in enumerate(x):
            z[i*3]=value
        y=[0.0]*(len(z)+4)
        for i,value in enumerate(z):
            for j,coefficient in enumerate(h):
                y[i+j]+=value*coefficient*3
        return [y[2*m+2] if 2*m+2<len(y) else 0.0 for m in range((n*3+1)//2)]
    return compact,dense


def measure():
    rows=[]
    for n in (32,128,512,2048):
        functions=float_paths(n)
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
        rows.append({"input_samples":n,"polyphase_samples_ms":samples[0],"dense_samples_ms":samples[1],
                     "polyphase_median_ms":statistics.median(samples[0]),"dense_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python float nonzero-phase gather versus explicit zero insertion and dense FIR convolution, U3/D2 fixed five taps",
            "excludes":"FIR design, native SciPy, streaming/chunk state, GPU, quality assessment","rows":rows}


def build_report(latency):
    x=[1,2,3,4,5]
    correct=polyphase(x,FIR,3,2);shifted=polyphase(x,FIR,3,2,1);unscaled=polyphase(x,FIR,3,2,scale=False)
    alternating=[1,-1]*8;lowpass=[Fraction(1,4),Fraction(1,2),Fraction(1,4)]
    filtered=polyphase(alternating,lowpass,1,2);naive=alternating[::2]
    dc=polyphase([1]*16,lowpass,1,2)
    return {"schema_version":1,"round":125,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"scipy/scipy","pin":PIN,"tree_date_utc":"2026-11-13T23:28:02Z",
                      "license":"BSD-3-Clause","signaltools_last_change":"2026-10-23 lfilter/sosfilt empty-input handling; resample_poly rules are established, not new launch"},
            "case":{"input":x,"fir":[str(v) for v in FIR],"up":3,"down":2,
                    "centered_output":[str(v) for v in correct],"dense_oracle":[str(v) for v in dense_oracle(x,FIR,3,2)],
                    "one_phase_shift_output":[str(v) for v in shifted],"unscaled_output":[str(v) for v in unscaled],
                    "phase_shift_matching_coordinates":sum(a==b for a,b in zip(correct,shifted)),
                    "unscaled_matching_coordinates":sum(a==b for a,b in zip(correct,unscaled)),
                    "output_coordinates":len(correct),"alternating_input":alternating,
                    "naive_decimation":naive,"filtered_decimation":[str(v) for v in filtered],
                    "interior_filtered_energy":str(sum(v*v for v in filtered[1:])),
                    "interior_naive_energy":sum(v*v for v in naive[1:]),
                    "interior_samples":len(filtered)-1,"dc_filtered":[str(v) for v in dc],
                    "boundary_policy":"zero extension only; centered custom odd symmetric FIR; no native default Kaiser design"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_scipy_latency_ms":None,"audio_quality":None,"gpu_peak_bytes":None}


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
    print("Round125 sealed artifacts match" if args.check else "Wrote Round125 original artifacts")


if __name__=="__main__":
    main()
