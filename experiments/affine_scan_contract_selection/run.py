"""Original exact ordered affine scan with explicit zero-reset controls."""

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
PIN="e9594ce1c732d97440f0332fdc43170a2294dbfa"
DEFAULT={"batch":8,"length":4096,"channels":64,"state_width":16,"element_bytes":4}


def pairs(values):
    if not values or any(len(value)!=2 or any(type(v) not in (int,Fraction) for v in value) for value in values):
        raise ValueError("nonempty exact rational affine pairs required")
    return [tuple(map(Fraction,value)) for value in values]


def compose(earlier,later):
    a,b=earlier;c,d=later
    return c*a,c*b+d


def prefix_scan(values,reverse=False):
    work=pairs(values)
    step=1
    while step<len(work):
        previous=list(work)
        for i in range(step,len(work)):
            work[i]=compose(previous[i],previous[i-step]) if reverse else compose(previous[i-step],previous[i])
        step*=2
    return work


def serial(values,initial=0):
    if type(initial) not in (int,Fraction):
        raise ValueError("exact rational initial state required")
    state=Fraction(initial);result=[]
    for a,b in pairs(values):
        state=a*state+b
        result.append(state)
    return result


def expanded_oracle(values,initial=0):
    values=pairs(values);out=[]
    for end in range(len(values)):
        base=Fraction(initial)*math.prod(a for a,_ in values[:end+1])
        out.append(base+sum((b*math.prod(a for a,_ in values[j+1:end+1]) for j,(_,b) in enumerate(values[:end+1])),Fraction()))
    return out


def apply(prefixes,initial):
    return [a*initial+b for a,b in prefixes]


def zero_resets(values,reset_indices):
    values=pairs(values)
    if any(type(i) is not int or not 0<=i<len(values) for i in reset_indices):
        raise ValueError("reset indices must address original elements")
    return [(Fraction() if i in reset_indices else a,b) for i,(a,b) in enumerate(values)]


def chunked(values,chunk_size,initial=0,wrong_reset=False):
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError("positive integer chunk size required")
    values=pairs(values);result=[];carry=initial
    for start in range(0,len(values),chunk_size):
        out=apply(prefix_scan(values[start:start+chunk_size]),initial if wrong_reset else carry)
        result.extend(out);carry=out[-1]
    return result


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(value) is not int or value<1 for value in p.values()) or p["element_bytes"] not in (1,2,4,8):
        raise ValueError("positive exact dimensions and selected byte width required")
    b,l,c,s,e=(p[k] for k in ("batch","length","channels","state_width","element_bytes"))
    out={"selected_last_state_bytes":b*c*s*e,"selected_materialized_all_states_bytes":b*l*c*s*e,
         "selected_one_affine_prefix_pair_bytes":2*b*c*s*e}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact budget range exceeded")
    return out


def float_paths(n):
    values=[(.5,float(i%7)) for i in range(n)]
    def sequential():
        state=0.;out=[]
        for a,b in values:
            state=a*state+b;out.append(state)
        return out
    def tree():
        work=list(values);step=1
        while step<n:
            previous=list(work)
            for i in range(step,n):
                work[i]=compose(previous[i-step],previous[i])
            step*=2
        return [b for _,b in work]
    return sequential,tree


def measure():
    rows=[]
    for n in (32,128,512,2048):
        functions=float_paths(n)
        assert all(abs(a-b)<1e-12 for a,b in zip(functions[0](),functions[1]()))
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
        rows.append({"length":n,"sequential_samples_ms":samples[0],"tree_samples_ms":samples[1],
                     "sequential_median_ms":statistics.median(samples[0]),"tree_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original serial Python float recurrence versus Hillis-Steele double-buffer prefix construction,not GPU parallel execution",
            "excludes":"native Mamba kernels,parameter discretization,SSMoutput projection,backward,GPU","rows":rows}


def build_report(latency):
    values=[(2,1),(3,4),(4,2),(5,3)];initial=7
    correct=serial(values,initial);tree=apply(prefix_scan(values),initial)
    reversed_values=apply(prefix_scan(values,True),initial)
    reset=zero_resets(values,{2});reset_values=serial(reset,initial)
    padded=serial([*values,(1,0),(1,0)],initial)
    badpad=serial([*values,(0,0),(0,0)],initial)
    return {"schema_version":1,"round":127,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"state-spaces/mamba","pin":PIN,"tree_date_utc":"2026-07-22T19:41:12Z",
                      "license":"Apache-2.0","scan_header_last_change":"2024-06-18ROCm support;ordered affine composition is established",
                      "scope":"original scalar affine math;zero-reset packing is our explicit extension,not claim of native reset API",
                      "latest_tree_change":"Mamba3Blackwell corruption fix;not executed or attributed to original selective_scan header"},
            "case":{"pairs":[list(value) for value in values],"initial_state":initial,"serial_states":[str(v) for v in correct],
                    "scan_states":[str(v) for v in tree],"expanded_oracle":[str(v) for v in expanded_oracle(values,initial)],
                    "reversed_composition_states":[str(v) for v in reversed_values],
                    "reversed_matching_states":sum(a==b for a,b in zip(correct,reversed_values)),"states":len(correct),
                    "reset_before_index":2,"zero_reset_states":[str(v) for v in reset_values],
                    "no_reset_leaked_final_state":str(correct[-1]),"reset_final_state":str(reset_values[-1]),
                    "chunk_size":2,"carried_chunk_states":[str(v) for v in chunked(values,2,initial)],
                    "wrong_chunk_restart_states":[str(v) for v in chunked(values,2,initial,True)],
                    "identity_padded_last_state":str(padded[-1]),"zero_pair_padded_last_state":str(badpad[-1]),
                    "final_prefix":[str(v) for v in prefix_scan(values)[-1]]},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_mamba_latency_ms":None,"floating_bitwise_equivalence":None,"model_quality":None,"peak_hbm_bytes":None}


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
    print("Round127 sealed artifacts match" if args.check else "Wrote Round127 original artifacts")


if __name__=="__main__":
    main()
