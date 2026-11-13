"""Original exact sampling laws and supplied-uniform identity; no native RNG/kernel reproduction."""

import argparse
import bisect
from fractions import Fraction
import itertools
import json
from pathlib import Path
import platform
import statistics
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "9b69c0e18a13c353bd6229d70780fd14dd19ff82"
MAX_EXACT = 2**53-1
DEFAULT = {"unique_rows":8,"output_rows":64,"vocabulary":131072,"index_bytes":4}


def exact(value):
    if type(value) not in (int,Fraction):
        raise ValueError("original model requires exact integer or rational input")
    return Fraction(value)


def normalize(weights):
    values = tuple(exact(w) for w in weights)
    if not values or any(w<0 for w in values) or sum(values)<=0:
        raise ValueError("nonempty nonnegative weights with positive mass required")
    total = sum(values)
    return tuple(value/total for value in values)


def nucleus(weights,candidates,threshold):
    mass = sum(weights[i] for i in candidates)
    chosen,cumulative = [],Fraction(0)
    for token in candidates:
        chosen.append(token)
        cumulative+=weights[token]
        if cumulative>=threshold*mass:
            break
    return chosen


def filtered(weights,k,p,order):
    weights = normalize(weights)
    p = exact(p)
    if type(k) is not int or not 1<=k<=len(weights) or not 0<p<=1:
        raise ValueError("original exact-k requires 1<=k<=vocabulary and 0<p<=1")
    ranking = sorted(range(len(weights)),key=lambda i:(-weights[i],i))
    top = ranking[:k]
    if order=="top_k_first":
        chosen = nucleus(weights,top,p)
    elif order=="joint":
        chosen = sorted(set(top)&set(nucleus(weights,ranking,p)))
    else:
        raise ValueError("explicit top_k_first or joint order required")
    return normalize([value if i in chosen else 0 for i,value in enumerate(weights)])


def inverse_cdf(weights,uniform):
    weights = normalize(weights)
    uniform = exact(uniform)
    if not 0<=uniform<1:
        raise ValueError("supplied exact uniform must be in [0,1)")
    cumulative = Fraction(0)
    for token,weight in enumerate(weights):
        cumulative+=weight
        if uniform<cumulative:
            return token
    raise AssertionError("normalized exact CDF must cover every allowed uniform")


def render_fractions(values):
    return [str(value) for value in values]


def supplied_uniform_outputs(requests,weights,uniforms):
    return {request:inverse_cdf(weights,uniforms[request]) for request in requests}


def estimate_budget(p):
    for key in DEFAULT:
        if type(p[key]) is not int or not 1<=p[key]<=MAX_EXACT:
            raise ValueError(f"{key} requires a positive bounded exact integer")
    if p["index_bytes"] not in (4,8):
        raise ValueError("selected index/output width must be four or eight bytes")
    result = {"selected_unique_probability_bytes":p["unique_rows"]*p["vocabulary"]*4,
              "selected_repeated_probability_bytes":p["output_rows"]*p["vocabulary"]*4,
              "selected_index_bytes":p["output_rows"]*p["index_bytes"],
              "selected_output_bytes":p["output_rows"]*p["index_bytes"]}
    if any(v>MAX_EXACT for v in result.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return result


def linear_lookup(cdf,draw):
    for token,end in enumerate(cdf):
        if draw<end:
            return token
    raise ValueError("draw outside integer CDF")


def measure():
    rows = []
    for width in (256,1024,4096,16384):
        cdf = list(itertools.accumulate((i%7+1 for i in range(width))))
        draws = [i*cdf[-1]//33 for i in range(1,33)]
        functions = (lambda:[bisect.bisect_right(cdf,u) for u in draws],
                     lambda:[linear_lookup(cdf,u) for u in draws])
        assert functions[0]()==functions[1]()
        samples = [[],[]]
        for _ in range(2):
            for function in functions:
                function()
        for repeat in range(7):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                started = time.perf_counter_ns()
                for _ in range(3):
                    functions[index]()
                samples[index].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"vocabulary":width,"draws_per_operation":32,"binary_samples_ms":samples[0],
                     "linear_samples_ms":samples[1],"binary_median_ms":statistics.median(samples[0]),
                     "linear_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original precomputed integer-CDF binary versus linear lookup, 32 supplied draws",
            "excludes":"filtering, sorting, CDF construction, random generation, canonical float sum tree and native/GPU execution",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    weights,k,p = [4,3,2,1],2,Fraction(1,2)
    first,joint = filtered(weights,k,p,"top_k_first"),filtered(weights,k,p,"joint")
    uniform = Fraction(3,5)
    uniforms = {"A":Fraction(1,10),"B":Fraction(3,5)}
    original = supplied_uniform_outputs(["A","B"],joint,uniforms)
    reordered = supplied_uniform_outputs(["B","A"],joint,uniforms)
    slot_outputs = {request:inverse_cdf(joint,u) for request,u in zip(["B","A"],uniforms.values())}
    tv = sum(abs(a-b) for a,b in zip(first,joint))/2
    return {"schema_version":1,"round":117,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"flashinfer-ai/flashinfer","pin":PIN,"date_utc":"2026-11-13T03:07:28Z",
                      "license":"Apache-2.0","recent_change":"basic categorical sampling batch_invariant=True request-keyed inverse CDF; existing filtered APIs not assumed upgraded"},
            "case":{"weights":weights,"k":k,"p":str(p),"uniform":str(uniform),
                    "top_k_first":render_fractions(first),"joint":render_fractions(joint),
                    "top_k_first_support":sum(v>0 for v in first),"joint_support":sum(v>0 for v in joint),
                    "top_k_first_token":inverse_cdf(first,uniform),"joint_token":inverse_cdf(joint,uniform),
                    "total_variation":str(tv),"total_variation_percent":float(100*tv),
                    "tie_policy":"original exact-k, descending weight then token ID; not a native tie-compatibility certificate"},
            "identity":{"original":original,"request_uniform_reordered":reordered,"slot_uniform_reordered":slot_outputs,
                        "requests":2,"request_key_matches":sum(original[key]==reordered[key] for key in original),
                        "slot_key_matches":sum(original[key]==slot_outputs[key] for key in original),
                        "note":"Uniforms supplied explicitly, no Philox/request-key RNG reproduction"},
            "bound_control":{"actual_k":3,"clipped_k":2,
                             "requested_distribution":render_fractions(filtered(weights,3,Fraction(1),"top_k_first")),
                             "clipped_distribution":render_fractions(filtered(weights,2,Fraction(1),"top_k_first")),
                             "note":"Original semantic clipping control; source fast-path bound conditions inspected, no native execution"},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_sampling_latency_ms":None,"native_rng_reproduction":False,
            "canonical_float_tree_reproduced":False,"model_quality":None,"peak_hbm_bytes":None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report = build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 117 artifacts match." if args.check else "Wrote Round 117 artifacts.")


if __name__=="__main__":
    main()
