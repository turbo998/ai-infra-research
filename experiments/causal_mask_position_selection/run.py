"""Original exact non-square mask and uniform-logit attention model."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff"
DEFAULT={"batch":4,"heads":32,"queries":2,"keys":8192,"additive_element_bytes":4}


def positions(values):
    if not values or any(type(v) is not int or abs(v)>2**53-1 for v in values):
        raise ValueError("nonempty bounded exact integer positions required")
    return list(values)


def positional_mask(query_positions,key_positions):
    queries,keys=positions(query_positions),positions(key_positions)
    return [[key<=query for key in keys] for query in queries]


def triangular_mask(queries,keys,lower_right=False):
    if type(queries) is not int or type(keys) is not int or queries<1 or keys<1:
        raise ValueError("positive integer query/key counts required")
    offset=keys-queries if lower_right else 0
    return [[j<=i+offset for j in range(keys)] for i in range(queries)]


def attend_uniform(values,mask):
    if not values or any(type(v) not in (int,Fraction) for v in values) or not mask:
        raise ValueError("exact values and nonempty mask rows required")
    output=[]
    for row in mask:
        if len(row)!=len(values) or any(type(v) is not bool for v in row):
            raise ValueError("boolean allow mask must match value axis")
        allowed=[Fraction(value) for value,keep in zip(values,row) if keep]
        output.append({"kind":"value","value":str(sum(allowed,Fraction())/len(allowed))}
                      if allowed else {"kind":"no_allowed_keys","value":None})
    return output


def independent_prefix_oracle(values,query_positions,key_positions):
    queries,keys=positions(query_positions),positions(key_positions)
    result=[]
    for query in queries:
        total=Fraction();count=0
        for i in sorted(range(len(keys)),key=lambda i:keys[i]):
            if keys[i]>query:
                break
            total+=Fraction(values[i]);count+=1
        result.append({"kind":"value","value":str(total/count)} if count else {"kind":"no_allowed_keys","value":None})
    return result


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()) or p["additive_element_bytes"] not in (2,4,8):
        raise ValueError("positive exact dimensions and selected additive width required")
    b,h,q,k,e=(p[key] for key in ("batch","heads","queries","keys","additive_element_bytes"))
    out={"selected_shared_2d_bool_mask_bytes":q*k,
         "selected_bh_bool_mask_if_materialized_bytes":b*h*q*k,
         "selected_shared_2d_additive_mask_bytes":q*k*e,
         "selected_int64_prefix_limits_bytes":q*8}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def float_paths(n):
    values=[float(i%13) for i in range(n)];q=8
    def dense():
        return [sum(value for j,value in enumerate(values) if j<=n-q+i)/(n-q+i+1) for i in range(q)]
    def prefix():
        sums=[0.0]
        for value in values:
            sums.append(sums[-1]+value)
        return [sums[n-q+i+1]/(n-q+i+1) for i in range(q)]
    return prefix,dense


def measure():
    rows=[]
    for n in (128,512,2048,8192):
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
        rows.append({"keys":n,"queries":8,"prefix_samples_ms":samples[0],"dense_samples_ms":samples[1],
                     "prefix_median_ms":statistics.median(samples[0]),"dense_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original uniform-logit scalar means,prefix sums rebuilt per operation versus repeated dense allow-coordinate filtering",
            "excludes":"QKmatmul,softmax,general attention,native SDPA/GPU,cache movement,quality","rows":rows}


def build_report(latency):
    values=[1,2,4,8,16];qpos=[3,4];kpos=list(range(5))
    correct=positional_mask(qpos,kpos)
    upper=triangular_mask(2,5);lower=triangular_mask(2,5,True)
    wrong_polarity=[[not v for v in row] for row in correct]
    gapkeys=[0,2,4,6,8];gapped=positional_mask(qpos,gapkeys)
    return {"schema_version":1,"round":130,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"tree_date_utc":"2026-11-14T00:44:56Z",
                      "license":"BSD-style","chronology":"Nov14regionalAOTI tree update is not new causal-mask semantics",
                      "verified":"SDPA non-square is_causal upper-left;lower-rightoffsetS-L;SDPATrueallow versus MHATrueblocked"},
            "case":{"values":values,"query_positions":qpos,"key_positions":kpos,
                    "upper_left_allow":upper,"lower_right_allow":lower,"positional_allow":correct,
                    "upper_left_allowed_counts":[sum(row) for row in upper],"lower_right_allowed_counts":[sum(row) for row in lower],
                    "correct_uniform_outputs":attend_uniform(values,correct),"upper_left_uniform_outputs":attend_uniform(values,upper),
                    "wrong_polarity_uniform_outputs":attend_uniform(values,wrong_polarity),
                    "independent_outputs":independent_prefix_oracle(values,qpos,kpos),
                    "gapped_key_positions":gapkeys,"gapped_correct_allow":gapped,
                    "gapped_correct_uniform_outputs":attend_uniform(values,gapped),
                    "gapped_wrong_lower_right_outputs":attend_uniform(values,lower),
                    "one_query_upper_left":attend_uniform(values,triangular_mask(1,5)),
                    "one_query_lower_right":attend_uniform(values,triangular_mask(1,5,True)),
                    "empty_row_policy":"explicit tagged no_allowed_keys,null;not native backend all-masked behavior"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_sdpa_latency_ms":None,"model_quality":None,"native_all_masked_behavior":None,"peak_hbm_bytes":None}


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
    print("Round130 sealed artifacts match" if args.check else "Wrote Round130 original artifacts")


if __name__=="__main__":
    main()
