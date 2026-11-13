"""Original exact split-conformal ranks and exhaustive holdout model."""

import argparse
from fractions import Fraction
import heapq
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="d5ea5e07aea09a897726fd8641c193850a5c540d"
DEFAULT={"calibration_count":1000,"query_count":128,"alpha_num":1,"alpha_den":20}


def exact(value):
    if type(value) not in (int,Fraction):
        raise ValueError("exact integer/rational input required")
    return Fraction(value)


def ceil_ratio(value):
    return -(-value.numerator//value.denominator)


def rank(n,alpha):
    alpha=exact(alpha)
    if type(n) is not int or n<1 or not 0<alpha<1:
        raise ValueError("positive calibration size and 0<alpha<1 required")
    return ceil_ratio((n+1)*(1-alpha))


def threshold(scores,alpha):
    scores=[exact(value) for value in scores]
    if not scores or any(value<0 for value in scores):
        raise ValueError("nonempty nonnegative exact conformity scores required")
    k=rank(len(scores),alpha)
    return {"kind":"unbounded","value":None,"rank":k,"calibration_count":len(scores)} if k>len(scores) else {
        "kind":"finite","value":str(sorted(scores)[k-1]),"rank":k,"calibration_count":len(scores)}


def contains(bound,score,inclusive=True):
    score=exact(score)
    if bound["kind"]=="unbounded":
        return True
    value=Fraction(bound["value"])
    return score<=value if inclusive else score<value


def exhaustive_holdout(world,alpha,mode="correct",inclusive=True):
    world=[exact(value) for value in world]
    if len(world)<2:
        raise ValueError("at least two scores required for holdout enumeration")
    successes=0
    for i,test in enumerate(world):
        calibration=world[:i]+world[i+1:]
        bound=threshold(calibration,alpha)
        if mode=="naive_n":
            k=ceil_ratio(len(calibration)*(1-exact(alpha)))
            bound={"kind":"finite","value":str(sorted(calibration)[k-1])}
        elif mode=="clip_max":
            bound={"kind":"finite","value":str(max(calibration))}
        elif mode!="correct":
            raise ValueError("explicit correct, naive_n or clip_max mode required")
        successes+=contains(bound,test,inclusive)
    return {"covered":successes,"heldouts":len(world),"coverage_percent":100*successes/len(world)}


def estimate_budget(p):
    if any(type(p[key]) is not int or not 1<=p[key]<=2**53-1 for key in DEFAULT):
        raise ValueError("positive bounded exact counts and alpha ratio required")
    alpha=Fraction(p["alpha_num"],p["alpha_den"])
    k=rank(p["calibration_count"],alpha)
    out={"selected_float64_score_bytes":p["calibration_count"]*8,
         "selected_optional_sorted_copy_bytes":p["calibration_count"]*8,
         "selected_float64_interval_output_bytes":p["query_count"]*2*8,
         "corrected_rank":k,
         "minimum_n_for_finite_single_threshold":ceil_ratio((1-alpha)/alpha)}
    if any(value>2**53-1 for value in out.values()):
        raise ValueError("selected result exceeds exact arithmetic range")
    return out


def measure():
    rows=[]
    for n in (32,128,512,2048):
        scores=[(i*37)%(n+7) for i in range(n)]
        k=rank(n,Fraction(1,5))
        functions=(lambda:heapq.nsmallest(k,scores)[-1],lambda:sorted(scores)[k-1])
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
        rows.append({"scores":n,"corrected_rank":k,"heap_samples_ms":samples[0],
                     "sort_samples_ms":samples[1],"heap_median_ms":statistics.median(samples[0]),
                     "sort_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original stdlib heapq order-statistic orchestration versus full sorted selection on precomputed integer scores; alpha=1/5",
            "excludes":"score/model fitting, score generation, model inference, native MAPIE and GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    world=list(range(1,11));alpha=Fraction(3,20)
    proper=exhaustive_holdout(world,alpha)
    naive=exhaustive_holdout(world,alpha,"naive_n")
    small_alpha=Fraction(1,20)
    return {"schema_version":1,"round":122,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"scikit-learn-contrib/MAPIE","pin":PIN,"tree_date_utc":"2026-11-10T09:25:13Z",
                      "license":"BSD-3-Clause","recent_tree_change":"educational documentation; finite-sample conformal theory is not new",
                      "inspected":"corrected quantile helpers, explicit unbounded/insufficient-sample handling; not every API modeled"},
            "case":{"calibration_scores":list(range(1,10)),"alpha":str(alpha),"target_percent":float(100*(1-alpha)),
                    "correct":threshold(list(range(1,10)),alpha),"naive_rank":ceil_ratio(9*(1-alpha)),
                    "naive_threshold":"8","exhaustive_correct":proper,"exhaustive_naive":naive,
                    "world":world,"scope":"all ten equally likely heldout identities, fixed score mechanism; synthetic enumeration, not real-data evaluation"},
            "small_calibration":{"alpha":str(small_alpha),"target_percent":float(100*(1-small_alpha)),
                                 "bound":threshold(list(range(1,10)),small_alpha),
                                 "correct":exhaustive_holdout(world,small_alpha),
                                 "wrong_clip_max":exhaustive_holdout(world,small_alpha,"clip_max")},
            "ties":{"inclusive":exhaustive_holdout([2]*10,alpha),
                    "wrong_strict":exhaustive_holdout([2]*10,alpha,inclusive=False)},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"real_data_coverage":None,"conditional_group_coverage":None,
            "native_mapie_latency_ms":None,"model_quality":None,"peak_hbm_bytes":None}


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
    print("Round 122 artifacts match." if args.check else "Wrote Round 122 artifacts.")


if __name__=="__main__":
    main()
