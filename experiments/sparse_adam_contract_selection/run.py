"""Original scalar SparseAdam contract and coalescing models; no native optimizer."""

import argparse
from decimal import Decimal,localcontext
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="22e0dfd2aed75aa2bc0a44e7ff4b49753f62c3ce"
DEFAULT={"rows":16384,"width":128,"stored_rows":128,"unique_rows":64}


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError("finite numeric scalar required")
    return float(value)


def validate_entries(entries,rows):
    if type(rows) is not int or rows<1:
        raise ValueError("positive exact row count required")
    for index,value in entries:
        if type(index) is not int or not 0<=index<rows:
            raise ValueError("stored row index outside table")
        finite(value)


def coalesce(entries,rows):
    validate_entries(entries,rows)
    groups={}
    for row,value in entries:
        groups.setdefault(row,[]).append(value)
    return {row:finite(math.fsum(values)) for row,values in groups.items()}


def sorted_coalesce(entries,rows):
    validate_entries(entries,rows)
    return {row:finite(math.fsum(value for _,value in values))
            for row,values in itertools.groupby(sorted(entries,key=lambda pair:pair[0]),key=lambda pair:pair[0])}


class ScalarTable:
    def __init__(self,weights,lr=.1,beta1=.5,beta2=.5,eps=0):
        self.weights=[finite(value) for value in weights]
        if not self.weights:
            raise ValueError("nonempty scalar table required")
        self.lr,self.beta1,self.beta2,self.eps=map(finite,(lr,beta1,beta2,eps))
        if self.lr<=0 or not 0<=self.beta1<1 or not 0<=self.beta2<1 or self.eps<0:
            raise ValueError("require lr>0, 0<=betas<1, eps>=0")
        self.m=[0.0]*len(weights)
        self.v=[0.0]*len(weights)
        self.step_number=0

    def step(self,entries):
        if entries is None:
            return
        gradients=coalesce(entries,len(self.weights))
        self.step_number+=1
        t=self.step_number
        scale=self.lr*math.sqrt(1-self.beta2**t)/(1-self.beta1**t)
        for row,g in gradients.items():
            moment=self.beta1*self.m[row]+(1-self.beta1)*g
            square=self.beta2*self.v[row]+(1-self.beta2)*g*g
            denominator=math.sqrt(square)+self.eps
            if denominator<=0 or not math.isfinite(denominator):
                raise ValueError("original model requires finite positive update denominator; native NaN behavior not reproduced")
            delta=finite(scale*moment/denominator)
            self.m[row],self.v[row]=moment,square
            self.weights[row]=finite(self.weights[row]-delta)

    def snapshot(self):
        return {"weights":list(self.weights),"first_moments":list(self.m),
                "second_moments":list(self.v),"parameter_step":self.step_number}


def decimal_delta(gradients,step,lr="0.1",beta1="0.5",beta2="0.5",eps="0"):
    with localcontext() as ctx:
        ctx.prec=50
        b1,b2=Decimal(beta1),Decimal(beta2)
        m=v=Decimal(0)
        for gradient in gradients:
            g=Decimal(str(gradient))
            m=b1*m+(1-b1)*g
            v=b2*v+(1-b2)*g*g
        return float(Decimal(lr)*(1-b2**step).sqrt()/(1-b1**step)*m/(v.sqrt()+Decimal(eps)))


def estimate_budget(p):
    if any(type(p[key]) is not int or not 0<=p[key]<=2**53-1 for key in DEFAULT):
        raise ValueError("bounded exact nonnegative counts required")
    if p["rows"]<1 or p["width"]<1 or p["unique_rows"]>min(p["rows"],p["stored_rows"]):
        raise ValueError("positive table dimensions and feasible unique/stored row counts required")
    out={"selected_float32_parameter_bytes":p["rows"]*p["width"]*4,
         "selected_dense_float32_two_moments_bytes":p["rows"]*p["width"]*8,
         "selected_raw_sparse_gradient_bytes":p["stored_rows"]*(8+p["width"]*4),
         "selected_coalesced_sparse_gradient_bytes":p["unique_rows"]*(8+p["width"]*4)}
    if any(value>2**53-1 for value in out.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return out


def measure():
    rows=[]
    for count in (32,128,512,2048):
        entries=[((i*17)%64,(i%13)-6) for i in range(count)]
        functions=(lambda:coalesce(entries,64),lambda:sorted_coalesce(entries,64))
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
        rows.append({"stored_entries":count,"hash_samples_ms":samples[0],"sorted_samples_ms":samples[1],
                     "hash_median_ms":statistics.median(samples[0]),"sorted_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python hash-group/f-sum versus sorted-group/f-sum scalar coalescing with input validation; exact small integer-valued inputs",
            "excludes":"optimizer moment/parameter updates, tensor sparse layout conversion and native/GPU execution",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    duplicate=ScalarTable([1,2])
    duplicate.step([(0,1),(0,2)])
    correct_delta=1-duplicate.weights[0]
    wrong_delta=.1*3/math.sqrt(1+4)
    absent=ScalarTable([1,2]);zero=ScalarTable([1,2])
    for table in (absent,zero):
        table.step([(1,4)])
    absent.step([])
    zero.step([(1,0)])
    late=ScalarTable([1,2])
    late.step([(0,3)])
    late.step([(1,4)])
    return {"schema_version":1,"round":120,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"BSD-style repository license",
                      "scope":"existing SparseAdam contracts, not a November algorithm launch",
                      "functional_api_change":"2026-08-28 public functional APIs; caller owns current per-parameter step"},
            "case":{"raw_entries":[[0,1],[0,2]],"coalesced_gradient":3,"coalesced_square":9,
                    "wrong_sum_of_squares":5,"correct_first_delta":correct_delta,
                    "wrong_square_first_delta":wrong_delta,"duplicate_state":duplicate.snapshot(),
                    "absent_row_after_second_step":absent.snapshot(),
                    "materialized_zero_after_second_step":zero.snapshot(),
                    "late_first_row_update":late.snapshot(),"late_row_delta":2-late.weights[1],
                    "wrong_row_local_step_delta":.1,
                    "hyperparameters":{"lr":.1,"beta1":.5,"beta2":.5,"eps":0}},
            "robustness":{"duplicate_permutations":2,
                          "coalesced_values":[coalesce(entries,2)[0] for entries in ([(0,1),(0,2)],[(0,2),(0,1)])],
                          "cancelled_materialized_zero":[{"row":row,"value":value} for row,value in coalesce([(1,1),(1,-1)],2).items()],
                          "note":"zero-valued stored key retained; not a promise of arbitrary floating coalescing order invariance"},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_optimizer_latency_ms":None,"native_precision_compatibility":False,
            "peak_hbm_bytes":None,"optimizer_state_compaction_claim":False}


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
    print("Round 120 artifacts match." if args.check else "Wrote Round 120 artifacts.")


if __name__=="__main__":
    main()
