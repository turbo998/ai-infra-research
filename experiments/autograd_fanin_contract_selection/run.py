"""Original backward-slot and rounding models, not a native autograd engine."""

import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="22e0dfd2aed75aa2bc0a44e7ff4b49753f62c3ce"
DEFAULT={"elements":1048576,"branches":3}
VALUES=[2**24,1,-2**24]


def f32(value):
    if type(value) not in (int,float):
        raise ValueError("finite numeric scalar required")
    try:
        result=struct.unpack("!f",struct.pack("!f",value))[0]
    except (OverflowError,struct.error) as error:
        raise ValueError("value does not fit finite float32") from error
    if not math.isfinite(result):
        raise ValueError("finite float32 required")
    return result


def fold(values,wide=False):
    if type(wide) is not bool:
        raise ValueError("explicit Boolean accumulation policy required")
    inputs=[f32(value) for value in values]
    total=0.0
    for value in inputs:
        total=total+value if wide else f32(total+value)
    return f32(total)


class BackwardSlots:
    def __init__(self,strict=False,parent=None,compiled=False):
        if type(strict) is not bool or type(compiled) is not bool:
            raise ValueError("explicit Boolean task flags required")
        if parent is not None and not isinstance(parent,BackwardSlots):
            raise ValueError("parent must be an original backward-slot task")
        self.strict=strict or (parent.strict if parent else False)
        if compiled and self.strict:
            raise NotImplementedError("original contract rejects strict compiled-autograd combination")
        self.slots={}

    def receive(self,slot,value,producer):
        if not isinstance(slot,str) or not slot or not isinstance(producer,str) or not producer:
            raise ValueError("nonempty slot and producer identity required")
        if value is None:
            return
        value=f32(value)
        if slot in self.slots:
            if self.strict:
                raise ValueError(f"original implicit fan-in rejected at {slot}; second producer={producer}")
            value=f32(self.slots[slot]+value)
        self.slots[slot]=value


def explicit_fanin(values):
    task=BackwardSlots(strict=True)
    for i,value in enumerate(values):
        task.receive(f"split-output:{i}",value,f"branch:{i}")
    combined=fold(list(task.slots.values()),wide=True)
    parent=BackwardSlots(strict=True)
    parent.receive("upstream:0",combined,"explicit-combiner")
    return combined,parent.slots


def estimate_budget(p):
    if any(type(p[key]) is not int or not 1<=p[key]<=2**53-1 for key in DEFAULT):
        raise ValueError("positive bounded exact element/branch counts required")
    out={"selected_float32_contribution_bytes":p["elements"]*p["branches"]*4,
         "selected_float64_accumulator_bytes":p["elements"]*8,
         "selected_float32_output_bytes":p["elements"]*4}
    if any(value>2**53-1 for value in out.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return out


def measure():
    rows=[]
    for count in (24,96,384,1536):
        values=VALUES*(count//3)
        functions=(lambda:fold(values,True),lambda:fold(values,False))
        assert functions[0]()==count//3 and functions[1]()==0
        samples=[[],[]]
        for _ in range(2):
            for function in functions:
                function()
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"contributions":count,"wide_samples_ms":samples[0],"rounded_samples_ms":samples[1],
                     "wide_median_ms":statistics.median(samples[0]),
                     "rounded_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python float64 fold/final float32 cast versus struct-rounded float32 after every add; outputs intentionally differ",
            "excludes":"autograd engine, task scheduling, view creation, native tensor operations and GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    permutations=[list(values) for values in itertools.permutations(VALUES)]
    outcomes=[fold(values) for values in permutations]
    exact=sum(Fraction(f32(value)) for value in VALUES)
    strict=BackwardSlots(strict=True)
    strict.receive("h:0",VALUES[0],"branch:0")
    error=None
    try:
        strict.receive("h:0",VALUES[1],"branch:1")
    except ValueError as caught:
        error=str(caught)
    combined,parent=explicit_fanin(VALUES)
    leaf=0.0
    for value in (1,2):
        task=BackwardSlots(strict=True)
        task.receive("leaf:0",value,"single-active-use")
        leaf=f32(leaf+task.slots["leaf:0"])
    return {"schema_version":1,"round":119,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"date_utc":"2026-11-12T16:23:08Z",
                      "license":"BSD-style repository license",
                      "recent_change":"disallow_implicit_grad_accumulation guard on a second defined incoming gradient within a GraphTask; separate backward .grad accumulation not covered"},
            "case":{"contributions":VALUES,"exact_encoded_sum":str(exact),"float32_left_fold":fold(VALUES),
                    "explicit_wide_final_float32":combined,"explicit_upstream_slots":parent,
                    "strict_second_arrival_error":error,"strict_slots_after_error":strict.slots,
                    "separate_tasks_leaf_grad":leaf,
                    "note":"Original scalar slots, not native autograd execution or transactional rollback"},
            "robustness":{"permutations":permutations,"float32_outputs":outcomes,
                          "wide_outputs":[fold(values,True) for values in permutations],
                          "orders":len(permutations),"float32_exact_matches":sum(Fraction(value)==exact for value in outcomes),
                          "wide_exact_matches":sum(Fraction(fold(values,True))==exact for values in permutations)},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_autograd_latency_ms":None,"native_precision_policy":None,
            "peak_hbm_bytes":None,"full_dag_engine_reproduced":False}


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
    print("Round 119 artifacts match." if args.check else "Wrote Round 119 artifacts.")


if __name__=="__main__":
    main()
