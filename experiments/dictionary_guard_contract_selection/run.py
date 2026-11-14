"""Original selective guard-cache model and native CPython-only observations."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="f407abb11f88c3b04bffba64a8f8d61e7c0207e2"
PROGRAMS=("named","order","unexpected","duplicate","type","constructor")
DEFAULT={"program":"order","calls":[{"mapping_type":"dict","pairs":[["alpha",2],["beta",5]],"base":3},
                                   {"mapping_type":"dict","pairs":[["beta",5],["alpha",2]],"base":3}]}


def integer(v,lo,hi):
    if type(v) is not int or not lo<=v<=hi:
        raise ValueError("bounded exact integer required")
    return v


def validate(program,call):
    if not isinstance(call,dict) or set(call)!={"mapping_type","pairs","base"} or call["mapping_type"] not in ("dict","OrderedDict"):
        raise ValueError("exact original mapping_type/pairs/base required")
    integer(call["base"],-10**6,10**6);pairs=call["pairs"]
    if not isinstance(pairs,list) or len(pairs)>16:
        raise ValueError("original maximum16pairs required")
    keys=[]
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2 or not isinstance(pair[0],str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,15}",pair[0]):
            raise ValueError("original unique bounded ASCII keyword/value pairs required")
        integer(pair[1],-10**6,10**6);keys.append(pair[0])
    if len(set(keys))!=len(keys):
        raise ValueError("original storage keys must be unique")
    if program in ("named","type","duplicate") and any(k not in ("alpha","beta") for k in keys):
        raise ValueError("original named/duplicate/type case requires alpha/beta keys")
    if program=="unexpected" and any(k in ("alpha","beta") for k in keys):
        raise ValueError("original unexpected case requires only unexpected names")
    return keys


def result(program,call,planned_keys=None,planned_type=None):
    keys=[p[0] for p in call["pairs"]] if planned_keys is None else planned_keys
    values=dict(call["pairs"]);value=None;order=None;error=None;error_key=None;kind=None
    if program in ("unexpected","duplicate") and keys:
        error=program;error_key=keys[0]
    elif program in ("order","constructor"):
        value=call["base"]+sum(values.values());order=list(keys)
    elif program=="duplicate":
        value=call["base"]+120
    else:
        value=call["base"]+10*values.get("alpha",0)+values.get("beta",0)
    if program=="type":
        kind=call["mapping_type"] if planned_type is None else planned_type
    return {"value":value,"order":order,"error_kind":error,"error_key":error_key,"mapping_type":kind}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!={"program","calls"} or p["program"] not in PROGRAMS:
        raise ValueError("supported original program/calls required")
    if not isinstance(p["calls"],list) or not 1<=len(p["calls"])<=16:
        raise ValueError("original1..16calls required")
    program=p["program"];observes_order=program in ("order","constructor","unexpected","duplicate")
    guarded={};naive={};rows=[]
    for i,call in enumerate(p["calls"]):
        keys=validate(program,call);key_set=tuple(sorted(keys))
        guard_key=(call["mapping_type"],len(keys),tuple(keys) if observes_order else key_set)
        expected=result(program,call)
        guard_hit=guard_key in guarded;naive_hit=key_set in naive
        guarded.setdefault(guard_key,(list(keys),call["mapping_type"]))
        naive.setdefault(key_set,(list(keys),call["mapping_type"]))
        correct=result(program,call,*guarded[guard_key]);bad=result(program,call,*naive[key_set])
        rows.append({"call":i,"inputs":call,"expected":expected,"guarded":correct,"naive":bad,
                     "guarded_cache_hit":guard_hit,"naive_cache_hit":naive_hit,"naive_matches":bad==expected})
    return {"program":program,"observes_order":observes_order,"rows":rows,
            "guarded_plans":len(guarded),"naive_plans":len(naive),"guarded_misses":sum(v["guarded"]!=v["expected"] for v in rows),
            "naive_misses":sum(not v["naive_matches"] for v in rows),
            "native_torch_frame_count":None}


def native_oracle():
    def named(base,alpha=0,beta=0):
        return base+10*alpha+beta
    def ordered(base,**kwargs):
        return {"value":base+sum(kwargs.values()),"order":list(kwargs)}
    observations=[]
    for program in PROGRAMS:
        for reverse in (False,True):
            pairs=[("omega",2),("sigma",5)] if program=="unexpected" else [("alpha",2),("beta",5)]
            if reverse:
                pairs.reverse()
            mapping=OrderedDict(pairs) if program=="type" and reverse else dict(pairs)
            try:
                if program=="duplicate":
                    output={"value":named(3,10,20,**mapping)}
                elif program=="order":
                    output=ordered(3,**mapping)
                elif program=="constructor":
                    copied=dict(**mapping);output={"value":3+sum(copied.values()),"order":list(copied)}
                elif program=="type":
                    output={"value":named(3,**mapping),"mapping_type":type(mapping).__name__}
                else:
                    output={"value":named(3,**mapping)}
            except TypeError as error:
                match=re.search(r"'([^']+)'",str(error))
                if match is None:
                    raise RuntimeError("native keyword exception did not expose expected key") from error
                output={"error_kind":program,"error_key":match.group(1)}
            observations.append({"program":program,"reverse":reverse,"output":output})
    events=[]
    class Projection(dict):
        def __iter__(self):
            return iter(dict.keys(self))
        def keys(self):
            events.append("keys");return ["alpha"]
        def __getitem__(self,key):
            events.append("get:"+key);return 7
    mapping=Projection(alpha=2,beta=5)
    projected=named(3,**mapping)
    class OverrideCount(dict):
        def __len__(self):
            return 77
    counted=OverrideCount(alpha=2,beta=5)
    class VisibleKind(dict):
        @property
        def __class__(self):
            return set
    disguised=VisibleKind(alpha=2)
    return {"provenance":"measured","kind":"native_cpython_only","python":platform.python_version(),
            "observations":observations,"mapping_protocol":{"value":projected,"events":events,"storage_shortcut_value":28},
            "length_contract":{"user_len":len(counted),"structural_dict_len":dict.__len__(counted)},
            "type_contract":{"isinstance_set":isinstance(disguised,set),"actual_type_is_dict_subclass":issubclass(type(disguised),dict)},
            "native_torch_compile":None}


def suite_inputs():
    inputs=[]
    for program in PROGRAMS:
        keys=("omega","sigma") if program=="unexpected" else ("alpha","beta")
        pairs=[[keys[0],2],[keys[1],5]]
        inputs.append({"program":program,"calls":[{"mapping_type":"dict","pairs":pairs,"base":3},
                      {"mapping_type":"OrderedDict" if program=="type" else "dict","pairs":list(reversed(pairs)),"base":3}]})
    return inputs


def order_matches(expected,current,scalar=False):
    return len(expected)==len(current) and (all(a==b for a,b in zip(expected,current)) if scalar else tuple(current)==expected)


def timing_paths(n):
    expected=tuple(f"field{i}" for i in range(n));current=list(expected)
    return lambda:order_matches(expected,current),lambda:order_matches(expected,current,True)


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n)
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):
                    if not funcs[i]():
                        raise RuntimeError("original equal-order fixture rejected")
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"keys":n,"tuple_samples_ms":samples[0],"scalar_samples_ms":samples[1],
                     "tuple_median_ms":statistics.median(samples[0]),"scalar_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original same-order guard,prebuilt expected tuple versus scalar zip comparison,both length checked",
            "excludes":"key/input preparation,dict/type/slot guards,compiler/cache execution,PyTorch/C++guards/GPU;timer exceeds UIbounds","rows":rows}


def build_report(latency,native):
    rows=[evaluate(p) for p in suite_inputs()]
    return {"schema_version":1,"round":152,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"BSD-3-Clause rights preserved;no implementation mirror",
                      "chronology":"actualNov14UTC04:00/BJT12:00 Dynamo dictionary semantic-guard correction",
                      "verified":"order guards for observed kwargs/iteration/errors;valid fixed-name binding can avoid order guard;structural dict length and actual type;custom mapping protocol",
                      "scope":"original guard-plan cache with dynamic scalar values;CPython oracle is not native Dynamo compilation/value-guard behavior"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},"matrix":rows,
            "summary":{"calls":sum(len(v["rows"]) for v in rows),"guarded_plans":sum(v["guarded_plans"] for v in rows),
                       "naive_plans":sum(v["naive_plans"] for v in rows),"guarded_misses":sum(v["guarded_misses"] for v in rows),
                       "naive_misses":sum(v["naive_misses"] for v in rows)},
            "native_python_oracle":native,"latency":latency,"native_torch_compile_verified":None,
            "native_recompilation_count":None,"native_compile_latency_ms":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
        (HERE/"native_oracle.json").write_text(json.dumps(native_oracle(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()),json.loads((HERE/"native_oracle.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round152 sealed artifacts match" if args.check else "Wrote Round152 original artifacts")


if __name__=="__main__":
    main()
