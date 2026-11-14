"""Original metadata layout/contracts,not native vLLM DTOs or worker throughput."""

import argparse
from dataclasses import asdict,dataclass,fields
from functools import cached_property
import gc
import json
from pathlib import Path
import pickle
import platform
import statistics
import sys
import time
import weakref

HERE=Path(__file__).resolve().parent
PIN="ce72586f8aded8dafa7d4c745a5cef3198070e3e"


@dataclass
class Plain:
    rid:int
    work:int
    payload:list[int]


@dataclass(slots=True)
class Slotted:
    rid:int
    work:int
    payload:list[int]


@dataclass(slots=True,weakref_slot=True)
class WeakSlotted:
    rid:int
    work:int
    payload:list[int]


class DictionaryBase:
    pass


@dataclass(slots=True)
class InheritedDictionary(DictionaryBase):
    rid:int
    work:int
    payload:list[int]


@dataclass(slots=True)
class CachedSlotted:
    value:int
    @cached_property
    def doubled(self):return self.value*2


CLASSES={"plain":Plain,"slots":Slotted,"weak_slots":WeakSlotted,"inherited_dict":InheritedDictionary}
DEFAULT={"layout":"slots","records":64,"operation":"pickle","rid":7,"work":123,"payload":[1,2,3]}


def integer(v,maxval=2**20):return type(v) is int and 0<=v<=maxval


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError("exact original metadata fields required")
    if not isinstance(p["layout"],str) or p["layout"] not in CLASSES or p["records"] not in (64,256,1024,4096) or type(p["records"]) is not int:
        raise ValueError("known layout and exactly measured count tier required")
    if p["operation"] not in ("fields","vars","dynamic","weakref","pickle"):raise ValueError("known original operation required")
    if not integer(p["rid"]) or not integer(p["work"]) or not isinstance(p["payload"],list) or len(p["payload"])>16 or any(not integer(x) for x in p["payload"]):
        raise ValueError("bounded original scalar/list metadata required")


def operation(obj,op):
    if op=="fields":return {"status":"ok","value":asdict(obj)}
    if op=="pickle":return {"status":"ok","value":asdict(pickle.loads(pickle.dumps(obj,protocol=5)))}
    if op=="vars":
        try:return {"status":"ok","value":dict(vars(obj))}
        except TypeError:return {"status":"expected_contract_error","error":"TypeError"}
    if op=="dynamic":
        try:obj.extra=9
        except AttributeError:return {"status":"expected_contract_error","error":"AttributeError"}
        return {"status":"ok","extra":obj.extra}
    if op=="weakref":
        try:r=weakref.ref(obj)
        except TypeError:return {"status":"expected_contract_error","error":"TypeError"}
        return {"status":"ok","alive":r() is obj}
    raise ValueError("unsupported operation")


def plan(p):
    validate(p)
    row=next(x for x in json.loads((HERE/"latency.json").read_text())["rows"] if x["records"]==p["records"])
    obj=CLASSES[p["layout"]](p["rid"],p["work"],p["payload"])
    return {"layout":p["layout"],"records":p["records"],"measured_canonical_batch_shell_bytes":row["shell_bytes"][p["layout"]],
            "operation":operation(obj,p["operation"]),"gc_tracked":gc.is_tracked(obj),
            "native_vllm_verified":None,"referent_bytes":None,"javascript_engine_object_bytes":None}


def layout_probe():
    profiles={}
    for name,cls in CLASSES.items():
        obj=cls(7,123,[1,2,3])
        profiles[name]={"declared_fields":[f.name for f in fields(obj)],"has_instance_dict":hasattr(obj,"__dict__"),
                        "gc_tracked":gc.is_tracked(obj),"vars":operation(obj,"vars"),"weakref":operation(obj,"weakref"),
                        "pickle":operation(obj,"pickle"),"dynamic":operation(obj,"dynamic")}
    c=CachedSlotted(2)
    try:c.doubled
    except TypeError:cache_error="TypeError"
    else:raise RuntimeError("slotted cached_property unexpectedly gained dictionary storage")
    payload=[1,2,3];pair=[Slotted(1,2,payload),Slotted(2,3,payload)]
    restored=pickle.loads(pickle.dumps(pair,protocol=5))
    mutable=Slotted(1,2,[]);mutable.work=3
    return {"provenance":"measured","python":platform.python_version(),"profiles":profiles,
            "cached_property_without_dict":cache_error,"pickle_preserves_shared_payload":restored[0].payload is restored[1].payload,
            "slotted_declared_fields_remain_mutable":mutable.work==3,
            "native_vllm_verified":None}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        payload=[1,2,3]
        def batch(cls):return [cls(i,i%128,payload) for i in range(n)]
        shell={}
        for name,cls in CLASSES.items():
            objects=batch(cls)
            shell[name]=sum(sys.getsizeof(o)+(sys.getsizeof(o.__dict__) if hasattr(o,"__dict__") else 0) for o in objects)
        funcs=(lambda:batch(Slotted),lambda:batch(Plain))
        for _ in range(2):
            for f in funcs:f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"records":n,"shell_bytes":shell,"slots_samples_ms":samples[0],"plain_samples_ms":samples[1],
                     "slots_median_ms":statistics.median(samples[0]),"plain_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original three-field record batch construction;object+instance-dict shell sizes at exact tiers",
            "excludes":"referents/list container,class/shared keys,RSS,deep heap,nativevLLM,Torch/tensors,IPC and GPU",
            "note":"shared payload is prebuilt;timed batch list creation/destruction included;canonical shell sizes do not predict arbitrary payload memory",
            "rows":rows}


def build_report(latency):
    cases=[("slotted_pickle",DEFAULT),("plain_vars",{**DEFAULT,"layout":"plain","operation":"vars"}),
           ("slotted_vars",{**DEFAULT,"operation":"vars"}),("slotted_dynamic",{**DEFAULT,"operation":"dynamic"}),
           ("weakref_optin",{**DEFAULT,"layout":"weak_slots","operation":"weakref"}),
           ("inherited_dictionary",{**DEFAULT,"layout":"inherited_dict","operation":"dynamic"})]
    return {"schema_version":1,"round":165,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"chronology":"actualNov14UTC09:18:41/BJT17:18:41 internal dataclass slot conversion",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"IterationDetails/KVConnectorBlockState/GrammarOutput and ModelRunnerOutput now slots;NewRequestData/CachedRequestData/SchedulerOutput still dictionaries;cached_property and __setstate__ remain",
                      "scope":"original three-field records,not native field counts/layout,full-repository conversion census or worker/IPC performance"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"plain":plan({**DEFAULT,"layout":"plain"})},
            "matrix":[{"name":name,"inputs":p,"outputs":plan(p)} for name,p in cases],
            "latency":latency,"layout_probe":json.loads((HERE/"probe.json").read_text()),
            "native_vllm_verified":None,"browser_visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
        (HERE/"probe.json").write_text(json.dumps(layout_probe(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:raise SystemExit(f"stale artifact:{file}")
        else:file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round165 sealed artifacts match" if a.check else "Wrote Round165 original artifacts")


if __name__=="__main__":main()
