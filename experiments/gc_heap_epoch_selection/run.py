"""Original GC epoch ledger and isolated CPython probes,not a vLLM worker."""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time
import weakref

HERE=Path(__file__).resolve().parent
PIN="449576cd736544461357e835ca74b44a993668ea"
DEFAULT={"startup_live":64,"init_collections":3,"drop_startup_cycles":16,
         "late_live":32,"late_garbage":8,"captures":0,"capture_gc_opt_out":False,
         "gc_initially_enabled":True}


def integer(v,maxval=2**20):return type(v) is int and 0<=v<=maxval


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError("exact original GC epoch inputs required")
    for k in ("startup_live","drop_startup_cycles","late_live","late_garbage"):
        if not integer(p[k]):raise ValueError("bounded original object counts required")
    if p["drop_startup_cycles"]>p["startup_live"] or not integer(p["init_collections"],64) or not integer(p["captures"],8):
        raise ValueError("bounded collection/capture count and reachable drops required")
    if type(p["capture_gc_opt_out"]) is not bool or type(p["gc_initially_enabled"]) is not bool:
        raise ValueError("explicit automatic-GC/opt-out witnesses required")


def plan(p,late_only=False,skip_epoch_reset=False):
    validate(p)
    f_live=0 if late_only else p["startup_live"];f_dead=0
    a_live=p["startup_live"] if late_only else 0;a_dead=0
    enabled=p["gc_initially_enabled"];reclaimed=0;trace=[]
    def snap(phase):
        trace.append({"phase":phase,"frozen_nodes":f_live+f_dead,"active_nodes":a_live+a_dead,
                      "unreachable_frozen_cycles":f_dead,"collected_cycle_nodes":reclaimed,"automatic_gc_enabled":enabled})
    def collect():
        nonlocal a_dead,reclaimed
        reclaimed+=a_dead;a_dead=0
    def unfreeze():
        nonlocal a_live,a_dead,f_live,f_dead
        a_live+=f_live;a_dead+=f_dead;f_live=0;f_dead=0
    def freeze():
        nonlocal a_live,a_dead,f_live,f_dead
        f_live+=a_live;f_dead+=a_dead;a_live=0;a_dead=0
    snap("startup")
    visits=a_live*p["init_collections"]
    drop=p["drop_startup_cycles"]
    if late_only:a_live-=drop;a_dead+=drop
    else:f_live-=drop;f_dead+=drop
    a_live+=p["late_live"];a_dead+=p["late_garbage"];snap("before_capture")
    for _ in range(p["captures"]):
        collect()
        if not p["capture_gc_opt_out"]:freeze();enabled=False
        snap("capture_enter")
        if not p["capture_gc_opt_out"]:
            unfreeze();collect();enabled=p["gc_initially_enabled"]
        snap("capture_exit")
    if not skip_epoch_reset:unfreeze()
    collect();freeze();snap("final_warmup")
    return {"modeled_init_object_visits":visits,"final_frozen_nodes":f_live+f_dead,
            "unreachable_frozen_cycles":f_dead,"collected_cycle_nodes":reclaimed,
            "automatic_gc_enabled":enabled,"trace":trace,"native_vllm_verified":None,
            "actual_gc_scanned_objects":None}


class Cycle:
    __slots__=("link","__weakref__")
    def __init__(self):self.link=self


def probe(size=64):
    """Called only in a disposable CLI child;never freeze the test-runner heap."""
    was=gc.isenabled();gc.disable();gc.unfreeze();gc.collect()
    try:
        roots=[Cycle() for _ in range(size)]
        refs=[weakref.ref(v) for v in roots]
        acyclic=Cycle();acyclic.link=None;ar=weakref.ref(acyclic)
        gc.collect();gc.collect(0);gc.freeze()
        freeze_count=gc.get_freeze_count()
        roots=None;acyclic=None
        gc.collect()
        held=sum(r() is not None for r in refs)
        later=Cycle();lr=weakref.ref(later);later=None;gc.collect()
        new_freed=lr() is None
        gc.unfreeze();gc.collect()
        released=sum(r() is not None for r in refs)
        return {"provenance":"measured","python":platform.python_version(),"implementation":platform.python_implementation(),
                "original_cycle_nodes":size,"actual_total_freeze_count":freeze_count,
                "frozen_cycles_alive_after_collection":held,"cycles_alive_after_unfreeze_collection":released,
                "frozen_acyclic_refcount_freed":ar() is None,"postfreeze_new_cycle_collected":new_freed,
                "native_vllm_verified":None}
    finally:
        gc.unfreeze();gc.collect()
        if was:gc.enable()
        else:gc.disable()


def finalizer_probe():
    was=gc.isenabled();gc.disable();gc.unfreeze();gc.collect();results={}
    try:
        for name in ("old_0_1_2","full_then_0"):
            created=[]
            class Finalizer:
                def __init__(self):self.loop=self
                def __del__(self):
                    child=Cycle();created.append(weakref.ref(child))
            root=Finalizer();gc.collect();root=None
            if name=="old_0_1_2":
                for generation in (0,1,2):gc.collect(generation)
            else:
                gc.collect();gc.collect(0)
            gc.freeze()
            results[name]={"finalizer_created_cycles":len(created),"alive_when_frozen":sum(r() is not None for r in created)}
            gc.unfreeze();gc.collect()
        return {"provenance":"measured","python":platform.python_version(),"results":results,"scope":"original promoted finalizer creates one new gen0cycle;not native vLLMobjects"}
    finally:
        gc.unfreeze();gc.collect()
        if was:gc.enable()
        else:gc.disable()


def measure():
    was=gc.isenabled();gc.disable();data=[]
    try:
        for size in (64,256,1024,4096):
            roots=[Cycle() for _ in range(size)]
            samples=[[],[]];freeze_counts=[]
            for repeat in range(9):
                for i in ((0,1) if repeat%2==0 else (1,0)):
                    gc.unfreeze();gc.collect()
                    if i==0:gc.collect(0);gc.freeze()
                    if repeat>=2 and i==0:freeze_counts.append(gc.get_freeze_count())
                    start=time.perf_counter_ns()
                    for _ in range(3):gc.collect()
                    elapsed=(time.perf_counter_ns()-start)/3/1e6
                    if repeat>=2:samples[i].append(elapsed)
            gc.unfreeze();roots=None;gc.collect()
            data.append({"original_live_cycle_nodes":size,"frozen_samples_ms":samples[0],"unfrozen_samples_ms":samples[1],
                         "frozen_median_ms":statistics.median(samples[0]),"unfrozen_median_ms":statistics.median(samples[1]),
                         "actual_total_freeze_counts":freeze_counts})
        return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
                "warmups":2,"repeats":7,"operations_per_sample":3,
                "method":"isolated original CPython gc.collect() on same rooted cycles;frozen versus unfrozen heap",
                "excludes":"setup/freeze cost,vLLMworker,weights,CUDA/NCCL/capture,GPU memory and server startup",
                "note":"freeze covers whole child interpreter,not only named cycle nodes;automatic GC disabled for probe isolation",
                "rows":data}
    finally:
        gc.unfreeze();gc.collect()
        if was:gc.enable()
        else:gc.disable()


def build_report(latency):
    cases=[("no_capture",DEFAULT),("default_capture",{**DEFAULT,"captures":1}),
           ("capture_opt_out",{**DEFAULT,"captures":1,"capture_gc_opt_out":True}),
           ("prior_gc_disabled",{**DEFAULT,"captures":1,"gc_initially_enabled":False}),
           ("several_captures",{**DEFAULT,"captures":3}),("late_freeze_only",DEFAULT)]
    return {"schema_version":1,"round":164,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"chronology":"actualNov14UTC05:00:46/BJT13:00:46 worker startup GCfreeze",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"freeze startup after distributed/cuBLAS init,before memory snapshot/weights;full collect then gen0;default capture exit unfreezes/restores prior GC;warmup explicitly unfreezes/refreezes",
                      "scope":"original epoch-count ledger plus separately measured isolated CPython probes,not native vLLMworker/CUDA performance"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"late_only":plan(DEFAULT,True),
                    "skip_reset_proposal":plan(DEFAULT,skip_epoch_reset=True)},
            "matrix":[{"name":name,"inputs":p,"outputs":plan(p,late_only=name=="late_freeze_only")} for name,p in cases],
            "latency":latency,"cpython_probe":json.loads((HERE/"probe.json").read_text()),
            "native_vllm_verified":None,"browser_visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true")
    ap.add_argument("--probe",action="store_true");ap.add_argument("--finalizer-probe",action="store_true");a=ap.parse_args()
    if a.probe:print(json.dumps(probe()));return
    if a.finalizer_probe:print(json.dumps(finalizer_probe()));return
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
        (HERE/"probe.json").write_text(json.dumps({"cycles":probe(),"finalizer":finalizer_probe()},indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:raise SystemExit(f"stale artifact:{file}")
        else:file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round164 sealed artifacts match" if a.check else "Wrote Round164 original artifacts")


if __name__=="__main__":main()
