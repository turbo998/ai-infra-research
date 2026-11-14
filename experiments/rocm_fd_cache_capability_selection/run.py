"""Original config/runtime-witness ledger;no VMM,FD,ROCm or vLLM execution."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="06c5e7c3ba2c05beb3b7b69274cfc6c71cc9fb82"
BITS={"REGISTERS":1,"EXPORTS_FD":2,"EXPORTS_FABRIC":4,"LEGACY_IPC":8}
DEFAULT={"cuda_alike":True,"rocm":True,"requested":None,"capabilities":["EXPORTS_FD"],
         "memory_override":False,"blocks_override":False,"elastic_ep":False,"custom_pool":False,
         "imex":False,"driver_probe":"unknown"}
SETS=[[],["REGISTERS"],["EXPORTS_FD"],["EXPORTS_FABRIC"],
      ["EXPORTS_FD","REGISTERS"],["EXPORTS_FD","EXPORTS_FABRIC"]]


class PolicyError(ValueError):
    def __init__(self,code,reason,stage="configuration",enabled=None):
        super().__init__(reason);self.code=code;self.stage=stage;self.enabled=enabled


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original platform/capability/override/probe fields required")
    for key in ("cuda_alike","rocm","memory_override","blocks_override","elastic_ep","custom_pool","imex"):
        if type(p[key]) is not bool:
            raise ValueError("explicit original booleans required")
    if p["requested"] is not None and type(p["requested"]) is not bool:
        raise ValueError("requested must be null,true or false")
    caps=p["capabilities"]
    if not isinstance(caps,list) or any(not isinstance(v,str) or v not in BITS for v in caps) or len(set(caps))!=len(caps):
        raise ValueError("unique known original capability names required")
    if not isinstance(p["driver_probe"],str) or p["driver_probe"] not in ("unknown","supported","unsupported"):
        raise ValueError("explicit original driver witness required")


def resolve(p,old=False):
    validate(p);caps=set(p["capabilities"]);fd="EXPORTS_FD" in caps
    single=bool(caps&{"REGISTERS","EXPORTS_FD","EXPORTS_FABRIC"})
    unsupported=None
    for condition,reason in ((not p["cuda_alike"],"NON_CUDA_ALIKE"),(p["memory_override"],"MEMORY_OVERRIDE"),
                            (p["blocks_override"],"BLOCKS_OVERRIDE"),(p["elastic_ep"],"ELASTIC_EP"),
                            ("LEGACY_IPC" in caps,"LEGACY_IPC"),(p["custom_pool"],"CUSTOM_POOL"),
                            (p["rocm"] and (single if old else "REGISTERS" in caps),"ROCM_REGISTRATION")):
        if condition:
            unsupported=reason;break
    if fd and (p["requested"] is False or unsupported is not None):
        raise PolicyError("FD_REQUIRES_EXTENSIBLE",unsupported or "explicitly disabled")
    fabric_unavailable="EXPORTS_FABRIC" in caps and not p["imex"]
    if p["requested"] is None:
        enabled=unsupported is None and (fd or not fabric_unavailable)
    else:
        if p["requested"] and unsupported is not None:
            raise PolicyError("UNSUPPORTED_EXPLICIT_REQUEST",unsupported)
        enabled=p["requested"]
    warnings=["NO_IMEX_LOCAL_TRANSFER_TCP_WARNING"] if enabled and fabric_unavailable else []
    reason=unsupported or ("NO_IMEX_AUTO_OFF" if not enabled and p["requested"] is None and fabric_unavailable else None)
    if not enabled:
        final=False;action="ordinary_allocator_without_probe"
    elif p["driver_probe"]=="unsupported":
        if fd:
            raise PolicyError("FD_DRIVER_REQUIRED","no ordinary-allocator fallback for FDimporter","runtime",True)
        final=False;action="model_runtime_fallback_ordinary_allocator"
    else:
        final=True;action="model_probe_passed" if p["driver_probe"]=="supported" else "driver_probe_not_observed"
    return {"status":"configured_enabled" if enabled else "configured_disabled","configuration_enabled":enabled,
            "reason_code":reason,"needs_single_physical_allocation_per_segment":single,
            "warnings":warnings,"runtime_enabled_in_model":final,"runtime_action":action,
            "native_driver_probe_verified":None,"native_startup_verified":None}


def capture(p,old=False):
    try:
        return resolve(p,old)
    except PolicyError as e:
        return {"status":"rejected","configuration_enabled":e.enabled,"reason_code":e.code,
                "detail":str(e),"stage":e.stage,"runtime_enabled_in_model":None,
                "native_driver_probe_verified":None,"native_startup_verified":None}


def classification_paths(n):
    inputs=[tuple(SETS[i%len(SETS)]) for i in range(n)]
    masks=[sum(BITS[v] for v in caps) for caps in inputs]
    def bitmask():
        return [(bool(m&7),bool(m&1),bool(m&2),bool(m&4),bool(m&8)) for m in masks]
    def names():
        return [(any(v in caps for v in ("REGISTERS","EXPORTS_FD","EXPORTS_FABRIC")),
                 "REGISTERS" in caps,"EXPORTS_FD" in caps,"EXPORTS_FABRIC" in caps,"LEGACY_IPC" in caps) for caps in inputs]
    return bitmask,names


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=classification_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original bitmask/name capability projections differ")
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                start=time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"capability_sets":n,"bitmask_samples_ms":samples[0],"names_samples_ms":samples[1],
                     "bitmask_median_ms":statistics.median(samples[0]),"names_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original prevalidated capability classification,bit projections versus name scans",
            "excludes":"input/mask/validation setup,full config resolution,driver/VMM/FD/UCX/RDMA/vLLM/GPU;not native throughput","rows":rows}


def build_report(latency):
    matrix=[]
    for caps in SETS:
        for requested in (None,True):
            p={**DEFAULT,"capabilities":caps,"requested":requested}
            matrix.append({"inputs":p,"before":capture(p,True),"after":capture(p)})
    changed=sum((v["before"]["status"],v["before"]["configuration_enabled"])!=(v["after"]["status"],v["after"]["configuration_enabled"]) for v in matrix)
    variants=[("FDexplicitfalse",{"requested":False}),("FD+registration",{"capabilities":["EXPORTS_FD","REGISTERS"]}),
              ("FDdriverfails",{"driver_probe":"unsupported"}),("Ordinarydriverfails",{"capabilities":[],"driver_probe":"unsupported","requested":True}),
              ("FDcustompool",{"custom_pool":True}),("FDlegacyIPC",{"capabilities":["EXPORTS_FD","LEGACY_IPC"]}),
              ("FabricIMEX",{"capabilities":["EXPORTS_FABRIC"],"imex":True}),("CUDAregistration",{"rocm":False,"capabilities":["REGISTERS"]})]
    return {"schema_version":1,"round":157,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"license":"Apache-2.0 rights retained,no implementation mirror",
                      "chronology":"actualNov14UTC06:08:48/BJT14:08:48 ROCm FD-exporting connector config correction",
                      "verified":"ROCmregistration gate tests REGISTERS,not coarse needs_single_chunk;FDrequires extensible;FABRICIMEXandruntimedriver remain separate",
                      "scope":"original configuration/runtime-witness decisions;not POSIXFD/ROCm/VMM/registration execution"},
            "case":{"inputs":DEFAULT,"before":capture(DEFAULT,True),"after":capture(DEFAULT)},
            "matrix":matrix,"summary":{"cases":len(matrix),"changed_configuration_results":changed,
                                      "before_enabled":sum(v["before"]["configuration_enabled"] is True for v in matrix),
                                      "after_enabled":sum(v["after"]["configuration_enabled"] is True for v in matrix)},
            "robustness":[{"name":name,"inputs":{**DEFAULT,**p},"outputs":capture({**DEFAULT,**p})} for name,p in variants],
            "latency":latency,"native_vllm_verified":None,"native_rocm_fd_export_verified":None,"native_memory_peak_bytes":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");args=p.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text()))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round157sealed artifacts match" if args.check else "Wrote Round157original artifacts")


if __name__=="__main__":
    main()
