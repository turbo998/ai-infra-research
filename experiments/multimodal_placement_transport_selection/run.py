"""Original placement/transport policy projection;no processors or IPC execution."""

import argparse
import json
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="04eece19ada1bf2fee46ab52f19e570f581f2118"
TRANSPORTS=("cpu","cuda_ipc","cuda_vmm")
DEFAULT={"platform":"cuda_alike","device_type":"cuda","base_gpu_id":3,"rl_cpu":False,"fast_disabled":False,
         "hf_image_processor":True,"processor_class":"GenericProcessor","declared_workers":8,"explicit_workers":None,
         "skip_pool":False,"transport":"cpu","budget_bytes":1025,"tokenizer_workers":3}
SCENARIOS=[("CUDAfast",{}),("CPU",{"platform":"cpu"}),("RLforcedCPU",{"rl_cpu":True}),
           ("NPUdeviceUnset",{"platform":"npu","processor_class":"Glm4vProcessor"}),
           ("FastDisabled",{"fast_disabled":True}),("OtherPlatform",{"platform":"other"})]


def bounded(v,lo,hi,label):
    if type(v) is not int or not lo<=v<=hi:
        raise ValueError(f"bounded original{label}integer required")


def placement(p):
    cpu=p["platform"]=="cpu" or p["rl_cpu"]
    accelerator=p["platform"] in ("cuda_alike","xpu","npu")
    contends=not cpu and not p["fast_disabled"] and p["hf_image_processor"] and accelerator
    if cpu:
        device="cpu"
    elif p["platform"]=="xpu":
        device="xpu"
    elif p["platform"]=="npu":
        device=None if p["processor_class"]=="Glm4vProcessor" else "npu"
    elif p["platform"]=="cuda_alike":
        device=f'{p["device_type"]}:{p["base_gpu_id"]}'
    else:
        device=None
    auto=1 if contends else 2 if p["declared_workers"] is None else p["declared_workers"]
    selected=1 if p["skip_pool"] else p["explicit_workers"] if p["explicit_workers"] is not None else auto
    return {"candidate_fast_device":device,"preprocessing_contends":contends,"auto_processor_workers":auto,
            "selected_processor_workers":selected,"worker_reason":"synchronous skip-pool" if p["skip_pool"] else "explicit override" if p["explicit_workers"] is not None else "auto placement",
            "candidate_device_not_actual_execution":True}


def transport(p):
    mode=p["transport"];keep=mode in ("cuda_ipc","cuda_vmm")
    share=p["budget_bytes"]//p["tokenizer_workers"];reserve=share*p["tokenizer_workers"]
    active=mode=="cuda_ipc" and not p["skip_pool"]
    return {"keep_features_on_device_policy":keep,"cuda_ipc_policy":mode=="cuda_ipc",
            "ipc_pool_configured":active,"per_tokenizer_budget_bytes":share,"aggregate_budget_split_bytes":reserve,
            "unused_budget_bytes":p["budget_bytes"]-reserve,"active_ipc_budget_split_bytes":reserve if active else 0,
            "naive_full_budget_per_tokenizer_bytes":p["budget_bytes"]*p["tokenizer_workers"],
            "cpu_executor_start_method_policy":"spawn" if mode=="cuda_vmm" else "fork"}


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original placement/transport/worker fields required")
    if not isinstance(p["platform"],str) or p["platform"] not in ("cpu","xpu","npu","cuda_alike","other"):
        raise ValueError("original declared platform category required")
    for field in ("device_type","processor_class"):
        if not isinstance(p[field],str) or not re.fullmatch(r"[A-Za-z0-9_]{1,32}",p[field]):
            raise ValueError("bounded original class/device label required")
    for field in ("rl_cpu","fast_disabled","hf_image_processor","skip_pool"):
        if type(p[field]) is not bool:
            raise ValueError("explicit original boolean required")
    bounded(p["base_gpu_id"],0,63,"base_gpu_id")
    bounded(p["budget_bytes"],1,1073741824,"budget_bytes")
    bounded(p["tokenizer_workers"],1,64,"tokenizer_workers")
    for field in ("declared_workers","explicit_workers"):
        if p[field] is not None:
            bounded(p[field],1,64,field)
    if not isinstance(p["transport"],str) or p["transport"] not in TRANSPORTS:
        raise ValueError("known transport required;prototype surfaces typos rather than native helper's CPUfallback")


def evaluate(p):
    validate(p);a=placement(p);b=transport(p)
    naive=1 if b["keep_features_on_device_policy"] else 2 if p["declared_workers"] is None else p["declared_workers"]
    return {**a,**b,"naive_transport_only_auto_workers":naive,"naive_auto_matches_placement":naive==a["auto_processor_workers"],
            "native_processor_execution":None,"native_ipc_pool_allocation":None,
            "native_feature_tensor_device":None,"native_zero_share_pool_acceptance":None}


def timing_paths(n):
    inputs=[{**DEFAULT,"transport":TRANSPORTS[i%3]} for i in range(n)]
    for p in inputs:
        validate(p)
    staged=placement(DEFAULT)
    def hoisted():
        return [{**staged,**transport(p)} for p in inputs]
    def repeated():
        return [{**placement(p),**transport(p)} for p in inputs]
    return hoisted,repeated


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original hoisted/repeated policy projections differ")
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
        rows.append({"projections":n,"hoisted_samples_ms":samples[0],"repeated_samples_ms":samples[1],
                     "hoisted_median_ms":statistics.median(samples[0]),"repeated_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original bulk policy projection,reused immutable placement versus repeated resolver",
            "excludes":"input/validation setup,actual preprocessing,worker threads/processes,IPC/VMMallocation,copying/hashing/SGLang/GPU;not native per-feature resolver cost","rows":rows}


def build_report(latency):
    matrix=[]
    for label,changes in SCENARIOS:
        for mode in TRANSPORTS:
            p={**DEFAULT,**changes,"transport":mode};matrix.append({"name":label,"inputs":p,"outputs":evaluate(p)})
    variations=[("Explicit4GPU",{"explicit_workers":4}),("NoDeclarationCPU",{"platform":"cpu","declared_workers":None}),
                ("InstanceGPU7",{"base_gpu_id":7}),("ZeroShare",{"transport":"cuda_ipc","budget_bytes":1}),
                ("SkipPoolSynchronous",{"transport":"cuda_ipc","skip_pool":True,"explicit_workers":4}),
                ("XPU",{"platform":"xpu"})]
    return {"schema_version":1,"round":156,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"license":"Apache-2.0 rights retained,no implementation mirror",
                      "chronology":"actualNov14UTC06:44:04/BJT14:44:04 module refactor,not invention of placement/transport policies",
                      "verified":"scheduler contention versus transport independent;NPUunset exception;auto cap/manual override;pool split by tokenizer processes not preprocessing threads",
                      "scope":"original declared policy projection,no native tensors/workers/IPC/driver support"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},"matrix":matrix,
            "summary":{"cases":len(matrix),"naive_transport_only_auto_misses":sum(not x["outputs"]["naive_auto_matches_placement"] for x in matrix)},
            "robustness":[{"name":name,"inputs":{**DEFAULT,**c},"outputs":evaluate({**DEFAULT,**c})} for name,c in variations],
            "latency":latency,"native_sglang_verified":None,"native_throughput_rps":None,"native_gpu_peak_bytes":None,
            "upstream_comment_performance_reproduced":False}


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
    print("Round156sealed artifacts match" if args.check else "Wrote Round156original artifacts")


if __name__=="__main__":
    main()
