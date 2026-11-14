"""Original optional-gradient byte ledger;no ROCm,CK or GPU allocation."""

import argparse
import copy
import json
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="b09cae0b541cc9b78691436c2b8bf1bf5be3feed"
DEFAULT={"layout":"varlen","q_lengths":[16384,2048,512,128],"k_lengths":[16384,1024,512,128],
         "query_heads":32,"kv_heads":8,"dtype":"bf16","causal":True,"bias_requires_grad":False,
         "grad_bias_provided":False,"provided_grad_bias_bytes":None,"other_live_bytes":"6442450944","budget_bytes":"8589934592"}


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError("bounded exact integer required")
    return value


def byte_count(value):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9]{1,19}",value):
        raise ValueError("bounded decimal byte-count string required")
    return integer(int(value),0,2**63-1)


def evaluate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original allocation-ledger fields required")
    if p["layout"] not in ("fixed","varlen") or p["dtype"] not in ("fp16","bf16"):
        raise ValueError("original fixed/varlen and16-bit dtype required")
    qs=p["q_lengths"];ks=p["k_lengths"]
    if not isinstance(qs,list) or not isinstance(ks,list) or len(qs)!=len(ks) or len(qs)>64:
        raise ValueError("matching original sequence-length lists,batch0..64 required")
    for length in [*qs,*ks]:
        integer(length,0,2**20)
    if p["layout"]=="fixed" and (len(set(qs))>1 or len(set(ks))>1):
        raise ValueError("original fixed layout requires uniform query/key lengths")
    heads=integer(p["query_heads"],1,128);kv=integer(p["kv_heads"],1,128)
    if heads%kv:
        raise ValueError("query heads must be divisible by key/value heads")
    for field in ("causal","bias_requires_grad","grad_bias_provided"):
        if type(p[field]) is not bool:
            raise ValueError("exact boolean policy flags required")
    if p["bias_requires_grad"] and not p["grad_bias_provided"]:
        raise ValueError("requested bias gradient requires provided grad_bias")
    if p["grad_bias_provided"]:
        provided=byte_count(p["provided_grad_bias_bytes"])
    else:
        if p["provided_grad_bias_bytes"] is not None:
            raise ValueError("absent grad_bias requires null byte metadata")
        provided=0
    other=byte_count(p["other_live_bytes"]);budget=byte_count(p["budget_bytes"])
    shape=[len(qs),heads,max(qs,default=0),max(ks,default=0)]
    dense_pairs=shape[0]*shape[2]*shape[3];actual_rectangles=sum(q*k for q,k in zip(qs,ks))
    dense_bytes=dense_pairs*heads*2
    old_extra=0 if p["bias_requires_grad"] else dense_bytes
    selected_base=other+provided
    return {"placeholder_shape":shape,"dense_placeholder_bytes":str(dense_bytes),
            "ragged_rectangular_pair_positions":str(actual_rectangles),"max_padded_pair_positions":str(dense_pairs),
            "padding_pair_positions":str(dense_pairs-actual_rectangles),
            "old_fresh_return_region_bytes":str(old_extra),"new_fresh_return_region_bytes":"0",
            "removed_unused_bytes":str(old_extra),"provided_caller_owned_bytes":str(provided),
            "old_selected_live_bytes":str(selected_base+old_extra),"new_selected_live_bytes":str(selected_base),
            "old_selected_fits_budget":selected_base+old_extra<=budget,"new_selected_fits_budget":selected_base<=budget,
            "new_dbias_return":"provided_grad_bias" if p["bias_requires_grad"] else "undefined_tensor",
            "kernel_has_dbias":p["bias_requires_grad"],"kernel_dbias_pointer":"provided" if p["bias_requires_grad"] else "null",
            "native_other_workspace_bytes":None,"native_peak_bytes":None,"native_oom_prediction":None}


def scenarios():
    requested={**DEFAULT,"bias_requires_grad":True,"grad_bias_provided":True,"provided_grad_bias_bytes":"68719476736"}
    return [("Varlen/no gradient",DEFAULT),("Varlen/noncausal",{**DEFAULT,"causal":False}),
            ("Fixed/no gradient",{**DEFAULT,"layout":"fixed","q_lengths":[4096,4096],"k_lengths":[4096,4096]}),
            ("Requested/provided",requested),("Provided/unrequested",{**requested,"bias_requires_grad":False}),
            ("Empty batch",{**DEFAULT,"q_lengths":[],"k_lengths":[],"other_live_bytes":"0"})]


def timing_paths(n):
    requested=False;payload=2*n*n
    def gated():
        if requested:
            raise RuntimeError("this original timing fixture is unrequested")
        return None
    def eager():
        unused=bytearray(payload)
        if len(unused)!=payload:
            raise RuntimeError("original CPU allocation size mismatch")
        return None
    return gated,eager


def measure():
    rows=[]
    for n in (64,128,256,512):
        funcs=timing_paths(n)
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
        rows.append({"toy_side":n,"toy_bytes":2*n*n,"gated_samples_ms":samples[0],"eager_samples_ms":samples[1],
                     "gated_median_ms":statistics.median(samples[0]),"eager_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),"warmups":2,"repeats":7,
            "operations_per_sample":3,"method":"original no-gradient Python branch versus discarded zero-initialized CPU bytearray",
            "excludes":"ROCm/CK/PyTorch kernels,uninitialized at::empty,GPU allocation or allocator peak,other scratch;maximum toy buffer512KiB",
            "rows":rows}


def build_report(latency):
    p=copy.deepcopy(DEFAULT);p["kv_heads"]=4
    return {"schema_version":1,"round":151,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"BSD-3-Clause rights preserved;no implementation mirror",
                      "chronology":"actual Nov14UTC03:22/BJT11:22 ROCm CK fixed/varlen backward correction",
                      "verified":"unrequested dbias pointer null,has_dbias false,old dense max-length placeholder removed;requested provided gradient returned",
                      "scope":"original exact byte subledger and CPU policy toy;not native gradients,allocator/peak,workspace or performance reproduction"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},
            "matrix":[{"scenario":name,"inputs":p,"outputs":evaluate(p)} for name,p in scenarios()],
            "robustness":{"fewer_kv_heads":evaluate(p),"length_permutation":evaluate({**DEFAULT,"q_lengths":list(reversed(DEFAULT["q_lengths"])),
                         "k_lengths":list(reversed(DEFAULT["k_lengths"]))})},
            "latency":latency,"native_rocm_execution_verified":None,"native_gradient_error":None,
            "native_backward_latency_ms":None,"native_allocator_peak_bytes":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round151 sealed artifacts match" if args.check else "Wrote Round151 original artifacts")


if __name__=="__main__":
    main()
