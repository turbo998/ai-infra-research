"""Original integer NCHW permutation, independent input-scatter oracle and inverse."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
CASE={"n":1,"c":8,"h":1,"w":2,"blocksize":2,"mode":"CRD","scalar_bytes":4}
DEFAULT={"n":1,"c":256,"h":128,"w":128,"blocksize":2,"mode":"CRD","scalar_bytes":4}


def budget(p):
    if set(p)!=set(CASE) or any(type(p[k]) is not int or not 1<=p[k]<=65536 for k in ("n","c","h","w","blocksize")) or type(p["scalar_bytes"]) is not int or p["scalar_bytes"] not in (1,2,4,8) or p["mode"] not in ("DCR","CRD"):
        raise ValueError("bounded positive exact NCHW/blocksize,mode and scalar width required")
    if p["c"]%(p["blocksize"]**2):
        raise ValueError("input channels must divide by blocksize squared")
    elements=p["n"]*p["c"]*p["h"]*p["w"]
    if elements*max(8,p["scalar_bytes"]*2)>2**53-1:
        raise ValueError("selected logical payload exceeds common exact integer budget")
    size=elements*p["scalar_bytes"]
    return {"input_shape":[p[k] for k in ("n","c","h","w")],
            "output_shape":[p["n"],p["c"]//p["blocksize"]**2,p["h"]*p["blocksize"],p["w"]*p["blocksize"]],
            "elements_preserved":elements,"input_payload_bytes":size,"output_payload_bytes":size,
            "selected_one_read_one_write_bytes":2*size,
            "selected_separate_int64_index_map_bytes":8*elements,
            "note":"logical materialized-copy/index-map accounting only;not native allocation,fusion,view,stride or GPU throughput"}


def permutation(p):
    contract=budget(p);n,c,h,w=contract["input_shape"];_,co,oh,ow=contract["output_shape"];b=p["blocksize"]
    if contract["elements_preserved"]>262144:
        raise ValueError("original permutation materialization limit262144elements")
    gather=[]
    for flat in range(n*co*oh*ow):
        q,ox=divmod(flat,ow);q,oy=divmod(q,oh);batch,channel=divmod(q,co)
        iy,by=divmod(oy,b);ix,bx=divmod(ox,b)
        ic=(by*b+bx)*co+channel if p["mode"]=="DCR" else channel*b*b+by*b+bx
        gather.append(((batch*c+ic)*h+iy)*w+ix)
    return gather


def depth_to_space(values,p):
    indices=permutation(p)
    if len(values)!=len(indices) or any(type(v) is not int or abs(v)>2**31 for v in values):
        raise ValueError("exact bounded integer values matching NCHW shape required")
    return [values[i] for i in indices]


def space_to_depth(values,p):
    indices=permutation(p)
    if len(values)!=len(indices) or any(type(v) is not int or abs(v)>2**31 for v in values):
        raise ValueError("exact bounded integer values matching selected inverse shape required")
    result=[None]*len(indices)
    for outindex,inindex in enumerate(indices):
        result[inindex]=values[outindex]
    return result


def scatter_oracle(values,p):
    contract=budget(p);n,ci,h,w=contract["input_shape"];_,co,oh,ow=contract["output_shape"];b=p["blocksize"]
    if len(values)!=n*ci*h*w:
        raise ValueError("oracle value count mismatch")
    result=[None]*len(values)
    for batch in range(n):
        for channel in range(ci):
            if p["mode"]=="CRD":
                oc,block=divmod(channel,b*b)
            else:
                block,oc=divmod(channel,co)
            by,bx=divmod(block,b)
            for iy in range(h):
                for ix in range(w):
                    source=((batch*ci+channel)*h+iy)*w+ix
                    destination=((batch*co+oc)*oh+iy*b+by)*ow+ix*b+bx
                    result[destination]=values[source]
    return result


def direct_space_to_depth_export_mode(mode,target_opset):
    if mode not in ("DCR","CRD") or type(target_opset) is not int or not 13<=target_opset<=29:
        raise ValueError("bounded direct-op target opset13..29 and explicit mode required")
    if mode=="CRD" and target_opset<28:
        raise ValueError("CRD cannot be represented by one pre28 SpaceToDepth operator;explicit lowering required")
    return {"mode":mode,"target_opset":target_opset,"direct_single_op_allowed":True}


def timing_paths(elements):
    p={**CASE,"h":2,"w":elements//16};values=list(range(elements));indices=permutation(p)
    def preplanned():
        return [values[i] for i in indices]
    def rederive():
        return [values[i] for i in permutation(p)]
    return preplanned,rederive


def measure():
    rows=[]
    for size in (64,256,1024,4096):
        paths=timing_paths(size);assert paths[0]()==paths[1]()
        for _ in range(2):
            for f in paths:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    paths[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"elements":size,"preplanned_samples_ms":samples[0],"rederive_samples_ms":samples[1],
                     "preplanned_median_ms":statistics.median(samples[0]),"rederive_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original Python integer gather using retained index list versus recomputed coordinate index list plus gather",
            "excludes":"retained-map construction for preplanned path,ONNX/PyTorch/native kernels,actual tensor strides,compiler fusion,GPU",
            "cost_boundary":"preplanned path retains index map;rederive allocates and computes it each call;not native shuffle speedup","rows":rows}


def build_report(latency):
    values=list(range(16));crd=depth_to_space(values,CASE);dcr=depth_to_space(values,{**CASE,"mode":"DCR"})
    g=list(range(1,17));inverse=space_to_depth(crd,CASE);wrong_inverse=space_to_depth(crd,{**CASE,"mode":"DCR"})
    one={**CASE,"c":4};ids=list(range(8))
    return {"schema_version":1,"round":138,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0",
                      "capability_commit":"24b862d924bf5112afb4fbfc5e886809dad004d8",
                      "capability_date":"2026-08-31","capability":"SpaceToDepth28 adds CRD mode and both28 operators function bodies;not a November invention",
                      "pytorch_pin":"7bbc31dd45f4e43a5e3899e7e328aedfd967f2ff","pytorch_license":"BSD-style",
                      "verified":"pinned schema divisibility,DCRdefault,SpaceToDepth28CRD;PyTorch TorchScript opset11 pixel_shuffle exports DepthToSpace modeCRD"},
            "case":{"inputs":CASE,"input_values":values,"contract":budget(CASE),"CRD_output":crd,"DCR_output":dcr,
                    "oracle_CRD":scatter_oracle(values,CASE),"oracle_DCR":scatter_oracle(values,{**CASE,"mode":"DCR"}),
                    "same_shape_wrong_mode_matching_values":sum(a==b for a,b in zip(crd,dcr)),"denominator":len(values),
                    "inverse_CRD":inverse,"inverse_wrong_DCR":wrong_inverse,
                    "adjoint_fixture":{"output_gradient":g,"dot_forward_gradient":sum(a*b for a,b in zip(crd,g)),
                                       "dot_input_inverse_gradient":sum(a*b for a,b in zip(values,space_to_depth(g,CASE)))},
                    "single_output_channel_mode_outputs_equal":depth_to_space(ids,one)==depth_to_space(ids,{**one,"mode":"DCR"}),
                    "direct_CRD_opset28":direct_space_to_depth_export_mode("CRD",28),
                    "direct_CRD_opset27":"rejected in original single-op contract;explicit reshape/transpose lowering may represent it"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_onnx_parity":None,"native_gpu_latency_ms":None,"native_memory_bytes":None,"superresolution_quality":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round138 sealed artifacts match" if a.check else "Wrote Round138 original artifacts")


if __name__=="__main__":
    main()
