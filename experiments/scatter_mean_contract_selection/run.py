"""Original flat-coordinate ScatterND integer-mean contract model."""

import argparse
from itertools import product
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
CHANGE="af56232bdc6f1ecbb27a5c9d2866263d6033bc38"
DEFAULT={"rows":4096,"width":128,"updates":1024,"index_depth":1,"index_element_bytes":4,"value_element_bytes":4}


def contract(shape,data,indices,updates,k,reduction,opset=29):
    if not shape or len(shape)>4 or any(type(d) is not int or d<1 for d in shape):
        raise ValueError("one to four positive exact dimensions required")
    if type(k) is not int or not 0<=k<=len(shape):
        raise ValueError("index depth outside data rank")
    if reduction not in ("none","add","mul","max","min","mean"):
        raise ValueError("unsupported reduction")
    if reduction=="mean" and opset!=29:
        raise ValueError("this pinned mean contract requires opset29")
    if len(data)!=math.prod(shape) or any(type(v) is not int for v in data):
        raise ValueError("flat exact integer data must match shape")
    width=math.prod(shape[k:])
    if len(updates)!=len(indices)*width or any(type(v) is not int for v in updates):
        raise ValueError("updates shape must equal [M]+data.shape[k:]")
    canonical=[]
    for row in indices:
        if len(row)!=k:
            raise ValueError("index tuple width must match depth")
        dest=[]
        for value,size in zip(row,shape):
            if type(value) is not int or not -size<=value<size:
                raise ValueError("index outside [-size,size-1]")
            dest.append(value%size)
        canonical.append(tuple(dest))
    if reduction=="none" and len(set(canonical))!=len(canonical):
        raise ValueError("none reduction forbids duplicate normalized locations")
    return canonical,width


def scatter(shape,data,indices,updates,k,reduction="mean",opset=29):
    canonical,width=contract(shape,data,indices,updates,k,reduction,opset)
    output=list(data);sums={};counts={}
    for row,destination in enumerate(canonical):
        prefix=0
        for value,size in zip(destination,shape):
            prefix=prefix*size+value
        for column in range(width):
            target=prefix*width+column;value=updates[row*width+column]
            if reduction=="mean":
                sums[target]=sums.get(target,data[target])+value
                counts[target]=counts.get(target,1)+1
            elif reduction=="none":
                output[target]=value
            elif reduction=="add":
                output[target]+=value
            elif reduction=="mul":
                output[target]*=value
            elif reduction=="max":
                output[target]=max(output[target],value)
            else:
                output[target]=min(output[target],value)
    for target,total in sums.items():
        output[target]=total//counts[target]
    return output


def independent_destination_oracle(shape,data,indices,updates,k,reduction="mean"):
    canonical,width=contract(shape,data,indices,updates,k,reduction)
    output=[]
    for target,coordinate in enumerate(product(*(range(d) for d in shape))):
        matching=[]
        for row,destination in enumerate(canonical):
            if coordinate[:k]==destination:
                suffix=coordinate[k:];column=0
                for value,size in zip(suffix,shape[k:]):
                    column=column*size+value
                matching.append(updates[row*width+column])
        values=[data[target],*matching]
        if not matching:
            output.append(data[target])
        elif reduction=="mean":
            output.append(sum(values)//len(values))
        elif reduction=="none":
            output.append(matching[0])
        elif reduction=="add":
            output.append(sum(values))
        elif reduction=="mul":
            output.append(math.prod(values))
        elif reduction=="max":
            output.append(max(values))
        else:
            output.append(min(values))
    return output


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<0 for v in p.values()):
        raise ValueError("exact nonnegative budget inputs required")
    if p["rows"]<1 or p["width"]<1 or p["index_depth"] not in (0,1,2) or p["index_element_bytes"] not in (4,8) or p["value_element_bytes"] not in (2,4,8):
        raise ValueError("invalid shape or selected byte widths")
    r,w,m,k,ix,e=(p[key] for key in ("rows","width","updates","index_depth","index_element_bytes","value_element_bytes"))
    slice_elements=math.prod((r,w)[k:])
    out={"selected_data_payload_bytes":r*w*e,"selected_updates_payload_bytes":m*slice_elements*e,
         "selected_indices_payload_bytes":m*k*ix,"selected_dense_int64_sum_count_payload_bytes":r*w*16}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def timing_paths(n):
    data=list(range(64));destinations=[i%64 for i in range(n)];values=[i%11-5 for i in range(n)]
    def grouped():
        sums=list(data);counts=[1]*64
        for d,v in zip(destinations,values):
            sums[d]+=v;counts[d]+=1
        return [s//c for s,c in zip(sums,counts)]
    def repeated():
        out=[]
        for d,original in enumerate(data):
            contributions=[v for target,v in zip(destinations,values) if target==d]
            out.append((original+sum(contributions))//(1+len(contributions)))
        return out
    return grouped,repeated


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        f=timing_paths(n);assert f[0]()==f[1]()
        for _ in range(2):
            for fn in f:
                fn()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    f[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"updates":n,"destinations":64,"grouped_samples_ms":samples[0],"repeated_samples_ms":samples[1],
                     "grouped_median_ms":statistics.median(samples[0]),"repeated_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original bounded integer means;grouped dense sum/count versus destination-first repeated scan;setup sums/counts included",
            "excludes":"nativeONNX,parallel atomics,FP order/overflow,native allocator,model quality,GPU","rows":rows}


def build_report(latency):
    shape=[2,2];data=[10,-4,30,40];indices=[[0],[-2]];updates=[2,-3,8,-2]
    answer=scatter(shape,data,indices,updates,1)
    wrong=list(data)
    for row in range(2):
        for col in range(2):
            wrong[col]=(wrong[col]+updates[row*2+col])//2
    reverse=list(data)
    for row in (1,0):
        for col in range(2):
            reverse[col]=(reverse[col]+updates[row*2+col])//2
    return {"schema_version":1,"round":131,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0","mean_change":CHANGE,
                      "mean_change_date_utc":"2026-11-12T04:44:48Z","opset":29,
                      "contract":"mean includes original once;integer floor;negative aliases;none duplicate forbidden;updates prefix plus trailing slice"},
            "case":{"shape":shape,"data_flat":data,"indices":indices,"updates_flat":updates,"index_depth":1,
                    "canonical_destinations":[[0],[0]],"selected_total_updates":2,"selected_unique_slice_destinations":1,
                    "correct_integer_mean":answer,"oracle_integer_mean":independent_destination_oracle(shape,data,indices,updates,1),
                    "exact_touched_means":["20/3","-3"],"wrong_update_only_mean":[5,-3,30,40],
                    "wrong_repeated_pair_mean":wrong,"wrong_reversed_pair_mean":reverse,
                    "correct_reversed_mean":scatter(shape,data,list(reversed(indices)),updates[2:]+updates[:2],1),
                    "none_duplicate_policy":"explicit validation error,not last-write guarantee",
                    "whole_tensor_mean":scatter([2],[10,-4],[[],[]],[2,-3,8,-2],0),
                    "negative_floor_control":{"sum":-5,"count":2,"integer_floor":-3,"wrong_truncation":-2}},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_onnx_latency_ms":None,"floating_bitwise_determinism":None,"native_peak_hbm_bytes":None,"model_quality":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    print("Round131 sealed artifacts match" if a.check else "Wrote Round131 original artifacts")


if __name__=="__main__":
    main()
