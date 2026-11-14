"""Original exact class-aware greedy NMS,not ONNX execution."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="3c41120b9c69657a1dd13407b080bd451ed06c9c"
DEFAULT={"batches":1,"classes":80,"boxes":8400,"max_per_class":100,"element_bytes":4}


def optional_scalar(value):
    if value is None or value==[]:
        return None
    if isinstance(value,list):
        if len(value)!=1:
            raise ValueError("original model accepts scalar or one-element vector only")
        value=value[0]
    if type(value) not in (int,Fraction):
        raise ValueError("exact finite rational optional scalar required")
    return value


def canonical(box,center=False):
    if len(box)!=4 or any(type(v) not in (int,Fraction) for v in box):
        raise ValueError("four exact rational box coordinates required")
    if center:
        x,y,w,h=map(Fraction,box)
        if w<0 or h<0:
            raise ValueError("original center-format width/height must be nonnegative")
        return y-h/2,x-w/2,y+h/2,x+w/2
    y1,x1,y2,x2=map(Fraction,box)
    return min(y1,y2),min(x1,x2),max(y1,y2),max(x1,x2)


def iou(a,b):
    y1,x1,y2,x2=a;v1,u1,v2,u2=b
    intersection=max(Fraction(),min(y2,v2)-max(y1,v1))*max(Fraction(),min(x2,u2)-max(x1,u1))
    union=(y2-y1)*(x2-x1)+(v2-v1)*(u2-u1)-intersection
    return intersection/union if union>0 else Fraction()


def validate(boxes,scores,maximum,threshold,score_threshold,center):
    maximum=optional_scalar(maximum)
    maximum=0 if maximum is None else maximum
    if type(maximum) is not int:
        raise ValueError("integer output cap required")
    maximum=max(maximum,0)
    threshold=optional_scalar(threshold)
    threshold=Fraction() if threshold is None else Fraction(threshold)
    score_threshold=optional_scalar(score_threshold)
    if not 0<=threshold<=1:
        raise ValueError("original IoU threshold must be in[0,1]")
    coords=[canonical(box,center) for box in boxes]
    if not boxes or not scores or any(len(row)!=len(boxes) for row in scores):
        raise ValueError("nonempty boxes/classes and matching score axes required")
    if any(type(v) not in (int,Fraction) for row in scores for v in row):
        raise ValueError("exact rational scores required")
    return coords,maximum,threshold,score_threshold


def nms(boxes,scores,maximum=None,threshold=None,score_threshold=None,center=False,wrong_equal=False):
    coords,maximum,threshold,score_threshold=validate(boxes,scores,maximum,threshold,score_threshold,center)
    result=[]
    for class_index,row in enumerate(scores):
        candidates=sorted((i for i,v in enumerate(row) if score_threshold is None or v>score_threshold),key=lambda i:(-row[i],i))
        chosen=[]
        for i in candidates:
            if len(chosen)>=maximum:
                break
            overlaps=[iou(coords[i],coords[j]) for j in chosen]
            if not any(value>=threshold if wrong_equal else value>threshold for value in overlaps):
                chosen.append(i);result.append([0,class_index,i])
    return result


def candidate_removal_oracle(boxes,scores,maximum,threshold,score_threshold=None):
    coords,maximum,threshold,score_threshold=validate(boxes,scores,maximum,threshold,score_threshold,False)
    result=[]
    for class_index,row in enumerate(scores):
        remaining={i for i,score in enumerate(row) if score_threshold is None or score>score_threshold}
        selected=0
        while remaining and selected<maximum:
            i=min(remaining,key=lambda k:(-row[k],k));result.append([0,class_index,i]);selected+=1
            remaining={j for j in remaining if j!=i and iou(coords[i],coords[j])<=threshold}
    return result


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()) or p["element_bytes"] not in (1,2,4,8):
        raise ValueError("positive bounded dimensions/cap and selected byte width required")
    b,c,n,k,e=(p[key] for key in ("batches","classes","boxes","max_per_class","element_bytes"))
    out={"box_payload_bytes":b*n*4*e,"score_payload_bytes":b*c*n*e,
         "selected_index_upper_bound_rows":b*c*min(n,k),"selected_int64_index_upper_bound_bytes":b*c*min(n,k)*3*8}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def float_paths(n):
    raw=[[0.,float(2*i),1.,float(2*i+1)] for i in range(n)]
    cached=[tuple(box) for box in raw]
    def overlap(a,b):
        y1,x1,y2,x2=a;v1,u1,v2,u2=b
        area=max(0.,min(y2,v2)-max(y1,v1))*max(0.,min(x2,u2)-max(x1,u1))
        union=(y2-y1)*(x2-x1)+(v2-v1)*(u2-u1)-area
        return area/union if union>0 else 0.
    def normalize(box):
        y1,x1,y2,x2=box
        return min(y1,y2),min(x1,x2),max(y1,y2),max(x1,x2)
    def run(use_cache):
        chosen=[]
        for i in range(n):
            if not any(overlap(cached[i],cached[j])>.5 if use_cache else overlap(normalize(raw[i]),normalize(raw[j]))>.5 for j in chosen):
                chosen.append(i)
        return chosen
    return lambda:run(True),lambda:run(False)


def measure():
    rows=[]
    for n in (16,32,64,128):
        functions=float_paths(n)
        assert functions[0]()==functions[1]()==list(range(n))
        for _ in range(2):
            for f in functions:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"boxes":n,"cached_samples_ms":samples[0],"normalize_samples_ms":samples[1],
                     "cached_median_ms":statistics.median(samples[0]),"normalize_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python float greedy IoU of disjoint boxes,precanonicalized coordinates versus normalization per pair;all selected",
            "excludes":"cached canonicalization setup,sorting,score thresholding,native ONNX/GPU/quality","rows":rows}


def build_report(latency):
    boxes=[[0,0,2,2],[0,1,2,3],[0,0,2,2],[10,10,12,12]]
    scores=[[Fraction(9,10),Fraction(8,10),Fraction(7,10),Fraction(-1,10)],
            [Fraction(1,10),Fraction(2,10),Fraction(95,100),0]]
    threshold=Fraction(1,3)
    disjoint=[[0,0,1,1],[0,2,1,3],[0,4,1,5]]
    signed=[[Fraction(-1,10),0,Fraction(1,10)]]
    return {"schema_version":1,"round":128,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0","tree_date_utc":"2026-11-12T23:39:19Z",
                      "optional_input_fix":"70f6a89920a3a17d0ea1cd84c0812a90d1e566e2","fix_date_utc":"2026-10-28T15:15:37Z",
                      "scope":"pinned reference strict explicit score predicate and index tie ordering;not universal backend tie certificate"},
            "case":{"boxes":boxes,"scores":[[str(v) for v in row] for row in scores],"iou_boundary":str(iou(canonical(boxes[0]),canonical(boxes[1]))),
                    "max_per_class":2,"strict_iou_selected":nms(boxes,scores,2,threshold),
                    "oracle_selected":candidate_removal_oracle(boxes,scores,2,threshold),
                    "wrong_equal_iou_selected":nms(boxes,scores,2,threshold,wrong_equal=True),
                    "omitted_iou_selected":nms(boxes,scores,2),
                    "cap1_selected":nms(boxes,scores,1,threshold),
                    "omitted_max_selected":nms(boxes,scores),"omitted_max_output_shape":[0,3],
                    "signed_omitted_score_selected":nms(disjoint,signed,3,Fraction(1,2)),
                    "signed_explicit_zero_score_selected":nms(disjoint,signed,3,Fraction(1,2),0)},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_onnx_latency_ms":None,"detection_quality":None,"native_gpu_tie_equivalence":None,"peak_hbm_bytes":None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true")
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
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    print("Round128 sealed artifacts match" if args.check else "Wrote Round128 original artifacts")


if __name__=="__main__":
    main()
