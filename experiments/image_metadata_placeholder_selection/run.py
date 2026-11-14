"""Original bounded image-metadata planner; no tokenizer, tensors or image encoder."""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PIN="24fe0a99f1062f7351852804f2cd38668f7423bd"
DEFAULT={"sizes":[[65,33],[9,7],[65,33]],"patch":8,"merge":2,"patch_budget":256,
         "side_patches":16,"fixed_tokens":None,"allow_missing":True,"embedding_shape":None}


def integer(value,lo=1,hi=8192):
    return type(value) is int and lo<=value<=hi


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("Exactly the original dimension/configuration fields are required")
    if not isinstance(p["sizes"],list) or not 1<=len(p["sizes"])<=64 or any(
        not isinstance(s,list) or len(s)!=2 or any(not integer(v) for v in s) for s in p["sizes"]):
        raise ValueError("1..64 positive integer [width,height] pairs,each <=8192,required")
    for key,limit in (("patch",128),("merge",8),("patch_budget",4096),("side_patches",64)):
        if not integer(p[key],hi=limit):
            raise ValueError("Positive bounded original resize configuration required")
    if p["fixed_tokens"] is not None and not integer(p["fixed_tokens"],hi=1024):
        raise ValueError("Fixed output must be null or an original positive count <=1024")
    if p["side_patches"]%p["merge"]:
        raise ValueError("Original profiles require a merge-aligned side limit")
    if type(p["allow_missing"]) is not bool:
        raise ValueError("allow_missing is an explicit deployment-profile boolean")
    shape=p["embedding_shape"]
    if shape is not None and (not isinstance(shape,list) or len(shape)!=2 or
                              not integer(shape[0],hi=262144) or not integer(shape[1],hi=128)):
        raise ValueError("Original embedding shape is null or [positive rows,positive columns]")
    if shape is None and not p["allow_missing"]:
        raise ValueError("Embedding values are required when the deployment disallows missing values")


def geometry(width,height,p):
    cap=p["side_patches"]*p["patch"]
    density=max(1,width//p["patch"])*max(1,height//p["patch"])
    scale=min(1,math.sqrt(p["patch_budget"]/density),cap/width,cap/height)
    w=min(cap,max(1,int(width*scale)));h=min(cap,max(1,int(height*scale)))
    factor=p["patch"]*p["merge"]
    padded_w=((w+factor-1)//factor)*factor
    padded_h=((h+factor-1)//factor)*factor
    tokens=p["fixed_tokens"] if p["fixed_tokens"] is not None else (padded_w//factor)*(padded_h//factor)
    return {"resized":[w,h],"padded":[padded_w,padded_h],"tokens":tokens}


def geometry_rows(sizes,p,memo):
    cache={};rows=[]
    for width,height in sizes:
        key=(width,height)
        if memo and key in cache:
            value=cache[key]
        else:
            value=geometry(width,height,p)
            if memo:cache[key]=value
        rows.append(value)
    return rows


def plan(p,memo=True,proposal="original"):
    validate(p)
    if proposal not in ("original","padded_headers","collapse_occurrences"):
        raise ValueError("Known original counterfactual required")
    rows=geometry_rows(p["sizes"],p,memo)
    total=sum(x["tokens"] for x in rows)
    if p["embedding_shape"] is not None and p["embedding_shape"][0]!=total:
        raise ValueError("Declared embedding rows do not sum to resize-derived token counts")
    indices=list(range(len(rows)))
    if proposal=="collapse_occurrences":
        seen=set();indices=[]
        for i,size in enumerate(p["sizes"]):
            if tuple(size) not in seen:indices.append(i);seen.add(tuple(size))
    headers=[];ranges=[];offset=0
    for i in indices:
        size=rows[i]["padded"] if proposal=="padded_headers" else p["sizes"][i]
        headers.append({"occurrence":i,"original_size":size,"pad_tokens":rows[i]["tokens"]})
        ranges.append([offset,offset+rows[i]["tokens"]]);offset+=rows[i]["tokens"]
    expected=[{"occurrence":i,"original_size":size,"pad_tokens":rows[i]["tokens"]}
              for i,size in enumerate(p["sizes"])]
    misses=sum(x not in headers for x in expected)
    packet=[{"image_sizes":size} for size in p["sizes"]]
    return {"mode":"metadata_only_deferred" if p["embedding_shape"] is None else "declared_embedding_rows",
            "resize_rows":rows,"headers":headers,"planned_embedding_ranges":ranges,
            "total_pad_tokens":total,"retained_pad_tokens":offset,"placeholder_misses":misses,
            "occurrences":len(rows),"retained_occurrences":len(headers),
            "compact_descriptor_bytes":len(json.dumps(packet,separators=(",",":")).encode()),
            "declared_uncompressed_rgb_bytes":sum(w*h*3 for w,h in p["sizes"]),
            "native_tokenizer_token_ids":None,"native_embedding_execution":None,
            "native_transport_bytes":None}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        sizes=[[65,33],[9,7],[128,128]]*(n//3)+[[65,33],[9,7],[128,128]][:n%3]
        if geometry_rows(sizes,DEFAULT,True)!=geometry_rows(sizes,DEFAULT,False):
            raise AssertionError("Memoized geometry changed ordered occurrence outputs")
        samples={True:[],False:[]}
        for mode in (True,False):
            for _ in range(2):geometry_rows(sizes,DEFAULT,mode)
        for repeat in range(7):
            for mode in ((True,False) if repeat%2==0 else (False,True)):
                start=time.perf_counter_ns()
                for _ in range(3):geometry_rows(sizes,DEFAULT,mode)
                samples[mode].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"occurrences":n,"memo_geometry_calls":3,"direct_geometry_calls":n,
                     "memo_samples_ms":samples[True],"direct_samples_ms":samples[False],
                     "memo_median_ms":statistics.median(samples[True]),
                     "direct_median_ms":statistics.median(samples[False])})
    return {"provenance":"measured","scope":"Original Python metadata geometry only;both paths retain every occurrence",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows,
            "warmups":2,"repeats":7,"batches_per_sample":3,"native_vllm_latency_ms":None}


def report(latency):
    variants=[("metadata-only",{}),("flat embedding rows",{"embedding_shape":[31,2]}),
              ("fixed output",{"fixed_tokens":4}),("side-limited aspect",{"sizes":[[8192,1]]}),
              ("missing values rejected",{"allow_missing":False}),
              ("row sum rejected",{"embedding_shape":[30,2]})]
    matrix=[]
    for name,changes in variants:
        p={**DEFAULT,**changes}
        try:out=plan(p)
        except ValueError as error:out={"error":str(error)}
        matrix.append({"name":name,"inputs":p,"outputs":out})
    return {"schema_version":1,"round":166,"track":"Image metadata / EPD",
            "provenance":"synthetic","independently_reproduced":False,
            "source_commit":PIN,"source_date_utc":"2026-11-14T04:13:13Z",
            "source_url":f"https://github.com/vllm-project/vllm/commit/{PIN}",
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),
                    "padded_headers":plan(DEFAULT,proposal="padded_headers"),
                    "collapsed":plan(DEFAULT,proposal="collapse_occurrences")},
            "matrix":matrix,"latency":latency,"native_checkpoint_configuration":None,
            "native_image_encoder_or_tokenizer_verified":False,"visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");args=ap.parse_args()
    path=HERE/"latency.json"
    if args.measure or not path.exists():
        path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=report(json.loads(path.read_text(encoding="utf-8")))
    from experiments.image_metadata_placeholder_selection.render import outputs
    files={HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}
    for path,value in files.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=value:raise AssertionError(f"Unsealed artifact:{path}")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    print("Round166 sealed artifacts match" if args.check else "Rendered original image metadata contracts")


if __name__=="__main__":main()
