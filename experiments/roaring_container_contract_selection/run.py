"""Original Roaring-like container payloads, exact identities and interval oracle."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import struct
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="188a7c75a9cea6b2b6d31efb604fab621e6a1b28"
CAPABILITY_PIN="ef50ca287eb6e775a4e90f012e1e089d193b610c"


def validated(values):
    values=list(values)
    if len(values)>65536 or any(type(v) is not int or not 0<=v<2**32 for v in values):
        raise ValueError("at most65536 unsigned32bit exact item IDs required")
    return sorted(set(values))


def runs_of(values):
    intervals=[]
    for v in values:
        if intervals and intervals[-1][1]+1==v:
            intervals[-1][1]=v
        else:
            intervals.append([v,v])
    return intervals


def encode(values,allow_runs=True):
    if type(allow_runs) is not bool:
        raise ValueError("explicit run-optimization boolean required")
    groups={}
    for v in validated(values):
        groups.setdefault(v>>16,[]).append(v&65535)
    containers=[]
    for key,low in groups.items():
        count=len(low);intervals=runs_of(low)
        kind="array" if count<=4096 else "bitset"
        size=2*count if kind=="array" else 8192
        if allow_runs and 2+4*len(intervals)<size:
            kind="run"
        if kind=="array":
            payload=struct.pack("<"+"H"*count,*low)
        elif kind=="run":
            payload=struct.pack("<H",len(intervals))+b"".join(struct.pack("<HH",start,end-start) for start,end in intervals)
        else:
            storage=bytearray(8192)
            for value in low:
                storage[value//8]|=1<<(value%8)
            payload=bytes(storage)
        containers.append({"key":key,"cardinality":count,"kind":kind,"runs":len(intervals),"payload_hex":payload.hex()})
    return containers


def decode(containers):
    result=[];lastkey=-1
    for c in containers:
        key,n,kind=c["key"],c["cardinality"],c["kind"]
        if type(key) is not int or not lastkey<key<=65535 or type(n) is not int or not 1<=n<=65536:
            raise ValueError("sorted unique high16bit keys and positive cardinality required")
        raw=bytes.fromhex(c["payload_hex"]);lastkey=key
        if kind=="array":
            if n>4096 or len(raw)!=2*n:
                raise ValueError("array cardinality/payload mismatch")
            low=list(struct.unpack("<"+"H"*n,raw))
            if low!=sorted(set(low)):
                raise ValueError("array values must be strictly increasing")
        elif kind=="bitset":
            if n<=4096 or len(raw)!=8192:
                raise ValueError("bitset threshold/payload mismatch")
            bits=int.from_bytes(raw,"little")
            if bits.bit_count()!=n:
                raise ValueError("bitset cardinality mismatch")
            low=[]
            while bits:
                bit=bits&-bits;low.append(bit.bit_length()-1);bits-=bit
        elif kind=="run":
            if len(raw)<2:
                raise ValueError("run header missing")
            count=struct.unpack_from("<H",raw)[0]
            if count==0 or len(raw)!=2+4*count:
                raise ValueError("run count/payload mismatch")
            low=[];last=-1
            for i in range(count):
                start,length=struct.unpack_from("<HH",raw,2+4*i);end=start+length
                if start<=last or end>65535:
                    raise ValueError("run overlap/order/low16bit overflow")
                low.extend(range(start,end+1));last=end
            if len(low)!=n:
                raise ValueError("run cardinality mismatch")
        else:
            raise ValueError("unknown container kind")
        result.extend((key<<16)|value for value in low)
    return result


def describe(values,allow_runs=True):
    unique=validated(values);containers=encode(unique,allow_runs)
    summaries=[{k:v for k,v in c.items() if k!="payload_hex"}|{"payload_bytes":len(bytes.fromhex(c["payload_hex"]))} for c in containers]
    return {"unique_count":len(unique),"containers":summaries,
            "container_payload_bytes":sum(c["payload_bytes"] for c in summaries),
            "uint32_identity_payload_bytes":4*len(unique),
            "flat_bitmap_through_max_id_bytes":(unique[-1]+8)//8 if unique else 0,
            "note":"container payload only;cookies/keys/cardinality/offset tables/allocator excluded;original strict run-saving policy"}


def intersection_sorted(a,b):
    i=j=0;result=[]
    while i<len(a) and j<len(b):
        if a[i]==b[j]:
            result.append(a[i]);i+=1;j+=1
        elif a[i]<b[j]:
            i+=1
        else:
            j+=1
    return result


def interval_intersection_oracle(a,b):
    left=runs_of(validated(a));right=runs_of(validated(b));result=[]
    for x0,x1 in left:
        for y0,y1 in right:
            if max(x0,y0)<=min(x1,y1):
                result.extend(range(max(x0,y0),min(x1,y1)+1))
    return result


def fixtures():
    return [5,9,13]+list(range(65536+100,65536+200)),[9,13]+list(range(65536+150,65536+250))


def timing_paths(n):
    a=list(range(0,n*3,3));b=list(range(0,n*5,5))
    ba=sum(1<<v for v in a);bb=sum(1<<v for v in b)
    def integer_and():
        return (ba&bb).bit_count()
    def merge_count():
        return len(intersection_sorted(a,b))
    return integer_and,merge_count


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        paths=timing_paths(n);assert paths[0]()==paths[1]()
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
        rows.append({"items_per_set":n,"integer_and_samples_ms":samples[0],"sorted_merge_samples_ms":samples[1],
                     "integer_and_median_ms":statistics.median(samples[0]),"sorted_merge_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"prebuilt Python arbitrary-width integer AND+bit_count versus original sorted two-pointer intersection materialization+len",
            "excludes":"construction,encoding,decoding,Roaring/CRoaring,container dispatch,SIMD,native rank/select,ANN/ACL latency",
            "unequal_work":"C-backed integer cardinality versus Python list materialization;not native algorithm speedup","rows":rows}


def build_report(latency):
    a,b=fixtures();decoded_a=decode(encode(a));decoded_b=decode(encode(b))
    intersection=intersection_sorted(decoded_a,decoded_b)
    cases=[]
    for name,ids in (("array_threshold4096",range(0,8192,2)),("bitset_threshold4097",range(0,8194,2)),("contiguous4097",range(4097)),("full_low16",range(65536))):
        cases.append({"name":name,"without_runs":describe(ids,False),"with_runs":describe(ids,True)})
    return {"schema_version":1,"round":137,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"RoaringBitmap/RoaringFormatSpec","pin":PIN,"license":"Apache-2.0",
                      "capability_repository":"RoaringBitmap/CRoaring","capability_pin":CAPABILITY_PIN,
                      "chronology":"Nov9CRoaring64rank/select/set-op cardinality capabilities;Nov10format-specwebpage,notnew32bitcontainerthreshold",
                      "verified":"unsigned32bit high16/low16,non-runarray<=4096,bitset8192B,runlength-minus-one,payload2+4r;whole-fileheadersseparate"},
            "case":{"A_ids":a,"B_ids":b,"A":describe(a),"B":describe(b),"A_without_runs":describe(a,False),
                    "intersection":intersection,"intersection_count":len(intersection),"union_count":len(set(a)|set(b)),
                    "interval_oracle":interval_intersection_oracle(a,b),"reversed_intersection":intersection_sorted(decoded_b,decoded_a),
                    "reversed_duplicate_input_summary":describe(list(reversed(a))+a),
                    "same_low_different_high":{"A":[7],"B":[65543],"true_intersection":intersection_sorted([7],[65543]),"wrong_low_only_intersection":[7]},
                    "unsigned_boundaries_roundtrip":decode(encode([0,65535,65536,2**31,2**32-1])),
                    "empty_summary":describe([])},
            "boundaries":cases,"latency":latency,"native_croaring_latency_ms":None,"native_roaring64_rank_select":None,
            "native_memory_bytes":None,"whole_file_portable_bytes":None,"security_revocation_latency_ms":None}


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
    print("Round137 sealed artifacts match" if a.check else "Wrote Round137 original artifacts")


if __name__=="__main__":
    main()
