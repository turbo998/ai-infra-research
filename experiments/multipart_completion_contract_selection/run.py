"""Original multipart/manifest state model,not an S3 client or network test."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEFAULT={"object_bytes":1099511627776,"part_bytes":134217728,"concurrency":16}


class Upload:
    def __init__(self):
        self.parts={};self.revision=0;self.object=None;self.active=True

    def put(self,number,payload):
        if not self.active or type(number) is not int or not 1<=number<=10000 or not isinstance(payload,bytes):
            raise ValueError("active upload,part1..10000 and byte payload required")
        self.revision+=1
        token=f"opaque-revision-{self.revision}"
        self.parts[number]=(token,payload)
        return token

    def complete(self,manifest,expected_sha256=None,expected_length=None,consecutive=True):
        if not self.active or not manifest:
            raise ValueError("active upload and nonempty completion manifest required")
        numbers=[entry[0] for entry in manifest]
        if any(type(n) is not int for n in numbers) or numbers!=sorted(set(numbers)):
            raise ValueError("completion list must have unique ascending part numbers")
        if consecutive and numbers!=list(range(1,len(numbers)+1)):
            raise ValueError("selected checksum mode requires consecutive numbers starting1")
        for number,token in manifest:
            if number not in self.parts or self.parts[number][0]!=token:
                raise ValueError("missing part or stale opaque ETag token")
        data=b"".join(self.parts[number][1] for number in numbers)
        if expected_length is not None and len(data)!=expected_length:
            raise ValueError("application expected-length mismatch")
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest()!=expected_sha256:
            raise ValueError("application expected full-object SHA256 mismatch")
        self.object=data;self.active=False;self.parts.clear()
        return data


def parse_completion(status,body):
    if type(status) is not int or status!=200:
        raise ValueError("non-success HTTP completion status")
    root=ET.fromstring(body)
    if root.tag.rsplit("}",1)[-1]!="CompleteMultipartUploadResult":
        raise ValueError("completion body is not a success result")
    return {"kind":"completion_result","http_status":status}


def independent_assembly(parts,manifest):
    ordered=[parts[number][1] for number,_ in manifest]
    return bytes(value for part in ordered for value in part)


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or not 1<=v<=2**53-1 for v in p.values()):
        raise ValueError("positive bounded exact bytes/concurrency required")
    b,s,c=(p[k] for k in ("object_bytes","part_bytes","concurrency"))
    if s>5*1024**3:
        raise ValueError("selected part exceeds5GiB")
    parts=(b+s-1)//s
    if parts>10000 or (parts>1 and s<5*1024**2):
        raise ValueError("selected plan exceeds10000parts or nonfinal minimum5MiB")
    out={"selected_part_count":parts,"selected_final_part_bytes":b-s*(parts-1),
         "selected_create_upload_complete_calls":parts+2,
         "selected_equal_size_buffer_upper_bytes":min(c,parts)*s}
    if max(out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def timing_paths(n):
    parts=[bytes((i%256 for _ in range(1024))) for i in range(n)]
    def joining():
        return b"".join(parts)
    def copying():
        data=bytearray(n*1024)
        for i,part in enumerate(parts):
            data[i*1024:(i+1)*1024]=part
        return bytes(data)
    return joining,copying


def measure():
    rows=[]
    for n in (16,64,256,1024):
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
        rows.append({"parts":n,"part_bytes":1024,"join_samples_ms":samples[0],"copy_samples_ms":samples[1],
                     "join_median_ms":statistics.median(samples[0]),"copy_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original local immutable bytes assembly:join versus preallocated bytearray slice copy plus bytes conversion",
            "excludes":"S3/network/SDK/checksums/server assembly/disk/retries;1KiBparts are toy not valid multi-partS3nonfinalsizes","rows":rows}


def build_report(latency):
    u=Upload();t2=u.put(2,b"BBBB");t1old=u.put(1,b"AAAA");t1new=u.put(1,b"CCCC");t3=u.put(3,b"DD")
    manifest=[[1,t1new],[2,t2],[3,t3]];expected=b"CCCCBBBBDD"
    original=list(expected);oracle=independent_assembly(u.parts,manifest)
    before=u.object;assembled=u.complete(manifest,hashlib.sha256(expected).hexdigest(),len(expected))
    omitted=Upload();a=omitted.put(1,b"CCCC");omitted.put(2,b"BBBB")
    subset=omitted.complete([[1,a]])
    return {"schema_version":1,"round":135,"provenance":"synthetic","independently_reproduced":False,
            "source":{"service":"Amazon S3","accessed_bjt":"2026-11-14","urls":[
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html",
                "https://docs.aws.amazon.com/AmazonS3/latest/API/API_CompleteMultipartUpload.html",
                "https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html"],
                "verified":"part replacement,supplied list ascending assembly,opaque ETag/full checksum distinction,HTTP200embeddederror,10000parts and5MiB/5GiB",
                "version_scope":"live AWS documentation,not pinned source;no daily popularity/new protocol claim"},
            "case":{"upload_order":[2,1,1,3],"old_part1_token":t1old,"current_part1_token":t1new,
                    "completion_manifest":manifest,"part_numbers":[1,2,3],"toy_part_lengths":[4,4,2],
                    "before_complete_object":before,"assembled_bytes":list(assembled),"expected_bytes":original,
                    "oracle_bytes":list(oracle),"assembled_length":len(assembled),"expected_sha256":hashlib.sha256(expected).hexdigest(),
                    "subset_supplied_manifest_bytes":list(subset),
                    "http_200_error_control":{"status":200,"body":"<Error><Code>InternalError</Code></Error>","acceptable":False},
                    "success_body":"<CompleteMultipartUploadResult><ETag>opaque-object-token</ETag></CompleteMultipartUploadResult>",
                    "etag_policy":"synthetic revision tokens,not MD5;no native ETag byte reproduction",
                    "checksum_mode":"original strict consecutive list option;application expected length/hash separate from native completion list"},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_s3_latency_ms":None,"native_transfer_throughput_bytes_per_s":None,"checkpoint_global_atomicity":None}


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
    print("Round135 sealed artifacts match" if a.check else "Wrote Round135 original artifacts")


if __name__=="__main__":
    main()
