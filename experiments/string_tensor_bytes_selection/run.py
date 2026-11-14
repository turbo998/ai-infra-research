"""Original byte-preserving STRING tensor and bounded wire-subset model."""

import argparse
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="48fc9e0b96520db0b056b74f1af09398790f2eac"
DEFAULT={"dims":[2,2],"raw":False,"values":[{"kind":"bytes","hex":"00750000"},
         {"kind":"text","text":"猫\u0000"},{"kind":"bytes","hex":"fe4100"},{"kind":"bytes","hex":""}]}


def bounded(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError("bounded exact integer required")
    return value


def dimensions(dims):
    if not isinstance(dims,list) or len(dims)>2:
        raise ValueError("original subset requires rank0..2 dims")
    for dim in dims:
        bounded(dim,0,16)
    return math.prod(dims)


def byte_value(record):
    if not isinstance(record,dict):
        raise ValueError("typed original text/bytes record required")
    if record.get("kind")=="bytes" and set(record)=={"kind","hex"}:
        value=record["hex"]
        if not isinstance(value,str) or len(value)>1024 or not re.fullmatch(r"(?:[0-9a-fA-F]{2})*",value):
            raise ValueError("bounded complete hex-byte pairs required")
        return bytes.fromhex(value)
    if record.get("kind")=="text" and set(record)=={"kind","text"}:
        text=record["text"]
        if not isinstance(text,str) or len(text)>512:
            raise ValueError("bounded Unicode scalar text required")
        value=text.encode("utf-8")
        if len(value)>512:
            raise ValueError("original UTF-8 byte bound exceeded")
        return value
    raise ValueError("exact typed original record fields required")


def varint(value):
    bounded(value,0,4096)
    out=bytearray()
    while value>=128:
        out.append((value&127)|128);value>>=7
    out.append(value)
    return bytes(out)


def encode(dims,values):
    count=dimensions(dims)
    if not isinstance(values,list) or len(values)!=count or any(not isinstance(v,bytes) for v in values):
        raise ValueError("original byte values must match shape product,including scalar/zero dimension")
    if sum(map(len,values))>4096:
        raise ValueError("original total payload bound exceeded")
    return b"".join([*(b"\x08"+varint(dim) for dim in dims),b"\x10\x08",
                    *(b"\x32"+varint(len(v))+v for v in values)])


def decode(wire):
    if not isinstance(wire,bytes) or len(wire)>8192:
        raise ValueError("bounded original wire bytes required")
    pos=0;dims=[];values=[];dtype=None
    def read():
        nonlocal pos
        result=0
        for shift in (0,7):
            if pos>=len(wire):
                raise ValueError("truncated original varint")
            byte=wire[pos];pos+=1;result|=(byte&127)<<shift
            if byte<128:
                bounded(result,0,4096)
                return result
        raise ValueError("original varint subset exceeded")
    while pos<len(wire):
        tag=read()
        if tag==8:
            dims.append(read())
        elif tag==16:
            if dtype is not None:
                raise ValueError("duplicate datatype outside original subset")
            dtype=read()
        elif tag==50:
            length=read()
            if pos+length>len(wire):
                raise ValueError("truncated original bytes field")
            values.append(wire[pos:pos+length]);pos+=length
        else:
            raise ValueError("unsupported field/wire type in original subset")
    if dtype!=8 or len(values)!=dimensions(dims):
        raise ValueError("original STRING type/shape count mismatch")
    if sum(map(len,values))>4096:
        raise ValueError("original total payload bound exceeded")
    return dims,values


def inspect_utf8(value):
    try:
        text=value.decode("utf-8")
    except UnicodeDecodeError:
        return {"utf8_text":None,"utf8_status":"invalid_utf8_bytes_preserved"}
    return {"utf8_text":text,"utf8_status":"valid_utf8_without_normalization"}


def evaluate(p):
    if not isinstance(p,dict) or set(p)!={"dims","values","raw"} or type(p["raw"]) is not bool:
        raise ValueError("exact original dims/values/raw input required")
    if p["raw"]:
        raise ValueError("STRING uses repeated string_data;raw_data forbidden")
    count=dimensions(p["dims"])
    if not isinstance(p["values"],list) or len(p["values"])!=count:
        raise ValueError("flat typed records must match shape product")
    values=[byte_value(v) for v in p["values"]]
    wire=encode(p["dims"],values);restored_dims,restored=decode(wire)
    naive=[v.rstrip(b"\0") for v in values]
    rows=[{"index":i,"kind":p["values"][i]["kind"],"hex":v.hex(),"byte_length":len(v),
           "restored_hex":restored[i].hex(),"naive_trim_hex":naive[i].hex(),
           "lost_bytes":len(v)-len(naive[i]),**inspect_utf8(v)} for i,v in enumerate(values)]
    return {"dims":restored_dims,"element_count":count,"rows":rows,
            "preserved_elements":sum(v==a for v,a in zip(values,restored)),
            "naive_corrupted_elements":sum(v!=a for v,a in zip(values,naive)),
            "payload_bytes":sum(map(len,values)),"naive_payload_bytes":sum(map(len,naive)),
            "wire_hex":wire.hex(),"wire_bytes":len(wire),
            "framing_bytes":len(wire)-sum(map(len,values)),
            "naive_wire_bytes":len(encode(p["dims"],naive)),
            "original_distinct_values":len(set(values)),"naive_distinct_values":len(set(naive))}


def timing_paths(n):
    parts=[b"\x32\x20"+bytes((i%251,))*32 for i in range(n)]
    def joined():
        return b"".join(parts)
    def repeated():
        result=b""
        for part in parts:
            result=result+part
        return result
    return joined,repeated


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n)
        if funcs[0]()!=funcs[1]():
            raise RuntimeError("original assembly paths disagree")
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
        rows.append({"fragments":n,"assembled_bytes":34*n,"join_samples_ms":samples[0],"concat_samples_ms":samples[1],
                     "join_median_ms":statistics.median(samples[0]),"concat_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"prebuilt original 34-byte field fragments,bytes.join versus repeated immutable concatenation",
            "excludes":"UTF-8 conversion,varint/fragment preparation,shape validation,ONNX/protobuf serialization,disk/network/GPU;timer exceeds UI bounds",
            "rows":rows}


def build_report(latency):
    collision={"dims":[3],"raw":False,"values":[{"kind":"bytes","hex":v} for v in ("73","7300","730000")]}
    scalar={"dims":[],"raw":False,"values":[{"kind":"bytes","hex":""}]}
    zero={"dims":[0],"raw":False,"values":[]}
    unicode={"dims":[2],"raw":False,"values":[{"kind":"text","text":"é"},{"kind":"text","text":"e\u0301"}]}
    return {"schema_version":1,"round":149,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"onnx/onnx","pin":PIN,"license":"Apache-2.0 rights preserved;no implementation mirrored",
                      "chronology":"Nov12UTC16:15 = Nov13BJT00:15 helper correction,reviewed Nov14;not a new tensor format",
                      "verified":"str UTF-8 encoding,bytes unchanged,object dtype at both conversion steps,string_data field6 versus forbidden STRING raw_data",
                      "scope":"original typed flat records and bounded protobuf-compatible subset;not native make_tensor/NumPy/protobuf execution"},
            "case":{"inputs":DEFAULT,"outputs":evaluate(DEFAULT)},
            "robustness":{"collision":evaluate(collision),"scalar_empty":evaluate(scalar),"zero_dimension":evaluate(zero),"unicode":evaluate(unicode)},
            "latency":latency,"native_make_tensor_verified":None,"native_protobuf_roundtrip_verified":None,
            "native_legacy_numpy_outputs":None,"native_serialization_latency_ms":None,"network_payload_bytes":None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round149 sealed artifacts match" if args.check else "Wrote Round149 original artifacts")


if __name__=="__main__":
    main()
