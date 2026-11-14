"""Original field-specific64KiB page accounting and isolated raw-page evidence."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import platform
import sqlite3
import statistics
import struct
import tempfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="9b0f74e9615f1c79e43fa876b84f8a5952030e8a"
DEFAULT={"page_bytes":65536,"first_page":False,"reserved_bytes":0,"cells":0,"content_raw":0,"fragmented_bytes":0}


def integer(v,low,high):
    if type(v) is not int or not low<=v<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return v


def page_size(v):
    integer(v,512,65536)
    if v&(v-1):
        raise ValueError("power-of-two SQLite page size required")
    return v


def decode_size(raw):
    integer(raw,0,65535)
    return 65536 if raw==1 else page_size(raw)


def layout(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original leaf-header fields required")
    size=page_size(p["page_bytes"])
    if type(p["first_page"]) is not bool:
        raise ValueError("explicit first_page boolean required")
    reserved=integer(p["reserved_bytes"],0,255);cells=integer(p["cells"],0,32767)
    raw=integer(p["content_raw"],0,65535);fragments=integer(p["fragmented_bytes"],0,60)
    usable=size-reserved;base=100 if p["first_page"] else 0
    if usable<480:
        raise ValueError("usable SQLite page bytes must be at least480")
    content=65536 if raw==0 else raw
    pointer_end=base+8+2*cells
    if not pointer_end<=content<=usable or fragments>usable-content:
        raise ValueError("content/pointer/fragment geometry invalid in original no-freeblock leaf model")
    if cells==0 and (content!=usable or fragments):
        raise ValueError("original empty leaf model must have fully unused usable body")
    if cells and content==usable:
        raise ValueError("nonempty original leaf requires a content region")
    return {"file_header_bytes":base,"leaf_header_bytes":8,"pointer_array_bytes":2*cells,
            "usable_page_bytes":usable,"decoded_content_offset":content,"unallocated_bytes":content-pointer_end,
            "fragmented_bytes":fragments,"unused_without_freeblocks_bytes":content-pointer_end+fragments,
            "naive_raw_offset_unused_bytes":raw-pointer_end+fragments,
            "opaque_content_region_bytes":usable-content,
            "scope":"original header arithmetic only;leaf0x0d,no freeblocks;not cell/record/overflow/tree/file validation"}


def opaque_leaf(size,lengths,first=False,reserved=0):
    page_size(size);integer(reserved,0,255)
    if not isinstance(lengths,list) or len(lengths)>64:
        raise ValueError("up to64opaque original cell lengths required")
    for n in lengths:
        integer(n,1,65535)
    content=size-reserved-sum(lengths)
    p={"page_bytes":size,"first_page":first,"reserved_bytes":reserved,"cells":len(lengths),
       "content_raw":0 if content==65536 else content,"fragmented_bytes":0}
    result=layout(p);data=bytearray(size);base=result["file_header_bytes"]
    if first:
        data[:16]=b"SQLite format 3\x00"
        data[16:18]=(1 if size==65536 else size).to_bytes(2,"big")
        data[20]=reserved
    struct.pack_into(">BHHHB",data,base,13,0,len(lengths),p["content_raw"],0)
    cursor=size-reserved
    for i,n in enumerate(lengths):
        cursor-=n;data[cursor:cursor+n]=bytes([i%255+1])*n
        struct.pack_into(">H",data,base+8+2*i,cursor)
    return bytes(data),p


def decode_leaf(data,page_number,reserved=0):
    if not isinstance(data,bytes):
        raise ValueError("original immutable full page bytes required")
    size=page_size(len(data));integer(page_number,1,2**20);base=100 if page_number==1 else 0
    flag,freeblock,cells,raw,fragments=struct.unpack_from(">BHHHB",data,base)
    if flag!=13 or freeblock:
        raise ValueError("original decoder only supports table leaf0x0d with no freeblocks")
    p={"page_bytes":size,"first_page":page_number==1,"reserved_bytes":reserved,
       "cells":cells,"content_raw":raw,"fragmented_bytes":fragments}
    result=layout(p)
    for i in range(cells):
        pointer=struct.unpack_from(">H",data,base+8+2*i)[0]
        if not result["decoded_content_offset"]<=pointer<result["usable_page_bytes"]:
            raise ValueError("original cell pointer outside content region")
    return {"inputs":p,"outputs":result,"first_freeblock_raw":freeblock}


def sqlite_fixture():
    rows=[]
    with tempfile.TemporaryDirectory(prefix="original-page-",dir=HERE) as folder:
        for size in (4096,65536):
            path=Path(folder)/f"original-{size}.db"
            with closing(sqlite3.connect(path)) as db:
                db.execute(f"PRAGMA page_size={size}");db.execute("CREATE TABLE original(x)");db.commit()
                root=db.execute("SELECT rootpage FROM sqlite_schema WHERE name='original'").fetchone()[0]
                stat=None;reason=None
                try:
                    stat=db.execute("SELECT unused FROM dbstat WHERE name='original' AND path='/'").fetchone()[0]
                except sqlite3.OperationalError as error:
                    if str(error)!="no such table: dbstat":
                        raise
                    reason="dbstat module unavailable: "+str(error)
                data=path.read_bytes();raw_size=int.from_bytes(data[16:18],"big")
                if decode_size(raw_size)!=size:
                    raise RuntimeError("actual original DB header page size differs from requested size")
                leaf=data[(root-1)*size:root*size]
                decoded=decode_leaf(leaf,root,data[20])
                rows.append({"page_bytes":size,"root_page":root,"file_page_size_raw":raw_size,
                             "btree_header_bytes":list(leaf[:8]),"decoded":decoded,
                             "native_dbstat_unused_bytes":stat,"dbstat_available":reason is None,"dbstat_unavailable_reason":reason})
    return {"provenance":"measured","sqlite_version":sqlite3.sqlite_version,"rows":rows,
            "scope":"original empty-table database headers actually read;not patched-source build or native dbstat fix reproduction when module unavailable"}


def timing_paths(n):
    header=b"\r\0\0\0\0\0\0\0"
    def unpacked():
        out=[]
        for _ in range(n):
            flag,free,cells,raw,frag=struct.unpack(">BHHHB",header)
            out.append((flag,free,cells,65536 if raw==0 else raw,frag))
        return out
    def manual():
        out=[]
        for _ in range(n):
            raw=(header[5]<<8)|header[6]
            out.append((header[0],(header[1]<<8)|header[2],(header[3]<<8)|header[4],65536 if raw==0 else raw,header[7]))
        return out
    return unpacked,manual


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        funcs=timing_paths(n);assert funcs[0]()==funcs[1]()
        for _ in range(2):
            for f in funcs:
                f()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    funcs[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"headers":n,"unpacked_samples_ms":samples[0],"manual_samples_ms":samples[1],
                     "unpacked_median_ms":statistics.median(samples[0]),"manual_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"same prevalidated8-byte64KiB empty-leaf header repeated;struct.unpack versus manual shifts;both output tuples",
            "excludes":"full-page reads,validation,different physical pages,SQLite engine/dbstat,disk,cache misses or allocator peaks","rows":rows}


def build_report(latency,native):
    rows=[]
    for size,lengths,first,reserved in ((4096,[],False,0),(65536,[],False,0),(65536,[],True,0),(65536,[],False,32),(65536,[9,17,33],False,0)):
        data,p=opaque_leaf(size,lengths,first,reserved)
        rows.append({"inputs":p,"opaque_cell_lengths":lengths,"decoded":decode_leaf(data,1 if first else 2,reserved),
                     "independent_unused_bytes":size-reserved-(100 if first else 0)-8-2*len(lengths)-sum(lengths)})
    return {"schema_version":1,"round":143,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sqlite/sqlite","pin":PIN,"license":"SQLite public-domain source;no implementation mirrored",
                      "verified":"Nov14dbstat empty64KiB page unused-byte correction and upstream regression;file header page-size1 differs from content-offset0sentinel",
                      "chronology":"Nov14actual dbstat accounting fix,not new SQLite page format",
                      "scope":"original leaf-header arithmetic and opaque cells;no freeblock chains,record varints,overflow pages,interior pages or full-file validation"},
            "case":{"inputs":DEFAULT,"outputs":layout(DEFAULT),"scenarios":rows,
                    "field_context":{"file_page_size_raw":1,"decoded_page_size":decode_size(1),"content_offset_raw":0,"decoded_content_offset":65536,
                                     "first_freeblock_raw":0,"meaning":"no freeblock,not65536","cell_count_raw":0,"decoded_cell_count":0}},
            "native_raw_page_fixture":native,"latency":latency,
            "patched_dbstat_binary_verified":None,"native_dbstat_throughput":None,"native_storage_latency_ms":None}


def main():
    p=argparse.ArgumentParser();p.add_argument("--measure",action="store_true");p.add_argument("--check",action="store_true");a=p.parse_args()
    if a.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
        (HERE/"native_oracle.json").write_text(json.dumps(sqlite_fixture(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")),json.loads((HERE/"native_oracle.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if a.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round143 sealed artifacts match" if a.check else "Wrote Round143 original artifacts")


if __name__=="__main__":
    main()
