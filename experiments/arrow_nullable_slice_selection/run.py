"""Original immutable List<int32> buffers: slicing, validity and ownership."""

import argparse
import json
import platform
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "0974e94ef5c2d510250923a7af239ec14a522700"
MAX_SAFE = 2**53 - 1
DEFAULT = dict(parent_rows=1048576,child_values=8388608,slice_rows=32,
               visible_child_values=240,offset_bytes=4,value_bytes=4,
               parent_bitmap=1,child_bitmap=1)


def integer(value,low=0):
    return isinstance(value,int) and not isinstance(value,bool) and value >= low


def valid(bitmap,index):
    return bitmap is None or bool(bitmap[index//8] & (1 << (index%8)))


def pack_bitmap(bits):
    result = bytearray((len(bits)+7)//8)
    for i,bit in enumerate(bits):
        if bit:
            result[i//8] |= 1 << (i%8)
    return bytes(result)


def check_bitmap(bitmap,length):
    if bitmap is not None and (not isinstance(bitmap,bytes) or len(bitmap) < (length+7)//8):
        raise ValueError("validity must be absent/all-valid or a sufficient immutable bitmap")


@dataclass(frozen=True)
class ChildInt32:
    data: bytes
    validity: bytes | None = None

    def __post_init__(self):
        if not isinstance(self.data,bytes) or len(self.data)%4:
            raise TypeError("the toy int32 child requires immutable four-byte-aligned data")
        check_bitmap(self.validity,self.length)

    @property
    def length(self):
        return len(self.data)//4

    def value(self,index):
        if not integer(index) or index >= self.length:
            raise ValueError("child index outside data")
        return struct.unpack_from("<i",self.data,4*index)[0] if valid(self.validity,index) else None


@dataclass(frozen=True)
class ListBuffers:
    offsets: bytes
    child: ChildInt32
    validity: bytes | None = None

    def __post_init__(self):
        if not isinstance(self.child,ChildInt32):
            raise TypeError("required child array object must not be None")
        if not isinstance(self.offsets,bytes) or len(self.offsets)<4 or len(self.offsets)%4:
            raise TypeError("int32 offsets require at least one immutable offset")
        positions = [item[0] for item in struct.iter_unpack("<i",self.offsets)]
        if positions[0]<0 or any(a>b for a,b in zip(positions,positions[1:])) or positions[-1]>self.child.length:
            raise ValueError("offsets must be nondecreasing and within the child")
        check_bitmap(self.validity,self.rows)

    @property
    def rows(self):
        return len(self.offsets)//4-1

    @property
    def buffer_bytes(self):
        return (len(self.offsets)+len(self.child.data)+len(self.validity or b"")+
                len(self.child.validity or b""))

    def position(self,index):
        if not integer(index) or index>self.rows:
            raise ValueError("offset index outside parent")
        return struct.unpack_from("<i",self.offsets,4*index)[0]


@dataclass(frozen=True)
class ListView:
    parent: ListBuffers
    offset: int
    length: int

    def __post_init__(self):
        if not isinstance(self.parent,ListBuffers):
            raise TypeError("a required parent array object must not be None")
        if (not integer(self.offset) or not integer(self.length) or
                self.offset+self.length>self.parent.rows):
            raise ValueError("strict toy slice must remain inside its parent")

    def slice(self,start,length):
        if not integer(start) or not integer(length) or start+length>self.length:
            raise ValueError("nested slice must remain inside the current view")
        return ListView(self.parent,self.offset+start,length)

    def logical_rows(self):
        result = []
        for i in range(self.length):
            row = self.offset+i
            if not valid(self.parent.validity,row):
                result.append(None)
            else:
                result.append([self.parent.child.value(j) for j in
                               range(self.parent.position(row),self.parent.position(row+1))])
        return result

    def flatten_logical(self):
        return [value for row in self.logical_rows() if row is not None for value in row]


def canonical_buffers(rows):
    offsets,values,parent_bits,child_bits = [0],[],[],[]
    for row in rows:
        if row is not None and not isinstance(row,(list,tuple)):
            raise TypeError("each logical row must be a list or null")
        parent_bits.append(row is not None)
        if row is not None:
            for value in row:
                if value is not None and (not isinstance(value,int) or isinstance(value,bool) or
                                          not -(2**31)<=value<2**31):
                    raise ValueError("child values must be int32 or null")
                child_bits.append(value is not None)
                values.append(-777 if value is None else value)
        offsets.append(len(values))
    child = ChildInt32(b"".join(struct.pack("<i",value) for value in values),
                       pack_bitmap(child_bits) if not all(child_bits) else None)
    return ListBuffers(b"".join(struct.pack("<i",value) for value in offsets),child,
                       pack_bitmap(parent_bits) if not all(parent_bits) else None)


def compact(view):
    parent = canonical_buffers(view.logical_rows())
    return ListView(parent,0,parent.rows)


def estimate_budget(p):
    for key in ("parent_rows","child_values","slice_rows","visible_child_values"):
        if not integer(p[key]):
            raise ValueError(f"{key} must be a nonnegative integer")
    if (not integer(p["offset_bytes"],1) or not integer(p["value_bytes"],1) or
            p["offset_bytes"] not in (4,8) or p["value_bytes"] not in (1,2,4,8)):
        raise ValueError("offset_bytes 4/8 and value_bytes 1/2/4/8 required")
    for key in ("parent_bitmap","child_bitmap"):
        if not integer(p[key]) or p[key] not in (0,1):
            raise ValueError(f"{key} must be integer 0 or 1")
    if p["slice_rows"]>p["parent_rows"] or p["visible_child_values"]>p["child_values"]:
        raise ValueError("visible counts exceed backing counts")
    if p["slice_rows"] == 0 and p["visible_child_values"] != 0:
        raise ValueError("an empty slice cannot expose child values")
    def buffers(rows,children):
        return dict(offsets=(rows+1)*p["offset_bytes"],values=children*p["value_bytes"],
                    parent_validity=((rows+7)//8)*p["parent_bitmap"],
                    child_validity=((children+7)//8)*p["child_bitmap"])
    backing = buffers(p["parent_rows"],p["child_values"])
    copied = buffers(p["slice_rows"],p["visible_child_values"])
    result = dict(backing_buffers=backing,canonical_copy_buffers=copied,
                  view_retained_buffer_bytes=sum(backing.values()),
                  view_new_data_buffer_bytes=0,canonical_copy_buffer_bytes=sum(copied.values()),
                  max_offset_fits_selected_signed_type=p["child_values"]<=2**(p["offset_bytes"]*8-1)-1,
                  python_object_bytes=None,peak_rss_bytes=None,peak_hbm_bytes=None,
                  native_arrow_latency_ms=None)
    values = list(backing.values())+list(copied.values())+[result["view_retained_buffer_bytes"],
                                                         result["canonical_copy_buffer_bytes"]]
    if any(value>MAX_SAFE for value in values):
        raise ValueError("budget exceeds exact JavaScript integer range")
    return result


def fixture():
    child = canonical_buffers([[10,11,99,98,20,None,30,31,40]]).child
    offsets = b"".join(struct.pack("<i",v) for v in (0,2,4,4,6,8,9))
    return ListBuffers(offsets,child,pack_bitmap([True,False,True,True,True,False]))


def measure_latency():
    rows = []
    operations = 25
    for count in (32,128,512,2048):
        parent = canonical_buffers([None if i%7 == 0 else [i,None,i+1,i+2] for i in range(count)])
        start,length = count//4,count//2
        view = ListView(parent,start,length)
        methods = (lambda: ListView(parent,start,length),lambda: compact(view))
        samples = [[],[]]
        for method in methods:
            for _ in range(2):
                method()
        for repeat in range(7):
            for index in ((0,1) if repeat%2 == 0 else (1,0)):
                begin = time.perf_counter_ns()
                for _ in range(operations):
                    methods[index]()
                samples[index].append((time.perf_counter_ns()-begin)/1e6/operations)
        rows.append(dict(parent_rows=count,slice_rows=length,
                         primary_samples_ms=samples[0],baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         logical_mismatches=sum(a!=b for a,b in zip(methods[0]().logical_rows(),methods[1]().logical_rows()))))
    data = dict(provenance="measured",scope="Original Python buffer-view handle versus canonical materialization",
                primary="Construct a bounded view handle; no row materialization",
                baseline="Read logical rows, drop hidden spans, build/validate new buffers",
                hardware=platform.machine(),processor=platform.processor(),python=platform.python_version(),
                warmups=2,repeats=7,operations_per_repeat=operations,
                sample_statistic="mean milliseconds per operation; medians across seven repeat means",
                alternating_order=True,rows=rows,native_arrow_measured=False,
                native_arrow_latency_ms=None)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    parent = fixture()
    view = ListView(parent,1,4)
    rows = view.logical_rows()
    copied = compact(view)
    span = [parent.child.value(j) for j in range(parent.position(1),parent.position(5))]
    wrong_bits,wrong_child = [],[]
    for i in range(view.length):
        row = view.offset+i
        wrong_bits.append([parent.child.value(j) for j in range(parent.position(row),parent.position(row+1))]
                          if valid(parent.validity,i) else None)
        wrong_child.append([parent.child.value(j+view.offset) for j in
                            range(parent.position(row),parent.position(row+1))]
                           if valid(parent.validity,row) else None)
    reference = [None,[],[20,None],[30,31]]
    variants = []
    original_rows = ListView(parent,0,parent.rows).logical_rows()
    for start,length in ((0,6),(1,4),(2,0),(2,3),(5,1)):
        selected = ListView(parent,start,length)
        variants.append(dict(start=start,length=length,
                             logical_mismatches=sum(a!=b for a,b in zip(selected.logical_rows(),
                                                                       original_rows[start:start+length])),
                             compact_mismatches=sum(a!=b for a,b in zip(selected.logical_rows(),
                                                                       compact(selected).logical_rows()))))
    if not (HERE/"latency.json").exists():
        raise FileNotFoundError("Run --measure once to seal original CPU timings")
    return dict(schema_version=1,round=108,provenance="synthetic",independently_reproduced=False,
                native_arrow_measured=False,
                source=dict(pin=PIN,date="2026-11-13",license="Apache-2.0",
                            capability="Required Arrow objects reject None; valid nullable buffers remain supported",
                            url=f"https://github.com/apache/arrow/blob/{PIN}/python/pyarrow/array.pxi"),
                slice=dict(parent_rows=parent.rows,parent_offset=1,length=4,logical_rows=rows,
                           independent_logical_reference=reference,
                           maximum_row_mismatches=sum(a!=b for a,b in zip(rows,reference)),
                           visible_offsets=[parent.position(i) for i in range(1,6)],
                           parent_validity_hex=parent.validity.hex(),
                           child_validity_hex=parent.child.validity.hex(),
                           values_property=[parent.child.value(j) for j in range(parent.child.length)],
                           contiguous_child_span=span,logical_flatten=view.flatten_logical(),
                           retained_buffer_bytes=parent.buffer_bytes,new_data_buffer_bytes=0,
                           shares_parent_object=view.parent is parent,
                           compact_logical_rows=copied.logical_rows(),
                           compact_offsets=[copied.parent.position(i) for i in range(5)],
                           compact_buffer_bytes=copied.parent.buffer_bytes),
                controls=dict(ignore_parent_row_offset=ListView(parent,0,4).logical_rows(),
                              ignore_validity_bit_offset=wrong_bits,
                              double_add_parent_offset_to_child=wrong_child,
                              raw_null_storage_sentinel=-777,
                              wrong_bit_row_mismatches=sum(a!=b for a,b in zip(rows,wrong_bits)),
                              wrong_child_row_mismatches=sum(a!=b for a,b in zip(rows,wrong_child))),
                variants=variants,budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
                latency=json.loads((HERE/"latency.json").read_text(encoding="utf-8")))


def outputs():
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 108 artifacts match." if args.check else "Wrote Round 108 artifacts.")


if __name__ == "__main__":
    main()
