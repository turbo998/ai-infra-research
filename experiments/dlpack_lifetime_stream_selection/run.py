"""Original DLPack contract models; no native capsule, C ABI or GPU execution."""

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import platform
import statistics
import struct
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "c1954b9becbd37d6c04e705cfb13e09331a9d762"
MAX_EXACT = 2**53-1
DEFAULT = {"rows":1024,"columns":768,"row_stride_elements":800,"element_bytes":2,
           "byte_offset":64,"owner_bytes":2097152}


@dataclass(frozen=True)
class Descriptor:
    shape: tuple
    strides: tuple | None
    byte_offset: int
    itemsize: int
    owner_bytes: int
    version: tuple = (1,4)

    def __post_init__(self):
        if (type(self.version) is not tuple or len(self.version)!=2 or
                any(type(v) is not int or v<0 for v in self.version) or self.version[0]!=1):
            raise ValueError("original descriptor model supports major one only")
        if type(self.shape) is not tuple or (self.strides is not None and type(self.strides) is not tuple):
            raise ValueError("immutable shape and stride tuples required")
        for value in (*self.shape,self.byte_offset,self.owner_bytes):
            if type(value) is not int or not 0<=value<=MAX_EXACT:
                raise ValueError("bounded nonnegative exact dimensions/offset/storage required")
        if self.itemsize not in (1,2,4,8) or type(self.itemsize) is not int:
            raise ValueError("only byte-addressable scalar widths 1/2/4/8 modeled")
        strides = self.strides
        if strides is None:
            if self.shape and self.version>=(1,2):
                raise ValueError("DLPack >=1.2 requires non-null strides for ndim>0")
            reverse,step = [],1
            for size in reversed(self.shape):
                reverse.append(step); step*=size
            strides = tuple(reversed(reverse))
            object.__setattr__(self,"strides",strides)
        if len(strides)!=len(self.shape) or any(type(s) is not int or abs(s)>MAX_EXACT for s in strides):
            raise ValueError("one bounded signed element stride per dimension required")
        count = 1
        for size in self.shape:
            count*=size
        if count*self.itemsize>MAX_EXACT:
            raise ValueError("logical payload exceeds exact arithmetic range")
        if count:
            low,high = self.bounds()
            if low<0 or high>self.owner_bytes or high>MAX_EXACT:
                raise ValueError("selected descriptor addresses outside selected owner")

    def bounds(self):
        if 0 in self.shape:
            return self.byte_offset,self.byte_offset
        deltas = [(size-1)*stride for size,stride in zip(self.shape,self.strides)]
        return (self.byte_offset+self.itemsize*sum(min(0,d) for d in deltas),
                self.byte_offset+self.itemsize*sum(max(0,d) for d in deltas)+self.itemsize)

    def addresses(self):
        count = 1
        for size in self.shape:
            count*=size
        if count>100000:
            raise ValueError("original address enumeration is limited to 100000 scalar entries")
        return [self.byte_offset+self.itemsize*sum(i*s for i,s in zip(index,self.strides))
                for index in itertools.product(*(range(size) for size in self.shape))]


def gather_i32(owner,descriptor,wrong_byte_strides=False):
    if descriptor.itemsize!=4 or len(owner)!=descriptor.owner_bytes:
        raise ValueError("original signed-int32 fixture requires matching owner and width")
    addresses = descriptor.addresses()
    if wrong_byte_strides:
        addresses = [descriptor.byte_offset+(address-descriptor.byte_offset)//descriptor.itemsize
                     for address in addresses]
    return [struct.unpack_from("<i",owner,address)[0] for address in addresses]


class Capsule:
    """Explicit toy state machine, not a PyCapsule or a C-ABI implementation."""

    def __init__(self,owner,*,copy=None,readonly=False,major=1):
        if copy is not None and type(copy) is not bool:
            raise ValueError("copy must be explicitly True/False/None")
        if type(readonly) is not bool:
            raise ValueError("explicit Boolean read-only flag required")
        self.owner = bytearray(owner) if copy else owner
        self.copied,self.readonly,self.major = copy is True,readonly,major
        self.name,self.deleter_calls = "dltensor_versioned",0

    def _delete(self):
        if self.deleter_calls:
            raise RuntimeError("duplicate deleter callback")
        self.deleter_calls+=1

    def consume(self):
        if self.name!="dltensor_versioned":
            raise ValueError("capsule can be consumed only once")
        self.name = "used_dltensor_versioned"
        if self.major!=1:
            self._delete()
            raise ValueError("major mismatch: release without interpreting payload")
        return Borrowed(self)

    def close(self):
        if self.name=="dltensor_versioned":
            self.name = "closed"
            self._delete()


class Borrowed:
    def __init__(self,capsule):
        self.capsule,self.closed = capsule,False

    def value(self,index):
        if self.closed:
            raise ValueError("borrowed handle is closed")
        return self.capsule.owner[index]

    def set(self,index,value):
        if self.closed or self.capsule.readonly:
            raise ValueError("closed or read-only borrowed handle")
        self.capsule.owner[index] = value

    def close(self):
        if not self.closed:
            self.closed = True
            self.capsule._delete()


def schedules(with_wait_edge=True):
    result = []
    for producer_slots in itertools.combinations(range(4),2):
        p,c = iter(("write","record")),iter(("wait","read"))
        sequence = tuple(next(p) if i in producer_slots else next(c) for i in range(4))
        if not with_wait_edge or sequence.index("record")<sequence.index("wait"):
            result.append(sequence)
    return result


def estimate_budget(p):
    for key in DEFAULT:
        minimum = 0 if key in ("rows","byte_offset","owner_bytes") else 1
        if type(p[key]) is not int or not minimum<=p[key]<=MAX_EXACT:
            raise ValueError(f"{key} requires a bounded exact integer >= {minimum}")
    if p["row_stride_elements"]<p["columns"]:
        raise ValueError("calculator selects nonoverlapping pitched rows")
    d = Descriptor((p["rows"],p["columns"]),(p["row_stride_elements"],1),
                   p["byte_offset"],p["element_bytes"],p["owner_bytes"])
    low,high = d.bounds()
    logical = p["rows"]*p["columns"]*p["element_bytes"]
    return {"logical_payload_bytes":logical,"addressed_span_bytes":high-low,
            "gaps_within_span_bytes":high-low-logical,"selected_retained_owner_bytes":p["owner_bytes"],
            "selected_compact_copy_payload_bytes":logical,
            "first_address":low if logical else None,"last_end":high if logical else None}


def measure():
    rows = []
    for size in (4096,65536,1048576,4194304):
        owner = bytearray((i%251 for i in range(size)))
        functions = (lambda:memoryview(owner).toreadonly(),lambda:bytes(owner))
        samples = [[],[]]
        for _ in range(2):
            for function in functions:
                function()
        for repeat in range(7):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                output = None
                started = time.perf_counter_ns()
                for _ in range(25):
                    output = functions[index]()
                samples[index].append((time.perf_counter_ns()-started)/25/1e6)
                assert bytes(output)==owner
        rows.append({"buffer_bytes":size,"view_samples_ms":samples[0],"copy_samples_ms":samples[1],
                     "view_median_ms":statistics.median(samples[0]),
                     "copy_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","method":"original Python readonly memoryview creation versus immutable bytes copy",
            "warmups":2,"repeats":7,"operations_per_sample":25,
            "excludes":"allocation of original owner, content checks, DLPack export/import, C ABI, GPU streams and kernels",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    owner = bytearray(struct.pack("<16i",*range(16)))
    descriptor = Descriptor((2,3),(4,1),4,4,len(owner))
    expected = [1,2,3,5,6,7]
    actual,wrong = gather_i32(owner,descriptor),gather_i32(owner,descriptor,True)
    assert actual==expected
    guarded,unguarded = schedules(),schedules(False)
    stale = sum(row.index("read")<row.index("write") for row in unguarded)
    lifetimes = []
    for order in ("capsule-first","consumer-first","unconsumed","major-mismatch"):
        capsule = Capsule(bytearray([1,2,3]),major=2 if order=="major-mismatch" else 1)
        if order=="unconsumed":
            capsule.close()
        elif order=="major-mismatch":
            try:
                capsule.consume()
            except ValueError as error:
                assert "major mismatch" in str(error)
        else:
            borrowed = capsule.consume()
            if order=="capsule-first":
                capsule.close(); borrowed.close()
            else:
                borrowed.close(); capsule.close()
        lifetimes.append({"order":order,"deleter_calls":capsule.deleter_calls})
    return {"schema_version":1,"round":115,"provenance":"synthetic",
            "independently_reproduced":False,"physical_gpu_measurement":False,
            "source":{"repository":"dmlc/dlpack","pin":PIN,"release_edit_utc":"2026-11-04T15:42:52Z",
                      "license":"Apache-2.0","version":[1,4],
                      "recent_change":"minor-version macro 3 to 4 only; ownership and stream rules are not new"},
            "case":{"shape":[2,3],"element_strides":[4,1],"byte_offset":4,"element_bytes":4,
                    "owner_bytes":64,"addresses":descriptor.addresses(),"expected_values":expected,
                    "correct_values":actual,"wrong_byte_stride_values":wrong,
                    "wrong_value_count":sum(a!=b for a,b in zip(expected,wrong)),
                    "logical_payload_bytes":24,"addressed_span_bytes":28},
            "schedule_model":{"guarded":[list(s) for s in guarded],"unguarded":[list(s) for s in unguarded],
                              "guarded_count":len(guarded),"unguarded_count":len(unguarded),
                              "stale_without_edge":stale,
                              "stale_with_edge":sum(row.index("read")<row.index("write") for row in guarded),
                              "note":"Enumeration fraction is not a GPU race probability or measured latency"},
            "lifetimes":lifetimes,"default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_dlpack_latency_ms":None,"gpu_synchronization_latency_ms":None,
            "peak_memory_bytes":None,"model_quality":None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    timing_path = HERE/"latency.json"
    if args.measure:
        timing_path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report = build_report(json.loads(timing_path.read_text(encoding="utf-8")))
    from .render import outputs
    artifacts = {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}
    for path,text in artifacts.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 115 artifacts match." if args.check else "Wrote Round 115 artifacts.")


if __name__=="__main__":
    main()
