"""Original stable float-key and byte-pass models; no native CUB execution."""

import argparse
import json
import math
from pathlib import Path
import platform
import random
import statistics
import struct
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "0dd2af496098ca890304ed4b8b3bc13e6b88851c"
MASK = 0xffffffff
DEFAULT = {"items":1048576,"value_bytes":4,"begin_bit":0,"end_bit":32}
FIXTURE = [0,0x80000000,0xc0000000,0x3fc00000,0xbf800000,
           0x7f800000,0xff800000,0x3fc00000,1,0x80000001]


def validate_range(begin,end):
    if type(begin) is not int or type(end) is not int or not 0<=begin<end<=32:
        raise ValueError("original selected range requires 0<=begin_bit<end_bit<=32")


def float_value(raw):
    if type(raw) is not int or not 0<=raw<=MASK:
        raise ValueError("require an exact uint32 bit pattern")
    if raw&0x7f800000==0x7f800000 and raw&0x7fffff:
        raise ValueError("original numeric-order model rejects NaN; native bitwise NaN handling is not reproduced")
    return struct.unpack("!f",struct.pack("!I",raw))[0]


def sortable_key(raw,descending=False,canonical_zero=True):
    float_value(raw)
    if type(descending) is not bool or type(canonical_zero) is not bool:
        raise ValueError("explicit Boolean sort direction and zero policy required")
    if canonical_zero and raw&0x7fffffff==0:
        raw=0
    result=(~raw&MASK) if raw&0x80000000 else raw^0x80000000
    return result^MASK if descending else result


def selected_keys(raws,begin=0,end=32,descending=False):
    validate_range(begin,end)
    return [(sortable_key(raw,descending)>>begin)&((1<<(end-begin))-1) for raw in raws]


def stable_lsd(keys,bits):
    if type(bits) is not int or not 1<=bits<=32:
        raise ValueError("original key width requires 1..32 bits")
    if any(type(key) is not int or not 0<=key<1<<bits for key in keys):
        raise ValueError("unsigned key must fit selected key width")
    order=list(range(len(keys)))
    for shift in range(0,bits,8):
        width=min(8,bits-shift)
        buckets=[[] for _ in range(1<<width)]
        for index in order:
            buckets[(keys[index]>>shift)&((1<<width)-1)].append(index)
        order=[index for bucket in buckets for index in bucket]
    return order


def radix_order(raws,begin=0,end=32,descending=False):
    return stable_lsd(selected_keys(raws,begin,end,descending),end-begin)


def numeric_oracle(raws,descending=False):
    values=[float_value(raw) for raw in raws]
    return sorted(range(len(values)),key=values.__getitem__,reverse=descending)


def estimate_budget(p):
    if type(p["items"]) is not int or not 1<=p["items"]<=2**53-1:
        raise ValueError("items requires a positive bounded exact integer")
    if type(p["value_bytes"]) is not int or p["value_bytes"] not in (0,4,8):
        raise ValueError("selected value width must be zero, four or eight bytes")
    validate_range(p["begin_bit"],p["end_bit"])
    result={"one_pair_buffer_bytes":p["items"]*(4+p["value_bytes"]),
            "two_pair_buffers_bytes":2*p["items"]*(4+p["value_bytes"]),
            "original_byte_passes":math.ceil((p["end_bit"]-p["begin_bit"])/8)}
    if any(value>2**53-1 for value in result.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return result


def measure():
    rows=[]
    generator=random.Random(118)
    for count in (64,256,1024,4096):
        keys=[generator.getrandbits(32) for _ in range(count)]
        functions=(lambda:stable_lsd(keys,32),
                   lambda:sorted(range(len(keys)),key=keys.__getitem__))
        assert functions[0]()==functions[1]()
        for _ in range(2):
            for function in functions:
                function()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(2):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/2/1e6)
        rows.append({"items":count,"byte_pass_samples_ms":samples[0],
                     "builtin_samples_ms":samples[1],
                     "byte_pass_median_ms":statistics.median(samples[0]),
                     "builtin_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":2,
            "method":"original Python stable byte buckets versus C-backed Python sorted, precomputed uint32 keys; includes implementation differences",
            "excludes":"float conversion, key generation, device transfer and native CUB/GPU execution",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    expected=numeric_oracle(FIXTURE)
    actual=radix_order(FIXTURE)
    unsigned=sorted(range(len(FIXTURE)),key=FIXTURE.__getitem__)
    unfolded=sorted(range(len(FIXTURE)),key=lambda i:sortable_key(FIXTURE[i],canonical_zero=False))
    partial=radix_order(FIXTURE,0,8)
    raw_low=sorted(range(len(FIXTURE)),key=lambda i:FIXTURE[i]&255)
    agree=lambda a,b:sum(x==y for x,y in zip(a,b))
    reverse_fixture=list(reversed(FIXTURE))
    return {"schema_version":1,"round":118,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"NVIDIA/cccl","pin":PIN,"date_utc":"2026-11-10T14:52:49Z",
                      "component_license":"header SPDX BSD-3; upstream rights retained",
                      "recent_change":"ADL-suppression qualification maintenance; existing stable numeric-key rules are not new"},
            "case":{"raw_hex":[f"{raw:08x}" for raw in FIXTURE],"items":len(FIXTURE),
                    "numeric_reference_ids":expected,"stable_radix_ids":actual,
                    "numeric_agreements":agree(expected,actual),
                    "unsigned_control_ids":unsigned,"unsigned_agreements":agree(expected,unsigned),
                    "uncanonical_zero_ids":unfolded,"uncanonical_zero_agreements":agree(expected,unfolded),
                    "zero_input_ids":[0,1],"zero_correct_ids":[i for i in actual if i in (0,1)],
                    "zero_wrong_ids":[i for i in unfolded if i in (0,1)],
                    "original_keys_hex":[f"{sortable_key(raw):08x}" for raw in FIXTURE]},
            "partial":{"range":[0,8],"transformed_order_ids":partial,"raw_low_order_ids":raw_low,
                       "raw_low_agreements":agree(partial,raw_low),
                       "note":"selection after type transform, not an original raw-mantissa ordering"},
            "robustness":{"reversed_input_numeric_agreements":agree(radix_order(reverse_fixture),numeric_oracle(reverse_fixture)),
                          "descending_numeric_agreements":agree(radix_order(FIXTURE,descending=True),numeric_oracle(FIXTURE,True)),
                          "wrong_reverse_ascending_agreements":agree(list(reversed(actual)),numeric_oracle(FIXTURE,True)),
                          "items":len(FIXTURE)},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_cub_latency_ms":None,"native_temporary_storage_bytes":None,
            "native_output_buffer_selector":None,"peak_hbm_bytes":None,"nan_order_compatibility":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
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
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 118 artifacts match." if args.check else "Wrote Round 118 artifacts.")


if __name__=="__main__":
    main()
