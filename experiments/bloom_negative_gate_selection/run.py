"""Original explicit-position Bloom negative-gate model and finite universe."""

import argparse
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="393f33d3fa9da26dd8be36f41cbca91655bdf8e9"
DEFAULT={"expected_items":1000000,"target_error_ppm":10000,"layers":3,"growth":2}


def validate_positions(positions,size):
    if type(size) is not int or size<1 or not positions or any(type(i) is not int or not 0<=i<size for i in positions):
        raise ValueError("nonempty bounded bit coordinates required")


def make_bits(inserted,size):
    bits=[False]*size
    for coordinates in inserted:
        validate_positions(coordinates,size)
        for i in coordinates:
            bits[i]=True
    return bits


def contains(bits,coordinates):
    validate_positions(coordinates,len(bits))
    return all(bits[i] for i in coordinates)


def independent_mask_oracle(inserted,coordinates):
    required=set(coordinates);occupied=set().union(*(set(row) for row in inserted))
    return required.issubset(occupied)


def counting(inserted,size,remove=None):
    counts=[0]*size
    for row in inserted:
        validate_positions(row,size)
        for i in row:
            counts[i]+=1
    if remove is not None:
        validate_positions(remove,size)
        if any(counts[i]<remove.count(i) for i in set(remove)):
            raise ValueError("counter underflow;trusted insertion provenance required")
        for i in remove:
            counts[i]-=1
    return counts


def layer_bounds(error_ppm,layers):
    if type(error_ppm) is not int or not 0<error_ppm<1000000 or type(layers) is not int or not 1<=layers<=16:
        raise ValueError("positive error below1 and1..16 layers required")
    p=Fraction(error_ppm,1000000)
    limits=[p/Fraction(2)**(i+1) for i in range(layers)]
    return {"selected_layer_error_limits":[str(v) for v in limits],
            "finite_union_bound":str(sum(limits,Fraction())),
            "constant_layer_union_bound":str(min(Fraction(1),p*layers)),
            "interpretation":"union bound requires per-layer bounds,not independence;not observed native FPR"}


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 or v>2**53-1 for v in p.values()) or p["expected_items"]>10**12 or p["growth"]>16:
        raise ValueError("bounded exact sizing inputs required")
    layer_bounds(p["target_error_ppm"],p["layers"])
    rows=[];capacity=p["expected_items"]
    for i in range(p["layers"]):
        if capacity>10**12:
            raise ValueError("selected layer capacity exceeds model limit")
        rate=p["target_error_ppm"]/1e6/2**(i+1)
        ideal_bits=math.ceil(-capacity*math.log(rate)/math.log(2)**2)
        rows.append({"layer":i+1,"selected_capacity":capacity,"ideal_approximation_bits":ideal_bits,
                     "selected_byte_rounded_payload":(ideal_bits+7)//8,
                     "selected_hash_count":max(1,math.ceil(math.log(2)*ideal_bits/capacity))})
        capacity*=p["growth"]
    return {"layers":rows,"selected_total_byte_rounded_payload":sum(v["selected_byte_rounded_payload"] for v in rows),
            "note":"ideal-hash sizing approximation,not native power-of-two/64-bit rounding,metadata or measured error"}


def timing_paths(n):
    size=4096;coordinates=[((i*37+3)%size,(i*71+5)%size) for i in range(n)]
    occupied=[i%3!=0 for i in range(size)];packed=bytearray(size//8)
    for i,value in enumerate(occupied):
        if value:
            packed[i>>3]|=1<<(i&7)
    def bitset():
        return sum(all(packed[i>>3]&(1<<(i&7)) for i in row) for row in coordinates)
    def booleans():
        return sum(all(occupied[i] for i in row) for row in coordinates)
    return bitset,booleans


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        functions=timing_paths(n);assert functions[0]()==functions[1]()
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
        rows.append({"queries":n,"bitset_samples_ms":samples[0],"bool_samples_ms":samples[1],
                     "bitset_median_ms":statistics.median(samples[0]),"bool_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original precomputed-coordinate two-probe membership,packed bytearray versus Python bool list",
            "excludes":"hashing,construction,layers,Redis/native commands,network,storage/database lookup,allocator footprint","rows":rows}


def build_report(latency):
    inserted=[(1,5),(5,9)];bits=make_bits(inserted,16)
    absent=[p for p in product(range(16),repeat=2) if p not in inserted]
    fp=[list(p) for p in absent if contains(bits,p)]
    cleared=list(bits)
    for i in inserted[0]:
        cleared[i]=False
    goodcounts=counting(inserted,16,list(inserted[0]))
    badcounts=counting(inserted,16,[1,9])
    return {"schema_version":1,"round":134,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"RedisBloom/RedisBloom","pin":PIN,"license":"sb.c RSALv2/SSPLv1/AGPLv3 choice;deps/bloom BSD header",
                      "chronology":"Nov12treechange bounds installer retries,not new Bloom algorithm",
                      "verified":"chain checks all links before insert;scalable firsterrorp/2 then half per link;capacity growth;non-scaling explicit full error",
                      "model_hash":"none;explicit coordinate-pair universe,not native Murmur hash/error reproduction"},
            "case":{"bit_count":16,"probe_count":2,"inserted_position_pairs":[list(p) for p in inserted],
                    "occupied_positions":[i for i,v in enumerate(bits) if v],"finite_absent_pair_count":len(absent),
                    "finite_false_positive_pairs":fp,"finite_false_positive_count":len(fp),"finite_true_negative_count":len(absent)-len(fp),
                    "finite_false_negative_count":sum(not contains(bits,p) for p in inserted),
                    "false_positive_witness":[1,9],"naive_clear_A_preserves_B":contains(cleared,inserted[1]),
                    "trusted_counting_delete_A_preserves_B":all(goodcounts[i]>0 for i in inserted[1]),
                    "untrusted_false_positive_counting_delete_preserves_A":all(badcounts[i]>0 for i in inserted[0]),
                    "untrusted_false_positive_counting_delete_preserves_B":all(badcounts[i]>0 for i in inserted[1]),
                    "gate":"negative skips only for a complete,current,consistent filter;positive always requires authoritative verification",
                    "native_add_zero":"may be false-positive already-present indication,not exact unique-item identity proof"},
            "selected_layer_bounds":layer_bounds(10000,3),"default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},
            "latency":latency,"native_redis_latency_ms":None,"production_false_positive_rate":None,"native_memory_bytes":None,"database_saved_latency_ms":None}


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
    print("Round134 sealed artifacts match" if a.check else "Wrote Round134 original artifacts")


if __name__=="__main__":
    main()
