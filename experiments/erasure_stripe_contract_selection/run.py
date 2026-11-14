"""Original GF(256) systematic Vandermonde erasure code;not Ceph byte parity."""

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="ec322a5e95ebcb46dec7d6a9d7813a34543c64ad"
DEFAULT={"object_bytes":1048577,"data_shards":4,"parity_shards":2,"stripe_unit_bytes":16384}


def mul(a,b):
    result=0
    for _ in range(8):
        if b&1:
            result^=a
        b>>=1;a<<=1
        if a&256:
            a^=0x11B
    return result


def power(a,n):
    result=1
    while n:
        if n&1:
            result=mul(result,a)
        a=mul(a,a);n>>=1
    return result


def inverse(a):
    if not a:
        raise ValueError("zero has no finite-field inverse")
    return power(a,254)


def invert(matrix):
    n=len(matrix)
    if n==0 or any(len(row)!=n for row in matrix):
        raise ValueError("nonempty square field matrix required")
    rows=[list(row)+[int(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for col in range(n):
        pivot=next((r for r in range(col,n) if rows[r][col]),None)
        if pivot is None:
            raise ValueError("singular survivor matrix")
        rows[col],rows[pivot]=rows[pivot],rows[col]
        factor=inverse(rows[col][col]);rows[col]=[mul(v,factor) for v in rows[col]]
        for r in range(n):
            if r==col:
                continue
            factor=rows[r][col]
            rows[r]=[a^mul(factor,b) for a,b in zip(rows[r],rows[col])]
    return [row[n:] for row in rows]


def profile(k,m):
    if type(k) is not int or type(m) is not int or k<1 or m<1 or k+m>16:
        raise ValueError("original bounded profile requires k,m>=1 and k+m<=16")


def generator(k,m):
    profile(k,m)
    v=[[power(x,j) for j in range(k)] for x in range(k+m)]
    base=invert(v[:k])
    return [[xor(mul(row[t],base[t][col]) for t in range(k)) for col in range(k)] for row in v]


def xor(values):
    result=0
    for value in values:
        result^=value
    return result


def validate_bytes(data):
    if not isinstance(data,(bytes,bytearray,list)) or any(type(v) is not int or not 0<=v<=255 for v in data):
        raise ValueError("exact bytes0..255 required")


def encode(data,k,m):
    profile(k,m);validate_bytes(data)
    width=(len(data)+k-1)//k
    padded=list(data)+[0]*(width*k-len(data))
    chunks=[padded[i*width:(i+1)*width] for i in range(k)]
    g=generator(k,m)
    return [[xor(mul(row[i],chunks[i][j]) for i in range(k)) for j in range(width)] for row in g]


def recover(shards,original_length,k,m):
    profile(k,m)
    if type(original_length) is not int or original_length<0 or len(shards)<k:
        raise ValueError("at least k identified survivors and original length required")
    width=(original_length+k-1)//k
    for index,data in shards.items():
        if type(index) is not int or not 0<=index<k+m or len(data)!=width:
            raise ValueError("survivor identity or chunk width mismatch")
        validate_bytes(data)
    chosen=sorted(shards)[:k];g=generator(k,m)
    decode=invert([g[i] for i in chosen])
    data=[xor(mul(row[i],shards[chosen[i]][j]) for i in range(k)) for row in decode for j in range(width)]
    return bytes(data[:original_length])


def lagrange_value(points,values,target):
    out=0
    for i,x in enumerate(points):
        numerator=1;denominator=1
        for j,y in enumerate(points):
            if j!=i:
                numerator=mul(numerator,target^y);denominator=mul(denominator,x^y)
        out^=mul(values[i],mul(numerator,inverse(denominator)))
    return out


def independent_lagrange_encode(data,k,m):
    profile(k,m);validate_bytes(data)
    width=(len(data)+k-1)//k;padded=list(data)+[0]*(width*k-len(data))
    return [[lagrange_value(list(range(k)),[padded[i*width+j] for i in range(k)],target) for j in range(width)] for target in range(k+m)]


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()):
        raise ValueError("positive exact payload/profile/stripe parameters required")
    b,k,m,u=(p[key] for key in ("object_bytes","data_shards","parity_shards","stripe_unit_bytes"))
    profile(k,m)
    stripes=(b+k*u-1)//(k*u);chunk=stripes*u
    out={"selected_classic_stripes":stripes,"selected_chunk_payload_bytes":chunk,
         "selected_classic_encoded_bytes":chunk*(k+m),"selected_data_padding_bytes":chunk*k-b,
         "selected_minimum_k_chunk_recovery_read_bytes":chunk*k,
         "selected_three_replica_payload_bytes":b*3}
    if max(*p.values(),*out.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return out


def lost_in_domains(domains,failed):
    return [i for i,domain in enumerate(domains) if domain in failed]


def timing_paths(width):
    k,m=3,2;g=generator(k,m)[k:];data=[[(i*37+j)%256 for j in range(width)] for i in range(k)]
    tables={c:[mul(c,b) for b in range(256)] for row in g for c in row}
    def lookup():
        return [[xor(tables[row[i]][data[i][j]] for i in range(k)) for j in range(width)] for row in g]
    def bitwise():
        return [[xor(mul(row[i],data[i][j]) for i in range(k)) for j in range(width)] for row in g]
    return lookup,bitwise


def measure():
    rows=[]
    for width in (64,256,1024,4096):
        funcs=timing_paths(width);assert funcs[0]()==funcs[1]()
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
        rows.append({"chunk_bytes":width,"k":3,"m":2,"lookup_samples_ms":samples[0],"bitwise_samples_ms":samples[1],
                     "lookup_median_ms":statistics.median(samples[0]),"bitwise_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original parity-only GF256 coefficient lookup versus bitwise multiply",
            "excludes":"coefficient-table/generator setup,Ceph/ISA-L/SIMD,network,disk,recovery,FastEC,checksums","rows":rows}


def build_report(latency):
    data=bytes(range(1,11));k,m=3,2;shards=encode(data,k,m);records=[]
    for missing in combinations(range(k+m),m):
        available={i:s for i,s in enumerate(shards) if i not in missing}
        records.append({"missing":list(missing),"matches":recover(available,len(data),k,m)==data})
    tampered={i:list(shards[i]) for i in (0,1,2)};tampered[0][0]^=1
    corrupted=recover(tampered,len(data),k,m)
    good=[f"rack{i}" for i in range(5)];bad=["rackA","rackA","rackA","rackB","rackC"]
    return {"schema_version":1,"round":133,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"ceph/ceph","pin":PIN,"documentation_license":"CC-BY-SA-3.0 per COPYING doc/*",
                      "chronology":"Nov13treeupdate drops completed push buffers,not new erasure-code theory",
                      "verified":"k/m storage amplification,failure domains,classic stripe unit,and TentacleFastEC version/plugin constraints",
                      "model_codec":"original systematic Vandermonde over GF256 polynomial0x11b,not native ISA-L/Ceph byte parity"},
            "case":{"k":k,"m":m,"original_bytes":list(data),"original_length":len(data),"generator":generator(k,m),
                    "encoded_chunks":shards,"oracle_encoded_chunks":independent_lagrange_encode(data,k,m),
                    "chunk_bytes":len(shards[0]),"padding_bytes":len(shards[0])*k-len(data),
                    "two_erasure_cases":records,"recovered_cases":sum(r["matches"] for r in records),"total_cases":len(records),
                    "good_domains":good,"bad_domains":bad,"bad_single_domain_missing":lost_in_domains(bad,["rackA"]),
                    "bad_domain_survivors":5-len(lost_in_domains(bad,["rackA"])),
                    "two_good_domains_missing":lost_in_domains(good,["rack0","rack1"]),
                    "corruption_control":{"available_ids":[0,1,2],"mutated_byte":"chunk0byte0xor1",
                                          "matches_expected":corrupted==data,"expected_sha256":hashlib.sha256(data).hexdigest(),
                                          "decoded_sha256":hashlib.sha256(corrupted).hexdigest(),
                                          "policy":"known erasure decoder does not locate/correct silent corruption;external expected checksum fixture only"}},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_ceph_latency_ms":None,"durability_probability":None,"native_rebuild_bandwidth_bytes":None,"native_storage_bytes":None}


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
    print("Round133 sealed artifacts match" if a.check else "Wrote Round133 original artifacts")


if __name__=="__main__":
    main()
