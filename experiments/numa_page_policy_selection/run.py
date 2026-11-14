"""Original ideal first-write placement and page-index weighted striping models."""

import argparse
from bisect import bisect_right
from collections import Counter
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="067c5c24cef23d913b67eaaf71ae67e7169a6620"
DEFAULT={"pages":12,"nodes":[0,1],"weights":[1,3],"initial_writer":0,
         "reader_nodes":[0,0,0]+[1]*9,"page_bytes":4096}


def integer(value,low,high):
    if type(value) is not int or not low<=value<=high:
        raise ValueError(f"exact integer in {low}..{high} required")
    return value


def policy(nodes,weights):
    if not isinstance(nodes,list) or not 1<=len(nodes)<=16:
        raise ValueError("one to sixteen explicit node IDs required")
    for node in nodes:
        integer(node,0,4095)
    if nodes!=sorted(set(nodes)):
        raise ValueError("node IDs must be unique and ascending")
    if not isinstance(weights,list) or len(weights)!=len(nodes):
        raise ValueError("one weight per node required")
    for weight in weights:
        integer(weight,1,255)
    return list(nodes),list(weights)


def weighted_node(index,nodes,weights):
    integer(index,0,2**32-1);nodes,weights=policy(nodes,weights)
    boundaries=[];total=0
    for weight in weights:
        total+=weight;boundaries.append(total)
    return nodes[bisect_right(boundaries,index%total)]


def effective_nodes(nodes,weights,allowed):
    nodes,weights=policy(nodes,weights)
    if not isinstance(allowed,list) or not allowed or any(type(n) is not int or not 0<=n<=4095 for n in allowed):
        raise ValueError("nonempty bounded allowed-node list required")
    pairs=[(n,w) for n,w in zip(nodes,weights) if n in allowed]
    if not pairs:
        raise ValueError("empty effective mask is outside the original model;native fallback not simulated")
    return [n for n,_ in pairs],[w for _,w in pairs]


class Pages:
    def __init__(self,mode,nodes,weights):
        self.owners={}
        self.set_policy(mode,nodes,weights)

    def set_policy(self,mode,nodes,weights):
        if mode not in ("local","interleave","weighted"):
            raise ValueError("original local/interleave/weighted policy required")
        nodes,weights=policy(nodes,weights)
        self.mode=mode;self.nodes=nodes;self.weights=weights

    def first_write(self,index,writer):
        integer(index,0,2**32-1);integer(writer,0,4095)
        if index not in self.owners:
            if self.mode=="local":
                self.owners[index]=writer
            elif self.mode=="interleave":
                self.owners[index]=self.nodes[index%len(self.nodes)]
            else:
                self.owners[index]=weighted_node(index,self.nodes,self.weights)
        return self.owners[index]


def summary(owners,readers,page_bytes):
    local=sum(a==b for a,b in zip(owners,readers))
    counts=Counter(owners)
    return {"owners":owners,"node_pages":{str(k):v for k,v in sorted(counts.items())},
            "node_payload_bytes":{str(k):v*page_bytes for k,v in sorted(counts.items())},
            "local_reads":local,"remote_reads":len(owners)-local,"read_events":len(owners),
            "remote_fraction":(len(owners)-local)/len(owners),"selected_payload_bytes":len(owners)*page_bytes}


def simulate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("exact original placement input fields required")
    pages=integer(p["pages"],1,4096);writer=integer(p["initial_writer"],0,4095)
    nodes,weights=policy(p["nodes"],p["weights"])
    size=integer(p["page_bytes"],512,2**21)
    if size&(size-1):
        raise ValueError("selected page payload size must be a power of two")
    readers=p["reader_nodes"]
    if not isinstance(readers,list) or len(readers)!=pages:
        raise ValueError("one explicit reader node per page required")
    for node in readers:
        integer(node,0,4095)
    result={}
    for name,mode in (("serial_first_write","local"),("parallel_owner_first_write","local"),
                      ("interleave","interleave"),("weighted","weighted")):
        memory=Pages(mode,nodes,weights)
        owners=[memory.first_write(i,readers[i] if name=="parallel_owner_first_write" else writer) for i in range(pages)]
        result[name]=summary(owners,readers,size)
    return result


def timing_paths(n):
    nodes=list(range(0,16,2));weights=[1,3,2,1,4,2,1,2]
    boundaries=[];total=0
    for w in weights:
        total+=w;boundaries.append(total)
    indexes=[(i*40503+17)%65536 for i in range(n)]
    def prefix():
        return [nodes[bisect_right(boundaries,i%total)] for i in indexes]
    def walk():
        out=[]
        for i in indexes:
            target=i%total
            for node,weight in zip(nodes,weights):
                if target<weight:
                    out.append(node);break
                target-=weight
        return out
    return prefix,walk


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
        rows.append({"lookups":n,"prefix_samples_ms":samples[0],"walk_samples_ms":samples[1],
                     "prefix_median_ms":statistics.median(samples[0]),"walk_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"nodes":8,
            "method":"original prebuilt cumulative boundaries plus bisect versus per-query weight walk;both output node-ID lists",
            "excludes":"validation/preprocessing,Pages ledger,Linux faults,NUMA bandwidth,remote-access latency,migration,syscalls","rows":rows}


def build_report(latency):
    outputs=simulate(DEFAULT)
    pages=Pages("local",[0,1],[1,1])
    for i in range(6):
        pages.first_write(i,0)
    old=dict(pages.owners);pages.set_policy("weighted",[0,1],[1,3])
    for i in reversed(range(12)):
        pages.first_write(i,1)
    changed=sum(pages.owners[i]!=node for i,node in old.items())
    nodes,weights=effective_nodes([2,5],[1,3],[5])
    reversed_memory=Pages("weighted",[0,1],[1,3])
    for i in reversed(range(12)):
        reversed_memory.first_write(i,0)
    reversed_owners=[reversed_memory.owners[i] for i in range(12)]
    return {"schema_version":1,"round":141,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"torvalds/linux","pin":PIN,"license":"mm/mempolicy.c SPDX GPL-2.0-only;upstream rights preserved;no source mirrored",
                      "verified":"weighted_interleave_nid page-index modulo and weight intervals;distinct dynamic per-task path;sysfs weights1..255 and auto/manual;new allocations only",
                      "chronology":"Nov10allocator/zonelist hotplug race fixes;weighted mode is not new this day;sysfs auto ABI dated May2025",
                      "scope":"original ideal preferred-node placement only;no capacity fallback,cpuset rebind,THP,shared page cache,auto weight derivation or migration"},
            "case":{"inputs":DEFAULT,"outputs":outputs,
                    "policy_change":{"old_pages":6,"old_pages_migrated":changed,"owners":[pages.owners[i] for i in range(12)],
                                     "note":"policy change affects only subsequent first writes in original model"},
                    "reversed_fault_order":{"owners":reversed_owners,"matches":sum(a==b for a,b in zip(outputs["weighted"]["owners"],reversed_owners)),"total":12},
                    "sparse_node_ids":{"nodes":[2,5],"weights":[1,3],"first_cycle":[weighted_node(i,[2,5],[1,3]) for i in range(4)],
                                       "effective_nodes":nodes,"effective_weights":weights,"effective_first_cycle":[weighted_node(i,nodes,weights) for i in range(4)]},
                    "limitations":"one explicit reader access per modeled page;equal-size synthetic pages;not bytes/sec,time or measured physical allocation"},
            "latency":latency,"native_memory_bandwidth_gbps":None,"native_remote_latency_ns":None,
            "native_page_placement_verified":None,"native_migration_ms":None}


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
    print("Round141 sealed artifacts match" if a.check else "Wrote Round141 original artifacts")


if __name__=="__main__":
    main()
