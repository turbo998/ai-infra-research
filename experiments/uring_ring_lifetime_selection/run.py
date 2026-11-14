"""Original single-producer ring-credit and ordinary one-shot read lifetime model."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="a10946dc51eeababc48ee10baf0d17489f647a80"
U32=2**32
DEFAULT={"sq_entries":8,"cq_entries":16,"buffer_count":32,"buffer_bytes":1048576,"sqe_bytes":64,"cqe_bytes":16}


def depth(tail,head):
    if any(type(v) is not int or not 0<=v<U32 for v in (tail,head)):
        raise ValueError("exact unsigned32bit counters required")
    return (tail-head)%U32


def capacity(value):
    if type(value) is not int or not 1<=value<=1024 or value&(value-1):
        raise ValueError("original ring entries must be power-of-two1..1024")
    return value


class Ring:
    def __init__(self,entries=2,cq_entries=4,buffers=3,start=0):
        capacity(entries);capacity(cq_entries);depth(start,start)
        if type(buffers) is not int or not 1<=buffers<=65536:
            raise ValueError("bounded positive buffer count required")
        self.entries=entries;self.cq_entries=cq_entries;self.buffers=buffers
        self.head=self.tail=self.cq_head=self.cq_tail=start
        self.sq=[None]*entries;self.cq=[None]*cq_entries
        self.pending={};self.owned={};self.used=set()

    def submit(self,request,buffer,requested_bytes=4):
        if type(request) is not int or not 0<=request<=2**53-1 or request in self.used:
            raise ValueError("original nonreused exact request ID required")
        if type(buffer) is not int or not 0<=buffer<self.buffers or buffer in self.owned:
            raise ValueError("buffer outside pool or still owned by an ordinary read")
        if type(requested_bytes) is not int or not 1<=requested_bytes<=2**30:
            raise ValueError("bounded positive requested read size required")
        if depth(self.tail,self.head)>=self.entries:
            raise ValueError("original SQ is full;submission credit unavailable")
        slot=self.tail&(self.entries-1)
        self.sq[slot]=[request,buffer];self.tail=(self.tail+1)%U32
        self.pending[request]={"buffer":buffer,"state":"queued","requested_bytes":requested_bytes}
        self.owned[buffer]=request;self.used.add(request)
        return slot

    def consume(self):
        if depth(self.tail,self.head)==0:
            raise ValueError("original SQ is empty")
        slot=self.head&(self.entries-1);request,buffer=self.sq[slot]
        self.sq[slot]=None;self.head=(self.head+1)%U32
        self.pending[request]["state"]="inflight"
        return request

    def complete(self,request,result):
        if type(request) is not int or request not in self.pending or self.pending[request]["state"]!="inflight":
            raise ValueError("unknown,unconsumed or already-completed original request")
        if type(result) is not int or not -4095<=result<=self.pending[request]["requested_bytes"]:
            raise ValueError("original ordinary-read result outside bounded contract")
        if depth(self.cq_tail,self.cq_head)>=self.cq_entries:
            raise ValueError("original CQ is full;native overflow/backlog not modeled")
        self.cq[self.cq_tail&(self.cq_entries-1)]=[request,result]
        self.cq_tail=(self.cq_tail+1)%U32;self.pending[request]["state"]="completion-queued"

    def reap(self):
        if depth(self.cq_tail,self.cq_head)==0:
            raise ValueError("original CQ is empty")
        slot=self.cq_head&(self.cq_entries-1);request,result=self.cq[slot]
        self.cq[slot]=None;self.cq_head=(self.cq_head+1)%U32
        item=self.pending.pop(request);buffer=item["buffer"];del self.owned[buffer]
        status="error" if result<0 else ("complete" if result==item["requested_bytes"] else "short_or_eof")
        return {"request_id":request,"buffer":buffer,"result":result,"requested_bytes":item["requested_bytes"],"status":status}

    def snapshot(self):
        return {"sq_head":self.head,"sq_tail":self.tail,"sq_depth":depth(self.tail,self.head),"sq_free":self.entries-depth(self.tail,self.head),
                "cq_head":self.cq_head,"cq_tail":self.cq_tail,"cq_depth":depth(self.cq_tail,self.cq_head),
                "sq_slots":list(self.sq),"pending":[{"request_id":k,**v} for k,v in sorted(self.pending.items())],
                "buffer_owners":[{"buffer":k,"request_id":v} for k,v in sorted(self.owned.items())]}


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()):
        raise ValueError("positive exact selected ring and buffer payload parameters required")
    capacity(p["sq_entries"]);capacity(p["cq_entries"])
    if p["buffer_count"]>65536 or p["buffer_bytes"]>2**30 or p["sqe_bytes"]>256 or p["cqe_bytes"]>256:
        raise ValueError("selected payload parameters exceed bounded contract")
    sq=p["sq_entries"]*p["sqe_bytes"];cq=p["cq_entries"]*p["cqe_bytes"]
    return {"selected_sqe_payload_bytes":sq,"selected_cqe_payload_bytes":cq,"selected_ring_entry_payload_bytes":sq+cq,
            "selected_buffer_pool_payload_bytes":p["buffer_count"]*p["buffer_bytes"],
            "note":"selected entry/pool payload only;no ring indexes,SQ indirection,mmap pages,kernel state,pinning,allocator or native peak memory"}


def timing_paths(n):
    items=list(range(n))
    def cursor():
        q=list(items);head=0;total=0
        while head<len(q):
            total+=q[head];head+=1
        return total
    def shifting():
        q=list(items);total=0
        while q:
            total+=q.pop(0)
        return total
    return cursor,shifting


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
        rows.append({"entries":n,"cursor_samples_ms":samples[0],"shifting_samples_ms":samples[1],
                     "cursor_median_ms":statistics.median(samples[0]),"shifting_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,"method":"original Python copied-list consumption by head cursor versus pop(0);both include list copy",
            "excludes":"ring-class bookkeeping,liburing,Linux/syscalls,atomic ordering,storage,buffer transfer,SQPOLL,IOPS",
            "baseline":"language-level list movement only,not an io_uring versus read/epoll benchmark","rows":rows}


def build_report(latency):
    ring=Ring();events=[]
    def snap(label):
        events.append({"event":label,**ring.snapshot()})
    ring.submit(100,0);ring.submit(101,1);snap("two queued requests fill SQ")
    ring.consume();snap("kernel consumed100;SQ credit available but buffer0still owned")
    reused=ring.submit(102,2);snap("slot0reused by102;request100still ownsbuffer0")
    naive=ring.sq[reused][0]
    ring.consume();ring.consume();snap("SQ empty;three ordinary read buffers still owned")
    for request,result in ((101,-5),(100,4),(102,4)):
        ring.complete(request,result)
    snap("three CQEs posted out of submission order")
    completions=[ring.reap(),ring.reap(),ring.reap()];snap("completion IDs reap exact owners")
    wrap=Ring(entries=4,cq_entries=4,buffers=4,start=U32-2)
    for i in range(4):
        wrap.submit(i,i)
    return {"schema_version":1,"round":140,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"axboe/liburing","pin":PIN,"license":"read manpages SPDX LGPL-2.0-or-later;repository LICENSE MIT and COPYING LGPL-2.1 inspected,not a blanket label",
                      "verified":"ring mask/releasepublication,out-of-order completion/user_data,ordinary read/write buffer valid until completion,submission metadata distinct,SQPOLL caveats",
                      "chronology":"Nov12SQPOLLtestsetupoutput-pointerfix,notnewringlifetimealgorithm",
                      "model_policy":"singleproducer,poweroftwohardCQlimit,ordinaryoneshotread,noSQPOLL/multishot/skip-success/zero-copy-send-notification;original IDnonreuse policy"},
            "case":{"events":events,"reused_sq_slot":reused,"wrong_slot_current_request":naive,"actual_completing_request":100,
                    "actual_request100_buffer":0,"wrong_slot102_buffer":2,"completions":completions,
                    "wraparound_full_snapshot":wrap.snapshot(),"wrong_plain_tail_minus_head":wrap.tail-wrap.head,
                    "wraparound_correct_unsigned_depth":depth(wrap.tail,wrap.head),
                    "publication_witness":{"correct":["fill payload","release publish tail","acquire observe tail","read payload"],
                                           "wrong":["publish tail before fill","observe tail","read unfilled payload","fill payload"],
                                           "scope":"original happens-before witness,not hardware race probability or native atomic simulation"}},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_io_uring_latency_ms":None,"native_iops":None,"native_memory_bytes":None,"race_probability":None}


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
    print("Round140 sealed artifacts match" if a.check else "Wrote Round140 original artifacts")


if __name__=="__main__":
    main()
