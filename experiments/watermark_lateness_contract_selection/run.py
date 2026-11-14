"""Original discrete watermark/alignment/window-boundary model,not Flink execution."""

import argparse
import heapq
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="cac69274d4e7a6ab4b47454bd1cdc42905f4daca"
DEFAULT={"partitions":64,"active_keys":10000,"retained_windows_per_key":3,"count_bytes":8}


def integer(value,minimum=None):
    if type(value) is not int or abs(value)>2**53-1 or (minimum is not None and value<minimum):
        raise ValueError("bounded exact integer required")
    return value


def bounded_watermark(max_timestamp,bound):
    integer(bound,0)
    if max_timestamp is None:
        return None
    result=integer(max_timestamp)-bound-1
    return integer(result)


class Valve:
    def __init__(self,n,flush_all_idle=True):
        integer(n,1)
        if n>1024:
            raise ValueError("original model permits at most1024partitions")
        self.local=[None]*n
        self.active=[True]*n
        self.aligned=[True]*n
        self.output=None
        self.flush_all_idle=flush_all_idle

    def _index(self,p):
        integer(p,0)
        if p>=len(self.local):
            raise ValueError("unknown partition")

    def _advance(self,value):
        if value is not None and (self.output is None or value>self.output):
            self.output=value
        return self.output

    def _minimum(self):
        values=[self.local[p] for p in range(len(self.local)) if self.active[p] and self.aligned[p]]
        if not values or any(v is None for v in values):
            return self.output
        return self._advance(min(values))

    def watermark(self,p,value):
        self._index(p);integer(value)
        if not self.active[p]:
            return self.output
        old=self.local[p]
        if old is not None and value<=old:
            return self.output
        self.local[p]=value
        if not self.aligned[p] and (self.output is None or value>=self.output):
            self.aligned[p]=True
        return self._minimum()

    def idle(self,p):
        self._index(p)
        if not self.active[p]:
            return self.output
        self.active[p]=False;self.aligned[p]=False
        if not any(self.active):
            known=[v for v in self.local if v is not None]
            return self._advance(max(known)) if self.flush_all_idle and known else self.output
        return self._minimum()

    def resume(self,p):
        self._index(p)
        if self.active[p]:
            return self.output
        self.active[p]=True
        self.aligned[p]=self.output is None or (self.local[p] is not None and self.local[p]>=self.output)
        return self._minimum()


class Windows:
    def __init__(self,size,allowed_lateness):
        self.size=integer(size,1);self.allowed=integer(allowed_lateness,0)
        self.watermark=None;self.counts={};self.fired=set()

    def times(self,timestamp):
        timestamp=integer(timestamp)
        start=(timestamp//self.size)*self.size
        end=start+self.size
        return start,integer(end-1),integer(end-1+self.allowed)

    def event(self,timestamp):
        start,fire,cleanup=self.times(timestamp)
        if self.watermark is not None and cleanup<=self.watermark:
            return {"status":"rejected_window_cleaned","window_start":start,"cleanup":cleanup}
        self.counts[start]=self.counts.get(start,0)+1
        return {"status":"accepted_late_update" if self.watermark is not None and fire<=self.watermark else "accepted_before_fire",
                "window_start":start,"count":self.counts[start],"cleanup":cleanup}

    def advance(self,watermark):
        integer(watermark)
        if self.watermark is not None and watermark<self.watermark:
            raise ValueError("original window watermark must not regress")
        self.watermark=watermark;transitions=[]
        for start,count in sorted(self.counts.items()):
            _,fire,cleanup=self.times(start)
            if fire<=watermark and start not in self.fired:
                self.fired.add(start);transitions.append({"kind":"fire","window_start":start,"count":count})
            if cleanup<=watermark:
                del self.counts[start];self.fired.discard(start)
                transitions.append({"kind":"cleanup","window_start":start})
        return transitions


def budget(p):
    if set(p)!=set(DEFAULT) or any(type(v) is not int or v<1 for v in p.values()) or p["count_bytes"] not in (4,8):
        raise ValueError("positive exact dimensions and selected4/8byte count required")
    result={"selected_partition_watermark_int64_bytes":p["partitions"]*8,
            "selected_window_count_payload_bytes":p["active_keys"]*p["retained_windows_per_key"]*p["count_bytes"]}
    if max(*p.values(),*result.values())>2**53-1:
        raise ValueError("exact browser budget range exceeded")
    return result


def minimum_trace(n,use_heap):
    values=list(range(n));heap=[(v,p) for p,v in enumerate(values)] if use_heap else []
    if use_heap:
        heapq.heapify(heap)
    out=[]
    for i in range(128):
        p=(i*17)%n;values[p]+=n+1
        if use_heap:
            heapq.heappush(heap,(values[p],p))
            while heap[0][0]!=values[heap[0][1]]:
                heapq.heappop(heap)
            out.append(heap[0][0])
        else:
            out.append(min(values))
    return out


def measure():
    rows=[]
    for n in (16,64,256,1024):
        functions=(lambda:minimum_trace(n,True),lambda:minimum_trace(n,False))
        assert functions[0]()==functions[1]()
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
        rows.append({"partitions":n,"updates":128,"heap_samples_ms":samples[0],"scan_samples_ms":samples[1],
                     "heap_median_ms":statistics.median(samples[0]),"scan_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original Python active-only minimum aggregation,128monotonic updates;lazy stale-entry heap versus full list min;initialization included",
            "excludes":"native Flink,network,source polling,idleness,timers,checkpointing,window backend","rows":rows}


def build_report(latency):
    valve=Valve(2);trace=[]
    trace.append({"step":"Amax29,Bbound2;emitA26","watermark":valve.watermark(0,bounded_watermark(29,2))})
    trace.append({"step":"Bmax9;emitB6","watermark":valve.watermark(1,bounded_watermark(9,2))})
    trace.append({"step":"Bmarkedidle","watermark":valve.idle(1)})
    trace.append({"step":"Bresumeold6unaligned","watermark":valve.resume(1)})
    trace.append({"step":"Aemit36;Bstillunaligned","watermark":valve.watermark(0,36)})
    trace.append({"step":"Bemit35below36","watermark":valve.watermark(1,35)})
    trace.append({"step":"Bemit40realigns","watermark":valve.watermark(1,40)})
    def all_idle(order,flush):
        v=Valve(2,flush);v.watermark(0,10);v.watermark(1,20)
        return [v.idle(p) for p in order]
    window=Windows(10,5);initial=[window.event(3),window.event(8)]
    fire=window.advance(9);late=window.event(8);window.advance(13);before=window.event(8)
    cleanup=window.advance(14);after=window.event(8)
    return {"schema_version":1,"round":129,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"apache/flink","pin":PIN,"tree_date_utc":"2026-11-13T18:22:59Z","license":"Apache-2.0",
                      "latest_tree_change":"savepoint format reporting,not watermark innovation",
                      "chronology":"bounded watermark code established;valve all-idle flush/realignment verified in pinned source;change-date lookup unavailable after TLS timeouts"},
            "case":{"source_bound":2,"source_maxA":29,"source_maxB":9,"partition_watermarks":[26,6],"trace":trace,
                    "all_idle_orders":[[0,1],[1,0]],"all_idle_pinned_flush":[all_idle([0,1],True),all_idle([1,0],True)],
                    "wrong_no_flush":[all_idle([0,1],False),all_idle([1,0],False)],
                    "window":{"interval":[0,10],"size":10,"allowed_lateness":5,"fire_timestamp":9,"cleanup_timestamp":14,
                              "initial_events":initial,"fire_transitions":fire,"after_first_fire_event":late,
                              "wm13_event":before,"cleanup_transitions":cleanup,"wm14_event":after},
                    "unknown_watermark":None,"bound_is_assumption_not_observed_guarantee":True},
            "default_budget":{"inputs":DEFAULT,"outputs":budget(DEFAULT)},"latency":latency,
            "native_flink_latency_ms":None,"exactly_once_reproduced":False,"model_quality":None,"peak_state_backend_bytes":None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true")
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
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    print("Round129 sealed artifacts match" if args.check else "Wrote Round129 original artifacts")


if __name__=="__main__":
    main()
