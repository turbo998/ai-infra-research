"""Signal-free original shutdown ledger; never spawns or kills a process."""
import argparse
import bisect
import json
import platform
import statistics
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PIN="ec94b6a7fc3b7141528113fbc89802829a914c25"
DEFAULT={"has_api":True,"reason":"signal","budget_ms":10000,"api_drain_ms":6000,
         "launcher_exit_after_ms":1000,"requests":[2000,6000],"stragglers":1,
         "pid_present":True,"already_started":False,"serve_error":False}


def integer(x,hi=600000):
    return type(x) is int and 0<=x<=hi


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):raise ValueError("Exactly the original shutdown profile fields required")
    if any(type(p[k]) is not bool for k in ("has_api","pid_present","already_started","serve_error")):
        raise ValueError("Explicit original lifecycle flags required")
    if p["reason"] not in ("signal","server","engine_exit"):raise ValueError("Known shutdown reason required")
    if not integer(p["budget_ms"]) or not integer(p["api_drain_ms"]) or not integer(p["stragglers"],16):
        raise ValueError("Bounded nonnegative original times/counts required")
    if p["launcher_exit_after_ms"] is not None and not integer(p["launcher_exit_after_ms"]):
        raise ValueError("Launcher exit is null for no observed timely exit or a bounded delay")
    if not isinstance(p["requests"],list) or not 1<=len(p["requests"])<=64 or any(not integer(x) for x in p["requests"]):
        raise ValueError("1..64 original finish times required")
    if not p["has_api"] and p["api_drain_ms"]!=0:raise ValueError("Headless original profiles have no frontend drain time")
    if p["reason"]=="engine_exit" and p["pid_present"]:
        raise ValueError("Original observed-exit profile cannot also claim a live launcher PID")


def plan(p,strategy="drain_first"):
    validate(p)
    if strategy not in ("drain_first","legacy_group_first"):raise ValueError("Known shutdown strategy required")
    if p["already_started"] or not p["pid_present"]:
        return {"outcome":"no_op_already_started" if p["already_started"] else "no_op_no_pid",
                "events":[],"engine_start_ms":None,"remaining_budget_ms":None,"launcher_wait_ms":None,
                "modeled_end_ms":None,"modeled_deadline_overrun_ms":None,
                "completed_requests":None,"failed_requests":None,"stragglers_remaining":None,
                "native_shutdown_verified":None}
    drain=p["api_drain_ms"] if p["has_api"] and p["reason"]!="server" else 0
    start=drain if strategy=="drain_first" else 0
    remaining=max(0,p["budget_ms"]-start)
    wait=max(remaining if strategy=="drain_first" else p["budget_ms"],5000)
    delay=p["launcher_exit_after_ms"]
    timeout=delay is None or delay>wait
    engine_end=start+(wait if timeout else delay)
    events=[{"at_ms":0,"event":"cancel_serving"}]
    if strategy=="drain_first" and drain:
        events.append({"at_ms":drain,"event":"frontend_await_complete"})
    events.append({"at_ms":start,"event":"TERM_launcher" if strategy=="drain_first" else "TERM_group"})
    if timeout:
        events.append({"at_ms":engine_end,"event":"KILL_group_timeout"})
    else:events.append({"at_ms":engine_end,"event":"launcher_exit_observed"})
    if strategy=="drain_first":
        events.append({"at_ms":engine_end,"event":"KILL_group_post_exit"})
    end=max(engine_end,drain)
    if strategy=="legacy_group_first" and drain:
        events.append({"at_ms":drain,"event":"frontend_await_complete"})
    if p["serve_error"] or p["reason"]=="server":
        events.append({"at_ms":end,"event":"preserve_serve_error"})
    stop=engine_end if strategy=="drain_first" else 0
    complete=sum(t<=stop for t in p["requests"])
    outcome="serve_error_after_cleanup" if p["serve_error"] or p["reason"]=="server" else "forced_group_cleanup" if timeout else "launcher_exit_cleanup"
    return {"outcome":outcome,"events":sorted(events,key=lambda x:x["at_ms"]),
            "engine_start_ms":start,"remaining_budget_ms":remaining,"launcher_wait_ms":wait,
            "modeled_end_ms":end,"modeled_deadline_overrun_ms":max(0,end-p["budget_ms"]),
            "completed_requests":complete,"failed_requests":len(p["requests"])-complete,
            "stragglers_remaining":0 if strategy=="drain_first" or timeout else p["stragglers"],
            "native_shutdown_verified":None}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        finishes=sorted((i*137)%10001 for i in range(n));cutoff=6000
        if bisect.bisect_right(finishes,cutoff)!=sum(x<=cutoff for x in finishes):raise AssertionError("Finish-count parity")
        samples={True:[],False:[]}
        for mode in (True,False):
            for _ in range(2):
                if mode:bisect.bisect_right(finishes,cutoff)
                else:sum(x<=cutoff for x in finishes)
        for repeat in range(7):
            for mode in ((True,False) if repeat%2==0 else (False,True)):
                start=time.perf_counter_ns()
                for _ in range(100):
                    if mode:bisect.bisect_right(finishes,cutoff)
                    else:sum(x<=cutoff for x in finishes)
                samples[mode].append((time.perf_counter_ns()-start)/100/1e6)
        rows.append({"requests":n,"prefix_samples_ms":samples[True],"scan_samples_ms":samples[False],
                     "prefix_median_ms":statistics.median(samples[True]),"scan_median_ms":statistics.median(samples[False])})
    return {"provenance":"measured","scope":"Original sorted-finish count only;sorting/event construction excluded,no process/signals",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows,
            "warmups":2,"repeats":7,"counts_per_sample":100,"native_drain_latency_ms":None}


def report(latency):
    variants=[("frontend drained",{}),("launcher never observed exiting",{"launcher_exit_after_ms":None}),
              ("zero caller budget",{"budget_ms":0,"launcher_exit_after_ms":None}),
              ("headless launcher drain",{"has_api":False,"api_drain_ms":0,"launcher_exit_after_ms":7000}),
              ("serve error preserved",{"serve_error":True}),
              ("already started",{"already_started":True}),("PID unavailable",{"pid_present":False})]
    matrix=[{"name":name,"inputs":{**DEFAULT,**changes},"outputs":plan({**DEFAULT,**changes})} for name,changes in variants]
    return {"schema_version":1,"round":168,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"vllm-project/vllm","pin":PIN,"chronology":"actualNov14UTC06:50:47/BJT14:50:47 frontend/launcher drain",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"serve drain before engine shutdown,remaining budget,five-second floor,positive launcher TERM and group KILL after exit/timeout",
                      "scope":"signal-free original timeline,not native Rust/Linux/Python/DP serving or wall-time guarantee"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"legacy":plan(DEFAULT,"legacy_group_first")},
            "matrix":matrix,"latency":latency,"native_exit_poll_ms":200,
            "native_min_shutdown_timeout_ms":5000,"native_signals_executed":False,"visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    path=HERE/"latency.json"
    if a.measure or not path.exists():path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=report(json.loads(path.read_text(encoding="utf-8")))
    from experiments.managed_engine_drain_selection.render import outputs
    files={HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}
    for path,value in files.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=value:raise AssertionError(f"Unsealed artifact:{path}")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    print("Round168 sealed artifacts match" if a.check else "Rendered original signal-free shutdown ledger")


if __name__=="__main__":main()
