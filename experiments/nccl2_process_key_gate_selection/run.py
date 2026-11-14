"""Original metadata-only NCCL2 configuration automaton;never accepts secret keys."""

import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="0a95a45a42fbec1ab15a5c8ad9b6f03ac2e225ef"


def key(cohort="synthetic-A",byte_length=32,nul=False):
    return {"cohort":cohort,"byte_length":byte_length,"nul":nul}


def event(option=None,environment=None,operation="create",api_ok=True):
    return {"operation":operation,"option":option,"environment":environment,"injected_api_ok":api_ok}


DEFAULT={"backend":"nccl2","compiled_nccl":[2,32,3],"rocm":False,"tls_build":True,
         "runtime_libraries":True,"events":[event(key(),key("synthetic-B")),event(None,key("synthetic-B")),
                                          event(operation="destroy"),event(key("synthetic-B"))]}


class GateError(ValueError):
    pass


def exact_fields(value,fields):
    if not isinstance(value,dict) or set(value)!=set(fields):
        raise ValueError("exact metadata-only fields required;never supply secrets")


def validate_key(value):
    if value is None:
        return
    exact_fields(value,("cohort","byte_length","nul"))
    if value["cohort"] not in ("synthetic-A","synthetic-B","synthetic-C","synthetic-D"):
        raise ValueError("only fixed synthetic cohort labels accepted;never supply keys")
    if type(value["byte_length"]) is not int or not 0<=value["byte_length"]<=256 or type(value["nul"]) is not bool:
        raise ValueError("bounded byte-length metadata and exact NUL boolean required")


def validate(p):
    exact_fields(p,("backend","compiled_nccl","rocm","tls_build","runtime_libraries","events"))
    if p["backend"] not in ("nccl2","nccl-legacy","torchcomms"):
        raise ValueError("resolved original backend label required")
    version=p["compiled_nccl"]
    if not isinstance(version,list) or len(version)!=3 or any(type(v) is not int or not 0<=v<=999 for v in version):
        raise ValueError("three nonnegative compiled-version integer components required")
    if type(p["rocm"]) is not bool or any(v is not None and type(v) is not bool for v in (p["tls_build"],p["runtime_libraries"])):
        raise ValueError("exact capability booleans or explicitly unknown/null required")
    if not isinstance(p["events"],list) or len(p["events"])>64:
        raise ValueError("at most64original sequential lifecycle events required")
    for e in p["events"]:
        exact_fields(e,("operation","option","environment","injected_api_ok"))
        if e["operation"] not in ("create","destroy") or type(e["injected_api_ok"]) is not bool:
            raise ValueError("original create/destroy event and injected API boolean required")
        validate_key(e["option"]);validate_key(e["environment"])
        if e["operation"]=="destroy" and (e["option"] is not None or e["environment"] is not None):
            raise ValueError("destroy event cannot configure metadata")


def apply(state,e,p):
    if e["operation"]=="destroy":
        return copy.deepcopy(state),"destroy_does_not_reset_process_key"
    if p["backend"]!="nccl2":
        if e["option"] is not None:
            raise GateError("explicit_option_unsupported_backend")
        return copy.deepcopy(state),"environment_has_no_effect_on_other_backend"
    applied=state["applied"]
    requested=e["option"] if e["option"] is not None else (e["environment"] if applied is None else None)
    if requested is None or requested==applied:
        return {"created_nccl2":True,"applied":copy.deepcopy(applied)},"inherit_or_same_key" if applied is not None else "keyless_group"
    if tuple(p["compiled_nccl"])<(2,32,3) or p["rocm"]:
        raise GateError("compiled_version_or_rocm_gate")
    if requested["nul"]:
        raise GateError("nul_metadata_rejected")
    if applied is not None:
        raise GateError("different_process_key")
    if state["created_nccl2"]:
        raise GateError("first_nccl2_group_required")
    if requested["byte_length"]<32 or not e["injected_api_ok"]:
        raise GateError("injected_native_api_failure")
    return {"created_nccl2":True,"applied":copy.deepcopy(requested)},"model_key_applied_not_native_execution"


def readiness(state,p):
    if state["applied"] is None:
        return "not_configured"
    if p["tls_build"] is False:
        return "blocked_tls_build"
    if p["runtime_libraries"] is False:
        return "blocked_runtime_libraries"
    if p["tls_build"] is None or p["runtime_libraries"] is None:
        return "requires_build_runtime_probe"
    return "metadata_eligible_not_authenticated"


def simulate(p):
    validate(p)
    state={"created_nccl2":False,"applied":None};steps=[]
    for e in p["events"]:
        try:
            updated,status=apply(state,e,p)
        except GateError as error:
            steps.append({"status":"rejected","error":str(error),"state":copy.deepcopy(state)})
        else:
            state=updated;steps.append({"status":status,"error":None,"state":copy.deepcopy(state)})
    return {"steps":steps,"final_state":state,"readiness":readiness(state,p),
            "rejections":sum(step["error"] is not None for step in steps)}


def scenario(name,events,**overrides):
    p={**copy.deepcopy(DEFAULT),"events":events,**overrides}
    return {"name":name,"inputs":p,"outputs":simulate(p)}


def cases():
    a=key();b=key("synthetic-B")
    return [
        scenario("option_precedes_environment",[event(a,b)]),
        scenario("environment_ignored_after_apply",[event(None,a),event(None,b)]),
        scenario("keyless_first_blocks_late_key",[event(),event(a)]),
        scenario("destroy_does_not_enable_rotation",[event(a),event(operation="destroy"),event(b)]),
        scenario("failed_short_key_does_not_consume_first",[event(key(byte_length=31)),event(a)]),
        scenario("injected_api_failure_rolls_back",[event(a,api_ok=False),event(a)]),
        scenario("compiled_2_32_2_rejected",[event(a)],compiled_nccl=[2,32,2]),
        scenario("rocm_rejected",[event(a)],rocm=True),
        scenario("legacy_explicit_rejected",[event(a)],backend="nccl-legacy"),
        scenario("legacy_environment_ignored",[event(None,a)],backend="nccl-legacy"),
        scenario("torchcomms_explicit_rejected",[event(a)],backend="torchcomms"),
        scenario("tls_build_absent",[event(a)],tls_build=False),
        scenario("runtime_libraries_unknown",[event(a)],runtime_libraries=None),
        scenario("nul_metadata_rejected",[event(key(nul=True)),event(a)]),
        scenario("64bytes_is_above_minimum_not_exact32",[event(key(byte_length=64))]),
    ]


def cohort_agreement(labels):
    return len(set(labels))<=1


def pairwise_agreement(labels):
    return all(labels[i]==labels[j] for i in range(len(labels)) for j in range(i+1,len(labels)))


def measure():
    rows=[]
    for n in (8,32,128,512):
        labels=("synthetic-A",)*n
        funcs=(lambda:cohort_agreement(labels),lambda:pairwise_agreement(labels))
        assert funcs[0]()==funcs[1]()
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
        rows.append({"rank_labels":n,"set_samples_ms":samples[0],"pairwise_samples_ms":samples[1],
                     "set_median_ms":statistics.median(samples[0]),"pairwise_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","python":platform.python_version(),"machine":platform.machine(),
            "warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"prebuilt homogeneous synthetic cohort-label agreement,set versus all unordered pairs;no early mismatch",
            "excludes":"key material,cryptographic equality,metadata collection,group construction,TLS,NCCL,GPU,network and distributed scaling","rows":rows}


def build_report(latency):
    return {"schema_version":1,"round":145,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"pytorch/pytorch","pin":PIN,"license":"PyTorch BSD-style source rights preserved;no implementation mirrored",
                      "verified":"Nov14nccl2 TCP TLS PSK option;compiled NCCL2.32.3+,no ROCm;first nccl2 group,process lifetime,environment precedence,>=32bytes,NUL rejection",
                      "scope":"original metadata-only sequential automaton with explicitly injected API outcomes;no credentials,crypto,communicators or backend support certification",
                      "chronology":"actual Nov14feature addition,not merely a new tree pin;public upstream tests skip missing TLS builds"},
            "case":{"inputs":DEFAULT,"outputs":simulate(DEFAULT)},"scenarios":cases(),
            "rank_metadata":{"matching_labels":["synthetic-A","synthetic-A"],"mismatched_labels":["synthetic-A","synthetic-B"],
                             "matching":cohort_agreement(["synthetic-A","synthetic-A"]),"mismatched":cohort_agreement(["synthetic-A","synthetic-B"]),
                             "scope":"synthetic labels are not hashes,secret verification,authentication or a native mismatch/hang test"},
            "latency":latency,"native_tls_verified":None,"native_crypto_throughput_bytes_s":None,
            "native_handshake_latency_ms":None,"encrypted_gpu_data_paths":None,
            "native_failure_stage":"not executed;group versus communicator failure timing is not predicted"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--measure",action="store_true");parser.add_argument("--check",action="store_true");args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for file,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if file.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact:{file}")
        else:
            file.parent.mkdir(parents=True,exist_ok=True);file.write_text(text,encoding="utf-8")
    print("Round145 sealed artifacts match" if args.check else "Wrote Round145 original artifacts")


if __name__=="__main__":
    main()
