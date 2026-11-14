"""Original literal-route registry and symbolic credential witnesses,not an auth service."""
import argparse
import copy
import json
import platform
import re
import statistics
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PIN="cb551180329b719f892580fe21d3534b45fbcbe6"
LEVELS=("normal","admin_optional","admin_force")
METHODS=("GET","POST","DELETE","HEAD","OPTIONS")


def leaf(path,level="normal",kind="http",methods=None):
    return {"path":path,"level":level,"typed":True,"kind":kind,
            "methods":methods or ["POST"],"matcher_error":False}


TREE=[leaf("/health",methods=["GET"]),leaf("/work"),{"prefix":"/api","routes":[
    {"prefix":"/control","routes":[leaf("/pause","admin_optional"),leaf("/weights","admin_force"),
                                  leaf("/socket","normal","websocket",["GET"])]}]}]
DEFAULT={"tree":TREE,"scope":{"path":"/api/control/pause","method":"POST","kind":"http"},
         "api_configured":True,"admin_configured":True,"credential":"api"}


def path_ok(value,prefix=False):
    return isinstance(value,str) and len(value)<=96 and (value=="" and prefix or
           re.fullmatch(r"/[A-Za-z0-9_/-]*",value) is not None)


def validate(p):
    if not isinstance(p,dict) or set(p)!=set(DEFAULT):
        raise ValueError("Exactly the original tree,scope and symbolic credential fields required")
    if type(p["api_configured"]) is not bool or type(p["admin_configured"]) is not bool or p["credential"] not in ("none","api","admin","both","other"):
        raise ValueError("Boolean key configuration and known credential witness required;no secrets")
    s=p["scope"]
    if not isinstance(s,dict) or set(s)!={"path","method","kind"} or not path_ok(s["path"]) or s["method"] not in METHODS or s["kind"] not in ("http","websocket"):
        raise ValueError("Original literal path/method/protocol scope required")
    count=0
    def visit(nodes,depth):
        nonlocal count
        if not isinstance(nodes,list) or not nodes or depth>8:
            raise ValueError("Nonempty bounded route arrays,maximum depth8,required")
        for x in nodes:
            count+=1
            if count>128 or not isinstance(x,dict):raise ValueError("At most128route nodes required")
            if set(x)=={"prefix","routes"}:
                if not path_ok(x["prefix"],True):raise ValueError("Literal include prefix required")
                visit(x["routes"],depth+1)
            elif set(x)==set(leaf("/x")):
                if not path_ok(x["path"]) or x["level"] not in (*LEVELS,"invalid") or type(x["typed"]) is not bool or type(x["matcher_error"]) is not bool or x["kind"] not in ("http","websocket"):
                    raise ValueError("Valid original leaf metadata required")
                if not isinstance(x["methods"],list) or not x["methods"] or any(m not in METHODS for m in x["methods"]):
                    raise ValueError("Declared finite leaf methods required")
            else:raise ValueError("Leaf or included-router node required")
    visit(p["tree"],0)


def compile_routes(nodes,prefix="",top_only=False):
    out=[]
    for x in nodes:
        if "routes" in x:
            if not top_only:out.extend(compile_routes(x["routes"],prefix+x["prefix"]))
        else:
            out.append({**copy.deepcopy(x),"prefix":prefix,
                        "resolved_level":x["level"] if x["typed"] and x["level"] in LEVELS else "normal"})
    return out


def resolve(rows,scope):
    trace=[]
    for x in rows:
        prefix=x["prefix"];result="NONE"
        if not scope["path"].startswith(prefix):result="PREFIX_SKIP"
        elif x["matcher_error"]:result="MATCHER_ERROR"
        elif scope["kind"]==x["kind"] and scope["path"][len(prefix):]==x["path"]:
            result="FULL" if x["kind"]=="websocket" or scope["method"] in x["methods"] else "PARTIAL"
        trace.append({"path":prefix+x["path"],"result":result})
        if result=="FULL":return x["resolved_level"],prefix+x["path"],trace
    return "normal",None,trace


def decision(p,level):
    s=p["scope"];w=p["credential"]
    if s["method"]=="OPTIONS" or s["path"].startswith(("/health","/ready","/metrics")):
        return {"allowed":True,"status":None,"reason":"preflight_or_probe_prefix"}
    expected=None
    if level=="admin_force":
        if not p["admin_configured"]:return {"allowed":False,"status":403,"reason":"admin_key_not_configured"}
        expected="admin"
    elif level=="admin_optional":
        expected="admin" if p["admin_configured"] else "api" if p["api_configured"] else None
    else:expected="api" if p["api_configured"] else None
    allowed=expected is None or w==expected or w=="both"
    return {"allowed":allowed,"status":None if allowed else 401,
            "reason":"no_key_required" if expected is None else "symbolic_credential_match" if allowed else "symbolic_credential_mismatch"}


def plan(p,strategy="included"):
    validate(p)
    if strategy not in ("included","top_only"):raise ValueError("Known registry strategy required")
    rows=compile_routes(p["tree"],top_only=strategy=="top_only")
    level,path,trace=resolve(rows,p["scope"])
    return {"resolved_level":level,"matched_path":path,"registered_leaves":len(rows),
            "examined_leaves":len(trace),"resolution_trace":trace,"decision":decision(p,level),
            "native_routing_verified":None,"credential_comparison_verified":None,
            "security_certification":None}


def measure():
    rows=[]
    for n in (64,256,1024,4096):
        tree=[{"prefix":"/api","routes":[leaf(f"/task{i}","admin_optional") for i in range(n)]}]
        scope={"path":f"/api/task{n-1}","method":"POST","kind":"http"}
        compiled=compile_routes(tree)
        if resolve(compiled,scope)!=resolve(compile_routes(tree),scope):raise AssertionError("Registry parity")
        samples={True:[],False:[]}
        for mode in (True,False):
            for _ in range(2):resolve(compiled if mode else compile_routes(tree),scope)
        for repeat in range(7):
            for mode in ((True,False) if repeat%2==0 else (False,True)):
                start=time.perf_counter_ns()
                for _ in range(3):resolve(compiled if mode else compile_routes(tree),scope)
                samples[mode].append((time.perf_counter_ns()-start)/3/1e6)
        rows.append({"leaves":n,"compiled_samples_ms":samples[True],"dynamic_samples_ms":samples[False],
                     "compiled_median_ms":statistics.median(samples[True]),
                     "dynamic_median_ms":statistics.median(samples[False])})
    return {"provenance":"measured","scope":"Original literal lookup/trace with versus without immutable registry rebuild;no framework or secrets",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows,
            "warmups":2,"repeats":7,"batches_per_sample":3,"native_http_latency_ms":None}


def report(latency):
    variants=[("nested admin/API witness",{}),("nested admin/admin witness",{"credential":"admin"}),
              ("optional API-only",{"admin_configured":False}),
              ("forced admin unset",{"scope":{"path":"/api/control/weights","method":"POST","kind":"http"},"admin_configured":False}),
              ("normal admin-only",{"scope":{"path":"/work","method":"POST","kind":"http"},"api_configured":False,"credential":"none"}),
              ("root probe",{"scope":{"path":"/health","method":"GET","kind":"http"},"credential":"none"}),
              ("websocket",{"scope":{"path":"/api/control/socket","method":"GET","kind":"websocket"}})]
    matrix=[{"name":name,"inputs":{**DEFAULT,**changes},"outputs":plan({**DEFAULT,**changes})} for name,changes in variants]
    return {"schema_version":1,"round":167,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"sgl-project/sglang","pin":PIN,"chronology":"actualNov14UTC06:03:38/BJT14:03:38 included-router resolution",
                      "license":"Apache-2.0 rights retained,no implementation mirror",
                      "verified":"recursive prefixes,first FULL leaf Enum metadata,normal fallback,existing auth decision policy and inventory tests",
                      "scope":"original literal registry and credential witnesses,not FastAPI/Starlette/Rust/live authentication/security review"},
            "case":{"inputs":DEFAULT,"current":plan(DEFAULT),"top_only":plan(DEFAULT,"top_only")},
            "matrix":matrix,"latency":latency,"native_framework_execution":False,"visual_acceptance":None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--measure",action="store_true");ap.add_argument("--check",action="store_true");a=ap.parse_args()
    path=HERE/"latency.json"
    if a.measure or not path.exists():path.write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    r=report(json.loads(path.read_text(encoding="utf-8")))
    from experiments.included_router_auth_selection.render import outputs
    files={HERE/"report.json":json.dumps(r,indent=2)+"\n",**outputs(r)}
    for path,value in files.items():
        if a.check:
            if path.read_text(encoding="utf-8")!=value:raise AssertionError(f"Unsealed artifact:{path}")
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding="utf-8")
    print("Round167 sealed artifacts match" if a.check else "Rendered original included-route metadata model")


if __name__=="__main__":main()
