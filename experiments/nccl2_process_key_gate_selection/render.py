import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar

ROOT=Path(__file__).resolve().parents[2]
SLUG="nccl2-tcp-tls-process-key-capability-gates"
JS=r"""
function exact(v,fields){if(!v||Array.isArray(v)||JSON.stringify(Object.keys(v).sort())!==JSON.stringify(fields.slice().sort()))throw Error("Exact metadata-only fields required;never supply secrets");}
function validateKey(v){
 if(v===null)return;exact(v,["cohort","byte_length","nul"]);
 if(!["synthetic-A","synthetic-B","synthetic-C","synthetic-D"].includes(v.cohort))throw Error("Only fixed synthetic cohort labels accepted;never supply keys");
 if(!Number.isInteger(v.byte_length)||v.byte_length<0||v.byte_length>256||typeof v.nul!=="boolean")throw Error("Bounded byte-length metadata and exact NUL boolean required");
}
function validate(p){
 exact(p,["backend","compiled_nccl","rocm","tls_build","runtime_libraries","events"]);
 if(!["nccl2","nccl-legacy","torchcomms"].includes(p.backend))throw Error("Resolved original backend label required");
 if(!Array.isArray(p.compiled_nccl)||p.compiled_nccl.length!==3||p.compiled_nccl.some(v=>!Number.isInteger(v)||v<0||v>999))throw Error("Compiled version triple required");
 if(typeof p.rocm!=="boolean"||[p.tls_build,p.runtime_libraries].some(v=>v!==null&&typeof v!=="boolean"))throw Error("Exact capability boolean or unknown/null required");
 if(!Array.isArray(p.events)||p.events.length>64)throw Error("At most64sequential lifecycle events required");
 p.events.forEach(e=>{exact(e,["operation","option","environment","injected_api_ok"]);
  if(!["create","destroy"].includes(e.operation)||typeof e.injected_api_ok!=="boolean")throw Error("Lifecycle and injected API boolean required");
  validateKey(e.option);validateKey(e.environment);
  if(e.operation==="destroy"&&(e.option!==null||e.environment!==null))throw Error("Destroy cannot configure metadata");
 });
}
function clone(v){return JSON.parse(JSON.stringify(v));}
function sameKey(a,b){return a!==null&&b!==null&&a.cohort===b.cohort&&a.byte_length===b.byte_length&&a.nul===b.nul;}
function apply(state,e,p){
 if(e.operation==="destroy")return [clone(state),"destroy_does_not_reset_process_key",null];
 if(p.backend!=="nccl2")return e.option!==null?[clone(state),"rejected","explicit_option_unsupported_backend"]:[clone(state),"environment_has_no_effect_on_other_backend",null];
 const applied=state.applied,requested=e.option!==null?e.option:(applied===null?e.environment:null);
 if(requested===null||sameKey(requested,applied))return [{created_nccl2:true,applied:clone(applied)},applied===null?"keyless_group":"inherit_or_same_key",null];
 const v=p.compiled_nccl,old=v[0]<2||(v[0]===2&&(v[1]<32||(v[1]===32&&v[2]<3)));
 let error=null;
 if(old||p.rocm)error="compiled_version_or_rocm_gate";
 else if(requested.nul)error="nul_metadata_rejected";
 else if(applied!==null)error="different_process_key";
 else if(state.created_nccl2)error="first_nccl2_group_required";
 else if(requested.byte_length<32||!e.injected_api_ok)error="injected_native_api_failure";
 return error?[clone(state),"rejected",error]:[{created_nccl2:true,applied:clone(requested)},"model_key_applied_not_native_execution",null];
}
function readiness(state,p){
 if(state.applied===null)return "not_configured";
 if(p.tls_build===false)return "blocked_tls_build";
 if(p.runtime_libraries===false)return "blocked_runtime_libraries";
 if(p.tls_build===null||p.runtime_libraries===null)return "requires_build_runtime_probe";
 return "metadata_eligible_not_authenticated";
}
function simulate(p){
 validate(p);let state={created_nccl2:false,applied:null};const steps=[];
 p.events.forEach(e=>{const [updated,status,error]=apply(state,e,p);state=updated;steps.push({status,error,state:clone(state)});});
 return {steps,final_state:state,readiness:readiness(state,p),rejections:steps.filter(v=>v.error!==null).length};
}
"""


def figure():
    # Parts:base8,frame8,tickets16,caps8,retainers8,stops4,latch4,seals4.
    solids=[]
    palette=("#d5dce7","#b5c1d2","#8b9bb3","#e8edf4")
    def box(x,y,z,w,d,h,accent=False):
        def point(a,b,c):
            return (800+3*(a-b),515+0.9*(a+b)-1.9*c)
        def polygon(points,fill):
            return '<polygon points="'+" ".join(f"{a:.2f},{b:.2f}" for a,b in points)+f'" fill="{fill}" stroke="#475569" stroke-width="0.8"/>'
        top=[point(x,y,z+h),point(x+w,y,z+h),point(x+w,y+d,z+h),point(x,y+d,z+h)]
        left=[point(x,y+d,z),point(x+w,y+d,z),point(x+w,y+d,z+h),point(x,y+d,z+h)]
        right=[point(x+w,y,z),point(x+w,y+d,z),point(x+w,y+d,z+h),point(x+w,y,z+h)]
        colors=("#2563eb","#1d4ed8","#60a5fa") if accent else (palette[1],palette[2],palette[3])
        content=polygon(left,colors[0])+polygon(right,colors[1])+polygon(top,colors[2])
        solids.append((x+y+z*0.2,content))
    box(-100,-46,0,200,92,8)
    for x in (-92,80):
        for y in (-38,26):
            box(x,y,-7,12,12,7)
    for x in (-96,0,92):
        box(x,-42,8,4,84,2)
    for x in (-98,92):
        for y in (-44,38):
            box(x,y,8,6,6,82)
    for y in (-44,38):
        for z in (28,84):
            box(-92,y,z,184,6,6)
    for i in range(4):
        for j in range(4):
            box(-80+i*44,-27+j*16,10,30,11,5+j*3)
    for i in range(4):
        box(-82+i*44,-30,42,34,65,4)
        box(-78+i*44,18,47,26,8,3)
    for x in (-94,86):
        for y in (-42,36):
            for z in (30,86):
                box(x,y,z,5,5,3)
    for x in (-91,-31,29,85):
        box(x,-39,78,6,9,6)
    for x in (-85,-47,-9,29):
        box(x,-20,61,34,8,7)
    for i,x in enumerate((-71,-27,17,61)):
        box(x,-15,70,12,18,5,accent=i==0)
    if len(solids)!=60:
        raise RuntimeError("original latching-cabinet parts specification violated")
    scene="".join(content for _,content in sorted(solids,key=lambda v:v[0]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" role="img">
<title>One-shot process-wide selector cabinet</title>
<desc>Original sixty-solid figure:one latched selector spans multiple group trays.The seal represents modeled process lifetime,not cryptographic equipment or actual keys.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG145 · ONE PROCESS-WIDE LATCH, NOT ONE SECRET PER GROUP</text>
<text x="88" y="158" font-size="34" font-weight="700">Destroying groups does not reset the process-wide key</text>{scene}
<text x="88" y="705" font-size="24">First nccl2 group gates the latch;rejected key application does not consume it</text>
<text x="88" y="754" font-size="24">Later groups inherit or match;environment changes do not rotate an applied key</text>
<text x="88" y="803" font-size="24">TCP TLS scope excludes GPU data paths,external net plugins and the c10d Store</text>
<text x="88" y="882" font-size="18">Original metadata-only teaching figure · no secret material,TLS/communicators/GPU execution or visual browser inspection</text></g></svg>\n"""


def workbench(r):
    return page("NCCL2 process-wide metadata gate workbench",
        f"""<h1>A destroyed group does not reset process-global configuration</h1>
        <p>Round145 · Original sequential metadata-only automaton.Never paste a PSK—only the four fixed synthetic cohort labels are accepted.</p>
        <form id="form"><label>Original metadata timeline JSON<textarea id="input" rows="16" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <button>Replay metadata gates—not native encryption</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>injected_api_ok is a synthetic stub result,not an observed NCCL call.Byte-length0..256is an original UI bound,not a native maximum.
        TLS/runtime null remains unknown.No credentials,cryptographic equality,communicators,concurrent group creation or key rotation execution.</p>""",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Metadata input required");out.textContent=JSON.stringify(simulate(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    rows=r["scenarios"];mx=max(v["pairwise_median_ms"] for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['rank_labels']}synthetic rank labels</b><p>Set {v['set_median_ms']:.6f}ms</p>{bar(v['set_median_ms'],mx)}"
                   f"<p>All unordered pairs {v['pairwise_median_ms']:.6f}ms</p>{bar(v['pairwise_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    short_names=("Option/env","Later env","Late first key","Destroy/change","31→32bytes","Injected failure","NCCL2.32.2",
                 "ROCm","Legacy option","Legacy env","Torchcomms","No TLS build","Unknown runtime","NUL/then valid","64bytes")
    readiness_names={"metadata_eligible_not_authenticated":"Metadata eligible","not_configured":"Not configured",
                     "blocked_tls_build":"TLS build blocked","requires_build_runtime_probe":"Runtime probe needed"}
    matrix=""
    for v,name in zip(rows,short_names):
        out=v["outputs"];answer=(f"✕{out['rejections']}rejected; " if out["rejections"] else "")+readiness_names[out["readiness"]]
        matrix+="<tr>"+"".join(f"<td style='font-size:14px;padding:7px'>{text}</td>" for text in (name,"Every step eligible?",answer,"Confirm"))+"</tr>"
    sections=[
        ("HANDS-ON · CONFIGURATION GATES","NCCL2's Nov14TCP TLS option requires compiled NCCL2.32.3+—not just a newer PyTorch tree",
         "<div class='grid'><div><h3>Scenarios</h3><p>First group<br>Environment precedence<br>Build/runtime readiness</p></div><div><h3>Structured input</h3><p>Fixed synthetic cohort IDs<br>Byte-length/NUL metadata<br>No key material</p></div>"
         "<div><h3>Decision pills</h3><p>Process-global latch<br>Reject late rotation<br>TLS probe before claims</p><span class='pill'>Native handshake:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Original CPU Python{r['latency']['python']} / {r['latency']['machine']};no cryptography or GPU execution.</div>"),
        ("SCENARIO: DESTROY THEN CHANGE","Two create attempts separated by destroy still share one process-wide latch",
         "<blockquote>Set synthetic-A → destroy groups → request synthetic-B</blockquote>"
         "<div class='metric'><span class='pill'>choice · deterministic model,not confidence</span><h3>✕ Different process key rejected</h3><p>Confirm · group destruction is not process restart or rotation.</p></div>"
         "<div class='metric'><span class='pill'>choice · deterministic model</span><h3>Unset options inherit the applied cohort</h3><p>Environment is not reread after a key has been applied.</p></div>"),
        ("WHAT I TESTED","15original timelines separate constructor policy from TLS build/runtime readiness",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>All15cases;✕ marks rejected steps,not model accuracy.No confidence thresholds or real PSKs.</div><div class='callout'>Build/runtime failures may surface at different native stages—not predicted here.</div></div>"),
        ("UNDER THE HOOD","Resolve option before environment;commit the latch only after injected API success",
         "<pre>Original metadata,not a real key:\ncohort: synthetic-A\nbyte_length:32;NUL:false\ncompiled NCCL:2.32.3\nAPI success:true (injected)\nTLS build/runtime:true (asserted metadata)</pre>"
         "<p>↓ Original sequential state verifier</p><pre>No applied key → option or environment\nNo request/same applied → inherit\nElse check compiled version/ROCm/NUL\nReject different or late first key\nInjected API failure → unchanged state\nDestroy groups → unchanged process latch</pre>"),
        ("ROBUSTNESS CHECK","A 31-byte failure leaves first-group eligibility;64bytes also satisfies the minimum",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Minimum byte length</td><td>32bytes</td><td>64bytes</td><td>Both model-eligible</td></tr>"
         "<tr><td>First attempt</td><td>31bytes,rejected</td><td>Then32bytes</td><td>First successful apply remains possible</td></tr>"
         "<tr><td>Applied environment</td><td>synthetic-A</td><td>Later synthetic-B env</td><td>No applied-latch change</td></tr>"
         "<tr><td>Runtime libraries</td><td>Metadata true</td><td>Unknown/null</td><td>Probe required,not zero/false</td></tr></table>"),
        ("LATENCY","Synthetic cohort-label preflight is not TLS handshake or encrypted throughput",
         timing+"<p>Prebuilt homogeneous labels,no early mismatch;set versus all unordered pairs.Two warmups,seven alternating repeats,three operations/sample,median repeat means.Excludes label collection,secret comparison,NCCL,TLS,GPUs and network.</p>"),
        ("MY TAKE","TCP socket protection does not certify GPU paths,external plugins or the c10d Store",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Version/backend gates,first-nccl2-group configuration,process lifetime,environment precedence and explicit build/runtime probes.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Per-group rotation,concurrent setting,passphrase/byte-length equivalence,synthetic IDs as authentication,or claiming complete data-path encryption.</p></div></div>"
         "<div class='dark'>Original metadata-only model.PyTorch BSD-style source rights preserved.No secret values,source implementation mirror,native TLS proof or independent reproduction.Visual browser inspection not performed.</div>"),
    ]
    return page("NCCL2 process key gates — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
