import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="slotted-worker-metadata-dictionary-inheritance-serialization"
JS=r"""
function integer(v){return Number.isInteger(v)&&v>=0&&v<=1048576;}
function plan(p,data){
 if(!p||Object.keys(p).sort().join(",")!=="layout,operation,payload,records,rid,work"||!["plain","slots","weak_slots","inherited_dict"].includes(p.layout)||![64,256,1024,4096].includes(p.records))throw Error("Known layout and exactly measured count tier required");
 if(!["fields","vars","dynamic","weakref","pickle"].includes(p.operation)||!integer(p.rid)||!integer(p.work)||!Array.isArray(p.payload)||p.payload.length>16||p.payload.some(x=>!integer(x)))throw Error("Bounded original operation/metadata required");
 const cap=data.layout_probe.profiles[p.layout],row=data.latency.rows.find(x=>x.records===p.records),value={rid:p.rid,work:p.work,payload:p.payload};let op;
 if(p.operation==="fields"||p.operation==="pickle")op={status:"ok",value};
 else if(p.operation==="vars")op=cap.has_instance_dict?{status:"ok",value:Object.fromEntries(Object.keys(cap.vars.value).map(k=>[k,value[k]]))}:{status:"expected_contract_error",error:"TypeError"};
 else if(p.operation==="dynamic")op=cap.dynamic.status==="ok"?{status:"ok",extra:9}:{status:"expected_contract_error",error:"AttributeError"};
 else op=cap.weakref.status==="ok"?{status:"ok",alive:true}:{status:"expected_contract_error",error:"TypeError"};
 return {layout:p.layout,records:p.records,measured_canonical_batch_shell_bytes:row.shell_bytes[p.layout],operation:op,gc_tracked:cap.gc_tracked,native_vllm_verified:null,referent_bytes:null,javascript_engine_object_bytes:null};
}
"""


def workbench(r):
    return page("Slotted metadata contract workbench","<h1>Slots remove an instance dictionary only if inheritance does not bring it back</h1>"
        "<p>Original three-field Python records.Canonical shell bytes come only from measured tiers,not arbitrary payload sizes.JS reads sealed Python contracts;it does not measure JSobject layout or execute pickle.</p>"
        f'<form id="form"><label>Layout,exact measured tier and operation<textarea id="input" rows="10" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Inspect sealed Python layout contract</button></form>'
        '<p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Expected vars/dynamic/weakref errors are explicit contracts.Slotted fields remain mutable;GC tracking and referenced tensor/payload memory are separate.</p>",
        JS+"const data="+json.dumps(r)+";"+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{out.textContent=JSON.stringify(plan(JSON.parse(document.getElementById("input").value),data),null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["plain"];caps=r["layout_probe"]["profiles"];row=r["latency"]["rows"][0]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (x["name"],x["inputs"]["operation"],("✕ "+x["outputs"]["operation"]["error"]) if x["outputs"]["operation"]["status"]!="ok" else "fields/contract OK","Confirm compatibility"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["slots_median_ms"],x["plain_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['records']}records</b><p>Slots{x['slots_median_ms']:.6f}ms</p>{bar(x['slots_median_ms'],mx)}"
                  f"<p>Plain{x['plain_median_ms']:.6f}ms</p>{bar(x['plain_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · METADATA LAYOUT","Internal record slots do not imply all scheduler/wire classes lost their dictionaries",
         "<div class='grid'><div>Scenarios<br>Slotted fields<br>Dict inheritance<br>Weakref opt-in</div><div>Input<br>Three original fields<br>Exact measured tiers<br>Serialization contract</div>"
         f"<div>Decision<br>{a['records']}canonical records<br>Slots shell{a['measured_canonical_batch_shell_bytes']}B<br>Native layout unknown</div></div>"
         f"<div class='dark'>Synthetic model,independently_reproduced=false;CPython{r['latency']['python']}/{r['latency']['machine']}layout and construction separately measured.No native vLLM/RSS/IPC.</div>"),
        ("SCENARIO: RECORD SHELL",f"{a['records']}original shells measure{b['measured_canonical_batch_shell_bytes']}Bplain versus{a['measured_canonical_batch_shell_bytes']}Bslots",
         "<blockquote>Object + per-instance dictionary only.Referents,batch container,shared class keys and RSS excluded.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm consumers · confidence unmeasured</span><p>Named-field pickle roundtrip remains equal.</p></div>"
         f"<div class='metric'><span class='pill'>score · measured canonical shell bytes</span><p>Plain{b['measured_canonical_batch_shell_bytes']}B</p>{bar(b['measured_canonical_batch_shell_bytes'],b['measured_canonical_batch_shell_bytes'])}"
         f"<p>Slots{a['measured_canonical_batch_shell_bytes']}B</p>{bar(a['measured_canonical_batch_shell_bytes'],b['measured_canonical_batch_shell_bytes'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original contracts distinguish fields/pickle from vars,dynamic attributes and weakrefs",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>A dict-bearing base restores instance dictionaries;slots=True alone is insufficient.</div><div class='callout'>Native NewRequestData/CachedRequestData/SchedulerOutput retain dictionaries at this pin.</div></div>"),
        ("UNDER THE HOOD","Named fields and ordinary pickle can survive without an instance dictionary",
         "<pre>Original rid/work/payload fields\nMeasure exact-tier object+dict shells\nasdict reads declared fields\npickle roundtrip preserves shared payload\nSlots without weakref slot reject weakref</pre><p>↓ Consumer audit</p>"
         "<pre>vars() and custom __dict__ loaders differ\ncached_property needs dictionary storage\nInheritance may reintroduce a dictionary\nslots does not mean frozen or GC-untracked\nNative payload and IPCcost remain unmeasured</pre>"),
        ("ROBUSTNESS CHECK",f"At{row['records']}records,weakref slots and inherited dictionaries have separately measured costs",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Weakref</td><td>Slots{row['shell_bytes']['slots']}B</td><td>Weakslots{row['shell_bytes']['weak_slots']}B</td><td>Explicit opt-in</td></tr>"
         f"<tr><td>Dictionary</td><td>Slots:{caps['slots']['has_instance_dict']}</td><td>Inherited:{caps['inherited_dict']['has_instance_dict']}</td><td>Base matters</td></tr>"
         "<tr><td>cached_property</td><td>Dictionary-backed</td><td>No dictionary</td><td>TypeError</td></tr>"
         f"<tr><td>GCtracked</td><td>Plain:{caps['plain']['gc_tracked']}</td><td>Slots:{caps['slots']['gc_tracked']}</td><td>Not untracking</td></tr></table>"),
        ("LATENCY","Original record construction includes batch-list lifetime,not native scheduler or IPC latency",timing+
         "<p>2warmups,7alternating repeats,3batches/sample;median repeat means.Precreated payload,referents/native vLLM/Torch/GPU/RSS excluded.</p>"),
        ("MY TAKE","Choose slots after checking dict consumers,inheritance and weakrefs,not by decorator alone",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Fixed metadata fields,no dynamic attributes,compatible pickle/named access and explicit weakref needs.</div>"
         "<div class='bad'><h3>⛔ Not for</h3>Blind conversion of cached_property/custom dict loaders,claiming tensor/RSS reduction or assuming immutable/untracked objects.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Original three-field records.Native fields/layout/performance and browser acceptance unknown.</div>")]
    return page("Slotted metadata — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["plain"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Fixed-field metadata cabinet</title><desc>Original sixty-solid cabinet;fixed compartments represent declared fields,not Python object geometry or tensor allocation.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG165 · FIXED FIELD CABINET</text><text x="88" y="158" font-size="34">Slots remove per-instance dictionaries,not referenced payloads</text>{sorter_scene()}<text x="88" y="710" font-size="24">Measured original{a['records']}record shells:plain{b['measured_canonical_batch_shell_bytes']}B / slots{a['measured_canonical_batch_shell_bytes']}B</text><text x="88" y="760" font-size="24">Dictionary inheritance,weakrefs and custom serialization remain separate contracts</text><text x="88" y="810" font-size="24">Named fields/pickle remain;vars and cached_property need explicit compatibility</text><text x="88" y="880" font-size="18">ActualNov14internal conversion · original CPython metadata,not native fields/RSS/IPC or visual acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
