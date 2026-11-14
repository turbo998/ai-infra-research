import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="worker-startup-gc-freeze-epochs-refcount-warmup"
JS=r"""
function integer(v,max=1048576){return Number.isInteger(v)&&v>=0&&v<=max;}
function validate(p){
 if(!p||Object.keys(p).sort().join(",")!=="capture_gc_opt_out,captures,drop_startup_cycles,gc_initially_enabled,init_collections,late_garbage,late_live,startup_live")throw Error("Exact GC epoch inputs required");
 for(const k of ["startup_live","drop_startup_cycles","late_live","late_garbage"])if(!integer(p[k]))throw Error("Bounded original object counts required");
 if(p.drop_startup_cycles>p.startup_live||!integer(p.init_collections,64)||!integer(p.captures,8)||typeof p.capture_gc_opt_out!=="boolean"||typeof p.gc_initially_enabled!=="boolean")throw Error("Bounded collection/capture counts and boolean witnesses required");
}
function plan(p,late=false,skip=false){
 validate(p);let fl=late?0:p.startup_live,fd=0,al=late?p.startup_live:0,ad=0,enabled=p.gc_initially_enabled,reclaimed=0;const trace=[];
 const snap=phase=>trace.push({phase,frozen_nodes:fl+fd,active_nodes:al+ad,unreachable_frozen_cycles:fd,collected_cycle_nodes:reclaimed,automatic_gc_enabled:enabled});
 const collect=()=>{reclaimed+=ad;ad=0;},unfreeze=()=>{al+=fl;ad+=fd;fl=0;fd=0;},freeze=()=>{fl+=al;fd+=ad;al=0;ad=0;};
 snap("startup");const visits=al*p.init_collections,d=p.drop_startup_cycles;
 if(late){al-=d;ad+=d;}else{fl-=d;fd+=d;}
 al+=p.late_live;ad+=p.late_garbage;snap("before_capture");
 for(let i=0;i<p.captures;i++){collect();if(!p.capture_gc_opt_out){freeze();enabled=false;}snap("capture_enter");
 if(!p.capture_gc_opt_out){unfreeze();collect();enabled=p.gc_initially_enabled;}snap("capture_exit");}
 if(!skip)unfreeze();collect();freeze();snap("final_warmup");
 return {modeled_init_object_visits:visits,final_frozen_nodes:fl+fd,unreachable_frozen_cycles:fd,collected_cycle_nodes:reclaimed,automatic_gc_enabled:enabled,trace,native_vllm_verified:null,actual_gc_scanned_objects:null};
}
"""


def workbench(r):
    return page("Startup GC epoch workbench","<h1>Freeze skips old cyclic-GC scans;unfreeze ends the epoch and releases unreachable cycles</h1>"
        "<p>Original object-count ledger,not measured scanner visits or native vLLMstartup.Automatic events are not simulated.CPython evidence was collected in disposable child processes only.</p>"
        f'<form id="form"><label>Static/dropped cyclic nodes,capture opt-out and prior automatic-GC state<textarea id="input" rows="10" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Compare startup freeze,late-only and missing epoch reset</button></form>'
        '<p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>freeze_gc_heap does not disable automatic GC.Default capture exit restores the prior enabled state;opt-out capture does not unfreeze startup heap.Warmup reset remains required.</p>",
        JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),late_only:plan(p,true),counterfactual_skip_reset:plan(p,false,true)},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    p=r["case"]["inputs"];a=r["case"]["current"];b=r["case"]["late_only"];s=r["case"]["skip_reset_proposal"];probe=r["cpython_probe"]["cycles"];f=r["cpython_probe"]["finalizer"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (x["name"],"epoch/GC state",x["outputs"]["final_frozen_nodes"],"Confirm "+str(x["outputs"]["automatic_gc_enabled"])))+"</tr>" for x in r["matrix"])
    matrix+=f"<tr><td>Skip reset proposal</td><td>Unreachable frozen cycles</td><td>✕ {s['unreachable_frozen_cycles']}remain</td><td>Unfreeze before final collection</td></tr>"
    mx=max(max(x["frozen_median_ms"],x["unfrozen_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['original_live_cycle_nodes']}original rooted cycles</b><p>Frozen{x['frozen_median_ms']:.6f}ms</p>{bar(x['frozen_median_ms'],mx)}"
                  f"<p>Unfrozen{x['unfrozen_median_ms']:.6f}ms</p>{bar(x['unfrozen_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · GC EPOCHS","Startup freeze occurs before weights;final warmup unfreezes and refreezes the final heap",
         "<div class='grid'><div>Scenarios<br>No CUDAgraphs<br>Capture opt-out<br>Prior GCdisabled</div><div>Input<br>Original node counts<br>Explicit collections<br>Capture witnesses</div>"
         f"<div>Decision<br>Final modeled nodes{a['final_frozen_nodes']}<br>Old cycles{a['unreachable_frozen_cycles']}<br>Native unknown</div></div>"
         f"<div class='dark'>Synthetic ledger,independently_reproduced=false.Separate measured CPython{r['latency']['python']}/{r['latency']['machine']}probe;no vLLMweights/CUDA.</div>"),
        ("SCENARIO: SKIP OLD SCANS",f"{b['modeled_init_object_visits']}modeled init visits become{a['modeled_init_object_visits']};not actual scanner telemetry",
         f"<blockquote>{p['startup_live']}declared startup nodes × {p['init_collections']}explicit init collections.The ledger assumes no unmodeled automatic GC.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm lifetime assumptions · confidence unmeasured</span><p>Initial GCenabled remains{a['automatic_gc_enabled']}</p></div>"
         f"<div class='metric'><span class='pill'>score · original modeled visits</span><p>Late-only{b['modeled_init_object_visits']}</p>{bar(b['modeled_init_object_visits'],b['modeled_init_object_visits'])}<p>Startup freeze{a['modeled_init_object_visits']}</p>{bar(a['modeled_init_object_visits'],b['modeled_init_object_visits'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original epochs separate freeze,automatic GC and capture cleanup",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Freeze is not immutability or disabled reference counting.</div><div class='callout'>Opt-out capture leaves startup frozen until mandatory warmup reset.</div></div>"),
        ("UNDER THE HOOD","A full collection followed by gen0can remove finalizer-created cyclic garbage before freezing",
         "<pre>Startup after distributed/cuBLAS init\ncollect(all) → collect(gen0) → freeze\nLater init collections skip frozen objects\nWeight/snapshot placement verified in source</pre><p>↓ End the startup epoch</p>"
         "<pre>Default capture exit:unfreeze → collect\nRestore prior automatic-GC enabled state\nOpt-out:do not freeze/disable/unfreeze there\nFinal warmup:unfreeze → full+gen0 → freeze</pre>"),
        ("ROBUSTNESS CHECK",f"Measured child retains{probe['frozen_cycles_alive_after_collection']}frozen cycles,then{probe['cycles_alive_after_unfreeze_collection']}after unfreeze",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Frozen cycles</td><td>{probe['original_cycle_nodes']}</td><td>Drop roots,collect</td><td>{probe['frozen_cycles_alive_after_collection']}alive</td></tr>"
         f"<tr><td>Epoch reset</td><td>Frozen</td><td>Unfreeze,collect</td><td>{probe['cycles_alive_after_unfreeze_collection']}alive</td></tr>"
         f"<tr><td>New finalizer cycle</td><td>Old0/1/2</td><td>Full then0</td><td>{f['results']['old_0_1_2']['alive_when_frozen']}→{f['results']['full_then_0']['alive_when_frozen']}</td></tr>"
         f"<tr><td>Acyclic refcount</td><td>Frozen</td><td>Drop root</td><td>Freed:{probe['frozen_acyclic_refcount_freed']}</td></tr></table>"),
        ("LATENCY","Isolated CPython collection timings exclude freeze setup and native server startup",timing+
         "<p>2warmups,7alternating repeats,3collections/sample;median repeat means.Whole child heap is frozen,not just named cycles.Automatic GCdisabled for isolation only.</p>"),
        ("MY TAKE","Treat freeze as a lifetime epoch with explicit reset,not permanent retention or a GPU memory release",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Mostly static startup/serving heaps,explicit warmup reset,correct capture opt-out and prior-GC restoration.</div>"
         "<div class='bad'><h3>⛔ Not for</h3>Permanent freezing of churn,claiming immutable objects,assuming GCcollect releases CUDA cache or promising startup speed.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Original ledger + isolated CPython evidence.Native vLLM/CUDA startup and browser acceptance unknown.</div>")]
    return page("GC startup epochs — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["late_only"];probe=r["cpython_probe"]["cycles"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>GC lifetime archive cabinet</title><desc>Original sixty-solid cabinet;frozen drawers stay outside cyclic scans until the epoch ends.Reference counting still works.Not native GC object layout.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG164 · LIFETIME EPOCH CABINET</text><text x="88" y="158" font-size="34">Freeze skips cyclic scans;warmup unfreeze releases stale cycles</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original init modeled visits{b['modeled_init_object_visits']}→{a['modeled_init_object_visits']};not actual scan telemetry</text><text x="88" y="760" font-size="24">Measured child frozen cycles{probe['frozen_cycles_alive_after_collection']}→{probe['cycles_alive_after_unfreeze_collection']}after unfreeze/collect</text><text x="88" y="810" font-size="24">Freeze does not disable GC or refcount;capture restores prior automatic state</text><text x="88" y="880" font-size="18">ActualNov14worker placement · synthetic ledger + isolated CPython;no native vLLM/CUDA or visual acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
