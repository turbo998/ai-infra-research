import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="lora-hrw-sticky-slot-fallback-placement"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==6||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![1,2,4,8].includes(p.element_bytes))throw Error("Positive exact dimensions/selected width required");
 const n=BigInt(p.rank)*(BigInt(p.input_width)+BigInt(p.output_width))*BigInt(p.selected_modules)*BigInt(p.element_bytes),total=n*BigInt(p.replicas);
 const out={selected_adapter_ab_payload_bytes:Number(n),selected_replica_payload_bytes:Number(total)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(workers,replicas,prior,sticky){
 const int=v=>Number.isSafeInteger(v)&&v>=0,identity=w=>JSON.stringify([w.id,w.dp]),full=w=>w.capacity!==null&&w.used>=w.capacity;
 if(!Array.isArray(workers)||workers.length>64||!int(replicas)||!Array.isArray(prior)||!prior.every(p=>Array.isArray(p)&&p.length===2&&p.every(int)))throw Error("Bounded workers and valid prior/replica count required");
 const seen=new Set();for(const w of workers){
 if(!w||Object.keys(w).sort().join()!=="capacity,dp,id,score,used"||!["id","dp","score"].every(k=>int(w[k]))||((w.used===null)!==(w.capacity===null))||(w.used!==null&&(!int(w.used)||!int(w.capacity))))throw Error("Exact worker score/identity and paired slot fields required");
 const key=identity(w);if(seen.has(key))throw Error("Duplicate worker/dp identity");seen.add(key);}
 const order=workers.slice().sort((a,b)=>a.score===b.score?(a.id===b.id?a.dp-b.dp:a.id-b.id):b.score-a.score),available=order.filter(w=>!full(w)),previous=new Set(sticky?prior.map(p=>JSON.stringify(p)):[]);
 const fallback=order.length&&replicas>0&&available.length===0?"all_workers_full_basic_hrw":null;
 const selected=(fallback?order:[...available.filter(w=>previous.has(identity(w))),...available.filter(w=>!previous.has(identity(w)))]).slice(0,replicas);
 return {selected:selected.map(w=>[w.id,w.dp]),fallback,selected_full_count:selected.filter(full).length,unknown_slot_workers:workers.filter(w=>w.capacity===null).length};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("LoRA affinity,sticky placement and slot fallback workbench",
        f"""<h1>Sticky affinity reduces churn;all-full fallback is not a capacity guarantee</h1>
        <p>Round132 · Original bounded score tables,not native BLAKE3/Rust/Dynamo execution.</p>
        <form id="form"><label>Worker score/slot records JSON<textarea id="workers" rows="10" required>{json.dumps(r['case']['workers'])}</textarea></label>
        <label>Prior worker/dp pairs JSON<input id="prior" value="[[9,0],[10,0]]" required></label>
        <label>Selected placement replicas<input id="replica_factor" value="2" type="number" min="0" step="1" required></label>
        <label>Retain eligible prior placements<input id="sticky" type="checkbox" checked></label>
        <h2>Independent selected adapter payload budget</h2><div class="grid">{fields}</div><button>Rank and inspect placement</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Null usage/capacity means unknown,not proven free. Partial capacity may return fewer replicas.
        All-full fallback explicitly returns basic HRW homes,possibly full;it is not admission or load success.
        Worker identity includes DP rank.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)])),workers=JSON.parse(document.getElementById("workers").value),prior=JSON.parse(document.getElementById("prior").value),replicas=Number(document.getElementById("replica_factor").value);
 out.textContent=JSON.stringify({result:simulate(workers,replicas,prior,document.getElementById("sticky").checked),nonsticky:simulate(workers,replicas,prior,false),budget:budget(p),
 note:"Synthetic explicit scores,not native BLAKE3 allocation/admission/residency proof"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(max(v["sort_median_ms"],v["topk_median_ms"]) for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['workers']} workers ·top4</b><p>Heap topK {v['topk_median_ms']:.6f}ms</p>{bar(v['topk_median_ms'],mx)}"
                  f"<p>Full sort {v['sort_median_ms']:.6f}ms</p>{bar(v['sort_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · LORA AFFINITY","Sticky placement retains2/2 eligible prior homes where fresh top2 retains0/2",
         "<div class='grid'><div><h3>Scenarios</h3><p>Pair identity/ties<br>Prior retention<br>Full-slot fallback</p></div>"
         "<div><h3>Input</h3><p>Four score records<br>R2,prior9:0/10:0<br>Known0/2slots</p></div>"
         "<div><h3>Decisions</h3><p>Retain eligible prior<br>Skip full if possible<br>Tag all-full exception</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned Dynamo LoRA HRW allocator;original explicit scores,not BLAKE3 execution. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: STICKY HOMES","Eligible prior workers9:0/10:0 survive even below natural score leaders7:0/7:1",
         f"<blockquote>Natural{c['natural']['selected']} · sticky{c['sticky']['selected']} · slot factor2.</blockquote>"
         f"<div class='metric'><span class='pill'>score · retained homes,not confidence</span><h3>Sticky{c['sticky_prior_retained']}/2</h3>{bar(c['sticky_prior_retained'],2)}<p>Act automatically only within selected policy</p></div>"
         f"<div class='metric'><span class='pill'>score · retained homes,not confidence</span><h3>Natural{c['natural_prior_retained']}/2</h3>{bar(c['natural_prior_retained'],2,True)}<p>Confirm churn versus fresh score order</p></div>"),
        ("WHAT I TESTED","One non-full worker yields1home;all-full fallback may select2full workers",
         f"<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Partialcapacity</td><td>Always fillR2?</td><td>✕ {c['only_one_nonfull']['selected']}</td><td>Accept smaller set</td></tr>"
         f"<tr><td>Allfull</td><td>Hardcapacity?</td><td>✕ {c['all_full']['selected_full_count']}full</td><td>Explicit fallback tag</td></tr>"
         "<tr><td>Missingusage</td><td>Proven free?</td><td>✕ Eligible by policy</td><td>Inspect telemetry</td></tr>"
         "<tr><td>Same workerID</td><td>Same destination?</td><td>✕ DP rank differs</td><td>Pair identity</td></tr>"
         "<tr><td>Forced scoreties</td><td>Input-order stable?</td><td>IDthenDPrank</td><td>Total tie order</td></tr></table>"
         "<div class='grid two'><div class='callout'>Placement candidates are not adapter load/residency success.</div><div class='callout'>All-full exception is pinned policy,not universal capacity safety.</div></div>"),
        ("UNDER THE HOOD","Native HRW hashes name/worker/DP rank;the original model isolates policy with score tables",
         "<pre>Native:BLAKE3(name bytes,worker_id LE,dp_rank LE)\nscore:first8bytes as little-endian u64\nrank:descending score,ascending ID/DP tie\n\nOriginal model:fixed exact score records\nNo native hash bit parity/distribution test</pre>"
         "<p>↓ Original discrete policy / subset oracle</p><pre>Keep present non-full prior homes in score order\nFill from other non-full workers\nIf zero eligible:tag basic-HRW fallback\nIf partially filled:keep smaller set\n\nIndependent:enumerate feasible subsets,\nmaximize retained prior count then rank preference</pre>"),
        ("ROBUSTNESS CHECK","Forced ties preserve order under reversal;making prior9:0full replaces only that home",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Allscoreties</td><td>{c['tie_order']}</td><td>{c['tie_reversed_order']}</td><td>Exact equality</td></tr>"
         f"<tr><td>Prior9:0becomesfull</td><td>{c['sticky']['selected']}</td><td>{c['sticky_prior_full']['selected']}</td><td>Not probability points</td></tr>"
         f"<tr><td>Newscore110worker</td><td>{c['natural']['selected']}</td><td>{c['added_high_basic']['selected']}</td><td>One old home replaced</td></tr></table>"),
        ("LATENCY","Precomputed-score top4 avoids full sorting only in this original Python microbenchmark",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Hashing,slots/stickiness,controller,Rust,Dynamo,adapter transfer/load and GPU excluded.</p>"),
        ("MY TAKE","Separate deterministic adapter affinity from capacity admission and actual loaded residency",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Worker/DP identity,tie-order audits,sticky-churn controls,partial capacity and explicit fallback reviews.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Hash balance certification,hard-cap admission,request fairness,native adapter load speed or global MCF reproduction.</p></div></div>"
         "<div class='dark'>Apache-2.0inspected Dynamo files;original policy/oracle/art.Nativehash/latency/residency/HBM unknown;browser inspection not performed.</div>"),
    ]
    return page("LoRA HRW sticky placement — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>LoRA home-ticket cabinet:deterministic ranks and sticky slot policy</title>
<desc>Original sixty-solid sorting cabinet metaphor for candidate adapter homes,prior retention and explicit fallback.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG132 · LORA STICKY PLACEMENT</text>
<text x="88" y="158" font-size="36" font-weight="700">A stable home is not a successful capacity admission</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Worker identity includes DP rank;score ties need a shared total order</text>
<text x="88" y="758" font-size="25">Retain eligible prior homes,then fill;partial capacity can underfill</text>
<text x="88" y="806" font-size="25">All-full basic-HRW fallback may select full homes—inspect explicitly</text>
<text x="88" y="880" font-size="18">Original discrete score tables/subset oracle/CPU timing · no native BLAKE3,Dynamo,load or GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
