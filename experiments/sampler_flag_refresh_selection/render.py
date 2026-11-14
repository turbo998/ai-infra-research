import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="sglang-sampler-row-drop-top-k-flag-refresh"
JS=r"""
const ALL=2**30;
function flags(ks){return {is_all_greedy:ks.every(k=>k===1),is_any_greedy:ks.includes(1),need_top_k_sampling:ks.some(k=>k!==ALL),
npu_top_k_top_p_eligible:ks.every(k=>k<=1024),max_top_k:ks.length&&!ks.includes(ALL)?Math.max(...ks):null};}
function weights(n){return Array.from({length:n},(_,i)=>(1777*i+19)%n+1);}
function support(scores,k,f,seeded,capture){
 const bound=f.max_top_k;if((bound!==null&&bound<k)||(f.is_all_greedy&&k!==1)||(!f.need_top_k_sampling&&k!==ALL))throw Error("Unsafe original metadata:required support cannot be skipped/truncated");
 const order=()=>Array.from({length:scores.length},(_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b);
 let route,width,ids;
 if(f.is_all_greedy){route="greedy_argmax";width=0;let best=0;for(let i=1;i<scores.length;i++)if(scores[i]>scores[best])best=i;ids=[best];}
 else if(!f.need_top_k_sampling){route="whole_vocab_direct";width=0;ids=Array.from({length:scores.length},(_,i)=>i);}
 else if(bound!==null&&bound<scores.length&&!seeded&&!capture){route="bounded_prefix";width=bound;ids=order().slice(0,bound).slice(0,Math.min(k,scores.length));}
 else{route="full_sort";width=scores.length;ids=order().slice(0,Math.min(k,scores.length));}
 return {route,planned_ranked_prefix_columns:width,support_token_ids:ids.sort((a,b)=>a-b)};
}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="capture,keep,seeded,top_ks,vocab")throw Error("Exact original fields required");
 const ks=p.top_ks,keep=p.keep,n=p.vocab;
 if(!Array.isArray(ks)||ks.length>16||ks.some(k=>!Number.isSafeInteger(k)||k<1||k>ALL))throw Error("Normalized positive top_k required;API-1must normalize first");
 if(!Number.isSafeInteger(n)||n<2||n>4096)throw Error("Original vocabulary2..4096 required");
 if(!Array.isArray(keep)||keep.some((v,i)=>!Number.isSafeInteger(v)||v<0||v>=ks.length||(i>0&&keep[i-1]>=v)))throw Error("Unique increasing in-range keep indices required");
 if(typeof p.seeded!=="boolean"||typeof p.capture!=="boolean")throw Error("Explicit seeded/capture booleans required");
 const before=flags(ks),retained=keep.map(i=>ks[i]),after=retained.length>0&&retained.length<ks.length?flags(retained):before,fresh=retained.length?flags(retained):null,scores=weights(n);
 const rows=keep.map((source,i)=>{const k=retained[i],old=support(scores,k,before,p.seeded,p.capture),next=support(scores,k,after,p.seeded,p.capture);
 return {source_row:source,top_k:k,sticky:old,refreshed:next,same_original_support:JSON.stringify(old.support_token_ids)===JSON.stringify(next.support_token_ids)};});
 return {before,after,fresh_nonempty_flags:fresh,matches_fresh_nonempty_flags:fresh!==null?JSON.stringify(after)===JSON.stringify(fresh):null,
 status:retained.length?"retained":"empty_no_forward",retained_rows:rows.length,support_misses:rows.filter(r=>!r.same_original_support).length,
 rows,native_sampled_tokens:null,native_npu_execution:null};
}
"""


def workbench(r):
    return page("Sampler row-drop routing and retained-support ledger",
        "<h1>Safe stale top-k flags can retain unnecessarily broad routing after rows finish</h1>"
        "<p>Round154 · Original support/plan ledger,not native sampled tokens.Fixedtop_p=1,min_p=0;resolvedtop_ks only.</p>"
        f'<form id="form"><label>Requests,monotonic kept rows and guard metadata<textarea id="input" rows="8" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare sticky and refreshed plans</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Seed/capture guard the general bounded-prefix route.NPUeligibility is not operator/device availability.Empty ScheduleBatch exits without refreshing stale fields."
        "JSuses full ordering to show canonical support,not a partial-sort benchmark.Prefix widths are plan metadata,not allocations/work estimates.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");out.textContent=JSON.stringify(evaluate(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    s=r["summary"];c=r["case"]["outputs"];row=c["rows"][0]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],x["outputs"]["rows"][0]["sticky"]["planned_ranked_prefix_columns"],
        x["outputs"]["rows"][0]["refreshed"]["planned_ranked_prefix_columns"],
        "Confirm native constraints;original support unchanged"))+"</tr>" for x in r["matrix"])
    robustness="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        "Seed guard" if x["inputs"]["seeded"] else "Capture guard" if x["inputs"]["capture"] else "Empty batch" if not x["inputs"]["keep"] else "Reversed request order",
        row["refreshed"]["planned_ranked_prefix_columns"],
        x["outputs"]["rows"][0]["refreshed"]["planned_ranked_prefix_columns"] if x["outputs"]["rows"] else "No forward",
        str(x["outputs"]["rows"][0]["refreshed"]["planned_ranked_prefix_columns"]-row["refreshed"]["planned_ranked_prefix_columns"])+"prefix columns,not score pts" if x["outputs"]["rows"] else "Not applicable"))+"</tr>" for x in r["robustness"])
    mx=max(max(x["bounded_median_ms"],x["full_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>Vocab{x['vocab']},top{x['k']}</b><p>Bounded{x['bounded_median_ms']:.6f}ms</p>{bar(x['bounded_median_ms'],mx)}"
                   f"<p>Full sort{x['full_median_ms']:.6f}ms</p>{bar(x['full_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SAMPLER FLAGS",f"{s['support_misses']}/{s['retained_rows']}original support misses;{s['tightened_plans']}selected plans tighten after row-drop",
         "<div class='grid'><div><h3>Scenarios</h3><p>Greedy/whole vocabulary<br>Bounded20/50/1024<br>Drop finished rows</p></div>"
         "<div><h3>Input</h3><p>Kept request params<br>TOP_K_ALL=2^30<br>Seed/capture guard</p></div>"
         "<div><h3>Decision</h3><p>Refresh5host flags<br>Preserve safe upper bound<br>NPUcapability separate</p><span class='pill'>Native tokens/latency unknown</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14SGLang correction.Original Python{r['latency']['python']}/{r['latency']['machine']}CPUtop20only.</div>"),
        ("SCENARIO: DROP WHOLE VOCAB",f"Retaining top20tightens the original ranked-prefix plan from{row['sticky']['planned_ranked_prefix_columns']}to{row['refreshed']['planned_ranked_prefix_columns']}",
         "<blockquote>Whole-vocabulary request finishes;remaining request still needs top20.Sticky max_top_k=None is safe but prevents a bounded-prefix fast path.</blockquote>"
         "<div class='metric'><span class='pill'>choice · confirm seed/capture/backend guards</span><h3>Read kept request parameters,not an arbitrary tensor snapshot</h3></div>"
         f"<div class='metric'><span class='pill'>score · selected plan columns,not GPUwork</span><h3>Sticky {row['sticky']['planned_ranked_prefix_columns']}</h3>{bar(row['sticky']['planned_ranked_prefix_columns'],row['sticky']['planned_ranked_prefix_columns'])}"
         f"<h3>Refreshed {row['refreshed']['planned_ranked_prefix_columns']}</h3>{bar(row['refreshed']['planned_ranked_prefix_columns'],row['sticky']['planned_ranked_prefix_columns'])}<p>Original support identical.Native sampled tokens remain unknown.</p></div>"),
        ("WHAT I TESTED",f"Six original row-drop cases preserve support while selected-prefix totals fall{s['sticky_prefix_columns']}→{s['refreshed_prefix_columns']}",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK:OLD WIDTH</th><th>MODEL ANSWER:NEW WIDTH</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Safe-but-sticky is a performance scope,not incorrect sampling.</div><div class='callout'>Zero greedy/direct prefix columns mean no ranked plan,not zero compute or memory.</div></div>"),
        ("UNDER THE HOOD","Five host flags follow kept request parameters;general seeded/capture paths retain full ordering",
         "<pre>Source contract:\nScheduleBatch filters nonempty proper subset\n→ SamplingBatchInfo.filter_batch\n→ refresh_top_k_flags(kept request top_k)\nEmpty/full keep exits before refresh</pre><p>↓ Original support verifier</p>"
         "<pre>Compare cached versus fresh flags\nmax_top_k must bound all kept rows\nGeneral prefix fast path only if:\nmax below vocab + unseeded + no capture\nCompare canonical retained token IDs\nNot native softmax,RNG or tie-order execution</pre>"),
        ("ROBUSTNESS CHECK","Seed/capture restores full ordering even when a finite top-k bound exists",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"+robustness+"</table>"
         "<p>Request-order perturbation retains the same logical request.Empty has no forward and no fabricated fresh flags.</p>"),
        ("LATENCY","Measured original CPUtop20selection is not native sampler speed or flag-refresh cost",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Score preparation,metadata refresh,softmax,RNG,SGLang/Torch/GPU/NPUexcluded.</p>"),
        ("MY TAKE","Refresh routing metadata after compaction without weakening determinism,capture or backend guards",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Host batch-flag freshness,safe upper bounds,greedy/direct eligibility and explicit guard comparisons.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Calling old safe flags incorrect tokens,forcing seeded partial sorts,inferring NPUavailability from eligibility or reporting CPUas GPUlatency.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.Stable token-IDties are original policy,not Torch tie guarantees.Native sampled tokens/backend execution/latency unknown.No browser visual inspection.</div>"),
    ]
    return page("Sampler flag refresh — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    s=r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Retained-row sampler routing cabinet</title><desc>Original sixty-solid cabinet metaphor;row compaction refreshes host top-k routing flags while preserving original support.Native SGLang sampled tokens and device latency are not verified.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG154 · SAFE STICKY FLAGS ARE NOT WRONG SAMPLED TOKENS</text>
<text x="88" y="158" font-size="34" font-weight="700">Refresh kept-row flags;keep seed,capture and capability guards</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original support misses{s['support_misses']}/{s['retained_rows']};tightened selected plans{s['tightened_plans']}</text>
<text x="88" y="760" font-size="24">Planned ranked-prefix columns{s['sticky_prefix_columns']}to{s['refreshed_prefix_columns']},not GPUwork or bytes</text>
<text x="88" y="810" font-size="24">General seeded/capture path retains full sort;empty batch does not forward</text>
<text x="88" y="880" font-size="18">ActualNov14SGLang correction · original CPUtop20only,native sampled tokens/GPU/NPU/latency unknown</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
