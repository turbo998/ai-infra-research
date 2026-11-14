import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="shortest-prefill-aging-clock-page-reservation"
JS=r"""
function integer(v,max=1073741824){return Number.isInteger(v)&&v>=0&&v<=max;}
function validate(p){
 if(!p||Object.keys(p).sort().join(",")!=="budget,continuation,events,factor_denominator,factor_numerator,initial_clock,page,policy,requests")throw Error("Exact SPF fields required");
 if(p.policy!=="shortest-prefill-first"||!integer(p.factor_numerator,64)||!integer(p.factor_denominator,1024)||!p.factor_denominator||!integer(p.initial_clock)||!integer(p.budget)||!integer(p.page)||!p.page)throw Error("Known SPFprofile and bounded rational inputs required");
 if(!Array.isArray(p.events)||p.events.length>128||p.events.some(e=>!e||Object.keys(e).sort().join(",")!=="is_extend,tokens"||typeof e.is_extend!=="boolean"||!integer(e.tokens)))throw Error("Explicit launch witnesses required");
 if(!Array.isArray(p.requests)||p.requests.length>128)throw Error("Bounded request ledger required");
 const keys="arrival,deferred,entry,matched,output,prompt,rid,staged";
 for(const r of p.requests)if(!r||Object.keys(r).sort().join(",")!==keys||typeof r.rid!=="string"||!r.rid||typeof r.deferred!=="boolean"||["arrival","entry","matched","output","prompt","staged"].some(k=>!integer(r[k])))throw Error("Exact request fields required");
 if(new Set(p.requests.map(r=>r.rid)).size!==p.requests.length)throw Error("Unique request identities required");
 if(!p.continuation||Object.keys(p.continuation).sort().join(",")!=="arrival,remaining"||!integer(p.continuation.arrival)||!integer(p.continuation.remaining))throw Error("Continuation fields required");
}
function work(r){return Math.max(1,r.prompt+r.output-Math.max(r.matched,r.staged));}
function rank(r,p,n,d){return work(r)*d-n*Math.max(0,p-r.arrival);}
function plan(p,legacy=false,unaged=false){
 validate(p);const clock=p.initial_clock+p.events.reduce((s,e)=>s+(legacy||e.is_extend?e.tokens:0),0),n=unaged?0:p.factor_numerator,d=p.factor_denominator;
 const rows=[...p.requests].sort((a,b)=>Number(a.deferred)-Number(b.deferred)||rank(a,clock,n,d)-rank(b,clock,n,d)||a.entry-b.entry);
 const ranks=rows.map(r=>({rid:r.rid,work:work(r),waited_prefill_tokens:Math.max(0,clock-r.arrival),rank_numerator:rank(r,clock,n,d),rank_denominator:d}));
 const active=p.continuation.remaining*d-n*Math.max(0,clock-p.continuation.arrival);let reserved=0,limit=null;const reservations=[];
 if(p.budget>=2*p.page){for(const r of rows){const charge=Math.ceil(work(r)/p.page)*p.page;
 if(rank(r,clock,n,d)>=active||reserved+charge>p.budget-p.page)break;reserved+=charge;reservations.push({rid:r.rid,work:work(r),page_charge:charge});}
 if(reserved)limit=Math.floor((p.budget-reserved)/p.page)*p.page;}
 return {processed_prefill_clock:clock,ordered_rids:rows.map(r=>r.rid),ranks,continuation_rank_numerator:active,reserved_tokens:reserved,reservations,continuation_limit:limit,native_schedule_verified:null,wall_clock_wait_bound_seconds:null};
}
"""


def workbench(r):
    return page("SPFaging clock/page workbench","<h1>Aging favors older work against new arrivals,not a moving order inside a fixed arrived backlog</h1>"
        "<p>Round161 · Original rational profile.Native factor default0;this example explicitly opts into1/2.No native scheduler or wall-time guarantee.</p>"
        f'<form id="form"><label>Prefill launch witnesses,request cache/arrival ledger and page budget<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare aged,unaged and legacy clocks</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Deferred-prefix flag precedes age;matched/staged cache lengths use max,not sum.Zero reservation is null,not a fabricated limit."
        "All-arrived common-clock shifts cancel from ordering;decode stale counts still must not charge the prefill clock.</p>",
        JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({aged:plan(p),unaged:plan(p,false,true),legacy_clock:plan(p,true)},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    c=r["case"];a=c["aged"];u=c["unaged"];clock=r["clock_case"]
    pg=next(x for x in r["matrix"] if x["name"]=="page_charge")
    charge=pg["outputs"]["reservations"][0]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (x["name"],"rank/clock/page",x["outputs"]["ordered_rids"],"Confirm "+str(x["outputs"]["continuation_limit"])))+"</tr>" for x in r["matrix"])
    matrix+=f"<tr><td>stale decode</td><td>processed-prefill clock</td><td>✕ legacy {clock['legacy']['processed_prefill_clock']} vs expected {clock['correct']['processed_prefill_clock']}</td><td>Correct counter</td></tr>"
    mx=max(max(v["adjusted_median_ms"],v["full_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['requests']}arrived requests</b><p>Adjusted{v['adjusted_median_ms']:.6f}ms</p>{bar(v['adjusted_median_ms'],mx)}"
                  f"<p>Full rank{v['full_median_ms']:.6f}ms</p>{bar(v['full_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · PREFILL FAIRNESS","One common processed-token offset cancels from every already-arrived rank",
         "<div class='grid'><div>Scenarios<br>New arrivals<br>Duplicate deferral<br>Page reservations</div><div>Input<br>Prefill clock<br>Cached work<br>Explicit factor</div><div>Decision<br>Age by processed work<br>Keep deferred gate<br>No wall-time claim</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14opt-in addition.CPUrational sorts onPython{r['latency']['python']}/{r['latency']['machine']},not queue latency.</div>"),
        ("SCENARIO: OLD WORK VS NEW ARRIVAL",f"Original aged order{a['ordered_rids']}reverses unaged{u['ordered_rids']}",
         "<blockquote>rank = work − factor×max(0,processed_prefill−arrival_prefill).Old work can become negative;it is not clipped to0.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · confirm workload assumptions · confidence unmeasured</span>{html.escape(str(a['ranks']))}</div>"
         f"<div class='metric'><span class='pill'>score · counted launch tokens,not time</span><h3>Correct{clock['correct']['processed_prefill_clock']}</h3>{bar(clock['correct']['processed_prefill_clock'],clock['legacy']['processed_prefill_clock'])}"
         f"<h3>Legacy stale decode{clock['legacy']['processed_prefill_clock']}</h3>{bar(clock['legacy']['processed_prefill_clock'],clock['legacy']['processed_prefill_clock'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original rank/page cases plus a stale-counter miss separate fairness and capacity",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Fixed arrived backlog order does not flip merely because a common clock advances.</div><div class='callout'>Aging cannot bypass the earlier duplicate-prefix deferral bit.</div></div>"),
        ("UNDER THE HOOD",f"{charge['work']}uncached tokens charge{charge['page_charge']}at page{pg['inputs']['page']};continuation retains{pg['outputs']['continuation_limit']}of budget{pg['inputs']['budget']}",
         "<pre>work=max(1,total−max(matched,staged))\nkey=(deferred,work−factor×waited,entry)\nOnly native-is_extend launches charge clock\nAging default0;finite nonnegative only</pre><p>↓ Page reservation</p>"
         "<pre>Compare waiting rank with continuation rank\nceil(work/page)×page charge\nStop at first ineligible/unfitting request\nLeave at least one page for continuation\nFloor remaining budget to page multiple\nNo reservation → null</pre>"),
        ("ROBUSTNESS CHECK","Age is processed-prefill progress,not elapsed seconds or a universal starvation bound",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Fixed arrived queue</td><td>Clock1000</td><td>Clock3000</td><td>Order unchanged</td></tr>"
         "<tr><td>Fresh arrival</td><td>Old work1000</td><td>New work10 at clock3000</td><td>Old age can win</td></tr>"
         "<tr><td>Old prefix</td><td>Eligible</td><td>Deferred</td><td>Deferral dominates</td></tr>"
         "<tr><td>Counter</td><td>Extend1000</td><td>Decode stale1000</td><td>Decode adds0</td></tr></table>"),
        ("LATENCY","CPUclock-cancelled sorts require arrived requests and do not measure native queue throughput",timing+
         "<p>2warmups,7alternating repeats,3sorts/sample;median repeat means.Setup,matching/prefetch/admission,GPUs and wall-time waits excluded.</p>"),
        ("MY TAKE","Use opt-in aging with a correct work clock and explicit reservation/fairness limits",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>New-arrival fairness,cache-overlap accounting,prefill-only clocks,page-compatible continuation.</div>"
         "<div class='bad'><h3>⛔ Not for</h3>Wall-time guarantees,duplicate bypass,adding overlapping cache hits,or claiming fixed queues reorder continuously.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Original rational ledger;native default0unchanged.Native scheduler performance,wall-time bound and browser acceptance unknown.</div>")]
    return page("SPFaging — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Prefill work-clock cabinet</title><desc>Original sixty-solid priority cabinet metaphor;age compares old work with new arrivals,not wall time.No native scheduler or visual acceptance.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG161 · WORK CLOCK,NOT WALL CLOCK</text><text x="88" y="158" font-size="34">Common arrived-clock offset cancels;new arrivals change fairness</text>{sorter_scene()}<text x="88" y="710" font-size="24">Deferred-prefix gate precedes aged rank;cached prefixes use max,not sum</text><text x="88" y="760" font-size="24">257tokens charge512at page256;budget4096leaves3584for continuation</text><text x="88" y="810" font-size="24">Only extend launches charge tokens;stale decode metadata adds0</text><text x="88" y="880" font-size="18">ActualNov14opt-in addition · rational oracle,not native queue/throughput or wall-time bound</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
