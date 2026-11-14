import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="bloom-negative-gate-layer-budget-safe-deletion"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==4||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||p.expected_items>1e12||p.target_error_ppm>=1e6||p.layers>16||p.growth>16)throw Error("Bounded positive sizing inputs required");
 const layers=[];let capacity=p.expected_items;
 for(let i=0;i<p.layers;i++){if(capacity>1e12)throw Error("Layer capacity exceeds model limit");
 const rate=p.target_error_ppm/1e6/2**(i+1),bits=Math.ceil(-capacity*Math.log(rate)/Math.log(2)**2);
 layers.push({layer:i+1,selected_capacity:capacity,ideal_approximation_bits:bits,selected_byte_rounded_payload:Math.ceil(bits/8),selected_hash_count:Math.max(1,Math.ceil(Math.log(2)*bits/capacity))});capacity*=p.growth;}
 return {layers,selected_total_byte_rounded_payload:layers.reduce((a,b)=>a+b.selected_byte_rounded_payload,0),note:"ideal-hash sizing approximation,not native power-of-two/64-bit rounding,metadata or measured error"};
}
function simulate(inserted,size){
 if(!Number.isInteger(size)||size<1||size>32||!Array.isArray(inserted)||inserted.length>64||!inserted.every(row=>Array.isArray(row)&&row.length===2&&row.every(i=>Number.isInteger(i)&&i>=0&&i<size)))throw Error("1..32bits,up to64 explicit coordinate pairs required");
 const occupied=new Set(inserted.flat()),known=new Set(inserted.map(JSON.stringify)),fp=[];
 let absent=0;for(let a=0;a<size;a++)for(let b=0;b<size;b++){if(known.has(JSON.stringify([a,b])))continue;absent++;if(occupied.has(a)&&occupied.has(b))fp.push([a,b]);}
 return {occupied_positions:[...occupied].sort((a,b)=>a-b),finite_absent_pair_count:absent,finite_false_positive_count:fp.length,finite_true_negative_count:absent-fp.length,finite_false_negative_count:inserted.filter(row=>!row.every(i=>occupied.has(i))).length,false_positive_examples:fp.slice(0,16),note:"Exact finite coordinate-pair universe,not native hash distribution or production FPR"};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Bloom negative gates and layered error budget",
        f"""<h1>A Bloom positive is a candidate—not authoritative membership</h1>
        <p>Round134 · Original explicit coordinate universe,not native Redis/Murmur hashing.</p>
        <form id="form"><label>Inserted coordinate pairs JSON<input id="inserted" value="[[1,5],[5,9]]" required></label>
        <label>Toy bit count<input id="bits" value="16" type="number" min="1" max="32" step="1" required></label>
        <h2>Independent ideal-hash sizing approximation</h2><div class="grid">{fields}</div><button>Enumerate absent-pair candidates</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Only a complete,current,consistent filter makes negative safe as a skip gate.
        Positive must check the authoritative store. Clearing shared bits creates false negatives;
        even counting deletion needs trusted insertion provenance. Ideal sizing is not native allocation.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("inserted").value),Number(document.getElementById("bits").value)),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(max(v["bool_median_ms"],v["bitset_median_ms"]) for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['queries']} two-probe queries</b><p>Packed bytearray {v['bitset_median_ms']:.6f}ms</p>{bar(v['bitset_median_ms'],mx)}"
                  f"<p>Python bool list {v['bool_median_ms']:.6f}ms</p>{bar(v['bool_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · MEMBERSHIP GATES","The16-bit teaching filter accepts7/254absent coordinate-pair tokens as candidates",
         "<div class='grid'><div><h3>Scenarios</h3><p>Negative skip gate<br>Layered error budgets<br>Unsafe deletion</p></div>"
         "<div><h3>Input</h3><p>16bits,2probes<br>Inserted[1,5]/[5,9]<br>Occupied1/5/9</p></div>"
         "<div><h3>Decisions</h3><p>Positive→verify<br>Negative→consistency gate<br>Deletion provenance</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned RedisBloom chain;original coordinate pairs,not native hash. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: FALSE POSITIVE","Uninserted[1,9]passes because other keys set both required bits",
         f"<blockquote>{c['finite_absent_pair_count']}absent tokens in the exact ordered-coordinate universe;not a production hash workload.</blockquote>"
         f"<div class='metric'><span class='pill'>score · absent candidates,not confidence</span><h3>False-positive candidates{c['finite_false_positive_count']}/{c['finite_absent_pair_count']}</h3>{bar(c['finite_false_positive_count'],c['finite_absent_pair_count'])}<p>Confirm in authoritative store</p></div>"
         f"<div class='metric'><span class='pill'>score · absent negatives,not confidence</span><h3>True negatives{c['finite_true_negative_count']}/{c['finite_absent_pair_count']}</h3>{bar(c['finite_true_negative_count'],c['finite_absent_pair_count'],True)}<p>Skip only if complete/current/consistent</p></div>"),
        ("WHAT I TESTED","Clearing A's shared bit makes insertedBnegative;counting deletion also needs trusted membership",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Standardinsert</td><td>Inserted negative?</td><td>0/2infixture</td><td>Preserve bits</td></tr>"
         "<tr><td>Positive</td><td>Exact existence?</td><td>✕ [1,9]notinserted</td><td>Verify store</td></tr>"
         "<tr><td>ClearA</td><td>Bpreserved?</td><td>✕ False</td><td>Do not clear shared bits</td></tr>"
         "<tr><td>CountingdeleteA</td><td>TrustedA,Bpreserved?</td><td>True</td><td>Track insertion</td></tr>"
         "<tr><td>DeletefalsepositiveC</td><td>Counting alwayssafe?</td><td>✕ A/Bbroken</td><td>Reject untrusted deletion</td></tr></table>"
         "<div class='grid two'><div class='callout'>Replica lag/missing writes invalidate a negative skip guarantee.</div><div class='callout'>Native add0may reflect false-positive existence,not exact dedup proof.</div></div>"),
        ("UNDER THE HOOD","Scalable chain starts atp/2and halves each link;finite error sum is bounded without independence",
         "<pre>Original coordinate model:\ninsert:OR selected bit positions\nquery:all selected positions set\nindependent oracle:required set subset of union\n\nNative chain:any link positive→candidate\ncheck all links before adding</pre>"
         f"<p>↓ Selected exact layer-bound arithmetic</p><pre>p=1/100\nlimits:{r['selected_layer_bounds']['selected_layer_error_limits']}\nfinite union bound:{r['selected_layer_bounds']['finite_union_bound']}\nconstant-per-layer bound:{r['selected_layer_bounds']['constant_layer_union_bound']}\nBounds require valid layer assumptions\nNot measured native FPR</pre>"),
        ("ROBUSTNESS CHECK","Trusted counting deletion preservesB;deleting an uninserted false positive corrupts A/Bmembership",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>BafterclearA</td><td>Present</td><td>{c['naive_clear_A_preserves_B']}</td><td>False negative</td></tr>"
         f"<tr><td>BaftertrustedcountdeleteA</td><td>Present</td><td>{c['trusted_counting_delete_A_preserves_B']}</td><td>Preserved</td></tr>"
         f"<tr><td>Aafterdeleteuninserted[1,9]</td><td>Present</td><td>{c['untrusted_false_positive_counting_delete_preserves_A']}</td><td>Not probability points</td></tr></table>"),
        ("LATENCY","Packed bits trade Python probe work for representation density—not automatic lower query latency",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Precomputed coordinates;hashing,construction,layers,Redis/network/database lookup and allocator footprint excluded.</p>"),
        ("MY TAKE","Use Bloom as an authoritative-store negative accelerator,not exact deduplication or deletion evidence",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Cache-miss prefilters,complete index negatives,layer budgets and explicit false-positive verification.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Dropping positive training records,authorization,stale-negative certainty,arbitrary bit clearing or native FPR/SLA claims.</p></div></div>"
         "<div class='dark'>Linked RedisBloom licensing documented;original model/oracle/art.NativeRedis memory/latency/productionFPR unknown;browser inspection not performed.</div>"),
    ]
    return page("Bloom negative gates — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Membership ticket cabinet:negative gates,candidate positives and safe deletion provenance</title>
<desc>Original sixty-solid sorting cabinet metaphor for shared membership bits and authoritative verification.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG134 · BLOOM NEGATIVE GATES</text>
<text x="88" y="158" font-size="36" font-weight="700">A positive candidate is not an authoritative key</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Negative skips require a complete,current,consistent filter</text>
<text x="88" y="758" font-size="25">Layer error budgets compose;native layout differs from ideal sizing</text>
<text x="88" y="806" font-size="25">Shared-bit deletion creates false negatives;counting needs provenance</text>
<text x="88" y="880" font-size="18">Original explicit-coordinate universe and CPU timing · no native Murmur/Redis/database reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
