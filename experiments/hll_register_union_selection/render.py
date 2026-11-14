import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="hll-register-max-union-cardinality-estimator"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==3||!Object.values(p).every(Number.isInteger)||p.precision_bits<4||p.precision_bits>20||p.register_bits<1||p.register_bits>8||p.shards<1||p.shards>1e6||2**p.register_bits-1<65-p.precision_bits)throw Error("Bounded precision/register width/shards with enough rank bits required");
 const n=2**p.precision_bits,b=Math.ceil(n*p.register_bits/8);
 return {selected_register_count:n,selected_dense_register_payload_bytes:b,selected_all_shards_dense_payload_bytes:b*p.shards,ideal_uniform_hash_asymptotic_rse:1.04/Math.sqrt(n),note:"ideal-hash asymptotic standard error,not a per-case bound;metadata/sparse encoding/native allocations excluded"};
}
function simulate(A,B){
 if(![A,B].every(a=>Array.isArray(a)&&a.length<=256&&a.every(v=>Number.isInteger(v)&&v>=0&&v<65536)))throw Error("Up to256 bounded exact item IDs0..65535 per shard required");
 const sketch=a=>{const regs=Array(16).fill(0);for(const id of a){const d=(id*40503+17)%65536,i=d&15;let x=d>>4,rank=1;while(rank<=12&&!(x&1)){rank++;x>>=1;}regs[i]=Math.max(regs[i],rank);}return regs;};
 const estimate=regs=>{const raw=.673*256/regs.reduce((s,v)=>s+2**-v,0),zeros=regs.filter(v=>v===0).length,linear=zeros?16*Math.log(16/zeros):null;return {raw_classic_estimate:raw,empty_registers:zeros,linear_counting_estimate:linear,selected_teaching_estimate:raw<=40&&zeros?linear:raw,selected_mode:raw<=40&&zeros?"linear_counting":"classic_raw"};};
 const a=sketch(A),b=sketch(B),merged=a.map((v,i)=>Math.max(v,b[i]));
 return {A_registers:a,B_registers:b,merged_registers:merged,direct_union_registers:sketch([...A,...B]),true_union_unique:new Set([...A,...B]).size,wrong_sum_exact_counts:new Set(A).size+new Set(B).size,estimate:estimate(merged),note:"Synthetic16bit affine permutation+classic estimator,not native Murmur/Redis Ertl or production error"};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("HLL register union and estimator workbench",
        f"""<h1>Merge register maxima—not local cardinality estimates</h1>
        <p>Round136 · Original16bitdigest permutation,16registers,classic teaching estimator. Not Redis Murmur/Ertl.</p>
        <form id="form"><label>Shard A item IDs JSON<textarea id="A" rows="3" required>{json.dumps(r['case']['A_items'])}</textarea></label>
        <label>Shard B item IDs JSON<textarea id="B" rows="3" required>{json.dumps(r['case']['B_items'])}</textarea></label>
        <h2>Independent dense payload/ideal asymptotic error budget</h2><div class="grid">{fields}</div><button>Merge compatible register states</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Max merge is associative,commutative,idempotent for compatible identity/hash/rank contracts.
        No membership or per-item deletion is encoded. Approximate error is not a guaranteed bound;
        this toy's correlated finite digests do not validate native estimation accuracy.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("A").value),JSON.parse(document.getElementById("B").value)),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["string_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['digests']} precomputed digests</b><p>Bitwise rank {v['bitwise_median_ms']:.6f}ms</p>{bar(v['bitwise_median_ms'],mx)}"
                  f"<p>Binary-string rank {v['string_median_ms']:.6f}ms</p>{bar(v['string_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · DISTINCT COUNTS","Two20-item shards overlap10items;true union30is not summed local count40",
         "<div class='grid'><div><h3>Scenarios</h3><p>Compatible register merge<br>Duplicate delivery<br>Estimator boundaries</p></div>"
         "<div><h3>Input</h3><p>A0..19,B10..29<br>16registertoy<br>Explicit finite digests</p></div>"
         "<div><h3>Decisions</h3><p>Register MAX union<br>Hash/precisioncontract<br>Native estimator separate</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned Redis rank/MAX and Ertl source;original classic toy,not native estimator. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: OVERLAPPING SHARDS","Register union equals direct ingestion while local-count addition double-counts10overlap items",
         f"<blockquote>Teaching merged estimate{c['merged_estimate']['selected_teaching_estimate']:.6f};true union{c['true_union_unique']}. Correlated finite toy,not accuracy validation.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact fixture identities,not confidence</span><h3>True union{c['true_union_unique']}</h3>{bar(c['true_union_unique'],c['wrong_sum_exact_counts'])}<p>Estimate only after compatible MAX union</p></div>"
         f"<div class='metric'><span class='pill'>score · exact fixture identities,not confidence</span><h3>✕ Summed local counts{c['wrong_sum_exact_counts']}</h3>{bar(c['wrong_sum_exact_counts'],c['wrong_sum_exact_counts'],True)}<p>Not a distinct union</p></div>"),
        ("WHAT I TESTED","Max-union is duplicate-idempotent;the sketch cannot tell which individual key was present",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Reversemerge</td><td>Same registers?</td><td>{c['reversed_merge_equal']}</td><td>Commutative</td></tr>"
         f"<tr><td>Repeatedshard</td><td>Same union?</td><td>{c['duplicate_merge_equal']}</td><td>Idempotent</td></tr>"
         "<tr><td>Precision/hash</td><td>Arbitrarymerge?</td><td>✕ Incompatible</td><td>Validate contract</td></tr>"
         "<tr><td>Membership/delete</td><td>Per-key truth?</td><td>✕ Not encoded</td><td>Separate structure</td></tr>"
         "<tr><td>Classicformula</td><td>NativePFCOUNT?</td><td>✕ RedisusesErtl</td><td>No parity claim</td></tr></table>"
         "<div class='grid two'><div class='callout'>Ideal1.04/√m standard error is not an all-case error bound.</div><div class='callout'>Digest quality/native estimator accuracy not measured.</div></div>"),
        ("UNDER THE HOOD","Index bits and rank sentinel determine register meaning before any estimator runs",
         "<pre>Original16bitdigest:\nindex = digest &amp; 15\ntail = digest &gt;&gt; 4\nrank = trailing-zero count(tail)+1\nall-zero tail rank=13\nregister[index] = max(old,rank)</pre>"
         "<p>↓ Original teaching estimator,not native Ertl</p><pre>Union register[i] = max(all shard register[i])\nclassic raw = alpha*m*m / sum(2^-register)\nsmall case:linear count m*ln(m/zero_registers)\nall-empty model→0\n\nNativeRedis:histrogram sigma/tau Ertl estimator\nNo forced numeric parity</pre>"),
        ("ROBUSTNESS CHECK","Reversing or duplicating delivery preserves the exact same compatible register union",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>MAX(A,B)vsMAX(B,A)</td><td>Unionregisters</td><td>{c['reversed_merge_equal']}</td><td>Exact equality</td></tr>"
         f"<tr><td>MAX(A,A,B)</td><td>Unionregisters</td><td>{c['duplicate_merge_equal']}</td><td>Exact equality</td></tr>"
         "<tr><td>Hashfamilychanged</td><td>Contractmatches</td><td>Invalidsemanticmerge</td><td>Not probability points</td></tr></table>"),
        ("LATENCY","Bitwise rank avoids binary-string work only in the original prehashed register-update toy",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Hashing,estimator,merge,Redis/Murmur/Ertl,SIMD,sparse conversion and network excluded.</p>"),
        ("MY TAKE","Use compatible max-register state for approximate distinct unions;keep exact identity tasks elsewhere",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Approximate telemetry cardinality,duplicate-tolerant union and explicit register/estimator contracts.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Exact dedup,membership,per-key deletion,adding PFCOUNT scalars or treating asymptotic error as SLA.</p></div></div>"
         "<div class='dark'>Redis source license documented;original model/oracle/art.Native PFCOUNT/memory/accuracy/latency unknown;browser inspection not performed.</div>"),
    ]
    return page("HLL register union — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Distinct-count ticket cabinet:register maxima before cardinality estimation</title>
<desc>Original sixty-solid sorting cabinet metaphor for compatible rank registers and duplicate-idempotent union.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG136 · HLL REGISTER UNION</text>
<text x="88" y="158" font-size="36" font-weight="700">Two local counts are not a distinct union</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Compatible register MAX union is associative,commutative,idempotent</text>
<text x="88" y="758" font-size="25">Hash/precision/rank contracts matter;membership and deletion are absent</text>
<text x="88" y="806" font-size="25">Original classic estimator is not native Redis Ertl accuracy</text>
<text x="88" y="880" font-size="18">Original finite16bitdigest,register/string oracle,CPU timing · no Murmur/Redis/native estimator reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
