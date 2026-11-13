import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "categorical-topk-topp-request-identity"
JS = r"""
function estimate(p){
 for(const key of Object.keys(DEFAULT)){
  if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" requires positive bounded exact integer");
 }
 if(![4,8].includes(p.index_bytes))throw Error("index/output width must be four/eight bytes");
 const out={selected_unique_probability_bytes:p.unique_rows*p.vocabulary*4,
  selected_repeated_probability_bytes:p.output_rows*p.vocabulary*4,
  selected_index_bytes:p.output_rows*p.index_bytes,selected_output_bytes:p.output_rows*p.index_bytes};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
 return out;
}
function calculate(values,k,pn,pd,un,ud){
 if(values.length<1||values.length>64||!values.every(v=>Number.isSafeInteger(v)&&v>=0))throw Error("select 1..64 bounded nonnegative integer weights");
 if(!Number.isSafeInteger(k)||k<1||k>values.length)throw Error("exact-k must be within vector width");
 if(![pn,pd,un,ud].every(Number.isSafeInteger)||pn<=0||pd<=0||pn>pd||ud<=0||un<0||un>=ud)throw Error("require 0<p<=1 and 0<=u<1 as bounded exact ratios");
 const w=values.map(BigInt),mass=w.reduce((a,b)=>a+b,0n);
 if(mass<=0n)throw Error("positive mass required; no degenerate success fallback");
 const rank=w.map((_,i)=>i).sort((a,b)=>w[a]===w[b]?a-b:w[a]>w[b]?-1:1);
 const top=rank.slice(0,k);
 function nucleus(ids){
  const total=ids.reduce((sum,i)=>sum+w[i],0n);let cumulative=0n;const chosen=[];
  for(const i of ids){chosen.push(i);cumulative+=w[i];if(cumulative*BigInt(pd)>=total*BigInt(pn))break}
  return chosen;
 }
 const jointSet=new Set(nucleus(rank)),sets=[new Set(nucleus(top)),new Set(top.filter(i=>jointSet.has(i)))];
 const gcd=(a,b)=>b?gcd(b,a%b):a;
 function describe(set){
  const total=w.reduce((sum,v,i)=>sum+(set.has(i)?v:0n),0n);
  const distribution=w.map((v,i)=>{if(!set.has(i)||v===0n)return "0";const g=gcd(v,total);return total===g?String(v/g):String(v/g)+"/"+String(total/g)});
  let cumulative=0n,token=-1;
  for(let i=0;i<w.length;i++){if(set.has(i))cumulative+=w[i];if(BigInt(un)*total<cumulative*BigInt(ud)){token=i;break}}
  if(token<0)throw Error("exact CDF did not cover supplied uniform");
  return {distribution,token,support:w.filter((v,i)=>set.has(i)&&v>0n).length};
 }
 return {top_k_first:describe(sets[0]),joint:describe(sets[1])};
}
"""


def workbench(r):
    fields = "".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                     for k,v in r["default_budget"]["inputs"].items())
    controls = {"k":2,"p_num":1,"p_den":2,"u_num":3,"u_den":5}
    ratios = "".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                     for k,v in controls.items())
    body = f"""<h1>Same k/p, different filter law</h1><p>Round 117 · Exact integer-weight/rational-threshold teaching workbench.
    No native sampler, Philox or canonical float-tree reproduction.</p>
    <form id="form"><label>Token weights (1..64 integer entries, token IDs are positions)
    <input id="weights" value="4,3,2,1" required></label><div class="grid">{ratios}</div>
    <h2>Selected float32 probability-row reuse budget</h2><div class="grid">{fields}</div>
    <button>Compare both laws with the same supplied uniform</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Tie policy: original exact-k, descending weight then token ID; not native tie compatibility.
    Probability rows and output identities are different axes. Logical bytes are not peak HBM.</p>"""
    script = f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const pieces=document.getElementById("weights").value.split(",").map(v=>v.trim());
  if(!pieces.every(v=>/^(0|[1-9][0-9]*)$/.test(v)))throw Error("weights require comma-separated nonnegative integers");
  const number=id=>Number(document.getElementById(id).value);
  const laws=calculate(pieces.map(Number),number("k"),number("p_num"),number("p_den"),number("u_num"),number("u_den"));
  const budget=estimate(Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,number(k)])));
  out.textContent=JSON.stringify({laws,budget,note:"Uniform is supplied, not generated. No calibrated confidence, native tie/float/RNG parity, model quality or native allocation claim."},null,2);
 }catch(e){error.textContent=e.message;}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Exact sampling filter/identity workbench",body,script)


def cards(r):
    c,identity = r["case"],r["identity"]
    timings=r["latency"]["rows"];maximum=max(v["linear_median_ms"] for v in timings)
    charts="".join(f"<div class='metric'><b>Vocabulary {v['vocabulary']:,} · 32 supplied draws</b>"
                   f"<p>Binary CDF lookup {v['binary_median_ms']:.6f} ms</p>{bar(v['binary_median_ms'],maximum)}"
                   f"<p>Linear CDF lookup {v['linear_median_ms']:.6f} ms</p>{bar(v['linear_median_ms'],maximum,True)}</div>" for v in timings)
    contract=html.escape(json.dumps({"weights":[4,3,2,1],"k":2,"p":"1/2","uniform":"3/5",
                                    "tie_policy":"original exact-k, stable token ID"},indent=2))
    sections=[
        ("HANDS-ON · CATEGORICAL SAMPLING","The same k=2, p=0.5 selects one token or two, depending on composition",
         "<div class='grid'><div><h3>Scenarios</h3><p>Filter order · exact CDF · request identity</p></div>"
         "<div><h3>Structured input</h3><p>weights 4:3:2:1<br>k=2, p=1/2</p></div>"
         "<div><h3>Top decisions</h3><p>Pin conditional mass.<br>Pin randomness address.</p><span class='pill'>Supplied uniforms, no native RNG</span></div></div>"
         "<div class='dark'>November 13 basic categorical batch-invariant sampler update verified. Existing filtered paths are separately scoped. Synthetic exact model; independently_reproduced=false.</div>"),
        ("SCENARIO: FILTER COMPOSITION",f"Filter composition moves {c['total_variation_percent']:.3f}% total-variation mass",
         "<blockquote>Top-k first renormalizes 4:3; its top probability is 4/7 &gt; 1/2. Joint filtering uses the original nucleus, which needs both 4 and 3.</blockquote>"
         f"<div class='metric'><span class='pill'>choice</span><h3>Top-k first → support one</h3><p>{c['top_k_first']}; supplied u=3/5 → token {c['top_k_first_token']}</p>{bar(1,1)}</div>"
         f"<div class='metric'><span class='pill'>choice</span><h3>Joint → support two</h3><p>{c['joint']}; same u → token {c['joint_token']}</p>{bar(4,7)}{bar(3,7,True)}</div>"
         "<p>Bars are exact distribution mass, not confidence or model-quality scores. Original stable exact-k tie policy is explicit.</p>"),
        ("WHAT I TESTED","Two different contracts govern repeatability: distribution law and random address",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Top-k first</td><td>Conditional nucleus</td><td>[1,0,0,0]</td><td>Declare order</td></tr>"
         "<tr><td>Joint</td><td>Original nucleus ∩ top-k</td><td>[4/7,3/7,0,0]</td><td>Different law</td></tr>"
         "<tr><td>Exact CDF boundary</td><td>u equals previous endpoint</td><td>Next nonzero interval</td><td>Use [start,end)</td></tr>"
         "<tr><td>✕ Invalid zero mass</td><td>Can a valid token be invented?</td><td>Original explicit error</td><td>No fallback</td></tr>"
         "<tr><td>✕ Understated fast-path bound</td><td>k=3 clipped to 2</td><td>Changed distribution</td><td>Validate bound</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: numerical batch invariance does not select your filter law.</div><div class='metric'>Failure analysis: per-row seed arguments are not automatically request-addressed in every path.</div></div>"),
        ("UNDER THE HOOD","Renormalization changes the nucleus denominator before inverse-CDF lookup",
         f"<pre>{contract}</pre><p>↓ Original exact verifier</p><pre>top_k_first:\n  candidate mass = 4+3 = 7\n  first prefix 4/7 >= 1/2 → keep token 0\n\njoint:\n  original mass = 10\n  first prefix 4/10 < 1/2\n  second prefix 7/10 >= 1/2 → keep 0,1\n\nCDF token = first index with u < cumulative mass</pre>"),
        ("ROBUSTNESS CHECK","Rebatching preserves 2/2 outputs only when supplied uniforms follow requests",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>REORDERED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Request-addressed uniform</td><td>2/2</td><td>{identity['request_key_matches']}/2</td><td>0</td></tr>"
         f"<tr><td>✕ Batch-slot uniform</td><td>2/2</td><td>{identity['slot_key_matches']}/2</td><td>−100</td></tr></table>"
         "<p>A→u=1/10; B→u=3/5. Reorder [A,B] to [B,A]. The wrong control reassigns the slot's supplied uniform.</p>"
         "<div class='dark'>This is not a Philox implementation or native float32 canonical-tree test. The inspected new basic sampler explicitly keys by request seed/offset; existing filtered paths have separate contracts.</div>"),
        ("LATENCY","Binary lookup reduces original integer-CDF search work, not GPU sampling cost",
         charts+"<p>Precomputed integer CDF; 32 supplied draws/operation. C-backed stdlib bisect versus Python linear loop: timings include implementation differences. Two warmups, seven alternating repeats, three operations/sample; median repeat means. Filtering, sorting, CDF building, RNG generation and native/GPU kernels excluded.</p>"),
        ("MY TAKE","Pin the probability law and request identity before claiming batch invariance",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Sampling contract audits, filter composition, exact interval tests and explicit request/output identity.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Native RNG/statistical validation, tie compatibility, model quality, canonical float-tree reproduction or GPU speedup.</p></div></div>"
         "<div class='dark'>Apache-2.0 inspected sources; original rational models and CPU lookup timing. Native sampling latency, peak HBM and model quality remain null. Native batch-invariant support is operator/fingerprint-specific. Browser check not performed.</div>"),
    ]
    return page("Sampling law and identity — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Sampling ticket cabinet: preserve filter law and request random address</title>
    <desc>Original static 60-solid magazine sorter as distribution and request-ticket routing cabinet.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 117 · SAMPLING CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Same thresholds do not guarantee the same probability law</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Top-k first: conditional nucleus mass; joint: original nucleus</text>
    <text x="88" y="758" font-size="25">Request identity must carry randomness across batch-slot changes</text>
    <text x="88" y="806" font-size="25">New basic batch-invariant sampler ≠ every filtered path upgraded</text>
    <text x="88" y="880" font-size="18">Original exact model · supplied uniforms · no native RNG / canonical float-tree / GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
