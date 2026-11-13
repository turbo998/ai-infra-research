import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="float-radix-signed-zero-transformed-bit-range"
JS=r"""
function estimate(p){
 if(!Number.isSafeInteger(p.items)||p.items<1)throw Error("positive bounded exact item count required");
 if(![0,4,8].includes(p.value_bytes))throw Error("value width must be zero/four/eight bytes");
 if(!Number.isSafeInteger(p.begin_bit)||!Number.isSafeInteger(p.end_bit)||p.begin_bit<0||p.begin_bit>=p.end_bit||p.end_bit>32)throw Error("require 0<=begin<end<=32");
 const out={one_pair_buffer_bytes:p.items*(4+p.value_bytes),two_pair_buffers_bytes:2*p.items*(4+p.value_bytes),
 original_byte_passes:Math.ceil((p.end_bit-p.begin_bit)/8)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
 return out;
}
function calculate(hex,begin,end,descending){
 estimate({items:1,value_bytes:0,begin_bit:begin,end_bit:end});
 if(typeof descending!=="boolean")throw Error("explicit Boolean direction required");
 if(hex.length<1||hex.length>64||!hex.every(s=>/^[0-9a-fA-F]{8}$/.test(s)))throw Error("require 1..64 exact eight-digit float32 hex patterns");
 const raw=hex.map(s=>parseInt(s,16));
 const keys=raw.map(bits=>{
  if((bits&0x7f800000)===0x7f800000&&(bits&0x7fffff)!==0)throw Error("original numeric model rejects NaN; no native NaN-order claim");
  if((bits&0x7fffffff)===0)bits=0;
  let key=(bits>=0x80000000?~bits:bits^0x80000000)>>>0;
  if(descending)key=(~key)>>>0;
  return Math.floor(key/2**begin)%2**(end-begin);
 });
 return {selected_keys:keys,order_ids:keys.map((_,i)=>i).sort((a,b)=>keys[a]-keys[b]||a-b)};
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>Numeric float order is not raw uint32 order</h1>
    <p>Round 118 · Original numeric key model; no native CUB/GPU execution.
    Hex input preserves signed zero and subnormal bits; NaNs explicitly rejected.</p>
    <form id="form"><label>Float32 raw hex (comma-separated, 1..64)
    <input id="hex" value="{','.join(r['case']['raw_hex'])}" required></label>
    <label>Direction<select id="direction"><option value="asc">Ascending</option><option value="desc">Descending</option></select></label>
    <div class="grid">{fields}</div><button>Inspect transformed selected keys</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>One/two pair-buffer bytes are selected logical payloads, not queried CUB temporary storage or peak HBM.
    Byte-pass count belongs to our model; native policy/output selector unknown.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  const result=calculate(document.getElementById("hex").value.split(",").map(s=>s.trim()),p.begin_bit,p.end_bit,document.getElementById("direction").value==="desc");
  out.textContent=JSON.stringify({result,budget:estimate(p),note:"Transform first, select bits second. Equal zeros retain input order; partial keys need not give full numeric order. NaN compatibility/native workspace unknown."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Float32 radix key and selected-buffer workbench",body,script)


def cards(r):
    c,rob=r["case"],r["robustness"]
    maximum=max(v["byte_pass_median_ms"] for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>{v['items']:,} precomputed integer keys</b>"
                   f"<p>Original Python byte buckets {v['byte_pass_median_ms']:.6f} ms</p>{bar(v['byte_pass_median_ms'],maximum)}"
                   f"<p>C-backed Python sorted {v['builtin_median_ms']:.6f} ms</p>{bar(v['builtin_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · RADIX CONTRACTS",f"Raw unsigned sorting agrees in only {c['unsigned_agreements']}/{c['items']} output positions",
         "<div class='grid'><div><h3>Scenarios</h3><p>Negative keys<br>Equal signed zeros<br>Transformed bit range</p></div>"
         "<div><h3>Structured input</h3><p>Float32 hex tickets<br>Original stable IDs</p></div>"
         "<div><h3>Decisions</h3><p>Transform first<br>Keep equal-key stability</p><span class='pill'>CPU model only</span></div></div>"
         "<div class='dark'>November 10 source change suppresses ADL, not a new floating-order algorithm. Existing CUB stable-key rules inspected; synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: SIGNED ZERO",f"Canonical zero restores {c['numeric_agreements']}/{c['items']} numeric positions",
         f"<blockquote>Input zero IDs {c['zero_input_ids']}; correct {c['zero_correct_ids']}; uncanonicalized flip {c['zero_wrong_ids']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact position agreement</span><h3>Original transformed numeric key</h3>{bar(c['numeric_agreements'],c['items'])}</div>"
         f"<div class='metric'><span class='pill'>score · negative control</span><h3>No zero canonicalization: {c['uncanonical_zero_agreements']}/{c['items']}</h3>{bar(c['uncanonical_zero_agreements'],c['items'],True)}</div>"
         "<p>Not model-quality scores or calibrated confidence. Preserve original raw payloads and IDs; canonicalization is for comparison keys only.</p>"),
        ("WHAT I TESTED","Stable order has three gates: numeric keys, equal zeros and selected transformed bits",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Numeric key</td><td>Independent float oracle</td><td>{c['numeric_agreements']}/{c['items']}</td><td>Act automatically</td></tr>"
         f"<tr><td>✕ Raw uint32</td><td>Numeric output order</td><td>{c['unsigned_agreements']}/{c['items']}</td><td>Reject proxy</td></tr>"
         "<tr><td>Signed zero</td><td>Preserve equal input order</td><td>Canonical comparison key</td><td>Keep original payload</td></tr>"
         "<tr><td>Partial range</td><td>Truncate before transform?</td><td>✕ Different selected keys</td><td>Confirm range law</td></tr>"
         "<tr><td>NaN</td><td>Numeric oracle valid?</td><td>Original explicit rejection</td><td>Send to a human</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: stable comparison keys do not authorize a new NaN policy.</div><div class='metric'>Failure analysis: reversing an ascending result reverses equal-key IDs.</div></div>"),
        ("UNDER THE HOOD","Type transformation precedes the [begin_bit,end_bit) selection",
         "<pre>original numeric model:\n  reject NaN explicitly\n  signed zeros → one comparison key\n  positive: flip sign bit\n  negative: invert all bits\n  descending: invert transformed key</pre>"
         "<p>↓ Original stable verifier</p><pre>selected = (key >> begin_bit) & mask\n\nfor each low-to-high byte:\n  append IDs to digit buckets in input order\n  concatenate buckets\n\nretain original float bits and IDs\ncompare full-width result to numeric float oracle</pre>"),
        ("ROBUSTNESS CHECK",f"Stable descending passes {rob['descending_numeric_agreements']}/{rob['items']}; reversing ascending does not",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Reverse input, stable numeric oracle</td><td>{c['numeric_agreements']}/{c['items']}</td><td>{rob['reversed_input_numeric_agreements']}/{rob['items']}</td><td>{100*(rob['reversed_input_numeric_agreements']-c['numeric_agreements'])/c['items']:.0f}</td></tr>"
         f"<tr><td>✕ Reverse ascending output</td><td>{rob['descending_numeric_agreements']}/{rob['items']}</td><td>{rob['wrong_reverse_ascending_agreements']}/{rob['items']}</td><td>{100*(rob['wrong_reverse_ascending_agreements']-rob['descending_numeric_agreements'])/rob['items']:.0f}</td></tr></table>"
         "<p>Equal zeros and duplicate 1.5 values must retain their current input order in both directions. Stable sorting is not permutation-invariant tie breaking.</p>"),
        ("LATENCY","The original Python byte-pass model is slower than C-backed sorted in these CPU samples",
         charts+"<p>Precomputed uint32 keys; original Python loops versus C-backed builtin, including implementation differences. Two warmups, seven alternating repeats, two operations/sample; median repeat means. Float conversion, key generation and native CUB/GPU excluded.</p>"),
        ("MY TAKE","Prove sort semantics first; query native workspace and final buffer separately",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Float-key audits, stable token IDs, transformed ranges and explicit buffer accounting.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>NaN-order compatibility, GPU throughput, actual CUB workspace or predicting its buffer selector from our pass count.</p></div></div>"
         "<div class='dark'>Pinned CUB header SPDX BSD-3; original model/art. Native latency, temporary storage, final buffer selector and peak HBM remain null. Query matching native API and read DoubleBuffer.Current(). Browser inspection not performed.</div>"),
    ]
    return page("Stable float radix contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Stable float-key ticket sorter: transformed comparisons retain payload identity</title>
    <desc>Figure-tier original 60-solid magazine sorter: four feet, base, three trays, six rails, 21 cards, 21 tabs and four fasteners.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 118 · STABLE FLOAT RADIX CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Transform comparison keys; preserve original payloads</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Equal signed zeros retain input order, ascending and descending</text>
    <text x="88" y="758" font-size="25">Select bit ranges after the type-specific transformation</text>
    <text x="88" y="806" font-size="25">Two logical buffers are not a queried native workspace size</text>
    <text x="88" y="880" font-size="18">Original byte-pass model · NaNs rejected · no native CUB / GPU / buffer-selector reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
