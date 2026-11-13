import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "dlpack-versioned-capsule-strides-stream-lifetime"
JS = r"""
function estimate(p){
  for(const key of Object.keys(DEFAULT)){
    const min=["rows","byte_offset","owner_bytes"].includes(key)?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" requires bounded exact integer >= "+min);
  }
  if(![1,2,4,8].includes(p.element_bytes))throw Error("only byte-addressable scalar widths 1/2/4/8 modeled");
  if(p.row_stride_elements<p.columns)throw Error("select nonoverlapping pitched rows");
  const logical=p.rows*p.columns*p.element_bytes;
  const span=p.rows===0?0:((p.rows-1)*p.row_stride_elements+p.columns)*p.element_bytes;
  const end=p.byte_offset+span;
  if(![logical,span,end].every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
  if(logical&&end>p.owner_bytes)throw Error("descriptor addresses outside selected owner");
  return {logical_payload_bytes:logical,addressed_span_bytes:span,gaps_within_span_bytes:span-logical,
    selected_retained_owner_bytes:p.owner_bytes,selected_compact_copy_payload_bytes:logical,
    first_address:logical?p.byte_offset:null,last_end:logical?end:null};
}
"""


def workbench(r):
    fields = "".join(f'<label>{key}<input id="{key}" type="number" step="1" value="{value}" required></label>'
                     for key,value in r["default_budget"]["inputs"].items())
    body = f"""<h1>Element strides are not byte strides</h1>
    <p>Round 115 · Original descriptor budget, not native DLPack/C ABI/GPU execution.</p>
    <p>Positive nonoverlapping pitched rows; byte offset plus element strides.
    DLPack ≥1.2 requires non-null strides for ndim &gt; 0. Copy payload is not allocator/RSS measurement.</p>
    <form id="form"><div class="grid">{fields}</div><button>Check owner bounds</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>A zero-copy borrow still needs valid lifetime and stream ordering.
    Read-only access is not an immutable snapshot. The recent November 4 change only stamped version 1.4.</p>"""
    script = f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
  if(event)event.preventDefault();
  const error=document.getElementById("error"),out=document.getElementById("result");
  error.textContent="";out.textContent="";
  try{
    const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
    if(!document.getElementById("form").checkValidity())throw Error("Complete all required integer controls");
    out.textContent=JSON.stringify({inputs:p,outputs:estimate(p),
      note:"Selected logical bytes and owner bounds only. No capsule export, GPU synchronization, allocation peak, native dtype/stride support or autograd guarantee."},null,2);
  }catch(e){error.textContent=e.message;}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("DLPack descriptor budget",body,script)


def cards(r):
    case,schedule = r["case"],r["schedule_model"]
    stale_pct = 100*schedule["stale_without_edge"]/schedule["unguarded_count"]
    lifetimes = "".join(f"<tr><td>{v['order']}</td><td>one managed record</td><td>{v['deleter_calls']} callback</td><td>Release once</td></tr>"
                        for v in r["lifetimes"])
    timing = r["latency"]["rows"]
    maximum = max(v["copy_median_ms"] for v in timing)
    charts = "".join(f"<div class='metric'><b>{v['buffer_bytes']:,} bytes</b>"
                     f"<p>Readonly view {v['view_median_ms']:.6f} ms</p>{bar(v['view_median_ms'],maximum)}"
                     f"<p>Immutable copy {v['copy_median_ms']:.6f} ms</p>{bar(v['copy_median_ms'],maximum,True)}</div>"
                     for v in timing)
    contract = html.escape(json.dumps({"shape":case["shape"],"element_strides":case["element_strides"],
                                      "byte_offset":4,"element_bytes":4,"owner_bytes":64},indent=2))
    sections = [
        ("HANDS-ON · INTERCHANGE","Six values need the right strides, not a copied-looking pointer",
         "<div class='grid'><div><h3>Scenarios</h3><p>Strides · one-shot capsule · release · stream edge</p></div>"
         f"<div><h3>Structured input</h3><p>shape {case['shape']}<br>element strides {case['element_strides']}<br>offset 4 bytes</p></div>"
         "<div><h3>Decisions</h3><p>Alias does not mean snapshot.<br>Reuse array, not consumed capsule.</p><span class='pill'>CPU buffer timing only</span></div></div>"
         "<div class='dark'>DLPack 1.4 pin · November 4 version stamp only. Synthetic protocol model, independently_reproduced=false. No native/GPU execution.</div>"),
        ("SCENARIO: ELEMENT STRIDES","Treating element strides as bytes changes five of six values",
         f"<blockquote>{case['expected_values']}</blockquote><div class='metric'><span class='pill'>choice</span><h3>Respect byte offset and element strides</h3><p>{case['correct_values']}</p><p>6/6 fixture values agree</p>{bar(6,6)}</div>"
         f"<div class='metric'><span class='pill'>choice</span><h3>✕ Reinterpret strides as bytes</h3><p>{case['wrong_byte_stride_values']}</p><p>5/6 values wrong</p>{bar(case['wrong_value_count'],6)}</div>"
         "<p>Original little-endian int32 fixture; no calibrated confidence or model-quality score.</p>"),
        ("WHAT I TESTED","Four modeled paths invoke the provided deleter exactly once",
         f"<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>{lifetimes}</table>"
         "<div class='grid'><div class='metric'>Gate: consume one capsule once; fresh array export may make another capsule.</div>"
         "<div class='metric'>Failure analysis: managed-record ownership moves; producer allocation is borrowed, not magically transferred.</div></div>"
         "<p>Original explicit state machine. Major mismatch releases before payload interpretation. This does not execute a C ABI or a real PyCapsule destructor.</p>"),
        ("UNDER THE HOOD","Address = base + byte offset + element size × stride dot index",
         f"<pre>{contract}</pre><p>↓ Original deterministic address verifier</p>"
         "<pre>index (1, 2)\nbyte address = 4 + 4 * (1*4 + 2*1)\n             = 28\nread signed int32 at byte 28: 7\n\nDLPack >=1.2: strides required if ndim > 0\nMajor mismatch: release, do not inspect payload</pre>"
         "<p>The pinned header is stricter than its older null-stride Python prose. Version-aware consumers must distinguish legacy compact layouts.</p>"),
        ("ROBUSTNESS CHECK","Removing one wait edge permits a stale read in one of six schedules",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Enumerated stale schedules</td><td>0 / {schedule['guarded_count']}</td><td>{schedule['stale_without_edge']} / {schedule['unguarded_count']}</td><td>+{stale_pct:.3f}</td></tr>"
         "<tr><td>Capsule/consumer destruction order</td><td>1 callback</td><td>1 callback</td><td>0</td></tr></table>"
         "<p>Producer order write→record and consumer order wait→read are fixed. Guard adds record→wait.</p>"
         "<div class='dark'>Enumeration fraction, not GPU race probability. Fast C Exchange no-sync methods require the producer's current work stream; temporary DLTensor lifetime is limited to the call.</div>"),
        ("LATENCY","A buffer view avoids copying the selected bytes, not DLPack synchronization",
         charts+"<p>Original Python readonly memoryview creation vs immutable bytes copy. Seven alternating repeats, 25 operations/sample, two warmups; median repeat means. Owner creation and content checks excluded. No native DLPack/C ABI/GPU measured.</p>"),
        ("MY TAKE","Zero-copy needs three contracts: description, ownership and execution order",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Version-aware interop audits; explicit byte-address checks; one-shot records; declared stream context.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>GPU throughput claims, autograd bridging, arbitrary sub-byte formats or unverified native consumer support.</p></div></div>"
         "<div class='dark'>Apache-2.0 source, original models/artwork. Read-only is not snapshot isolation. Copies have explicit IS_COPIED semantics. Native interop latency, GPU synchronization, peak memory and quality remain null. Browser visual inspection not performed.</div>"),
    ]
    body = "".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{content}</section>'
                   for i,(label,title,content) in enumerate(sections,1))
    return page("DLPack lifetime/stride/stream contract — seven cards",body)


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>DLPack ticket cabinet: borrow data, transfer one managed record</title>
    <desc>Original static 60-solid magazine sorter analogy; producer, capsule and consumer tickets preserve one release path.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g font-family="system-ui,sans-serif" fill="#0f172a"><text x="88" y="104" font-size="18">FIG 115 · INTERCHANGE CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">A capsule hands off a release ticket, not the producer allocation</text>
    {sorter_scene()}
    <text x="88" y="710" font-size="25">Description: element strides + byte offset + negotiated version</text>
    <text x="88" y="758" font-size="25">Lifetime: one consumed capsule, one eventual managed-record deleter</text>
    <text x="88" y="806" font-size="25">Execution: establish the declared stream ordering before a read</text>
    <text x="88" y="880" font-size="18">Synthetic original analogy · no native ABI/GPU execution · November 4 version stamp is not a new lifetime rule</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
