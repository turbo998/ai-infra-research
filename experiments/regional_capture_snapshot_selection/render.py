import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page, bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "regional-aoti-capture-detach-clone-storage-contract"
JS = r"""
function budget(p){
 if(Object.keys(p).length!==4||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>=0)||p.length<1||![1,2,4,8].includes(p.element_bytes))throw Error("Nonempty exact layout and 1/2/4/8 byte width required");
 const out={logical_bytes:p.length*p.element_bytes,layout_preserving_buffer_bytes:(p.offset+1+(p.length-1)*p.stride)*p.element_bytes};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(p,replacement){
 const bytes=budget(p);if(p.length>64||p.offset>256||p.stride>64||!Number.isFinite(replacement))throw Error("Teaching controls: length<=64, offset<=256, stride<=64 and finite replacement");
 const span=p.offset+1+(p.length-1)*p.stride,backing=Array.from({length:span},(_,i)=>i),before=Array.from({length:p.length},(_,i)=>backing[p.offset+i*p.stride]);
 const frozen=backing.slice();backing[p.offset]=replacement;
 return {before,detached_after:Array.from({length:p.length},(_,i)=>backing[p.offset+i*p.stride]),cloned_after:Array.from({length:p.length},(_,i)=>frozen[p.offset+i*p.stride]),budget:bytes};
}
"""


def workbench(r):
    fields = "".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>'
                     for k,v in r["case"]["layout"].items())
    return page("Regional capture: live storage or frozen input?",
        f"""<h1>Detached storage remains live; a layout snapshot costs more than logical bytes</h1>
        <p>Round124 · Original bounded one-dimensional storage model, not PyTorch execution.</p>
        <form id="form"><div class="grid">{fields}</div><label>New value at the first view storage coordinate
        <input id="replacement" type="number" value="50" step="any" required></label><button>Compare snapshots</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>The original model permits zero stride. Overlapping logical coordinates can change together.
        No native object traversal, autograd, compilation or GPU claim. Native clone rejects distinct input views
        sharing storage and region-state alias conflicts; the original Python tests exercise these bounded guards.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(["offset","length","stride","element_bytes"].map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify(simulate(p,Number(document.getElementById("replacement").value)),null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c = r["case"]
    maximum = max(v["clone_median_ms"] for v in r["latency"]["rows"])
    chart = "".join(f"<div class='metric'><b>{v['logical_elements']} logical elements</b>"
                    f"<p>Live detach {v['detach_median_ms']:.6f} ms</p>{bar(v['detach_median_ms'],maximum)}"
                    f"<p>Prefix clone {v['clone_median_ms']:.6f} ms</p>{bar(v['clone_median_ms'],maximum,True)}</div>"
                    for v in r["latency"]["rows"])
    sections = [
        ("HANDS-ON · REGIONAL CAPTURE", "One storage mutation changes a detached capture; cloning preserves all four original coordinates",
         "<div class='grid'><div><h3>Scenarios</h3><p>Live versus cloned<br>Alias rejection<br>Pre-hook boundary</p></div>"
         "<div><h3>Input</h3><p>Offset3, length4, stride2<br>Selected float32 width<br>Original [3,5,7,9]</p></div>"
         "<div><h3>Decision</h3><p>Live inputs ≠ frozen examples</p><span class='pill'>Native compilation: unmeasured</span></div></div>"
         f"<div class='dark'>Nov14 PyTorch capture change, pinned source. Original CPU: Python {r['latency']['python']} / {r['latency']['machine']}; synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: MUTATED INPUT", "Detached capture preserves 3/4 original values; cloned capture preserves 4/4",
         f"<blockquote>{c['before']} → source coordinate mutation → {c['live_after_mutation']}</blockquote>"
         f"<div class='metric'><span class='pill'>score · deterministic equality, not confidence</span><h3>Live detach {c['detach_preserved_coordinates']}/{c['coordinates']}</h3>{bar(c['detach_preserved_coordinates'],c['coordinates'])}<p>Confirm: intended live observation?</p></div>"
         f"<div class='metric'><span class='pill'>score · deterministic equality, not confidence</span><h3>Cloned layout {c['clone_preserved_coordinates']}/{c['coordinates']}</h3>{bar(c['clone_preserved_coordinates'],c['coordinates'],True)}<p>Act automatically only after alias guards</p></div>"),
        ("WHAT I TESTED", "Repeated tensor identity is supported; distinct same-storage views are rejected by the inspected clone path",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Live detach</td><td>Frozen values?</td><td>✕ Mutation visible</td><td>Confirm</td></tr>"
         "<tr><td>Same View twice</td><td>Repeated identity?</td><td>One memoized leaf</td><td>Preserve</td></tr>"
         "<tr><td>Distinct views, same storage</td><td>Clone alias preserved?</td><td>✕ Rejected</td><td>Explicit error</td></tr>"
         "<tr><td>Input aliases region state</td><td>Clone allowed?</td><td>✕ Rejected</td><td>Explicit error</td></tr>"
         "<tr><td>Unused region</td><td>Compilation failed?</td><td>NOT_REACHED ≠ failed</td><td>Inspect coverage</td></tr></table>"
         "<div class='two grid'><div class='callout'>Only bounded original containers/views modeled. Native arbitrary objects and subclasses not executed.</div><div class='callout'>Reached status None is pending, not compilation success; one eager forward is not full input coverage.</div></div>"),
        ("UNDER THE HOOD", "Capture precedes region pre-hooks so replay applies the original hook once",
         "<pre>original event model:\n capture(args) BEFORE selected +1 hook\n run selected +1 hook\n append every invocation per shared module\n\nleft/right alias one module → one capture hook\nunused module → NOT_REACHED</pre>"
         "<p>↓ Original independent value oracle</p><pre>read backing[offset + i*stride]\nclone copies prefix up to final addressed element\nmemoize same View identity\nreject distinct same-storage Views in clone\n\nnative next-layer contract validation NOT in this pin</pre>"),
        ("ROBUSTNESS CHECK", "Two calls through one aliased module produce two cloned pre-hook snapshots, not duplicate hooks",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Call 1 values</td><td>{c['cloned_event_capture']['left']['calls'][0]}</td><td>{c['live_event_capture']['left']['calls'][0]}</td><td>+2 each, live storage</td></tr>"
         f"<tr><td>Call 2 values</td><td>{c['cloned_event_capture']['left']['calls'][1]}</td><td>{c['live_event_capture']['left']['calls'][1]}</td><td>+1 each, live storage</td></tr></table>"
         f"<p>Points here are integer storage-value changes, not accuracy/probability. Logical payload {c['budget']['logical_bytes']} bytes; offset/stride-preserving prefix {c['budget']['layout_preserving_buffer_bytes']} bytes. Not native peak HBM.</p>"),
        ("LATENCY", "Original live-detach and prefix-clone timings compare different ownership semantics, not compiler throughput",
         chart+"<p>Two warmups, seven alternating repeats, five operations/sample; median repeat means. Includes original finite backing validation. Native hooks/tensors/AOT compilation/GPU excluded.</p>"),
        ("MY TAKE", "Select snapshot ownership and alias policy before trusting regional examples",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Mutation audits, capture boundaries, aliased module call logs and selected layout budgets.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Assuming detach freezes storage, treating pending as compiled, arbitrary-object/native parity or GPU performance.</p></div></div>"
         "<div class='dark'>Pinned BSD-style PyTorch source. Original bounded model and art, no implementation copied. Compiler behavior/GPU latency unmeasured; browser visual inspection not performed.</div>"),
    ]
    return page("Regional capture — seven-card report", "".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Regional capture snapshot archive: live tickets versus frozen copies</title>
<desc>Original sixty-solid ticket sorter analogy. Same-identity memoization, pre-hook records and storage ownership are distinct contracts.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 124 · REGIONAL INPUT CAPTURE</text>
<text x="88" y="158" font-size="36" font-weight="700">A detached ticket still points to live storage</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Capture before pre-hooks; record every call, one hook per shared module</text>
<text x="88" y="758" font-size="25">Clone preserves selected values and layout, not arbitrary alias relationships</text>
<text x="88" y="806" font-size="25">Reached / pending is not compilation success; unused is not failed</text>
<text x="88" y="880" font-size="18">Original bounded storage model · no native compiler / arbitrary-object / GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
