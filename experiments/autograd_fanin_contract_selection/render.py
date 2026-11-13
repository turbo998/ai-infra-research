import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="autograd-implicit-fanin-explicit-precision"
JS=r"""
function estimate(p){
 for(const key of Object.keys(DEFAULT))if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error("positive bounded exact counts required");
 const out={selected_float32_contribution_bytes:p.elements*p.branches*4,selected_float64_accumulator_bytes:p.elements*8,selected_float32_output_bytes:p.elements*4};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
 return out;
}
function f32(x){if(typeof x!=="number"||!Number.isFinite(x)||!Number.isFinite(Math.fround(x)))throw Error("finite float32 input/result required");return Math.fround(x)}
function calculate(values){
 if(values.length<1||values.length>64)throw Error("require 1..64 finite contributions");
 const inputs=values.map(f32);let rounded=0,wide=0;
 for(const x of inputs){rounded=f32(rounded+x);wide+=x}
 return {float32_left_fold:rounded,wide_final_float32:f32(wide),strict_would_reject_second_defined_arrival:inputs.length>1};
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>Backward fan-in is not cross-call .grad accumulation</h1>
    <p>Round 119 · Original scalar task-slot and precision model. No native PyTorch/autograd/GPU execution.</p>
    <form id="form"><label>Contributions to one backward slot (1..64)
    <input id="values" value="16777216,1,-16777216" required></label>
    <div class="grid">{fields}</div><button>Compare original fold policies</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>The inspected guard rejects a second defined arrival; it does not choose precision, prevent all reductions,
    or forbid .grad accumulation across separate backward calls. Logical payloads are not peak HBM.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const parts=document.getElementById("values").value.split(",").map(s=>s.trim());
  if(parts.some(s=>!s))throw Error("empty contribution is invalid");
  const values=parts.map(Number),result=calculate(values);
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  out.textContent=JSON.stringify({result,budget:estimate(p),note:"Float64 is a selected original policy, not implied by the native flag. Same-slot count is illustrative; no full DAG, broadcasting, compiled-autograd or optimizer reproduction."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Autograd fan-in precision contract workbench",body,script)


def cards(r):
    c,rob=r["case"],r["robustness"]
    maximum=max(v["rounded_median_ms"] for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>{v['contributions']} supplied contributions</b>"
                   f"<p>Float64 fold, final float32 {v['wide_median_ms']:.6f} ms</p>{bar(v['wide_median_ms'],maximum)}"
                   f"<p>Struct float32 each add {v['rounded_median_ms']:.6f} ms</p>{bar(v['rounded_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · AUTOGRAD FAN-IN","One backward slot can hide a precision decision at its second gradient arrival",
         "<div class='grid'><div><h3>Scenarios</h3><p>Active fan-in<br>Explicit combiner<br>Separate backward calls</p></div>"
         "<div><h3>Structured input</h3><p>[16777216,1,-16777216]<br>Float32 contributions</p></div>"
         "<div><h3>Decisions</h3><p>Guard actual arrivals<br>Declare accumulation policy</p><span class='pill'>Original CPU scalar model</span></div></div>"
         "<div class='dark'>November 12 PyTorch adds disallow_implicit_grad_accumulation. It guards within-task implicit sums, not cross-call .grad accumulation. Synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: CANCELLATION",f"The original narrow fold returns {c['float32_left_fold']:g}; the explicit wide fold returns {c['explicit_wide_final_float32']:g}",
         f"<blockquote>Exact sum of the supplied encoded inputs: {c['exact_encoded_sum']}. A per-add float32 rounding loses the middle one.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact-order agreement</span><h3>Float32 schedules: {rob['float32_exact_matches']}/{rob['orders']}</h3>{bar(rob['float32_exact_matches'],rob['orders'],True)}</div>"
         f"<div class='metric'><span class='pill'>score · original explicit policy</span><h3>Wide schedules: {rob['wide_exact_matches']}/{rob['orders']}</h3>{bar(rob['wide_exact_matches'],rob['orders'])}</div>"
         "<p>Not a promise of general float64 exactness or native scheduling frequency. The flag does not select this policy.</p>"),
        ("WHAT I TESTED","Undefined, zero and separate-task arrivals have different meanings",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>One active arrival</td><td>Forward fanout alone?</td><td>No second-slot sum</td><td>Act automatically</td></tr>"
         "<tr><td>✕ Second defined arrival</td><td>Strict task enabled</td><td>Explicit rejection</td><td>Inspect producer</td></tr>"
         "<tr><td>Undefined / zero</td><td>Equivalent?</td><td>No; zero is defined</td><td>Retain distinction</td></tr>"
         "<tr><td>Explicit split slots</td><td>One arrival per slot</td><td>Combiner owns sum</td><td>Declare precision</td></tr>"
         f"<tr><td>Separate backward tasks</td><td>Leaf .grad 1 then 2</td><td>{c['separate_tasks_leaf_grad']:g}</td><td>Different contract</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: guard actual traversed backward edges, not every forward use.</div><div class='metric'>Failure analysis: the guard is not a transaction or universal reduction ban.</div></div>"),
        ("UNDER THE HOOD","Give each use a distinct gradient slot, then make the one sum explicit",
         "<pre>original task-slot verifier:\n  undefined arrival: ignore\n  defined arrival + occupied same slot:\n    strict → explicit error\n    otherwise → selected narrow addition\n\nreentrant task:\n  effective strict = local OR parent</pre>"
         "<p>↓ Original explicit model</p><pre>branch 0 → split slot 0\nbranch 1 → split slot 1\nbranch 2 → split slot 2\n\nexplicit float64 sum → one float32 result\none upstream arrival\n\nNo native tensor views or full engine reproduced</pre>"),
        ("ROBUSTNESS CHECK",f"Reordering three contributions changes narrow exact agreement to {rob['float32_exact_matches']}/{rob['orders']}",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>REORDERED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Float32 exact-sum agreement</td><td>{int(c['float32_left_fold']==float(c['exact_encoded_sum']))}/1</td><td>{rob['float32_exact_matches']}/{rob['orders']}</td><td>{100*rob['float32_exact_matches']/rob['orders']:.2f}</td></tr>"
         f"<tr><td>Explicit wide agreement</td><td>1/1</td><td>{rob['wide_exact_matches']}/{rob['orders']}</td><td>{100*rob['wide_exact_matches']/rob['orders']-100:.2f}</td></tr></table>"
         "<p>All six permutations enumerated, not random scheduling probabilities. First-slot state survives the original model's rejection; do not assume rollback or silently retry.</p>"),
        ("LATENCY","The original wide fold avoids per-add packing; this is not native engine speed",
         charts+"<p>Original scalar policies intentionally produce different results. Input float32 conversion included; Python struct packing dominates narrow work. Two warmups, seven alternating repeats, three operations/sample; median repeat means. Native autograd/tensors/views/GPU excluded.</p>"),
        ("MY TAKE","Use the new guard to expose ownership of sums, not to promise no accumulation",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Active backward fan-in audits, explicit gradient-combiner precision and reentrant flag scope.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Cross-call .grad bans, broadcast-reduction detection, general higher-order differentiation, compiled autograd or guaranteed memory reduction.</p></div></div>"
         "<div class='dark'>Pinned BSD-style PyTorch sources; original scalar task/rounding/art. Native engine latency, native precision policy and peak HBM null. Compiled strict mode unsupported in inspected source. Browser inspection not performed.</div>"),
    ]
    return page("Autograd fan-in contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Gradient fan-in ticket cabinet: separate incoming slots before explicit combination</title>
    <desc>Figure-tier original 60-solid magazine sorter maps branch tickets to separate gradient slots and an explicit combining station.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 119 · BACKWARD FAN-IN CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Own the gradient sum; do not infer scope from the flag name</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Second defined arrival in one task is not separate-call .grad accumulation</text>
    <text x="88" y="758" font-size="25">Distinct slots make the combiner responsible for accumulation precision</text>
    <text x="88" y="806" font-size="25">Undefined is not zero; forward fanout is not active backward fan-in</text>
    <text x="88" y="880" font-size="18">Original scalar slot / rounding model · no full DAG engine / native autograd / GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
