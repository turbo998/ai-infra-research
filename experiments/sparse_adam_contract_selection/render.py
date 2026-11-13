import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="sparse-adam-coalescing-mask-step-state"
JS=r"""
function estimate(p){
 for(const key of Object.keys(DEFAULT))if(!Number.isSafeInteger(p[key])||p[key]<0)throw Error("bounded exact nonnegative counts required");
 if(p.rows<1||p.width<1||p.unique_rows>Math.min(p.rows,p.stored_rows))throw Error("positive dimensions and feasible unique/stored counts required");
 const out={selected_float32_parameter_bytes:p.rows*p.width*4,selected_dense_float32_two_moments_bytes:p.rows*p.width*8,
 selected_raw_sparse_gradient_bytes:p.stored_rows*(8+p.width*4),selected_coalesced_sparse_gradient_bytes:p.unique_rows*(8+p.width*4)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
 return out;
}
function calculate(entries,t,eps){
 if(!Number.isSafeInteger(t)||t<1||!Number.isFinite(eps)||eps<0)throw Error("positive exact parameter step and nonnegative finite eps required");
 if(entries.length<1||entries.length>64)throw Error("require 1..64 scalar stored entries");
 const gradients=new Map;
 for(const [row,g] of entries){
  if(!Number.isSafeInteger(row)||row<0||row>=2||!Number.isSafeInteger(g)||Math.abs(g)>2**32-1)throw Error("teaching input needs row 0/1 and bounded integer gradients (absolute value <=2^32-1)");
  gradients.set(row,(gradients.get(row)||0)+g);
 }
 const out=[];
 for(const [row,g] of gradients){
  const m=.5*g,v=.5*g*g,den=Math.sqrt(v)+eps;
  if(!Number.isFinite(g)||!Number.isFinite(den)||den<=0)throw Error("original finite positive update denominator required");
  const delta=.1*Math.sqrt(1-.5**t)/(1-.5**t)*m/den;
  if(!Number.isFinite(delta))throw Error("finite update required");
  out.push({row,coalesced_gradient:g,square:g*g,first_update_delta:delta,weight_after:row+1-delta});
 }
 return out;
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>Sparse gradients do not imply sparse Adam moments</h1>
    <p>Round 120 · Original two-row scalar first-update model. Initial weights [1,2], zero moments,
    lr=0.1, beta1=beta2=0.5. No native PyTorch optimizer/GPU execution.</p>
    <form id="form"><label>Stored row:integer-gradient pairs (1..64, each absolute gradient at most 2^32-1)
    <input id="entries" value="0:1,0:2" required></label>
    <label>Per-parameter step<input id="step" type="number" value="1" step="1" required></label>
    <label>Epsilon before bias rescaling<input id="eps" type="number" value="0.00000001" step="any" required></label>
    <div class="grid">{fields}</div><button>Coalesce before the nonlinear moment</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Budget selects float32 dense table/moments and row-sparse gradient with int64 row indices.
    Raw/unique counts are independent selected storage assumptions, not actual allocations or peak HBM.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const pieces=document.getElementById("entries").value.split(",").map(s=>s.trim().split(":"));
  if(pieces.some(pair=>pair.length!==2||pair.some(s=>!s.trim())))throw Error("entries require row:gradient pairs");
  const result=calculate(pieces.map(pair=>pair.map(Number)),Number(document.getElementById("step").value),Number(document.getElementById("eps").value));
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  out.textContent=JSON.stringify({result,budget:estimate(p),note:"First-update scalar model only. Materialized zeros differ from absent keys with history. Per-parameter step is not per-row age; dense moments remain dense."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("SparseAdam coalescing and dense-state workbench",body,script)


def cards(r):
    c=r["case"];budget=r["default_budget"]["outputs"]
    maximum=max(max(v["hash_median_ms"],v["sorted_median_ms"]) for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>{v['stored_entries']} stored scalar entries</b>"
                   f"<p>Original hash grouping {v['hash_median_ms']:.6f} ms</p>{bar(v['hash_median_ms'],maximum)}"
                   f"<p>Original sorted grouping {v['sorted_median_ms']:.6f} ms</p>{bar(v['sorted_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SPARSE OPTIMIZER","Sparse row gradients leave both full-table Adam moment buffers dense",
         "<div class='grid'><div><h3>Scenarios</h3><p>Duplicate row IDs<br>Absent versus zero<br>Per-parameter step</p></div>"
         "<div><h3>Structured input</h3><p>row 0: +1, +2<br>Coalesced g=3</p></div>"
         f"<div><h3>Selected bytes</h3><p>Moments {budget['selected_dense_float32_two_moments_bytes']:,}<br>Raw gradient {budget['selected_raw_sparse_gradient_bytes']:,}</p><span class='pill'>Logical payload, not HBM</span></div></div>"
         "<div class='dark'>Existing PyTorch SparseAdam contracts inspected at a pinned tree; not a November algorithm launch. Original scalar/CPU model; synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: DUPLICATE ROW",f"Coalesce first: square {c['coalesced_square']}, not sum-of-squares {c['wrong_sum_of_squares']}",
         f"<blockquote>g=1+2=3; the second moment must see 3². Squaring the two stored entries first changes the nonlinear update.</blockquote>"
         f"<div class='metric'><span class='pill'>score · original first-step delta</span><h3>Coalesce then square: {c['correct_first_delta']:.6f}</h3>{bar(c['correct_first_delta'],c['wrong_square_first_delta'])}</div>"
         f"<div class='metric'><span class='pill'>score · negative control</span><h3>Square before coalescing: {c['wrong_square_first_delta']:.6f}</h3>{bar(c['wrong_square_first_delta'],c['wrong_square_first_delta'],True)}</div>"
         "<p>Original lr=0.1, betas=0.5, epsilon=0; not the source defaults or native precision parity. Bars are deltas, not confidence.</p>"),
        ("WHAT I TESTED","Absent row, stored zero and absent gradient tensor are three different events",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Duplicate COO row</td><td>Nonlinear moment input</td><td>Coalesced gradient</td><td>Act automatically</td></tr>"
         "<tr><td>Absent row</td><td>Decay its moments?</td><td>Masked out</td><td>Preserve state</td></tr>"
         "<tr><td>Stored numeric zero</td><td>Same as absent?</td><td>✕ It is materialized</td><td>Retain key</td></tr>"
         "<tr><td>Empty sparse gradient</td><td>Parameter step</td><td>Advances in wrapper</td><td>Do not use row age</td></tr>"
         "<tr><td>No gradient tensor</td><td>Parameter step</td><td>Does not advance</td><td>Different event</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: dense parameters and sparse-layout gradients, not sparse parameters.</div><div class='metric'>Failure analysis: sparse gradients do not compact the two dense moments.</div></div>"),
        ("UNDER THE HOOD","Bias correction uses the parameter step, not the row's first appearance",
         "<pre>original scalar model:\n  coalesce duplicates, keep stored zero keys\n  t = current per-parameter step\n  m = beta1*m + (1-beta1)*g\n  v = beta2*v + (1-beta2)*g*g\n\nscale = lr*sqrt(1-beta2^t)/(1-beta1^t)\ndelta = scale*m/(sqrt(v)+eps)</pre>"
         f"<p>↓ First materialized row at t=2</p><pre>delta = {c['late_row_delta']:.9f}\nwrong row-local t=1 → {c['wrong_row_local_step_delta']:.9f}\n\nfunctional API reads current t\ncaller increments; not the functional kernel</pre>"),
        ("ROBUSTNESS CHECK","Reordering duplicates preserves this exact fixture; dropping a stored zero changes history",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Duplicate gradient</td><td>{r['robustness']['coalesced_values'][0]:g}</td><td>{r['robustness']['coalesced_values'][1]:g}</td><td>0 (value)</td></tr>"
         f"<tr><td>Row weight after t=2</td><td>absent: {c['absent_row_after_second_step']['weights'][1]:.9f}</td><td>stored 0: {c['materialized_zero_after_second_step']['weights'][1]:.9f}</td><td>Different contract</td></tr></table>"
         "<p>Duplicate cancellation retains a materialized zero key. Exact small-integer coalescing is not a guarantee of arbitrary floating-order invariance. No model-quality score.</p>"),
        ("LATENCY","Original coalescing timing excludes optimizer updates and native sparse kernels",
         charts+"<p>Python hash-group/f-sum versus sorted-group/f-sum, both with validation; supplied exact small integer-valued entries. Two warmups, seven alternating repeats, three operations/sample; median repeat means. Native layouts, moments, parameters and GPU excluded.</p>"),
        ("MY TAKE","Choose masked sparse-gradient semantics deliberately, not as dense Adam with fewer bytes",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Dense embeddings producing sparse-layout gradients, explicit duplicate coalescing and mask/step audits.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Dense-Adam-with-zero equivalence, compact optimizer-state claims, arbitrary layouts or native throughput predictions.</p></div></div>"
         "<div class='dark'>BSD-style pinned PyTorch sources; original scalar model and CPU grouping. Native precision/latency and peak HBM not reproduced. Positive-denominator errors are an explicit narrower original policy, not native NaN behavior. Browser inspection not performed.</div>"),
    ]
    return page("SparseAdam contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Sparse-gradient ticket cabinet: coalesce duplicates before updating dense moment drawers</title>
    <desc>Figure-tier original 60-solid magazine sorter maps duplicate gradient tickets into row-specific updates while retaining full moment drawers.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 120 · SPARSE ADAM CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Sparse gradient tickets still update dense moment drawers</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Coalesce duplicates before squaring; stored zero is not an absent row</text>
    <text x="88" y="758" font-size="25">Bias correction follows parameter time, not each row's local age</text>
    <text x="88" y="806" font-size="25">Sparse gradient payload is not compact optimizer-state allocation</text>
    <text x="88" y="880" font-size="18">Original scalar / coalescing model · no native optimizer precision / throughput / HBM reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
