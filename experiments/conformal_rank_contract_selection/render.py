import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="split-conformal-finite-rank-unbounded-boundary"
JS=r"""
function correctedRank(n,p,d){
 if(![n,p,d].every(Number.isSafeInteger)||n<1||p<1||p>=d)throw Error("positive exact n and 0<alpha_num<alpha_den required");
 return Number(((BigInt(n)+1n)*(BigInt(d)-BigInt(p))+BigInt(d)-1n)/BigInt(d));
}
function estimate(p){
 if(!Object.values(p).every(Number.isSafeInteger)||Object.values(p).some(x=>x<1))throw Error("positive bounded exact inputs required");
 const k=correctedRank(p.calibration_count,p.alpha_num,p.alpha_den);
 const out={selected_float64_score_bytes:p.calibration_count*8,selected_optional_sorted_copy_bytes:p.calibration_count*8,
 selected_float64_interval_output_bytes:p.query_count*16,corrected_rank:k,
 minimum_n_for_finite_single_threshold:Number((BigInt(p.alpha_den)-1n)/BigInt(p.alpha_num))};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("selected result exceeds exact arithmetic range");
 return out;
}
function calculate(scores,p,d){
 if(scores.length<1||scores.length>64||!scores.every(s=>Number.isSafeInteger(s)&&s>=0))throw Error("teaching scores require 1..64 bounded nonnegative integers");
 const k=correctedRank(scores.length,p,d);
 return {kind:k>scores.length?"unbounded":"finite",value:k>scores.length?null:String([...scores].sort((a,b)=>a-b)[k-1]),rank:k,calibration_count:scores.length};
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>Finite calibration can require an unbounded interval</h1>
    <p>Round 122 · Original exact rank/holdout model, not native MAPIE or real-data coverage.</p>
    <form id="form"><label>Calibration scores (1..64 nonnegative integers)
    <input id="scores" value="1,2,3,4,5,6,7,8,9" required></label>
    <div class="grid">{fields}</div><button>Compute the corrected order statistic</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Score-vector length determines toy calibration size; budget calibration_count is a separate logical storage assumption.
    Target coverage is marginal under exchangeability/frozen score construction, not confidence about this query or group coverage.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  const parts=document.getElementById("scores").value.split(",").map(s=>s.trim());
  if(parts.some(s=>!/^[0-9]+$/.test(s)))throw Error("nonnegative integer score entries required");
  const bound=calculate(parts.map(Number),p.alpha_num,p.alpha_den);
  out.textContent=JSON.stringify({bound,budget:estimate(p),note:"Unbounded is an explicit mathematical result, not missing data or zero. Do not silently clamp k to n. Native APIs can reject instead; distribution shift/conditional coverage not guaranteed."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Exact conformal rank and unbounded-boundary workbench",body,script)


def cards(r):
    c,small=r["case"],r["small_calibration"]
    maximum=max(max(v["heap_median_ms"],v["sort_median_ms"]) for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>{v['scores']} scores · corrected rank {v['corrected_rank']}</b>"
                   f"<p>Stdlib heap selection {v['heap_median_ms']:.6f} ms</p>{bar(v['heap_median_ms'],maximum)}"
                   f"<p>Full sorted selection {v['sort_median_ms']:.6f} ms</p>{bar(v['sort_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · CONFORMAL CALIBRATION",f"An {c['target_percent']:g}% marginal target needs corrected rank {c['correct']['rank']}, not naive rank {c['naive_rank']}",
         "<div class='grid'><div><h3>Scenarios</h3><p>Finite rank correction<br>Small calibration<br>Tied scores</p></div>"
         "<div><h3>Structured input</h3><p>scores 1..9<br>alpha=3/20</p></div>"
         "<div><h3>Decisions</h3><p>Use n+1 correction<br>Expose unbounded result</p><span class='pill'>Exact synthetic enumeration</span></div></div>"
         "<div class='dark'>Pinned MAPIE quantile helpers inspected. November 10 tree update is educational content, not new conformal theory. Synthetic, independently_reproduced=false; no real-data coverage claim.</div>"),
        ("SCENARIO: FINITE RANK","The corrected synthetic holdout coverage is 9/10; the naive rank gives 8/10",
         "<blockquote>Freeze the score mechanism. Enumerate ten equally likely heldout identities from scores 1..10; calibrate on the other nine each time.</blockquote>"
         f"<div class='metric'><span class='pill'>score · enumerated coverage</span><h3>Correct: {c['exhaustive_correct']['covered']}/{c['exhaustive_correct']['heldouts']}</h3>{bar(c['exhaustive_correct']['covered'],c['exhaustive_correct']['heldouts'])}</div>"
         f"<div class='metric'><span class='pill'>score · negative control</span><h3>Naive n: {c['exhaustive_naive']['covered']}/{c['exhaustive_naive']['heldouts']}</h3>{bar(c['exhaustive_naive']['covered'],c['exhaustive_naive']['heldouts'],True)}</div>"
         "<p>Not a posterior query probability or group-conditional guarantee. The target relies on exchangeability and valid score construction.</p>"),
        ("WHAT I TESTED","Unbounded and insufficient samples must be explicit, not disguised as the maximum score",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>n=9, alpha=3/20</td><td>Correct finite rank</td><td>{c['correct']}</td><td>Act automatically</td></tr>"
         f"<tr><td>n=9, alpha=1/20</td><td>k exceeds n</td><td>{small['bound']['kind']}</td><td>Confirm utility</td></tr>"
         "<tr><td>✕ Clip k to n</td><td>Preserve 95% target?</td><td>No in enumerated world</td><td>Reject fallback</td></tr>"
         "<tr><td>Tied score</td><td>Include boundary?</td><td>Use ≤ threshold</td><td>Keep inclusion law</td></tr>"
         "<tr><td>Distribution shift</td><td>Guaranteed unchanged?</td><td>No</td><td>Send to a human</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: marginal exchangeability is not per-user confidence.</div><div class='metric'>Failure analysis: calibration reuse/model selection can violate the score protocol.</div></div>"),
        ("UNDER THE HOOD","The extra n+1 rank represents the heldout test score in the exchangeable ordering",
         "<pre>original split-conformal threshold:\n  k = ceil((n+1)*(1-alpha))\n  if k > n:\n    kind = unbounded; value = null\n  else:\n    threshold = sorted_scores[k-1]\n\ninclude test when score <= threshold</pre>"
         "<p>↓ Original exhaustive verifier</p><pre>for each heldout identity:\n  calibration = other scores\n  compute corrected threshold\n  test inclusion\n\nfixed score mechanism, all identities counted\nnot real-data or conditional coverage</pre>"),
        ("ROBUSTNESS CHECK","Silently clipping an unbounded 95% threshold leaves only 90% in the synthetic world",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>95% target, k&gt;n</td><td>{small['correct']['coverage_percent']:g}%</td><td>{small['wrong_clip_max']['coverage_percent']:g}%</td><td>{small['wrong_clip_max']['coverage_percent']-small['correct']['coverage_percent']:g}</td></tr>"
         f"<tr><td>Tied scores, ≤ → &lt;</td><td>{r['ties']['inclusive']['coverage_percent']:g}%</td><td>{r['ties']['wrong_strict']['coverage_percent']:g}%</td><td>{r['ties']['wrong_strict']['coverage_percent']-r['ties']['inclusive']['coverage_percent']:g}</td></tr></table>"
         "<p>Unbounded covers all scores but may be operationally useless. Native APIs may explicitly reject insufficient calibration instead. Do not turn either outcome into a finite success-shaped fallback.</p>"),
        ("LATENCY","Two original CPU selection paths return the same corrected finite-rank threshold",
         charts+"<p>Precomputed integer scores; alpha=1/5 for load tiers. Stdlib heapq versus sorted includes implementation differences. Two warmups, seven alternating repeats, three operations/sample; median repeat means. Model/score fitting, inference, native MAPIE and GPU excluded.</p>"),
        ("MY TAKE","A target coverage level needs an exchangeability protocol, not a confidence label",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Frozen-score calibration audits, finite-sample rank boundaries and explicit unbounded/insufficient-data handling.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Per-user conditional guarantees, time-shift immunity, calibration-data tuning or real-data coverage/latency claims.</p></div></div>"
         "<div class='dark'>BSD-3-Clause MAPIE sources and cited conformal theory; original exact model/art. Real-data coverage, conditional coverage, model quality, native latency and peak HBM null. Browser inspection not performed.</div>"),
    ]
    return page("Conformal finite-rank contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Conformal calibration ticket sorter: reserve the test rank before selecting a threshold</title>
    <desc>Figure-tier original 60-solid magazine sorter maps calibration score tickets into finite ranks and an explicit unbounded overflow tray.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 122 · CONFORMAL FINITE-RANK CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Reserve the extra test rank; do not hide an unbounded result</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Corrected rank uses n+1; silently clipping k to n can under-cover</text>
    <text x="88" y="758" font-size="25">Equal scores belong inside the declared ≤ threshold boundary</text>
    <text x="88" y="806" font-size="25">Exchangeable marginal coverage is not per-query confidence</text>
    <text x="88" y="880" font-size="18">Original exact rank / exhaustive holdout model · no real-data / native MAPIE / GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
