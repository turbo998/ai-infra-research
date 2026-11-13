"""Static evidence deck and interactive shape-signature/padding calculator."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "symbolic-shape-export-dimension-contract"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic contracts · independently_reproduced=false · '
                     'Original Python CPU timing separately measured; no JAX compilation/GPU benchmark.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 109 · Symbolic shape contracts</span><span>{n} / 7</span></footer></section>')

    s,b,p = r["solve"],r["budget"],r["proof"]
    card("HANDS-ON · SHAPE-POLYMORPHIC EXPORT",
         f'One export need not mean <span class="accent">one executable</span>: {b["concrete_request_signatures"]} concrete signatures remain.',
         '<div class="grid"><div class="panel"><h2>Shape contract</h2>'
         f'<p>{s["equations"]}</p><p>Actual shapes: {s["actual_shapes"]}</p></div>'
         f'<div class="panel"><h2>Integer bindings</h2><p>{s["solution"]}</p>'
         f'<p>Residuals: {s["residuals"]}</p><span class="pill">No remainder / positive variables</span></div>'
         '<div class="panel"><h2>Signature workbench</h2>'
         f'<p>{b["concrete_request_signatures"]} concrete batch keys<br>{b["bucketed_request_signatures"]} padded batch keys</p>'
         f'<p>Padding adds {b["padding_bytes"]:,} logical bytes.</p></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; November 4, 2026 JAX '
         'symbolic polynomial optimization tracked. Signature counts are declared-key arithmetic, not observed compile events.')
    controls = '<blockquote>Input dimensions 25, 13, 10 must bind 3*b+1, b+c and 2*c using positive integers.</blockquote>'
    zero = sum(value==0 for value in s["residuals"])
    controls += f'<div class="panel"><span class="pill">score · deterministic verifier</span><span class="pill">Confirm</span>'
    controls += f'<h2>b={s["solution"]["b"]}, c={s["solution"]["c"]}</h2><p>Zero residual equations: {zero}/{len(s["residuals"])}. '
    controls += 'Calibrated confidence: null.</p>'+bar(zero,len(s["residuals"]))+'</div>'
    for row in r["negative_controls"][:2]:
        controls += '<div class="panel"><span class="pill">choice</span><span class="pill">Send to a human</span>'
        controls += f'<h2>{row["case"]}: {row["actual_shapes"]}</h2><p>✕ Rejected: {row["reason"]}</p></div>'
    card("SCENARIO: POSITIVE INTEGER DIMENSION BINDING",
         f'Three shape equations recover b={s["solution"]["b"]}, c={s["solution"]["c"]} without truncation.',
         controls,'<b>Never floor an invalid binding into validity.</b> A nonzero remainder, nonpositive variable or '
         'inconsistent repeated dimension is an error. Verifier bars are equation counts, not probabilities.')
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    rows = [("Valid input","Bind positive integers",str(s["solution"]),"Confirm")]
    rows += [(row["case"],str(row["actual_shapes"]),"✕ "+row["reason"],"Reject") for row in r["negative_controls"]]
    rows += [("Coupled equations","Is solver rejection infeasibility?",
              str(r["coupled"]["bounded_oracle_solutions"]),"Not equivalent"),
             ("Symbolic comparison","Does b=8 prove all b?","✕ Inconclusive without bound","Add valid constraint"),
             ("Scopes","Same variable name, different scope","✕ Distinct identity","Reuse intended scope")]
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Runtime binding is not universal symbolic proof.</div>'
    matrix += '<div class="callout">Unsupported equation strategy is not mathematical infeasibility.</div></div>'
    card("WHAT I TESTED","Invalid bindings fail; a solvable coupled system can still exceed solver scope.",
         matrix,'<b>Model boundary:</b> original affine univariate elimination plus finite exhaustive oracle. '
         'No JAX parser, general polynomial solver, StableHLO lowering or native comparison-engine reproduction.')
    card("UNDER THE HOOD","Solve one unknown at a time; export and executable caches remain separate.",
         '<pre>for each shape equation:\n'
         '  substitute solved variables\n'
         '  one unknown: divmod(residual, coefficient)\n'
         '  assert remainder == 0\n'
         '  assert variable &gt;= scope.lower_bound\n'
         '  no unknowns: assert exact consistency\n'
         '  no progress: report unsupported system</pre>'
         '<p>↓ a shared symbolic export does not erase concrete shape keys</p>'
         '<pre>EXPORT KEY: one selected symbolic family\n'
         'EXECUTABLE KEY: shape + dtype + target + layout ...\n'
         f'raw batch keys: {b["input"]["request_batches"]}\n'
         f'padded batch keys: {b["padded_batches"]}\n'
         'compile events / executable bytes: null</pre>',
         '<b>Source contract:</b> shape-polymorphic export traces/lowers once and can compile for many concrete shapes. '
         'This calculator counts one declared dtype/width/target/layout family, not a real compiler cache.')
    robustness = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    robustness += f'<tr><td>{p["expression"]}</td><td>b=1: {p["at_b1"]}</td><td>b=8: {p["at_b8"]}</td>'
    robustness += '<td>Truth depends on b</td></tr>'
    robustness += '<tr><td>Universal proof</td><td>b &gt;= 1: inconclusive</td><td>b &gt;= 3: proved</td><td>Domain narrowed</td></tr>'
    robustness += f'<tr><td>Coupled affine inputs</td><td>Univariate strategy rejects</td><td>{r["coupled"]["bounded_oracle_solutions"]}</td>'
    robustness += '<td>One bounded solution</td></tr>'
    robustness += '<tr><td>Equation order</td><td>Original order</td><td>Reversed order</td><td>Same bindings in tests</td></tr></table>'
    robustness += '<div class="panel"><h2>Scope is identity, not spelling</h2><p>Two separate scopes named b are not '
    robustness += 'interchangeable; use a shared scope intentionally.</p></div>'
    card("ROBUSTNESS CHECK","An example at b=8 cannot prove an inequality for every positive b.",
         robustness,'<b>Unknown is not false.</b> The bounded proof routine explicitly raises Inconclusive. '
         'Adding b &gt;= 3 restricts accepted inputs too; constraints are not performance hints.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: affine binding · Gray: enumerate every bounded candidate environment.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>Bound {row["oracle_variable_bound"]} · '
        timings += f'{row["oracle_candidate_environments"]} oracle candidates</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","The bounded exhaustive oracle scales with candidate environments, not JAX compile cost.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; each sample is mean '
         'of three operations, then median across seven repeat means. Oracle checks the same valid equations. '
         'Not the measured effect of the upstream optimization or a production solver benchmark.')
    card("MY TAKE",
         f'Bucketing {b["concrete_request_signatures"]} keys to {b["bucketed_request_signatures"]} adds {b["padding_bytes"]:,} logical input bytes.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Exporting a known symbolic family.</p>'
         '<p>Exact runtime shape guards.</p><p>Explicit buckets with padding-aware kernels.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>One export called one compiled executable.</p>'
         '<p>Inconclusive treated as false.</p><p>Flooring nonintegral dimensions.</p></div></div>'
         '<pre>READINESS: independent affine contract model\n'
         'SOURCE: pinned JAX, Apache-2.0 rights retained\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'NATIVE JAX / COMPILER / GPU: not measured\n'
         'EXECUTABLE MEMORY / PEAK HBM: unknown\n'
         'PAD MASK SEMANTICS: application-specific, unverified</pre>',
         f'<b>Selected payload:</b> {b["logical_input_bytes"]:,} actual versus {b["padded_logical_input_bytes"]:,} padded bytes. '
         'Padding can change reductions, normalization and attention unless valid-position semantics are preserved.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 109 · Symbolic shape contracts · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  if(!Array.isArray(p.request_batches)||p.request_batches.length<1||p.request_batches.length>512||
     !p.request_batches.every(b=>Number.isSafeInteger(b)&&b>0))
    throw Error("one to 512 positive exact integer request batches required");
  if(!Number.isSafeInteger(p.width)||p.width<1||!Number.isSafeInteger(p.bucket_multiple)||p.bucket_multiple<1)
    throw Error("positive exact integer width and bucket multiple required");
  if(![1,2,4,8].includes(p.value_bytes))throw Error("value_bytes must be 1/2/4/8");
  const m=BigInt(p.bucket_multiple),batches=p.request_batches.map(BigInt);
  const padded=batches.map(b=>((b+m-1n)/m)*m),actualRows=batches.reduce((a,b)=>a+b,0n),
    paddedRows=padded.reduce((a,b)=>a+b,0n),factor=BigInt(p.width)*BigInt(p.value_bytes);
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  const actual=exact(actualRows*factor),allocated=exact(paddedRows*factor);
  return {concrete_request_signatures:new Set(p.request_batches).size,assumed_export_artifact_signatures:1,
    bucketed_request_signatures:new Set(padded.map(String)).size,padded_batches:padded.map(exact),
    actual_batch_rows:exact(actualRows),padded_batch_rows:exact(paddedRows),
    logical_input_bytes:actual,padded_logical_input_bytes:allocated,padding_bytes:allocated-actual,
    cache_policy:"one dtype, width, target and shard layout; count batch signatures only",
    observed_jax_compile_events:null,native_jax_latency_ms:null,executable_bytes:null,peak_hbm_bytes:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const raw=form.elements.request_batches.value.split(",");
    if(raw.some(value=>value.trim()===""))throw Error("request list contains an empty batch");
    const p={request_batches:raw.map(Number)};
    for(const key of ["width","value_bytes","bucket_multiple"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic request batches and declared cache-key policy",
      ...estimateBudget(p),production_approved:false,
      note:"One assumed export artifact is not one executable. Signature counts are not observed compilation events. Padding requires application-correct valid-position semantics. Payloads exclude executable cache, intermediates, allocator and peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    controls = '<label for="request_batches">Batch sizes / 请求 batch 列表<input required id="request_batches" '
    controls += f'name="request_batches" maxlength="4096" value="{",".join(map(str,p["request_batches"]))}"></label>'
    for key,label in (("width","Feature width / 特征维度"),("value_bytes","Value bytes / 值字节"),
                      ("bucket_multiple","Bucket multiple / 补齐倍数")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{p[key]}"></label>'
    return page("Symbolic shapes / 符号 Shape 与 Padding 工作台",
                '<main><div class="eyebrow">ROUND 109 · SHAPE CONTRACTS · SYNTHETIC SIGNATURE POLICY</div>'
                '<h1>One export is not one executable.<br>一次导出，不等于一个编译产物。</h1>'
                '<div class="panel">Symbolic tracing/lowering 与 concrete compilation 不同层。'
                '本计算器固定 dtype/width/target/layout，只统计 batch 签名，不观察 native JAX compile events。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>整数契约：</b>3*b+1=25、b+c=13、2*c=10 解为 b=8,c=5。'
                '除法有余数、非正维度和重复维度矛盾都应报错；未知比较不是 false。'
                '<p>Bucket padding 必须配套正确的有效位置语义，不能假设对 reduction/attention 无影响。'
                '所有字节为 logical input payload，不含 executable cache、allocator 或 peak HBM。</p></div></main>',
                CALCULATOR)


def summary_svg(r):
    b = r["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Symbolic dimension contract gauge</title>'
        '<desc id="description">A detailed solid sorting gauge stands for exact integer shape checks. '
        'Dimension contracts admit shape families while concrete executable signatures remain distinct.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 109 · SYMBOLIC SHAPES · INTEGER CONTRACTS AND COMPILER LAYERS</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">One symbolic export does not erase concrete shape keys.</text>'
        '<text x="64" y="180" font-size="24">A contract-gauge analogy: accept exact bindings, reject remainder and keep scopes distinct.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">3*b+1=25; b+c=13; 2*c=10: b=8, c=5 with exact zero residuals.</text>'
        +f'<text x="80" y="700" font-size="25">Default {b["concrete_request_signatures"]} concrete batch keys become {b["bucketed_request_signatures"]} padded keys, adding {b["padding_bytes"]:,} logical bytes.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">One sampled dimension is not a universal proof; inconclusive is not false.</text>'
        +'<text x="96" y="842" font-size="23">A univariate strategy can reject a feasible coupled system; no native JAX failure is reproduced.</text>'
        +'<text x="96" y="883" font-size="21">Original affine CPU model and declared signature policy only; compiler time and peak HBM unknown.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/symbolic_shape_contract_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
