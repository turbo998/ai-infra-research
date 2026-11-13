"""Self-contained AUROC evidence cards, H5 budgets and original SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "distributed-auroc-ties-threshold-state"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic binary metric inputs · independently_reproduced=false · '
                     'Original Python CPU timing separately measured; no native TorchMetrics/distributed/GPU execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 111 · Distributed AUROC contracts</span><span>{n} / 7</span></footer></section>')

    c,b,e = r["case"],r["budget"],r["endpoint"]
    card("HANDS-ON · DISTRIBUTED EVALUATION",
         f'Local AUROCs {c["local_aucs"]} hide a <span class="accent">{c["exact_auc"]:.2f} global AUROC</span>.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>Rank pairs: {c["shards"]}</p><p>{c["positive_negative_pairs"]} global positive-negative pairs</p></div>'
         '<div class="panel"><h2>Structured verifier</h2>'
         f'<p>Grouped ROC: {c["exact_auc"]}<br>Pair oracle: {c["pairwise_oracle_auc"]}</p>'
         '<span class="pill">Ties receive half credit</span></div>'
         '<div class="panel"><h2>Selected state</h2>'
         f'<p>Raw gather {b["gathered_raw_payload_bytes_per_rank"]:,} B/rank<br>'
         f'Counters {b["fixed_counter_state_bytes_per_rank"]:,} B/rank</p></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; pinned September 2026 '
         'compatibility update. Exact concatenation and threshold-state SUM are longstanding, not new ROC algorithms.')
    deep = '<blockquote>Rank zero scores 0.1 negative / 0.2 positive; rank one scores 0.8 negative / 0.9 positive.</blockquote>'
    maximum = max(c["local_aucs"])
    for label,value,action in (("Global tied-group ROC",c["exact_auc"],"Confirm"),
                               ("Wrong mean of local AUROCs",c["mean_local_auc"],"Send to a human"),
                               ("Coarse threshold approximation",c["coarse_auc"],"Confirm")):
        deep += f'<div class="panel"><span class="pill">score · synthetic AUROC</span><span class="pill">{action}</span>'
        deep += f'<h2>{label}: {value:.3f}</h2><p>Calibrated model confidence: null. These are metric values, not probabilities.</p>'
        deep += bar(value,maximum)+'</div>'
    card("SCENARIO: CROSS-RANK PAIRS CHANGE THE METRIC",
         f'Averaging local AUROCs overstates this global result by {(c["mean_local_auc"]-c["exact_auc"])*100:.0f} points.',
         deep,'<b>Different contracts:</b> exact global ROC includes cross-rank pairs; local AUROC averaging does not. '
         'Fixed-grid approximation is explicitly lossy even when counter merging is exact.')
    rows = [("Cross-rank pairs","Mean local AUROC",f'✕ {c["mean_local_auc"]} versus {c["exact_auc"]}',"Reject"),
            ("Tie grouping","Pairwise half-credit oracle",str(r["ties"]["exact_auc"]),"Confirm"),
            ("Threshold SUM","Same grid on every rank",f'Coarse AUROC {c["coarse_auc"]}',"Exact state merge"),
            ("Refined grid","Eleven thresholds",f'AUROC {c["fine_auc"]}',"Fixture agreement only"),
            ("Score-one endpoint","Origin present?",f'✕ {e["origin_present"]}; binned {e["inspected_binned_rule_auc"]}',"Inspect convention"),
            ("Missing class","Mathematically defined AUROC?","✕ Original model raises explicitly","Do not impute zero")]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Merge raw scores or same-grid counts, not finished AUROCs.</div>'
    matrix += '<div class="callout">Persistent state size does not bound update workspace or error.</div></div>'
    card("WHAT I TESTED","Exact state merging does not make a fixed threshold grid exact.",
         matrix,'<b>Model boundary:</b> binary unweighted finite [0,1] scores. No logits normalization, multiclass/multilabel, '
         'partial AUROC, ignored labels or native missing-class warning-path reproduction.')
    card("UNDER THE HOOD","Raw states concatenate; same-grid 2×2 counters reduce by SUM.",
         '<pre>EXACT: keep scores and integer labels\n'
         'sort descending; group all equal scores\n'
         'ROC begins at (0,0)\n'
         'integrate grouped points\n'
         'independent oracle: P(score_pos &gt; score_neg)\n'
         '                  + 0.5*P(tie)</pre>'
         '<p>↓ threshold equality and endpoint handling are part of the contract</p>'
         '<pre>BINNED: predict positive when score &gt;= threshold\n'
         'state[T,2,2] = int64 TN,FP,FN,TP\n'
         'SUM matching-grid rank states\n'
         'reverse threshold order for ROC\n'
         'inspected binned rule adds no origin automatically\n'
         'source update: vectorized &lt;=50k, else threshold loop</pre>',
         '<b>Source-derived arithmetic, not native execution.</b> Our optional explicit-origin completion is separately '
         'labeled as an original diagnostic variant, not silently substituted into the inspected binned rule.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>{row["count"]} thresholds</td><td>{c["exact_auc"]}</td><td>{row["auc"]}</td><td>{100*row["error"]:+.0f}</td></tr>'
    robust += f'<tr><td>Both scores exactly one</td><td>Exact {e["exact_auc"]}</td>'
    robust += f'<td>Binned {e["inspected_binned_rule_auc"]}</td><td>{100*(e["inspected_binned_rule_auc"]-e["exact_auc"]):+.0f}</td></tr></table>'
    robust += '<div class="panel"><h2>Endpoint diagnostic</h2>'
    robust += f'<p>Inspected-rule points {e["roc_points"]}; original explicit origin completion returns '
    robust += f'{e["original_explicit_origin_completion_auc"]}. This is a declared convention change, not a native fix.</p></div>'
    card("ROBUSTNESS CHECK","Threshold refinement repairs this fixture, not every distribution or endpoint.",
         robust,'<b>No universal error guarantee from threshold count alone.</b> Score ties, distribution and ROC endpoints '
         'matter. The all-one numerical result is a synthetic integration control, not a reproduced native failure.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: rational grouped ROC · Gray: scalar pairwise oracle. Both validate inputs.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>{row["samples"]} samples · {row["positive_negative_pairs"]:,} pair comparisons</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","Pairwise wins at 16/64 samples; rational grouped ROC wins at 256/1024.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; three operations per sample, '
         'median repeat means. Fraction arithmetic and Python loops differ from tensor kernels. No native metric/distributed speedup.')
    card("MY TAKE",
         f'{b["fixed_counter_state_bytes_per_rank"]:,} counter bytes can coexist with a {b["selected_vectorized_int64_matrix_bytes"]:,}-byte update matrix.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Exact global ranking when raw state fits.</p>'
         '<p>Mergeable fixed-grid streaming counts.</p><p>Explicit tie and endpoint tests.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Average of local AUROCs called global.</p>'
         '<p>Fixed state called bounded peak memory.</p><p>Threshold count called an accuracy bound.</p></div></div>'
         '<pre>READINESS: independent binary metric contracts\n'
         'SOURCE: pinned TorchMetrics, Apache-2.0\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'NATIVE / DISTRIBUTED / GPU: not executed\n'
         'MISSING CLASS: explicitly undefined in original model\n'
         'WIRE BYTES / PEAK HBM: unknown</pre>',
         f'<b>Default selected raw payload:</b> {b["gathered_raw_payload_bytes_per_rank"]:,} bytes/rank; '
         f'threshold vector {b["threshold_vector_bytes_per_rank"]:,}. '
         'The int64 matrix is only one possible vectorized temporary, not a total allocation or peak measurement.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 111 · Distributed AUROC contracts · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  if(!Array.isArray(p.rank_counts)||p.rank_counts.length<1||p.rank_counts.length>512||
     !p.rank_counts.every(n=>Number.isSafeInteger(n)&&n>=0)||!p.rank_counts.some(n=>n>0))
    throw Error("one to 512 nonnegative exact rank counts, not all empty, required");
  if(!Number.isSafeInteger(p.threshold_count)||p.threshold_count<2||!Number.isSafeInteger(p.update_samples)||
     p.update_samples<0||p.update_samples>Math.max(...p.rank_counts))
    throw Error("at least two thresholds and update_samples within a declared rank total required");
  if(![2,4,8].includes(p.prediction_bytes)||![1,4,8].includes(p.target_bytes)||![4,8].includes(p.threshold_bytes))
    throw Error("unsupported selected element width");
  const total=p.rank_counts.reduce((s,n)=>s+BigInt(n),0n),t=BigInt(p.threshold_count),
    raw=BigInt(p.prediction_bytes+p.target_bytes);
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  return {world_size:p.rank_counts.length,global_samples:exact(total),
    largest_raw_state_bytes:exact(BigInt(Math.max(...p.rank_counts))*raw),
    gathered_raw_payload_bytes_per_rank:exact(total*raw),fixed_counter_state_bytes_per_rank:exact(t*4n*8n),
    threshold_vector_bytes_per_rank:exact(t*BigInt(p.threshold_bytes)),
    source_update_path:p.update_samples<=50000?"vectorized":"threshold loop",
    selected_vectorized_int64_matrix_bytes:p.update_samples<=50000?exact(BigInt(p.update_samples)*t*8n):null,
    source_counter_bytes:8,wire_bytes:null,peak_hbm_bytes:null,native_metric_latency_ms:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const raw=form.elements.rank_counts.value.split(",");
    if(raw.some(value=>value.trim()===""))throw Error("rank list contains an empty count");
    const p={rank_counts:raw.map(Number)};
    for(const key of ["threshold_count","prediction_bytes","target_bytes","threshold_bytes","update_samples"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic cumulative rank counts and selected element formats",
      ...estimateBudget(p),production_approved:false,
      note:"Counters use the pinned source int64 T*2*2 layout. Fixed persistent state is not bounded update workspace. A selected int64 matrix is not total peak memory. Threshold count alone gives no error bound; missing classes, ties and endpoints need separate checks. Wire bytes and native latency are unknown."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    controls = '<label for="rank_counts">Cumulative samples/rank / 各 rank 累计样本<input required id="rank_counts" '
    controls += f'name="rank_counts" maxlength="4096" value="{",".join(map(str,p["rank_counts"]))}"></label>'
    for key,label in (("threshold_count","Threshold count / 阈值数"),("prediction_bytes","Score bytes / 分数字节"),
                      ("target_bytes","Target bytes / 标签字节"),("threshold_bytes","Threshold bytes / 阈值字节"),
                      ("update_samples","One local update samples / 单次本地 update 样本")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{p[key]}"></label>'
    return page("Distributed AUROC / 跨 Rank ROC 与阈值状态工作台",
                '<main><div class="eyebrow">ROUND 111 · GLOBAL PAIRS · FIXED STATE IS NOT PEAK MEMORY</div>'
                '<h1>Merge scores or counters, not AUROCs.<br>合并全局状态，不平均局部 AUC。</h1>'
                '<div class="panel">两 rank 局部 AUC 都为 1，global AUC 却为 0.75。'
                'Exact state 使用 cat，固定阈值的 int64 2×2 counts 使用 SUM；merge grids 必须一致。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>Counter state 固定不等于 update workspace 固定。'
                '固定源码 update_samples≤50,000 选 vectorized path，否则 threshold loop；'
                '本表只估计一份 N×T int64 matrix，未计其他 temporaries。'
                '<p>原始模型要求概率分数与二值标签，不模拟 native logits/missing-class warning。'
                'Thresholds、ties、endpoint 与 absent class 需另查，不凭阈值数量给质量保证。</p></div></main>',CALCULATOR)


def summary_svg(r):
    c,b = r["case"],r["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Global ranking card sorter</title>'
        '<desc id="description">A detailed solid sorter interleaves score cards from ranks, preserving '
        'ties and cross-rank positive-negative ordering before computing a global metric.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 111 · DISTRIBUTED AUROC · CROSS-RANK PAIRS AND THRESHOLD STATE</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Two local AUROCs of one can hide a global AUROC of 0.75.</text>'
        '<text x="64" y="180" font-size="24">A ranking-card sorter analogy: merge scores or consistent counters before integrating the curve.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Exact global AUC {c["exact_auc"]}; coarse grid {c["coarse_auc"]}; ties require grouping and half credit.</text>'
        +f'<text x="80" y="700" font-size="25">Default counters {b["fixed_counter_state_bytes_per_rank"]:,} bytes/rank; one selected update matrix {b["selected_vectorized_int64_matrix_bytes"]:,} bytes.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Fixed-state SUM is exact for its grid; grid approximation and endpoint conventions remain separate.</text>'
        +'<text x="96" y="842" font-size="23">Same-score-one control exposes a missing origin in the inspected binned rule; no native failure claimed.</text>'
        +'<text x="96" y="883" font-size="21">Original binary metric model only; no native distributed/GPU latency, model quality or peak HBM.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/distributed_auroc_threshold_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
