"""Self-contained set-supervision cards, calculator and original solid figure."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "set-prediction-hungarian-matching-global-normalization"


def deck(r: dict) -> str:
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic costs · independently_reproduced=false · '
                     'Original CPU timings separately measured; no detector AP/FPS.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 102 · One-to-one supervision</span><span>{n} / 7</span></footer></section>')

    a,s = r["assignment"],r["scalar_supervision"]
    card("HANDS-ON · SET PREDICTION",
         'Choosing the <span class="accent">cheapest pair first</span> can cost 81 instead of 4.',
         '<div class="grid"><div class="panel"><h2>Fixture</h2><p>3 queries<br>2 targets<br>One-to-one constraint</p>'
         '<span class="pill">Exact gate: &lt;1e-12</span></div><div class="panel"><h2>Cost matrix</h2>'
         '<pre>[ 1,   2]\n[ 2, 100]\n[80,  80]</pre><p>Original abstract costs, not detector boxes.</p></div>'
         '<div class="panel"><h2>Decision</h2><p>Global optimum: 4</p><p>Greedy pairs: 81</p>'
         '<span class="pill">AP / GPU: null</span></div></div>',
         '<b>Established source contract.</b> RT-DETR uses no-grad weighted class/L1/GIoU costs and CPU '
         'SciPy assignment. Our independent exhaustive and subset-DP references are not Hungarian implementations.')
    scores = '<blockquote>Every target needs one distinct query. Independent argmin selects query 0 twice: '
    scores += 'its apparent cost 3 is infeasible, not an improvement.</blockquote>'
    for key,label,action in (("exact","Exact assignment","Confirm"),("greedy","Greedy pairs","Send to a human")):
        scores += f'<div class="panel"><span class="pill">score · lower is better</span><span class="pill">{action}</span>'
        scores += f'<h2>{label}: {a[key]["cost"]:g}</h2><p>Queries by target: {a[key]["query_for_target"]}; calibrated confidence: null</p>'
        scores += bar(a[key]["cost"],a["greedy"]["cost"])+'</div>'
    card("SCENARIO: GLOBAL VERSUS GREEDY",
         'A valid one-to-one assignment beats greedy cost by 77 on this fixture.',
         scores,'<b>Numerical cost only.</b> This counterexample says nothing about detector AP, model quality '
         'or whether another approximate assignment policy is acceptable for an application.')
    rows = [
        ("Subset DP","Same cost matrix","4","Confirm"),
        ("Greedy pairs","Same one-to-one contract","81 ✕","Review objective"),
        ("Per-target argmin","Distinct queries required","3, duplicate query ✕","Reject infeasible"),
        ("Stale query indices","Same scalar supervision","Loss 41 vs .5 ✕","Rematch identity"),
        ("Rank-local count","Uneven [1,3,0]","Gradient 2.6667 vs 5 ✕","Global count"),
        ("Global count only","DDP already averages","Gradient 1.6667 vs 5 ✕","World compensation"),
        ("N below world size","Pinned clamp contract","Gradient .5, not 2","Preserve policy"),
    ]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</table>'
    matrix += '<div class="grid two"><div class="callout">Match indices identify query and target together.</div>'
    matrix += '<div class="callout">Zero coordinate loss on empty ranks does not remove background classification.</div></div>'
    card("WHAT I TESTED","Correct assignment can still yield wrong gradients with local normalization.",
         matrix,'<b>Separate fixtures:</b> abstract assignment cost, original scalar squared supervision, and '
         'matched-coordinate gradient sums. None reproduces focal/GIoU loss or a detector training run.')
    card("UNDER THE HOOD","DDP averaging turns the source clamp into max(global objects, world size).",
         '<pre># Matching branch: no differentiation through discrete indices\n'
         'cost = weighted(class_cost, L1_cost, -GIoU)\n'
         'indices = one_to_one_assignment(cost.cpu())\n'
         'coordinate_loss = matched_pair_loss(indices)\n\n'
         '# Actual source also classifies unmatched queries as no-object.</pre>'
         '<p>↓ count scaling compensates for gradient averaging</p>'
         '<pre>N = all_reduce_sum(local_object_count)\n'
         'D = max(N / world_size, 1)\n'
         'local_coordinate_loss = local_sum / D\n'
         'ddp_gradient = sum(local_gradient_sum) / (world_size * D)\n'
         '             = total_gradient_sum / max(N, world_size)</pre>',
         '<b>Clamp is intentional source behavior.</b> For N ≥ world size this equals the object mean; '
         'below that boundary it does not. Generic global-mean advice must not erase the clamp.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for label,actual in (("Reverse target list",s["reversed_targets"]["loss"]),
                         ("Reorder queries, rematch",s["fresh_reordered"]["loss"]),
                         ("Reorder queries, stale indices",s["stale_reordered"]["loss"])):
        robust += f'<tr><td>{label}</td><td>Loss {s["baseline"]["loss"]:g}</td><td>Loss {actual:g}</td>'
        robust += f'<td>{actual-s["baseline"]["loss"]:+g}</td></tr>'
    robust += '</table><div class="panel"><h2>Fixed-assignment derivative region</h2>'
    robust += f'<p>All three scalar finite differences: max gradient error {s["finite_difference_error"]:.2e}.</p>'
    robust += '<p>No derivative claim at ties; source matching is no-grad.</p></div>'
    robust += '<div class="panel"><h2>Changing cost is changing the objective</h2>'
    robust += '<p>Replacing -probability with -log(probability) flips the optimal assignment on a sealed '
    robust += 'valid-class-probability fixture. Per-pair monotonicity does not preserve summed assignment order.</p></div>'
    card("ROBUSTNESS CHECK","Permutation invariance needs fresh query identity, not stale index reuse.",
         robust,'<b>Unique optimum in the derivative fixture.</b> Tie-breaking may select different '
         'identities at equal matching cost. No generalized smooth gradient through assignment is claimed.')
    latency = r["latency"]
    maximum = max(max(row["subset_dp_median_ms"],row["enumeration_median_ms"]) for row in latency["rows"])
    timing = '<p>Measured original CPU medians: blue subset DP, gray permutation enumeration.</p>'
    for row in latency["rows"]:
        timing += f'<div class="latency-row"><b>{row["queries"]} queries × {row["targets"]} targets</b>'
        for key,label,baseline in (("subset_dp_median_ms","Subset DP",False),
                                   ("enumeration_median_ms","Enumerate",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","The tiny DP reference is slower at 4×2, faster at 7×5.",
         timing,'<b>Windows ARM64, Python '+latency["environment"]["python"]+'.</b> Two warmups, seven '
         'alternating repeats, including validation. Both are toy exponential references, not native '
         'SciPy Hungarian timing, detector FPS or GPU latency.')
    card("MY TAKE","Matching, loss normalization and classification have different contracts.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>One-to-one identity review.</p><p>Uneven-rank coordinate losses.</p><p>Empty-batch and clamp tests.</p>'
         '</div><div class="panel bad"><h2>⛔ Not for</h2><p>Independent argmin as assignment.</p>'
         '<p>Assuming all criteria share one denominator.</p><p>Calling cost-array bytes peak memory.</p></div></div>'
         '<pre>READINESS: arithmetic and source-contract review\n'
         'UPSTREAM: RT-DETR Apache-2.0; rights retained\n'
         'REFERENCE SOLVERS: original exhaustive + subset DP\n'
         'LATENCY: measured original CPU functions only\n'
         'DETECTOR / GIoU / FOCAL / AP: not reproduced\n'
         'PROVENANCE: synthetic inputs, independently_reproduced=false</pre>',
         '<b>Source rematches auxiliary layers.</b> Denoising uses group-specific indices/counts. '
         'The calculator excludes denoising and counts raw cost arrays, not lifetime-overlapped peak allocation.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 102 · Set-prediction matching · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["batch","queries","targets_per_image","cost_bytes","auxiliary_layers"]){
    const min=["targets_per_image","auxiliary_layers"].includes(key)?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" outside valid integer range");
  }
  if(p.targets_per_image>p.queries||![4,8].includes(p.cost_bytes))
    throw Error("Enough queries and 4/8-byte costs required");
  const block=p.batch*p.queries*p.targets_per_image,flat=p.batch*block,passes=1+p.auxiliary_layers;
  const values=[block,flat,block*p.cost_bytes,flat*p.cost_bytes,flat*p.cost_bytes*passes,passes];
  if(!values.every(Number.isSafeInteger))throw Error("Cost-work counter exceeds exact numerical range");
  return {per_image_block_elements:block,flattened_cross_image_elements:flat,
    per_image_block_bytes:block*p.cost_bytes,flattened_cross_image_bytes:flat*p.cost_bytes,
    main_and_aux_matching_passes:passes,flattened_cost_bytes_written_across_passes:flat*p.cost_bytes*passes,
    peak_memory_bytes:null,gpu_latency_ms:null,detector_ap:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),
  error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const p={...BASE};
    for(const control of form.elements)if(control.name){
      if(control.value.trim()==="")throw Error(control.name+" is required");
      p[control.name]=Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic uniform-target raw cost-array formulas",
      ...estimateBudget(p),production_approved:false,
      note:"Only same-image blocks are matched; cross-image-sized allocation is not cross-image supervision. Written bytes across passes are not peak allocation."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r: dict) -> str:
    p = r["budget"]["input"]
    labels = {"batch":"Images in batch / 批量图像数","queries":"Queries per image / 每图 query 数",
              "targets_per_image":"Targets per image (uniform) / 每图 target 数",
              "cost_bytes":"Cost bytes (4/8) / 分数字节","auxiliary_layers":"Rematched auxiliary layers / 重新匹配辅助层"}
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'min="{0 if key in ("targets_per_image","auxiliary_layers") else 1}" step="1" value="{p[key]}"></label>'
                       for key,label in labels.items())
    return page("Set prediction / 集合监督工作台",
                '<main><div class="eyebrow">ROUND 102 · SET MATCHING · SYNTHETIC ARRAY BUDGET</div>'
                '<h1>One query per target, with global loss semantics.<br>匹配身份与全局分母不能混淆。</h1>'
                '<div class="panel">原创矩阵中全局匹配 cost=4，greedy=81；每个 target 独立 argmin 的 '
                'cost=3 重复占用 query，属于非法解。分数不代表 confidence 或 AP。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>默认 flatten cost 为 B²×Q×T，但 matcher 只提取同图块；'
                '不能误读为跨图匹配。按图构造是独立预算对照，不是已实现优化。'
                '<p>仅均匀 target 数、原始 cost arrays 和 auxiliary 次数；排除 denoising、模型、'
                '临时 cost 分量与 tensor lifetime。不能据此推导 peak HBM、FPS 或原生 Hungarian 延迟。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r: dict) -> str:
    a = r["assignment"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">One-to-one target card sorter</title>'
        '<desc id="description">A solid sorter gives each target one distinct query card. Matching identity '
        'and distributed count normalization are separate contracts.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 102 · SYNTHETIC COSTS · SET SUPERVISION</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A cheap pair is not necessarily a cheap assignment.</text>'
        '<text x="64" y="180" font-size="24">Distinct query identity first; then preserve the selected global-loss normalization.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Original 3-query / 2-target fixture: exact cost {a["exact"]["cost"]:g}; '
        +f'greedy cost {a["greedy"]["cost"]:g}; duplicate-query cost 3 is invalid.</text>'
        +'<text x="80" y="700" font-size="25">DDP source coordinate gradient = total gradient sum / max(global objects, world size).</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Empty matched-coordinate loss does not remove no-object classification supervision.</text>'
        +'<text x="96" y="842" font-size="23">Auxiliary layers rematch; reused indices after query changes can supervise the wrong identity.</text>'
        +'<text x="96" y="883" font-size="21">Original CPU references only; no native Hungarian timing, detector AP/FPS or peak HBM.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/set_prediction_matching_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r: dict) -> dict[Path,str]:
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
