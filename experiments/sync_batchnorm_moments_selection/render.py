"""Sealed scalar moment evidence, static cards, SVG and exact payload H5."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "sync-batchnorm-count-weighted-moments"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic scalar normalization · independently_reproduced=false · '
                     'Original Python CPU timing separately measured; no native PyTorch or network execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 110 · Count-aware synchronized moments</span><span>{n} / 7</span></footer></section>')

    m,g,b,c = r["moments"],r["gradient"],r["budget"],r["cancellation"]
    card("HANDS-ON · DISTRIBUTED NORMALIZATION",
         f'Counts {m["rank_counts"]} give mean <span class="accent">{m["mean"]:.4f}</span>, not {m["mean_of_rank_means"]:.4f}.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>{m["shards"]}</p><p>{len(m["rank_counts"])} ranks; {m["count"]} observations/channel</p></div>'
         '<div class="panel"><h2>Global estimators</h2>'
         f'<p>Population: {m["population_variance"]:.6f}<br>Unbiased: {m["unbiased_variance"]:.6f}</p>'
         '<span class="pill">Different denominators</span></div>'
         '<div class="panel"><h2>Backward</h2>'
         f'<p>FD max error {g["maximum_dx_finite_difference_error"]:.2e}</p>'
         f'<p>Local-only sums error {g["maximum_wrong_local_dx_error"]:.6f}</p></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; pinned July 2026 native '
         'source and June wrapper migration. Longstanding count-aware mechanism, not a November launch.')
    case = '<blockquote>One channel, four ranks, one empty rank: the global loss is a sum of six upstream-weighted outputs.</blockquote>'
    maximum = max(g["maximum_dx_finite_difference_error"],g["maximum_wrong_local_dx_error"])
    for label,key,action in (("Global backward sums","maximum_dx_finite_difference_error","Confirm"),
                             ("Wrong rank-local sums","maximum_wrong_local_dx_error","Send to a human")):
        case += f'<div class="panel"><span class="pill">score · deterministic max error</span><span class="pill">{action}</span>'
        case += f'<h2>{label}</h2><p>{g[key]:.8g} maximum input-gradient error.</p>'
        case += '<p>Calibrated confidence: null. Lower error is better.</p>'+bar(g[key],maximum)+'</div>'
    card("SCENARIO: GLOBAL MEAN REQUIRES GLOBAL BACKWARD SUMS",
         f'Local-only backward sums miss the correct input gradient by {g["maximum_wrong_local_dx_error"]:.6f}.',
         case,'<b>Different references are labeled:</b> correct derivative versus central finite differences; wrong local-only '
         'derivative versus the correct analytic derivative. Both use the same global forward moments.')
    rows = [("Unequal counts","Mean of rank means",f'✕ {m["mean_of_rank_means"]:.6f} versus {m["mean"]:.6f}',"Reject"),
            ("Variance merge","Count-weight local variances",f'✕ {m["count_weighted_local_variances"]:.6f} versus {m["population_variance"]:.6f}',"Add between-mean term"),
            ("Empty rank","May skip collective?",f'✕ Rank {m["empty_ranks"][0]} still participates',"Keep protocol"),
            ("Forward estimator","Population denominator",f'N={m["count"]}, variance {m["population_variance"]:.6f}',"Confirm"),
            ("Running estimator","Unbiased denominator",f'N-1, variance {m["unbiased_variance"]:.6f}',"Separate"),
            ("Global backward","Independent finite difference",f'{g["maximum_dx_finite_difference_error"]:.2e} max error',"Confirm")]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Count weights alone do not repair mean-of-variances.</div>'
    matrix += '<div class="callout">Empty numerical identity does not remove collective participation.</div></div>'
    card("WHAT I TESTED","Merged M2 needs the between-shard mean correction, not just count weighting.",
         matrix,'<b>Scope:</b> one-channel original scalar model. Native layout, mixed precision, graph capture, '
         'autograd/DDP scaling and actual synchronization require separate tests.')
    card("UNDER THE HOOD","Merge moments with a between-mean term; reduce two global backward sums.",
         '<pre>ORIGINAL SUMMARY: (n, mean, M2)\n'
         'delta = mean_b - mean_a\n'
         'mean = mean_a + delta*n_b/(n_a+n_b)\n'
         'M2 = M2_a + M2_b\n'
         '   + delta**2*n_a*n_b/(n_a+n_b)\n'
         'empty summary: identity, mean undefined</pre>'
         '<p>↓ source wire representation is different from the original M2 triple</p>'
         '<pre>SOURCE FORWARD PACKET: mean[C], invstd[C], count[1]\n'
         'SOURCE BACKWARD REDUCTION: sum(dy), sum(dy*(x-mean))\n'
         'dx = gamma*invstd*(dy - sum_dy/N\n'
         '   - (x-mean)*invstd**2*sum_dy_xmu/N)\n'
         'local parameter gradients: DDP handles synchronization</pre>',
         '<b>No wire-format substitution:</b> the original model transports exact Python counts and M2 conceptually; '
         'it does not reproduce finite-precision invstd reconstruction or native collective buffers.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    robust += f'<tr><td>Population variance</td><td>Exact {c["exact_variance"]}</td><td>Offset {c["offset"]:.0f}</td>'
    robust += f'<td>Merged {c["merged_variance"]}</td></tr>'
    robust += f'<tr><td>Raw E[x²]-E[x]²</td><td>Exact {c["exact_variance"]}</td><td>Large common offset</td>'
    robust += f'<td>✕ {c["raw_second_moment_variance"]}</td></tr>'
    robust += '<tr><td>Merge order</td><td>Original rank order</td><td>All permutations in tests</td><td>Within numeric tolerance</td></tr>'
    robust += '<tr><td>Count representation</td><td>Integer logical count</td><td>Float32 packet / saved int32</td><td>Separate fit checks</td></tr></table>'
    robust += '<div class="panel"><h2>Stable does not mean bitwise associative</h2>'
    robust += '<p>Floating-point merge order can change low bits; compare to declared tolerances, not exact bit equality.</p></div>'
    card("ROBUSTNESS CHECK",
         f'At offset {c["offset"]:.0f}, raw second moments report {c["raw_second_moment_variance"]} instead of variance {c["exact_variance"]}.',
         robust,'<b>Zero here is an observed wrong numeric result, not missing data.</b> The exact rational reference '
         'and original merged moments both give the declared variance. This is not a native float32/GPU stability guarantee.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: merge four precomputed summaries · Gray: two-pass feature rescan.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>{row["observations_per_rank"]} observations/rank · {row["world_size"]} ranks</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","Precomputed summary merge excludes local scanning and all communication.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; ten operations per sample, '
         'then median repeat means. Local summary construction and network/native kernels excluded. No NCCL latency or end-to-end speedup claim.')
    card("MY TAKE",f'A {b["input"]["channels"]}-channel forward packet is {b["forward_packet_bytes_per_rank"]:,} selected bytes per rank.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Unequal-shard global moments.</p>'
         '<p>Explicit empty-rank participation.</p><p>Correct population/unbiased denominators.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Averaging local variance only.</p>'
         '<p>Local-only global-BN derivatives.</p><p>Payload bytes called wire time.</p></div></div>'
         '<pre>READINESS: independent scalar moment/gradient model\n'
         'SOURCE: pinned PyTorch LICENSE; SPDX NOASSERTION\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'CUDA GRAPH EMPTY-RANK PATH: not certified\n'
         'NATIVE COLLECTIVES / DDP / GPU: not executed\n'
         'WIRE BYTES / PEAK HBM: unknown</pre>',
         f'<b>Selected gathered stats buffer:</b> {b["forward_gather_buffer_bytes_per_rank"]:,} bytes per rank; '
         f'backward payload {b["backward_reduction_payload_bytes_per_rank"]:,}. '
         'Buffers exclude other temporaries, network algorithm, transport overhead and allocator state.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 110 · Count-weighted normalization · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  if(!Array.isArray(p.rank_counts)||p.rank_counts.length<1||p.rank_counts.length>512||
     !p.rank_counts.every(n=>Number.isSafeInteger(n)&&n>=0)||!p.rank_counts.some(n=>n>0))
    throw Error("one to 512 nonnegative exact rank counts, not all empty, required");
  if(!Number.isSafeInteger(p.channels)||p.channels<1)throw Error("positive exact integer channel count required");
  if(![4,8].includes(p.stat_bytes)||![1,2,4,8].includes(p.value_bytes))
    throw Error("stat_bytes 4/8 and value_bytes 1/2/4/8 required");
  const c=BigInt(p.channels),w=BigInt(p.rank_counts.length),stat=BigInt(p.stat_bytes),
    total=p.rank_counts.reduce((s,n)=>s+BigInt(n),0n),packet=(2n*c+1n)*stat;
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  return {world_size:p.rank_counts.length,total_observations_per_channel:exact(total),
    nonempty_ranks:p.rank_counts.filter(n=>n>0).length,
    forward_packet_bytes_per_rank:exact(packet),forward_gather_buffer_bytes_per_rank:exact(w*packet),
    backward_reduction_payload_bytes_per_rank:exact(2n*c*stat),
    hypothetical_feature_gather_bytes_per_rank:exact(total*c*BigInt(p.value_bytes)),
    all_packet_counts_exact_in_selected_float:p.stat_bytes===8||p.rank_counts.every(n=>Math.fround(n)===n),
    all_saved_rank_counts_fit_int32:p.rank_counts.every(n=>n<=2147483647),
    native_collective_latency_ms:null,wire_bytes:null,peak_hbm_bytes:null};
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
    for(const key of ["channels","stat_bytes","value_bytes"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic per-channel observation counts and selected transport format",
      ...estimateBudget(p),production_approved:false,
      note:"Counts are observations PER CHANNEL, including spatial positions, not just batch size. Empty ranks still participate. Float-count exactness and saved int32 fit are limited representation checks, not native certification. Selected payloads are not wire bytes, collective time or peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    controls = '<label for="rank_counts">Observations/channel/rank / 每通道各 rank 计数<input required id="rank_counts" '
    controls += f'name="rank_counts" maxlength="4096" value="{",".join(map(str,p["rank_counts"]))}"></label>'
    for key,label in (("channels","Channels / 通道数"),("stat_bytes","Stats bytes 4/8 / 统计传输字节"),
                      ("value_bytes","Feature bytes / 输入值字节")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{p[key]}"></label>'
    return page("Count-aware SyncBN / 跨 Rank 归一化工作台",
                '<main><div class="eyebrow">ROUND 110 · MERGED MOMENTS · SELECTED COLLECTIVE PAYLOADS</div>'
                '<h1>Counts, not ranks, set the mean.<br>跨 rank 均值按 observation 数加权。</h1>'
                '<div class="panel">Count-weighted local variance 仍缺 between-mean correction。'
                'Global forward 必须配套 global backward sums；empty rank 数值上无贡献，不等于可以跳过 collective。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>每通道计数包括 batch 与 spatial positions。'
                'Stats format 固定 4/8 bytes，分别检查 float count 能否精确表示与保存的 int32 count 是否溢出。'
                '<p>原始模型使用 Python M2，不模拟 native invstd 重建、mixed precision、CUDA Graph empty-rank path 或 DDP scaling。'
                'Buffers 不等于 wire bytes、实测 latency 或 peak HBM。</p></div></main>',CALCULATOR)


def summary_svg(r):
    m,b = r["moments"],r["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Count-weighted moment balancing sorter</title>'
        '<desc id="description">A detailed solid sorter balances shard summaries by observation counts, '
        'including an empty participant, without treating rank count as sample count.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 110 · MERGED MOMENTS · COUNT WEIGHTS AND GLOBAL GRADIENTS</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Global moments need counts and between-mean correction.</text>'
        '<text x="64" y="180" font-size="24">A balancing-sorter analogy: an empty station contributes no observations but stays in the protocol.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Counts {m["rank_counts"]}: global mean {m["mean"]:.6f}; mean of rank means {m["mean_of_rank_means"]:.6f} is wrong.</text>'
        +f'<text x="80" y="700" font-size="25">Population variance {m["population_variance"]:.6f}; unbiased variance {m["unbiased_variance"]:.6f}: different estimators.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Global backward reduces sum(dy) and sum(dy*(x-mean)); local-only sums change dx.</text>'
        +f'<text x="96" y="842" font-size="23">Default packet {b["forward_packet_bytes_per_rank"]:,} bytes/rank; gathered stats {b["forward_gather_buffer_bytes_per_rank"]:,}: not wire bytes.</text>'
        +'<text x="96" y="883" font-size="21">Original scalar CPU model only; native CUDA Graph behavior, network timing and peak HBM unknown.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/sync_batchnorm_moments_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
