"""Seven sealed-data cards, a selected-counter H5 and original cabinet SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "exponential-histogram-signed-index-downscale"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic exact rational observations · independently_reproduced=false · '
                     'Original Python CPU timing separately measured; no native OpenTelemetry/collector execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 113 · Exponential histogram contracts</span><span>{n} / 7</span></footer></section>')

    c,b = r["case"],r["budget"]
    card("HANDS-ON · TELEMETRY DISTRIBUTIONS",
         f'A span of {c["original_span"]} buckets fits {c["fitted_span"]} slots while preserving <span class="accent">{c["count"]} observations</span>.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>Exact values {c["observations"]}</p><p>Max logical span {c["max_size"]}</p></div>'
         '<div class="panel"><h2>Structured state</h2><pre>'
         +json.dumps({"scale":c["fitted_scale"],"counts":c["fitted_counts"],"zero":c["zero_count"]},indent=1)
         +'</pre></div><div class="panel"><h2>Top decisions</h2>'
         f'<p>Scale {c["original_scale"]} → {c["fitted_scale"]}</p><p>Aligned merge equal: {c["aligned_merge_equal"]}</p>'
         '<span class="pill">Coarsening is irreversible</span></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; November 9 deduplication '
         'and October 30 circular-storage changes verified. Exponential histogram mathematics is longstanding.')
    deep = '<blockquote>Nearest-rank quantiles return a bucket interval, not an exact hidden observation or calibrated confidence.</blockquote>'
    for row in r["quantiles"]:
        deep += '<div class="panel"><span class="pill">score · nearest-rank lookup</span><span class="pill">Confirm</span>'
        deep += f'<h2>q={row["q"]}: rank {row["rank"]}; exact {row["exact"]}</h2>'
        deep += f'<p>Bucket ({row["lower_exclusive"]}, {row["upper_inclusive"]}]; confidence: null. '
        deep += 'Bar encodes requested rank, not confidence.</p>'+bar(row["rank"],c["count"])+'</div>'
    last = r["quantiles"][-1]
    card("SCENARIO: A COUNT-PRESERVING SKETCH STILL LOSES RESOLUTION",
         f'Exact tail {last["exact"]} lies in ({last["lower_exclusive"]},{last["upper_inclusive"]}] after fitting.',
         deep,'<b>Do not invent a point estimate.</b> Downscaling preserves counts but widens value intervals. '
         'Without separately retained extrema, these counters alone do not certify every SLO decision.')
    rows = [("Exact power of two","Lower exclusive / upper inclusive","1 maps to index -1 at scale 0","Confirm"),
            ("Zero","Positive index or separate count?",f'Separate zero count {c["zero_count"]}',"Confirm"),
            ("Signed downscale","Floor or truncate?",f'✕ Truncation misplaces {c["wrong_containment_violations"]}',"Reject truncation"),
            ("Mixed scales","Align before adding counts?",str(c["aligned_merge_equal"]),"Confirm"),
            ("Finer resolution","Recover from coarse counts?","✕ No inverse reconstruction","Reject upscale"),
            ("Memory","Logical span equals capacity?","✕ Capacity rounds to power of two","Budget selected counters")]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Count preservation is necessary, not sufficient, for correct bucket membership.</div>'
    matrix += '<div class="callout">A negative positive-bucket index is not a negative observation.</div></div>'
    card("WHAT I TESTED","Signed index alignment and value boundaries are independent invariants.",
         matrix,'<b>Model boundary:</b> nonnegative integers/Fractions; scales -10..0, zero threshold zero, '
         'immutable counts. No negative observations, positive scales, float rounding, native atomics or temporality.')
    card("UNDER THE HOOD","Lowering scale groups signed indices by mathematical floor.",
         '<pre>base = 2 ** (2 ** -scale)\n'
         'bucket i means (base**i, base**(i+1)]\n'
         'scale <= 0, exact rational observation:\n'
         'i = (ceil(log2(value)) - 1) // 2**(-scale)\n'
         'zero -> separate counter\n'
         'exact power-of-two boundaries stay upper-inclusive</pre>'
         '<p>↓ no floating logarithm in the original bounded model</p>'
         '<pre>downscale delta levels:\n'
         '    new_i = floor(old_i / 2**delta)\n'
         '    new_count[new_i] += old_count[old_i]\n'
         'merge: align to a common coarser scale, then SUM\n'
         'fit: lower scale until logical span <= max_size\n'
         'never reconstruct finer buckets from coarse counts</pre>',
         '<b>Original arithmetic, not copied SDK code.</b> Native Go uses exponent/fraction mapping and '
         'signed shifts; source merge has locking/mutation requirements not reproduced by immutable Python states.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>{row["name"]}; scale {row["scale"]}</td><td>{c["count"]}/{c["count"]} contained</td>'
        robust += f'<td>{row["containment_percent"]:.0f}%; {row["count"]} counts</td><td>{row["change_points"]:+.0f}</td></tr>'
    robust += '</table><div class="panel"><h2>Same count, wrong geometry</h2>'
    robust += f'<p>Wrong truncation retains {c["wrong_count"]} counts yet violates '
    robust += f'{c["wrong_containment_violations"]} observation intervals. Count-only tests would miss this.</p></div>'
    card("ROBUSTNESS CHECK",
         f'Both mappings keep {c["count"]} counts; only floor keeps every observation in its interval.',
         robust,'<b>Boundary oracle:</b> exact Fraction comparisons check lower &lt; value ≤ upper. '
         'Repartitioned scales and empty identities are tested at an explicit common target scale.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: precomputed state alignment/merge · Gray: raw exact-observation replay. Identical final state.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>{row["shards"]} shards × {row["samples_per_shard"]} observations · {row["total_observations"]:,} total</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","Eight precomputed states avoid replaying every historical observation.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; three operations per sample, '
         'median repeat means. Initial per-shard histograms excluded; not native ingestion/collector/serialization throughput.')
    card("MY TAKE",
         f'Logical positive span 160 needs {b["positive_fresh_capacity"]} selected fresh circular slots.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Count-conserving scale alignment.</p>'
         '<p>Explicit boundary and rank-interval tests.</p><p>Selected per-stream capacity accounting.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Truncating signed indices toward zero.</p>'
         '<p>Counts alone called exact tail values.</p><p>Fresh counter payload called total retained memory.</p></div></div>'
         '<pre>READINESS: original exact nonnegative-rational model\n'
         'SOURCE: pinned OpenTelemetry Go, Apache-2.0\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'NATIVE SDK / CONCURRENCY / OTLP: not executed\n'
         'FRESH COUNTERS: selected power-of-two capacity only\n'
         'RETAINED CAPACITY / TOTAL MEMORY / WIRE: unknown</pre>',
         f'<b>Default selected counters:</b> {b["selected_fresh_counter_bytes_per_state"]:,} bytes/state, '
         f'{b["selected_live_counter_payload_bytes"]:,} across selected live states. '
         'Old capacity may survive downscaling/clear; attributes, sum/extrema, exemplars, maps and snapshots excluded.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 113 · Exponential histogram contracts · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["streams","live_states_per_stream","max_size","positive_span","negative_span"]){
    const min=(key==="positive_span"||key==="negative_span")?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" requires an exact integer >= "+min);
  }
  if(Math.max(p.positive_span,p.negative_span)>p.max_size)throw Error("logical span exceeds selected max_size");
  const capacity=n=>{if(n===0)return 0n;let c=1n;while(c<BigInt(n))c<<=1n;return c;};
  const pos=capacity(p.positive_span),neg=capacity(p.negative_span),per=(pos+neg+1n)*8n;
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  return {positive_fresh_capacity:exact(pos),negative_fresh_capacity:exact(neg),
    selected_logical_counter_bytes_per_state:exact((BigInt(p.positive_span)+BigInt(p.negative_span)+1n)*8n),
    selected_fresh_counter_bytes_per_state:exact(per),
    selected_live_counter_payload_bytes:exact(per*BigInt(p.streams)*BigInt(p.live_states_per_stream)),
    native_retained_capacity_bytes:null,total_sdk_memory_bytes:null,wire_bytes:null,native_collection_latency_ms:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const p={};
    for(const key of ["streams","live_states_per_stream","max_size","positive_span","negative_span"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic spans and selected fresh uint64 circular counter capacity",
      ...estimateBudget(p),production_approved:false,
      note:"Span includes zero-count gaps. Power-of-two fresh capacities are not retained capacity after history. One zero counter included per selected live state; attributes, sum/min/max, maps, exemplars, snapshots, transport and allocator overhead excluded. This does not predict SDK totals or quantile error."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    controls = ""
    for key,label in (("streams","Attribute streams / 属性序列数"),("live_states_per_stream","Selected live states/stream / 同时存活状态"),
                      ("max_size","Max logical span / 最大逻辑跨度"),("positive_span","Positive span / 正值侧跨度"),
                      ("negative_span","Negative span / 负值侧跨度")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{r["budget"]["input"][key]}"></label>'
    return page("Exponential histogram / 指数直方图与 Counter Capacity 工作台",
                '<main><div class="eyebrow">ROUND 113 · SIGNED FLOOR · COUNTS ARE NOT TAIL VALUES</div>'
                '<h1>Preserve boundaries, not only counts.<br>计数对了，还要检查桶边界。</h1>'
                '<div class="panel">十条 exact observations 的 span 从 8 缩至 3，scale 从 0 降到 -2。'
                'Floor 映射保留全部边界；错误 signed truncation 虽保留 10 counts，却有 3 条 interval violations。'
                '<p>Positive bucket 的负 index 表示较小正数，不是 negative observation；两种“负”不可混用。</p></div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>选定 fresh circular counters 按 2 的幂向上取 capacity，'
                '每状态另加一个 uint64 zero counter。历史 clear/downscale 后可能保留更大容量，不能据此推总内存。'
                '<p>Live states/stream 是用户预算假设，不声称 hot/cold 两份永远都满。'
                '原创模型只含非负整数/Fraction、scale -10..0、zero threshold=0，不实现 native SDK concurrency、'
                'negative observations、positive-scale float mapping 或 DELTA/CUMULATIVE reset。</p></div></main>',CALCULATOR)


def summary_svg(r):
    c = r["case"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Scale-alignment card sorting cabinet</title>'
        '<desc id="description">A detailed original sorting cabinet groups observation cards into '
        'coarser drawers; signed floor mapping preserves interval membership while drawer capacity differs from occupied cards.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 113 · EXPONENTIAL HISTOGRAM · SIGNED INDEX DOWNSCALING</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A conserved count can still hide a wrong bucket boundary.</text>'
        '<text x="64" y="180" font-size="24">A sorting-cabinet analogy: coarser drawers combine counts, not recover hidden values.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Scale {c["original_scale"]} to {c["fitted_scale"]}: span {c["original_span"]} to {c["fitted_span"]}, all {c["count"]} observations retained.</text>'
        +f'<text x="80" y="700" font-size="25">Wrong signed truncation preserves the same count but violates {c["wrong_containment_violations"]} exact intervals.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Buckets are lower-exclusive, upper-inclusive. Zero has a separate counter.</text>'
        +'<text x="96" y="842" font-size="23">Align scales before SUM; use floor for negative indices; do not infer exact tail values.</text>'
        +'<text x="96" y="883" font-size="21">Fresh circular-counter capacity is not retained SDK memory; native collection and wire bytes remain unknown.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/exponential_histogram_downscale_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
