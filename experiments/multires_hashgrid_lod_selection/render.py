"""Static evidence cards, interactive allocation calculator and original solid figure."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "multires-hash-grid-collision-lod"


def deck(r):
    cards = []

    def card(eyebrow, title, body, note):
        n = len(cards) + 1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic scalar fields · independently_reproduced=false · '
                     'Original CPU timings separately measured; no native/GPU or reconstruction-quality result.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 105 · Hash-grid and LoD</span><span>{n} / 7</span></footer></section>')

    b, c, g, lod = r["budget"], r["collision"], r["gradient"], r["lod"]
    card("HANDS-ON · MULTIRESOLUTION ENCODING",
         '<span class="accent">Hard/soft LoD gates</span> change work, not allocated table size.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>{b["input"]["levels"]} levels, {b["input"]["dimensions"]}D<br>'
         f'{b["input"]["features"]} features per entry</p>'
         '<span class="pill">Hash cap + alignment</span></div><div class="panel"><h2>Parameters</h2>'
         f'<p>{b["allocated_table_entries"]:,} entries</p>'
         f'<p>{b["parameter_bytes"]:,} bytes</p><p>Peak HBM: null</p></div>'
         '<div class="panel"><h2>Soft wrapper buffer</h2>'
         f'<p>{b["soft_unweighted_base_output_bytes"]:,} bytes</p>'
         '<span class="pill">Unweighted base output</span>'
         f'<p class="pill">First tier CPU: {r["latency"]["rows"][0]["primary_median_ms"]:.4f} ms</p>'
         '<p>Tokens: not applicable</p></div></div>',
         '<b>Source capability:</b> the pinned September 29, 2026 tiny-cuda-nn change generalizes native '
         'hard/soft LoD across multilevel encodings. Our scalar grid arithmetic is an original reference. '
         f'CPU: {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; synthetic field, separately measured timing.')
    scores = '<blockquote>Four corners from one bilinear query map to only two shared table buckets.</blockquote>'
    for label, values, action in (("Accumulate every corner", c["parameter_gradient"], "Confirm"),
                                  ("Overwrite collided gradients", c["wrong_overwrite_gradient"], "Send to a human")):
        scores += f'<div class="panel"><span class="pill">score · table VJP</span><span class="pill">{action}</span>'
        scores += f'<h2>{label}</h2><p>Calibrated confidence: null</p>'
        for index in (0, 3):
            scores += f'<p>Bucket {index}: {values[index]:.4f}</p>' + bar(values[index], 1)
        scores += '</div>'
    card("SCENARIO: COLLISIONS SHARE PARAMETERS",
         f'Overwriting collided gradients loses {c["overwrite_max_error"]:.4f} in one bucket.',
         scores, '<b>Collisions are parameter tying, not duplicate records to discard.</b> '
         'The forward must gather a shared parameter and its adjoint must sum every occurrence.')
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    rows = [
        ("Unfolded corners", "Same tied field?", f'{r["parity"]["maximum_error"]:.2e} max error', "Confirm"),
        ("Table VJP", "Finite differences", f'{g["maximum_parameter_error"]:.2e}', "Confirm"),
        ("Coordinate VJP", "Within-cell derivatives", f'{g["maximum_coordinate_error"]:.2e}', "Confirm"),
        ("Collision overwrite", "Preserve all occurrences?", f'✕ {c["overwrite_max_error"]:.4f} error', "Reject"),
        ("Soft backward", "Weight exactly once?", f'✕ omit: {lod["wrong_missing_weight"][1]:.6f}', "Reject"),
        ("Soft backward", "Avoid double weighting?", f'✕ double: {lod["wrong_double_weight"][1]:.6f}', "Reject"),
    ]
    for values in rows:
        matrix += "<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>"
    matrix += '</table><div class="grid two"><div class="callout">Gradients tested away from cell boundaries.</div>'
    matrix += '<div class="callout">Hash collisions change representational capacity, not the chain rule.</div></div>'
    card("WHAT I TESTED", "One shared field passes independent values and both adjoint gates.",
         matrix, '<b>Reference boundary:</b> 2D scalar float64 linear interpolation, bounded nonnegative coordinates. '
         'No float32 FMA, half-precision atomics, stochastic interpolation or JIT parity claim.')
    card("UNDER THE HOOD", "A soft forward needs the same weight in its backward.",
         '<pre>z = lod * number_of_levels + 1e-3\n'
         'hard[level] = 1 if level &lt; z else 0\n'
         'soft[level] = clamp(z - level, 0, 1)\n'
         'output[level] = soft[level] * raw[level]\n'
         'table[index] is shared by all collided corners</pre>'
         '<p>↓ objective = sum(output²) / 2</p>'
         '<pre>dL/draw = soft² * raw\n'
         f'raw middle feature = {lod["raw"][1]:.8f}\n'
         f'soft middle weight = {lod["soft_weights"][1]:.8f}\n'
         f'correct gradient = {lod["correct_gradient"][1]:.10f}\n'
         f'finite difference = {lod["finite_difference"][1]:.10f}\n'
         'table VJP: add upstream * weight for EVERY corner</pre>',
         '<b>Unweighted output matters.</b> The source retains an unweighted base output and '
         'weights upstream adjoints; accidentally feeding a weighted value back to a base API can weight twice. '
         'This study does not implement the source wrapper’s LoD-input derivatives or full double backward.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>Same tied field</td><td>Direct gather</td><td>R={row["resolution"]}, '
        robust += f'T={row["capacity"]}, unfolded</td><td>{row["maximum_error"]:.2e}</td></tr>'
    robust += '</table><div class="panel"><h2>Zero fraction is not zeroed in this source convention</h2>'
    robust += f'<p>At lod=0, hard weights {lod["zero_fraction_hard"]}; soft weights {lod["zero_fraction_soft"]}.</p>'
    robust += f'<p>For four levels at lod={lod["boundary_lod"]}, z=1: level 1 is off in both forward and backward.</p></div>'
    card("ROBUSTNESS CHECK", "The +1e−3 convention activates the first hard level even at lod=0.",
         robust, '<b>Boundary consistency:</b> forward and parameter/input derivatives must use the same '
         'level &lt; z predicate. These are mathematical float64 boundary controls, not native float32 bitwise tests.')
    largest = max(max(row["primary_median_ms"], row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timing = '<p>Blue: direct gathers · Gray: unfold the scalar field once per batch.</p>'
    for row in r["latency"]["rows"]:
        timing += f'<div class="latency-row"><h2>{row["batch"]} points · R={row["resolution"]}, T={row["table_entries"]}</h2>'
        for key, baseline in (("primary_median_ms", False), ("baseline_median_ms", True)):
            timing += f'<div class="latency-pair"><b>{row[key]:.4f} ms</b>{bar(row[key],largest,baseline)}</div>'
        timing += '</div>'
    card("LATENCY", "At larger CPU batches, a tiny unfolded field amortizes its construction.",
         timing, f'<b>Measured original Python CPU:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; '
         'two warmups, seven alternating repeats, median. Direct builds Python term tuples including derivative '
         'coefficients; baseline allocates a 17×17 scalar field. Not tuned, not equivalent memory, not GPU speedup.')
    card("MY TAKE", "Allocation, collision semantics and LoD derivatives are separate readiness gates.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Budget every level before allocation.</p><p>Sum collided parameter adjoints.</p>'
         '<p>Test hard/soft epsilon boundaries.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Collision count called PSNR.</p><p>Lower LoD called smaller weights allocation.</p>'
         '<p>CPU toy ratios called CUDA throughput.</p></div></div>'
         '<pre>READINESS: independent scalar values + adjoints\n'
         'SOURCE: pinned tiny-cuda-nn, LICENSE.txt inspected\n'
         'LICENSE: BSD-3-Clause terms; API says NOASSERTION\n'
         'MODEL: 2D scalar linear interpolation, fixed LoD\n'
         'GPU / PSNR / FULL DOUBLE BACKWARD: not measured\n'
         'PROVENANCE: synthetic; independently_reproduced=false</pre>',
         '<b>Logical bytes are not peak HBM.</b> Parameters, output and the soft wrapper’s unweighted output '
         'are separate buffers. Gradients, optimizer states, layout staging and allocator/workspace are omitted.')
    styles = ('<style>.latency-row{padding:12px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 105 · Hash-grid collision and LoD · seven cards", styles + '<main>' + "".join(cards) + '</main>')


CALCULATOR = r"""
function estimateBudget(p){
  const limits={dimensions:[2,3],levels:[1,32],base_resolution:[2,64],growth:[1,2],
    log2_cap:[1,24],features:[1,8],parameter_bytes:[2,4],batch:[1,Number.MAX_SAFE_INTEGER]};
  for(const [key,[low,high]] of Object.entries(limits)){
    if(!Number.isSafeInteger(p[key])||p[key]<low||p[key]>high)throw Error(key+" outside supported integer range");
  }
  if(![2,4].includes(p.parameter_bytes))throw Error("parameter_bytes must be 2 or 4");
  if(!Number.isFinite(p.lod)||p.lod<0||p.lod>1)throw Error("lod must be finite in [0,1]");
  const z=p.lod*p.levels+1e-3,rows=[];
  for(let level=0;level<p.levels;level++){
    const resolution=p.base_resolution*p.growth**level;
    if(resolution>65536)throw Error("resolution exceeds this bounded exact calculator's 65536 limit");
    const dense=resolution**p.dimensions;
    const aligned=Math.ceil(Math.min(dense,2147483647)/8)*8;
    rows.push({level,resolution,dense_vertices:dense,table_entries:Math.min(aligned,2**p.log2_cap),
      hard_weight:Number(level<z),soft_weight:Math.min(1,Math.max(0,z-level))});
  }
  const entries=rows.reduce((s,r)=>s+r.table_entries,0),scalars=entries*p.features,
    output=p.batch*p.levels*p.features*p.parameter_bytes;
  const result={levels:rows,allocated_table_entries:entries,parameter_scalars:scalars,
    parameter_bytes:scalars*p.parameter_bytes,
    ideal_dense_vertex_bytes:rows.reduce((s,r)=>s+r.dense_vertices,0)*p.features*p.parameter_bytes,
    output_feature_bytes:output,soft_unweighted_base_output_bytes:output,
    hard_active_levels:rows.reduce((s,r)=>s+r.hard_weight,0),
    table_entries_fit_uint32:entries<=4294967295,parameter_scalars_fit_uint32:scalars<=4294967295,
    peak_hbm_bytes:null,gpu_latency_ms:null,quality:null};
  const integers=[...Object.values(result).filter(v=>typeof v==="number"),
    ...rows.flatMap(r=>[r.resolution,r.dense_vertices,r.table_entries])];
  if(!integers.every(Number.isSafeInteger))throw Error("budget exceeds exact JavaScript integer range");
  return result;
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
    result.textContent=JSON.stringify({provenance:"synthetic allocation inputs",...estimateBudget(p),
      production_approved:false,note:"Lower LoD does not deallocate tables. The dense number is an ideal mathematical vertex payload, not a supported native dense allocation. Outputs exclude gradients, optimizer, staging and peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    labels = dict(dimensions="Dimensions / 维度", levels="Levels / 层数",
                  base_resolution="Base vertices / 基层顶点数", growth="Integer level growth / 整数倍率",
                  log2_cap="log₂ hash cap / 哈希容量", features="Features per entry / 每项特征",
                  parameter_bytes="Parameter bytes / 参数字节", batch="Batch points / 查询点数", lod="LoD fraction / 层级比例")
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'step="{0.001 if key == "lod" else 1}" value="{p[key]}"></label>'
                       for key, label in labels.items())
    return page("Hash-grid / 哈希网格与 LoD 工作台",
                '<main><div class="eyebrow">ROUND 105 · ORIGINAL SCALAR GRID · SYNTHETIC ALLOCATION</div>'
                '<h1>Hash collisions share gradients.<br>碰撞共享参数，LoD 不回收参数。</h1>'
                '<div class="panel">四个插值角点可落到两个共享桶。反向必须累加每次出现的贡献；'
                'soft LoD 的上游梯度需要同一个权重。lod=0 的首层仍受 +1e-3 约定影响。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>仅支持二维/三维、整数倍率 1/2、每层分辨率 ≤65536。'
                '输出是 exact integer logical payload，不认证 native feature specialization 或浮点边界。'
                '<p>Ideal dense vertices 不是上游支持的 dense allocation；'
                'hash tables 全部分配，降低 LoD 不减少该数字。未计 gradients、optimizer、staging、峰值 HBM。</p></div></main>',
                "const BASE=" + json.dumps(p) + ";\n" + CALCULATOR)


def summary_svg(r):
    c, lod = r["collision"], r["lod"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Shared-bin hash-grid adjoint sorter</title>'
        '<desc id="description">A detailed solid sorter maps incoming corner cards to shared bins. '
        'The bin accumulates every card contribution; a LoD gate changes contributions, not installed bin capacity.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 105 · HASH-GRID COLLISIONS · HARD AND SOFT LOD</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Shared buckets sum adjoints; LoD does not deallocate them.</text>'
        '<text x="64" y="180" font-size="24">A mechanical analogy: corner cards share a bin, and every card adds to its total.</text>'
        + sorter_scene()
        + f'<text x="80" y="650" font-size="26" font-weight="700">Four corners, two bins: bucket gradients {c["parameter_gradient"][0]:.4f} and {c["parameter_gradient"][3]:.4f}.</text>'
        + f'<text x="80" y="700" font-size="25">Soft middle weight {lod["soft_weights"][1]:.3f}; the same gate must appear in its adjoint.</text>'
        + '<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        + '<g fill="#ffffff"><text x="96" y="799" font-size="23">September 2026 native LoD capability: hard and soft paths preserve base encoding contracts.</text>'
        + '<text x="96" y="842" font-size="23">Independent scalar interpolation and adjoints verified; native CUDA and quality not reproduced.</text>'
        + '<text x="96" y="883" font-size="21">Parameter, output and unweighted soft buffer payloads are distinct; peak HBM remains unknown.</text></g>'
        + '<text x="64" y="955" font-size="20">Data: experiments/multires_hashgrid_lod_selection/report.json · independently_reproduced=false</text>'
        + '</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT / "docs" / "reports" / f"{SLUG}-selection-cards.html": deck(r),
            ROOT / "docs" / "reports" / f"{SLUG}-sizer.html": workbench(r),
            ROOT / "assets" / "svg" / f"{SLUG}-selection-faq-summary.svg": summary_svg(r)}
