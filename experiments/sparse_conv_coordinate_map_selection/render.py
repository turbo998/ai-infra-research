"""Self-contained sparse-coordinate evidence cards, H5 calculator and SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "sparse-convolution-coordinate-map-submanifold-inverse"


def deck(r):
    cards = []

    def card(eyebrow, title, body, note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic scalar coordinates · independently_reproduced=false · '
                     'Original CPU timings separately measured; no native/GPU or model-quality result.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 106 · Sparse coordinate maps</span><span>{n} / 7</span></footer></section>')

    b, t, inv, cache = r["budget"], r["topology"], r["inverse"], r["cache"]
    card("HANDS-ON · SPARSE CONVOLUTION",
         '<span class="accent">Submanifold support</span> is an operator choice, not just fewer FLOPs.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>{b["input"]["active_input"]:,} active input rows<br>{b["kernel_volume"]} kernel offsets</p>'
         f'<p>{b["input"]["valid_pairs"]:,} synthetic valid pairs</p></div>'
         '<div class="panel"><h2>Native pair capacity</h2>'
         f'<p>{b["native_padded_pair_bytes"]:,} bytes</p>'
         '<span class="pill">2 × K × N indices</span><p>Peak HBM: null</p></div>'
         '<div class="panel"><h2>Coordinate work</h2>'
         f'<p>Compact pair IDs: {b["compact_valid_pair_id_bytes"]:,} bytes</p>'
         f'<span class="pill">First CPU tier: {r["latency"]["rows"][0]["primary_median_ms"]:.4f} ms</span>'
         '<p>Tokens: not applicable</p></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; synthetic maps. '
         'Pinned spconv mechanics are longstanding (conv.py last changed in 2023), not a newly announced 2026 capability.')
    scores = '<blockquote>Input coordinate rows are permuted together with their features. '
    scores += 'Fresh topology preserves the function; stale row-number pairs do not.</blockquote>'
    maximum = max(abs(v) for key in ("fresh_output","stale_unchecked_output") for v in cache[key])
    for key,label,action in (("fresh_output","Fresh map","Confirm"),("stale_unchecked_output","Stale row map","Send to a human")):
        scores += f'<div class="panel"><span class="pill">score · scalar output</span><span class="pill">{action}</span>'
        scores += f'<h2>{label}</h2><p>Calibrated confidence: null</p>'
        for i,value in enumerate(cache[key]):
            scores += f'<p>Output row {i}: {value:g}</p>'+bar(abs(value),maximum)
        scores += '</div>'
    card("SCENARIO: ROW IDENTITY IS PART OF A MAP",
         f'Equal coordinate counts can still hide output error {cache["maximum_stale_error"]:g}.',
         scores,'<b>Guard the complete contract.</b> Our original model checks ordered coordinates exactly. '
         'This is stronger than the inspected source reuse checks for shape/count and is not a claim that source compares every coordinate.')
    rows = [
        ("Regular layer", "Structural support", f'{len(t["regular_coordinates"])} active rows', "Choose semantics"),
        ("Submanifold", "Same active input support", f'{len(t["submanifold_coordinates"])} active rows', "Confirm"),
        ("Second layer", "Same values at original sites?", f'✕ {t["second_regular_at_input"]} vs {t["second_submanifold_output"]}', "Not interchangeable"),
        ("Zero features", "Do coordinates disappear?", f'{t["zero_feature_active_count"]} active rows remain', "Keep topology"),
        ("Tied adjoint", "Independent finite differences", f'{inv["maximum_gradient_error"]:.2e}', "Confirm"),
        ("Adjoint as inverse", "Reconstruct original input?", f'✕ {inv["adjoint_of_forward"]}', "Reject inference"),
    ]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Active is a coordinate property, not nonzero feature value.</div>'
    matrix += '<div class="callout">Learned inverse-conv weights need not be the forward adjoint.</div></div>'
    card("WHAT I TESTED","Two sparse operators already disagree after their second layer.",
         matrix,'<b>Original 1D scalar model.</b> Batched coordinates, regular stride/dilation and centered submanifold '
         'convolution agree with independent dense scans at declared output sites. No native multi-channel kernel parity.')
    card("UNDER THE HOOD","Cached inverse connectivity is neither new transpose support nor a matrix inverse.",
         '<pre>input x = output y * stride - padding + k*dilation\n'
         'keep valid integer coordinates, including batch identity\n'
         'pair = (kernel_offset, input_row, output_row)\n'
         'regular: union of reachable output coordinates\n'
         'submanifold: retain only original active coordinates</pre>'
         '<p>↓ reverse stored edges with tied scalar weights</p>'
         f'<pre>A x = {inv["forward"]}\n'
         f'A^T u on original support = {inv["cached_tied_weight_adjoint"]}\n'
         f'ordinary transpose support = {len(inv["ordinary_transposed"]["coordinates"])} rows\n'
         f'cached original support = {len(inv["input_coordinates"])} rows\n'
         f'&lt;Ax,u&gt; = &lt;x,A^Tu&gt; = {inv["dot_left"]:g}\n'
         f'A^T A x = {inv["adjoint_of_forward"]}, not {inv["features"]}</pre>',
         '<b>Tied-weight condition:</b> reversing the cached map implements the adjoint only with the matching '
         'transposed channel/kernel weights and no extra nonlinear/bias terms. A separately learned SparseInverseConv need not satisfy that.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>Same output sites</td><td>Dense scan</td><td>s={row["stride"]}, '
        robust += f'd={row["dilation"]}, subm={row["submanifold"]}</td><td>{row["maximum_error"]:.2e}</td></tr>'
    robust += f'<tr><td>Permute input rows</td><td>Fresh map</td><td>Fresh permuted map</td><td>0</td></tr>'
    robust += f'<tr><td>Permute input rows</td><td>Fresh map</td><td>Stale cached pairs</td><td>{cache["maximum_stale_error"]:g} ✕</td></tr></table>'
    robust += '<div class="panel"><h2>Batch isolation and zero-valued rows</h2>'
    robust += '<p>Identical x positions in separate batches never share an edge. Zero-valued features do not erase coordinate membership.</p></div>'
    card("ROBUSTNESS CHECK","Stride, dilation and batch identity preserve a declared operator—not an arbitrary cache.",
         robust,'<b>Topology changes invalidate maps.</b> Coalescing duplicates needs an explicit policy; '
         'the toy rejects them. Its stronger ordered-coordinate checks are not a benchmark of production cache bookkeeping.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timing = '<p>Blue: immutable map reuse · Gray: rebuild identical map then apply.</p>'
    for row in r["latency"]["rows"]:
        timing += f'<div class="latency-row"><h2>{row["active_input"]} inputs · {row["valid_pairs"]} valid edges</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timing += f'<div class="latency-pair"><b>{row[key]:.4f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timing += '</div>'
    card("LATENCY","Reusing identical submanifold maps removes construction work in every measured CPU tier.",
         timing,'<b>Measured Python CPU:</b> two warmups, seven alternating repeats, median. '
         'Primary excludes initial map construction and requires unchanged coordinates/operator; both include exact row checks. '
         'Irregular pair counts vary by tier. Not CUDA speedup, end-to-end inference or peak-memory evidence.')
    card("MY TAKE","Support, cached row identity and numerical inverse are three different contracts.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Preserve sparse support deliberately.</p><p>Validate immutable map identity.</p>'
         '<p>Budget pair capacity separately from features.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Submanifold called identical dense network.</p><p>Inverse connectivity called matrix inversion.</p>'
         '<p>CPU reuse ratio called native GPU speedup.</p></div></div>'
         '<pre>READINESS: scalar support and adjoint checks\n'
         'SOURCE: pinned spconv, Apache-2.0 rights retained\n'
         'MODEL: 1D batched scalar cross-correlation, no bias\n'
         'GPU / 3D KERNEL / MODEL QUALITY: not reproduced\n'
         'PAIR BYTES: selected Native path, not implicit GEMM\n'
         'PROVENANCE: synthetic; independently_reproduced=false</pre>',
         '<b>Selected payloads only.</b> Compact valid-pair IDs omit grouping metadata; Native padded pairs '
         'omit hash/unique workspace, masks, sorting and allocator state. Neither equals total peak HBM.')
    styles = ('<style>.latency-row{padding:12px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 106 · Sparse coordinate maps · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  if(!Number.isSafeInteger(p.dimensions)||p.dimensions<1||p.dimensions>3)throw Error("dimensions must be 1, 2 or 3");
  for(const key of ["batches","edge","kernel","channels_in","channels_out"]){
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive exact integer");
  }
  for(const key of ["active_input","active_output","valid_pairs"]){
    if(!Number.isSafeInteger(p[key])||p[key]<0)throw Error(key+" must be a nonnegative exact integer");
  }
  if(![2,4].includes(p.feature_bytes)||![4,8].includes(p.index_bytes))throw Error("feature_bytes 2/4 and index_bytes 4/8 required");
  const volume=p.batches*p.edge**p.dimensions,k=p.kernel**p.dimensions;
  if(Math.max(p.active_input,p.active_output)>volume)throw Error("active coordinates exceed declared batched spatial volume");
  if(p.valid_pairs>k*Math.min(p.active_input,p.active_output))throw Error("valid pairs exceed kernel-volume per-row bound");
  const result={spatial_volume:volume,kernel_volume:k,
    native_padded_pair_bytes:2*k*p.active_input*p.index_bytes,
    compact_valid_pair_id_bytes:2*p.valid_pairs*p.index_bytes,kernel_pair_count_bytes:k*p.index_bytes,
    input_coordinate_bytes:p.active_input*(p.dimensions+1)*p.index_bytes,
    output_coordinate_bytes:p.active_output*(p.dimensions+1)*p.index_bytes,
    input_feature_bytes:p.active_input*p.channels_in*p.feature_bytes,
    output_feature_bytes:p.active_output*p.channels_out*p.feature_bytes,
    source_selects_int64_hash_keys:volume>=2147483647,
    active_row_ids_fit_signed_int32:Math.max(p.active_input,p.active_output)<=2147483648,
    peak_hbm_bytes:null,gpu_latency_ms:null,model_quality:null};
  if(!Object.values(result).filter(v=>typeof v==="number").every(Number.isSafeInteger))
    throw Error("budget exceeds exact JavaScript integer range");
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
    result.textContent=JSON.stringify({provenance:"synthetic active-row and valid-pair counts",
      ...estimateBudget(p),production_approved:false,
      note:"Assumes a shared declared spatial volume for the payload comparison, not shape-changing operator inference. Valid pairs are user inputs, not predicted occupancy. Native [2,K,N] is one selected path; compact pair IDs exclude grouping metadata, both exclude full workspace/peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    labels = dict(dimensions="Dimensions / 维度",batches="Batches / 批数",edge="Declared grid edge / 声明边长",
                  kernel="Kernel edge / 卷积核边长",active_input="Active input rows / 输入坐标数",
                  active_output="Active output rows / 输出坐标数",valid_pairs="Synthetic valid pairs / 假设有效边数",
                  channels_in="Input channels / 输入通道",channels_out="Output channels / 输出通道",
                  feature_bytes="Feature bytes / 特征字节",index_bytes="Pair index bytes / Pair 索引字节")
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'step="1" value="{p[key]}"></label>' for key,label in labels.items())
    return page("Sparse convolution / 稀疏坐标图工作台",
                '<main><div class="eyebrow">ROUND 106 · COORDINATE MAPS · SYNTHETIC PAYLOADS</div>'
                '<h1>Topology is not a nonzero test.<br>坐标支持、行身份和数值求逆不能混为一谈。</h1>'
                '<div class="panel">Regular convolution 扩展活动坐标；submanifold 保留输入支持。'
                'Cached inverse 反转历史边，不等于重新生成 transpose support，也不意味着恢复输入数值。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>活动行数和有效边数是 synthetic 输入，不从 occupancy 猜测。'
                '采用共享声明 grid volume 比较 payload，不推断 stride 后形状。'
                '<p>Native [2,K,N] 只代表所查一路径；不是 implicit GEMM masks/workspace。'
                '64-bit hash keys 不代表 pair row IDs 全变宽；不含实际 allocator、排序临时区、peak HBM。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r):
    t,inv,c = r["topology"],r["inverse"],r["cache"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Immutable coordinate-map sorter</title>'
        '<desc id="description">A detailed solid sorter represents coordinate rows and stored routing. '
        'Reordering cards without rebuilding their row map sends contributions to the wrong output.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 106 · SPARSE CONVOLUTION · SUPPORT AND ROW IDENTITY</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A cached route identifies rows, not just their count.</text>'
        '<text x="64" y="180" font-size="24">A mechanical analogy: card order changes require rebuilding the routing map.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Second regular values {t["second_regular_at_input"]}; submanifold {t["second_submanifold_output"]} at the same sites.</text>'
        +f'<text x="80" y="700" font-size="25">Stale row map: error {c["maximum_stale_error"]:g}. Cached tied adjoint: {len(inv["input_coordinates"])} sites; ordinary transpose: {len(inv["ordinary_transposed"]["coordinates"])}.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Sparse support is structural: even exactly zero features retain their active coordinates.</text>'
        +'<text x="96" y="842" font-size="23">Reversing cached connectivity is not numerical inversion; learned weights need not be adjoint.</text>'
        +'<text x="96" y="883" font-size="21">Original batched scalar CPU model only; selected map payloads are not native GPU peak HBM.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/sparse_conv_coordinate_map_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
