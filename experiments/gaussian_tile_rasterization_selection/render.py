"""Self-contained Gaussian rasterization evidence and intersection budget."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "gaussian-splat-tile-alpha-wide-offset"


def deck(r: dict) -> str:
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic RGB/geometry · independently_reproduced=false · '
                     'Original CPU timings separately measured; no gsplat/GPU or image-quality result.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 104 · Ordered tile compositing</span><span>{n} / 7</span></footer></section>')

    b = r["budget"]
    card("HANDS-ON · GAUSSIAN RASTERIZATION",
         '<span class="accent">64-bit intersection offsets</span> do not widen every identity.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         '<p>1 image, 1920×1080<br>16×16 tiles<br>1M Gaussians, 8 tiles each</p>'
         '<span class="pill">Synthetic multiplicity</span></div><div class="panel"><h2>Raw payload</h2>'
         f'<p>{b["intersection_entries"]:,} entries</p>'
         f'<p>Key64 + ID32: {b["one_key_and_id_payload_bytes"]:,} bytes</p>'
         '<p>Peak HBM: null</p></div><div class="panel"><h2>Offset array</h2>'
         f'<p>64-bit: {b["tile_offset_int64_bytes"]:,} bytes</p>'
         f'<p>32-bit comparison: {b["tile_offset_int32_comparison_bytes"]:,}</p>'
         '<span class="pill">Other widths still matter</span></div></div>',
         '<b>Concrete capability:</b> the pinned September 1, 2026 gsplat change widens intersection offsets. '
         'Flattened Gaussian identities and tile-relative last IDs remain int32 in the inspected path.')
    blends = r["compositing"]
    scores = '<blockquote>Effective alphas .6 and .5, red then blue, background [.1,.2,.3]. '
    scores += 'Reversing depth leaves alpha .8 but changes RGB by .3.</blockquote>'
    for key,label,action in (("correct","Front-to-back","Confirm"),("reversed_depth","Reversed depth","Send to a human")):
        values = blends[key]["rgb"]
        scores += f'<div class="panel"><span class="pill">score · RGB channel value</span><span class="pill">{action}</span>'
        scores += f'<h2>{label}: [{", ".join(f"{v:.2f}" for v in values)}]</h2>'
        scores += '<p>Red and blue channel values; calibrated confidence: null</p>'
        scores += bar(values[0],1.0)+bar(values[2],1.0)+'</div>'
    card("SCENARIO: ORDER IS PART OF THE OPERATOR",
         'Identical render alpha .8 can hide a .3 color error.',
         scores,'<b>Premultiplied contribution:</b> accumulate T×alpha×color, then update T. '
         'The remaining T multiplies the background; additive color is not the same operator.')
    names = {"tile-versus-same-support-dense":"Tile / same-support dense","reverse-depth":"Reverse depth",
             "additive-no-transmittance":"Drop transmittance","omit-background":"Omit background",
             "exclusive-transmittance-cutoff":"Exclusive tail cutoff","alpha-floor-many-small-layers":"Many sub-floor layers"}
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in r["cases"]:
        status = "Intentional approximation" if row["intentional_approximation"] else "Match" if row["maximum_rgb_error"] < 1e-12 else "Mismatch ✕"
        matrix += f'<tr><td>{names[row["id"]]}</td><td>Max RGB difference</td>'
        matrix += f'<td>{row["maximum_rgb_error"]:.6g}<br>{status}</td><td>'
        matrix += ('Evaluate policy' if row["intentional_approximation"] else
                   'Confirm' if row["maximum_rgb_error"] < 1e-12 else 'Review blend')+'</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout">Exact gate compares the same finite support.</div>'
    matrix += '<div class="callout">Approximation controls are not claims of a source implementation bug.</div></div>'
    card("WHAT I TESTED","Tile indexing preserves declared support; opacity shortcuts change the target.",
         matrix,'<b>Original float64 CPU model.</b> Not bitwise float32 CUDA parity near thresholds; '
         '3D projection, antialiasing and the source ellipse/tile culling algorithm are excluded.')
    cutoff = r["exclusive_cutoff"]
    card("UNDER THE HOOD","Exclusive termination can omit more color than its next-T threshold.",
         '<pre>alpha = min(.99, opacity * exp(-sigma))\n'
         'skip if sigma &lt; 0 or alpha &lt; 1/255\n'
         'next_T = T * (1 - alpha)\n'
         'if next_T &lt;= 1e-4:\n'
         '    stop BEFORE committing this Gaussian\n'
         'color += T * alpha * RGB\n'
         'T = next_T\n'
         'color += remaining_T * background  # after the loop</pre>'
         '<p>↓ three .98-opacity RGB layers, black background</p>'
         f'<pre>pre-stop T = {cutoff["unit_rgb_error_bound_pre_T"]:.6f}\n'
         f'exact blue = {cutoff["exact"]["rgb"][2]:.6f}\n'
         f'exclusive blue = {cutoff["source_style_exclusive"]["rgb"][2]:.6f}\n'
         'omitted blue exceeds 1e-4\n'
         'For bounded unit RGB, use pre-stop T as the tail bound.\n'
         'Do not generalize this bound to unbounded/HDR features.</pre>',
         '<b>Gate order is meaningful.</b> Since the current Gaussian is excluded, hypothetical '
         'next_T is not a bound on all omitted color. The retained T also controls output alpha/background.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["tile_variant_checks"]:
        robust += f'<tr><td>Same-support RGB</td><td>Dense scan</td><td>Toy tile {row["tile"]}</td>'
        robust += f'<td>{row["maximum_rgb_error"]:.2e}</td></tr>'
    robust += '</table><div class="panel"><h2>Effective-alpha derivative gate</h2>'
    robust += f'<p>Independent tail-product derivatives and finite differences: max error {r["gradient"]["maximum_error"]:.2e}.</p>'
    robust += '<p>Fixed depth order, away from thresholds; no covariance/projection/sort derivative claim.</p></div>'
    robust += '<div class="panel"><h2>Wide cursor without a large allocation</h2>'
    robust += '<p>Logical offsets above 2³¹−1 wrap negative in signed int32. '
    robust += 'This arithmetic test allocates no billion-entry array.</p></div>'
    card("ROBUSTNESS CHECK","Four toy tile sizes preserve RGB while changing candidate work.",
         robust,'<b>Toy launch independence.</b> Tile sizes 1/2 here are CPU partitions, not claims '
         'of native kernel support. The source also has packed-key metadata and tile-relative counter limits.')
    latency = r["latency"]
    maximum = max(max(row["tile_indexed_median_ms"],row["dense_scan_median_ms"]) for row in latency["rows"])
    timing = '<p>Measured original CPU medians: blue tile index, gray dense candidate scan.</p>'
    for row in latency["rows"]:
        timing += f'<div class="latency-row"><b>{row["gaussians"]} Gaussians · 16×12 pixels</b>'
        for key,label,baseline in (("tile_indexed_median_ms","Tile indexed",False),
                                   ("dense_scan_median_ms","Dense scan",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","Tile indexing reduces time in all four measured original CPU cases.",
         timing,'<b>Windows ARM64, Python '+latency["environment"]["python"]+'.</b> Two warmups, seven '
         'alternating repeats; bin construction and validation included. Same finite support, early '
         'termination disabled; not production rasterizer throughput or GPU speedup.')
    card("MY TAKE","Address width, compositing order and approximation policy are separate gates.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Review depth/blend identities.</p><p>Validate same-support tile candidates.</p>'
         '<p>Audit every index-width boundary.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>All indices assumed int64.</p><p>Small per-alpha floor called a global quality bound.</p>'
         '<p>Raw sort bytes called peak HBM.</p></div></div>'
         '<pre>READINESS: original arithmetic and source review\n'
         'SOURCE: pinned gsplat, Apache-2.0 rights retained\n'
         'MODEL: finite square support, float64 CPU, unit RGB\n'
         'TIMING: original dense/tile references only\n'
         '3D PROJECTION / PSNR / GPU PERFORMANCE: not measured\n'
         'PROVENANCE: synthetic inputs, independently_reproduced=false</pre>',
         '<b>Approximation needs task evidence.</b> Five hundred alphas of .003 can accumulate '
         'substantial color even though each is below 1/255. That is not an observed image-quality failure.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 104 · Gaussian tile compositing · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["images","width","height","tile","gaussians_per_image","intersections_per_gaussian"]){
    const min=["gaussians_per_image","intersections_per_gaussian"].includes(key)?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" outside exact integer range");
  }
  const ceilDiv=(a,b)=>Number((BigInt(a)+BigInt(b)-1n)/BigInt(b));
  const tx=ceilDiv(p.width,p.tile),ty=ceilDiv(p.height,p.tile),tiles=tx*ty;
  if(p.intersections_per_gaussian>tiles)throw Error("Per-Gaussian tile count exceeds image tiles");
  const entries=p.images*p.gaussians_per_image*p.intersections_per_gaussian,ids=p.images*p.gaussians_per_image;
  const values={tiles_x:tx,tiles_y:ty,tiles_per_image:tiles,intersection_entries:entries,
    one_key_and_id_payload_bytes:12*entries,two_sort_buffer_payload_bytes:24*entries,
    tile_offset_int64_bytes:8*p.images*tiles,tile_offset_int32_comparison_bytes:4*p.images*tiles};
  if(![...Object.values(values),ids].every(Number.isSafeInteger))throw Error("Logical budget exceeds exact numerical range");
  return {...values,offsets_exceed_signed_int32:entries>2147483647,
    flattened_identity_fits_signed_int32:ids<=2147483648,
    packed_image_tile_metadata_fits_32_bits:Math.ceil(Math.log2(tiles))+Math.ceil(Math.log2(p.images))<=32,
    peak_memory_bytes:null,gpu_latency_ms:null,image_quality:null};
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
    result.textContent=JSON.stringify({provenance:"synthetic uniform intersection multiplicity",
      ...estimateBudget(p),production_approved:false,
      note:"64-bit offsets do not widen Gaussian identities, tile-relative last IDs or packed image/tile metadata. Two sort payloads exclude temporary workspace and actual peak allocation."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r: dict) -> str:
    p = r["budget"]["input"]
    labels = {"images":"Images / 图像数","width":"Image width / 宽","height":"Image height / 高",
              "tile":"Tile edge / Tile 边长","gaussians_per_image":"Gaussians per image / 每图高斯数",
              "intersections_per_gaussian":"Modeled tiles per Gaussian / 假设每高斯相交 Tile 数"}
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'min="{0 if key in ("gaussians_per_image","intersections_per_gaussian") else 1}" step="1" value="{p[key]}"></label>'
                       for key,label in labels.items())
    return page("Gaussian splat / Tile 与 Opacity 工作台",
                '<main><div class="eyebrow">ROUND 104 · GAUSSIAN TILES · SYNTHETIC PAYLOADS</div>'
                '<h1>Widen offsets without forgetting other indices.<br>Offset 变宽，不代表全部索引变宽。</h1>'
                '<div class="panel">同一 alpha=.8 可对应 RGB [.62,.04,.26] 或反序 [.32,.04,.56]。'
                'Depth order 是算子的一部分；exclusive cutoff 必须用 pre-stop T 审查尾部误差。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>相交倍数是输入假设，不是实际投影观测。'
                'Key64+ID32 payload 与两个排序 buffer 不含临时 workspace、真实 allocator 或 peak HBM。'
                '<p>原始 CPU square-support 分桶不是上游 3D projection/ellipse culling；'
                'tile knob 不认证 native launch configuration。未测 PSNR、GPU latency 或训练质量。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r: dict) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Gaussian depth-card tile sorter</title>'
        '<desc id="description">A solid sorter groups Gaussian cards by tile and depth. '
        'Wide list offsets do not change Gaussian identity or compositing order.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 104 · SYNTHETIC GAUSSIANS · TILE ORDER AND ADDRESS WIDTH</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Wide offsets do not make every identity a 64-bit index.</text>'
        '<text x="64" y="180" font-size="24">Sort by tile and depth; accumulate T * alpha * color; retain background transmittance.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">Same alpha .8, different depth order: RGB [.62,.04,.26] versus [.32,.04,.56].</text>'
        +'<text x="80" y="700" font-size="25">Exclusive stopping excludes the current Gaussian; bound the omitted tail using pre-stop T.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">September 2026 source: global intersection offsets int64, flattened identities int32.</text>'
        +'<text x="96" y="842" font-size="23">Original same-support tile/dense RGB agrees; geometry and image quality are not reproduced.</text>'
        +'<text x="96" y="883" font-size="21">CPU scalar timing and logical sort payloads are not GPU throughput or peak HBM.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/gaussian_tile_rasterization_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r: dict) -> dict[Path,str]:
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
