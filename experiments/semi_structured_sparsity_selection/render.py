"""Self-contained logical 2:4 sparse-matrix evidence and assumption calculator."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "semi-structured-2-4-sparse-gemm-metadata"


def deck(report: dict) -> str:
    b = report["budget"]
    cases = {row["id"]:row for row in report["cases"]}
    cards = []

    def card(eyebrow,title,evidence,takeaway):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7">'
                     f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                     '<p class="muted">SYNTHETIC · independently_reproduced=false · no GPU, quality or physical-packing measurement</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 100 · logical 2:4 sparse GEMM</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · STRUCTURED SPARSITY",
         '<span class="accent">2:4 values</span> still need metadata, compatible shapes and reuse.',
         '<div class="grid"><div class="panel"><h2>Fixture</h2>'
         '<p>4 × 8 weight matrix<br>8 × 3 dense operand<br>Two retained slots per quartet</p>'
         '<span class="pill">Gate: error &lt; 1e-12</span></div><div class="panel"><h2>Logical input</h2>'
         '<p>Explicit row pruning</p><p>Four-bit index nibble</p><p>Same pruned-dense reference</p>'
         '</div><div class="panel"><h2>Separate budget</h2>'
         f'<p>{b["compressed_to_original_ratio"]*100:.2f}% logical forward weight bytes<br>'
         f'{b["minimum_reuse_for_strict_saving"]} modeled reuse threshold</p>'
         '<span class="pill">Model accuracy: null</span></div></div>',
         '<b>Original scalar logical encoding, not a backend format.</b> '
         'Pruning changes weights; compression must preserve the same pruned matrix. '
         'PyTorch source is inspected, not executed.')
    exact = cases["logical-packed-product"]
    wrong = cases["reverse-metadata-index-order"]
    detail = '<blockquote>Nibble low bits identify the first stored value; high bits identify the second. '
    detail += 'Keeping the values but swapping index interpretation changes the matrix.</blockquote>'
    for row in (exact,wrong):
        detail += '<div class="panel"><span class="pill">score · max product error</span>'
        detail += '<span class="pill">'+('Confirm' if row["matches_reference"] else 'Send to a human')+'</span>'
        detail += f'<h2>{row["id"]}</h2><p>{row["maximum_absolute_error"]:.3e} · confidence: null</p>'
        detail += bar(row["maximum_absolute_error"],wrong["maximum_absolute_error"])+'</div>'
    card("SCENARIO: METADATA IDENTITY",
         f'Reversing logical index order creates {wrong["maximum_absolute_error"]:.1f} output error.',
         detail,'<b>Numerical reference:</b> the same row-pruned dense matrix, not the original '
         'unpruned weights. Logical nibbles do not claim CUTLASS/cuSPARSELt physical compatibility.')
    names = {
        "logical-packed-product":"Packed row-pruned product",
        "reverse-metadata-index-order":"Reverse index order",
        "pruned-vs-original-dense":"Pruned versus original",
        "bidirectional-pruned-product":"Packed bidirectional policy",
        "bidirectional-vs-row-pruning":"Bidirectional versus row policy",
    }
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        answer = "Intentional change" if row["intentional_pruning"] else "Match" if row["matches_reference"] else "Mismatch ✕"
        matrix += f'<tr><td>{names[row["id"]]}</td><td>{row["reference"]}</td>'
        matrix += f'<td>{answer}<br>{row["maximum_absolute_error"]:.3e}</td><td>'
        matrix += ('Evaluate quality' if row["intentional_pruning"] else 'Confirm' if row["matches_reference"] else 'Review metadata')+'</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout"><b>Gate:</b> same pruned operator and error &lt; 1e-12.</div>'
    matrix += '<div class="callout"><b>Failure scope:</b> row-valid does not imply transpose-valid.</div></div>'
    card("WHAT I TESTED","Pruned-reference products match; pruning policies need separate quality checks.",
         matrix,'<b>No model-accuracy score.</b> A nonzero change from original weights is a pruning '
         'difference, not a broken compressed multiplication.')
    card("UNDER THE HOOD","Two FP16 values plus four bits occupy 56.25% of aligned dense bytes.",
         '<pre>for each adjacent quartet:\n'
         '    retain two positions i0 &lt; i1\n'
         '    values = [weight[i0], weight[i1]]\n'
         '    logical_code = i0 | (i1 &lt;&lt; 2)\n\n'
         'FP16 values: 2 * 16 bits\n'
         'metadata:    4 bits\n'
         'dense:       4 * 16 bits\n'
         'ratio:       36 / 64 = 56.25%</pre>'
         '<p>↓ physical backend format is a separate contract</p>'
         '<pre>CUTLASS FP16/BF16 sparse shape: M multiple 32, K multiple 64\n'
         'cuSPARSELt FP16/BF16:          M multiple 16, K multiple 16\n'
         'Caller padding and cropping are explicit here.\n'
         'Backend swizzles, masks and actual packed allocation: unknown.</pre>',
         '<b>Dtype boundary:</b> pinned CUTLASS FP32 takes a different 1:2 conversion branch. '
         'FP8 requires separate scaled operations and platform gates; neither is modeled.')
    checks = report["transpose_checks"]
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    robust += f'<tr><td>Row-only pattern</td><td>Forward valid: {checks["row_pruned_forward_valid"]}</td>'
    robust += f'<td>Transpose valid: {checks["row_pruned_transpose_valid"]}</td><td>Constraint fails</td></tr>'
    robust += f'<tr><td>Both-direction toy policy</td><td>Forward valid: {checks["bidirectional_forward_valid"]}</td>'
    robust += f'<td>Transpose valid: {checks["bidirectional_transpose_valid"]}</td><td>Both pass</td></tr></table>'
    robust += '<div class="panel"><h2>Storing both orientations is not free</h2>'
    robust += f'<p>Default forward logical pack: {b["logical_compressed_bytes"]:,} bytes.</p>'
    robust += f'<p>Two orientation packs: {b["both_orientation_logical_bytes"]:,} bytes.</p>'
    robust += f'<p>Original dense weights: {b["original_dense_weight_bytes"]:,} bytes.</p></div>'
    card("ROBUSTNESS CHECK","A row-valid 2:4 mask can fail after transpose.",
         robust,'<b>Training needs its own plan.</b> The source stores separate packed orientations '
         'for its tile-pruning path; our bidirectional greedy mask is an original policy, not that implementation.')
    maximum = max(max(row["dense_modeled_kernel_ms"],row["sparse_modeled_kernel_ms"]) for row in report["scaling"])
    timing = '<p>Modeled per-call kernel times only; blue padded sparse, gray original dense. Packing excluded here.</p>'
    for row in report["scaling"]:
        threshold = row["minimum_reuse_for_strict_saving"]
        timing += f'<div class="latency-row"><b>{row["input"]["tokens"]} tokens · '
        timing += ('no strict break-even' if threshold is None else f'minimum reuse {threshold}')+'</b>'
        for key,label,baseline in (("sparse_modeled_kernel_ms","Padded sparse",False),
                                   ("dense_modeled_kernel_ms","Dense",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.5f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","Padding can erase sparse savings; packing requires repeated reuse.",
         timing,'<b>Assumed effective rates:</b> dense 500, sparse 800 dense-equivalent TFLOP/s; '
         '5 µs launch, 0.2 ms one-time pack. Not hardware timing or peak sparse FLOPS. Median/repeats N/A.')
    card("MY TAKE","Choose a valid pruned operator before claiming sparse acceleration.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Stable, reusable compatible weights.</p><p>Pruned-dense reference and index tests.</p>'
         '<p>Separate forward/backward format planning.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Equating 50% zeros with 2× speed.</p><p>Blindly transposing a row mask.</p>'
         '<p>Calling logical bytes backend allocations.</p></div></div>'
         '<pre>READINESS: original logical prototype\n'
         'SOURCE: pinned PyTorch sparse classes/conversions\n'
         'RIGHTS: upstream LICENSE retained; no implementation copied\n'
         'PHYSICAL PACK / GPU / MODEL QUALITY: not measured\n'
         'SCOPE: FP16/BF16 formulas, not FP32/FP8 support certification</pre>',
         '<b>Synthetic, independently_reproduced=false.</b> The August 2026 ROCm FP8 FNUZ '
         'change is source context only, not a hardware result from this study.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:11px 6px}'
              'td{overflow-wrap:anywhere}</style>')
    return page("Round 100 · semi-structured sparse GEMM · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["output_features","input_features","tokens","reuse"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(!["cutlass","cusparselt"].includes(p.backend))throw Error("Unknown backend");
  for(const key of ["pack_ms","launch_us","dense_effective_tflops","sparse_effective_tflops"])
    if(!Number.isFinite(p[key])||p[key]<0||(key.endsWith("tflops")&&p[key]===0))throw Error("Invalid rate/time: "+key);
  const align=(v,a)=>Math.ceil(v/a)*a,[ra,ca]=p.backend==="cutlass"?[32,64]:[16,16];
  const m=p.output_features,k=p.input_features,n=p.tokens,pm=align(m,ra),pk=align(k,ca),pn=align(n,8);
  const tm=align(k,ra),tk=align(m,ca),values=pm*pk,metadata=Math.ceil(pm*pk/8);
  const compressed=values+metadata,transposed=tm*tk+Math.ceil(tm*tk/8),dense=2*m*k;
  if(![pm,pk,pn,tm,tk,values,metadata,compressed,transposed,dense,compressed+transposed].every(Number.isSafeInteger))
    throw Error("Byte budget exceeds exact numerical range");
  const denseMs=2*m*k*n/(p.dense_effective_tflops*1e9)+p.launch_us/1000;
  const sparseMs=2*pm*pk*pn/(p.sparse_effective_tflops*1e9)+p.launch_us/1000;
  const saving=denseMs-sparseMs;
  let minimum=null;
  if(saving>0){
    minimum=Math.floor(p.pack_ms/saving)+1;
    if(!Number.isSafeInteger(minimum))throw Error("Reuse threshold exceeds exact numerical range");
    while(p.pack_ms+minimum*sparseMs>=minimum*denseMs){
      minimum++;
      if(!Number.isSafeInteger(minimum))throw Error("Reuse threshold exceeds exact numerical range");
    }
  }
  const denseTotal=p.reuse*denseMs,sparseTotal=p.pack_ms+p.reuse*sparseMs;
  if(![denseMs,sparseMs,denseTotal,sparseTotal].every(Number.isFinite))throw Error("Modeled times must remain finite");
  return {forward_padded_shape:[pm,pk],transpose_padded_shape:[tm,tk],padded_tokens:pn,
    source_shape_gate_passes_without_padding:m%ra===0&&k%ca===0,original_dense_weight_bytes:dense,
    padded_dense_weight_bytes:2*pm*pk,retained_value_bytes:values,logical_metadata_bytes:metadata,
    logical_compressed_bytes:compressed,both_orientation_logical_bytes:compressed+transposed,
    compressed_to_original_ratio:compressed/dense,dense_modeled_kernel_ms:denseMs,sparse_modeled_kernel_ms:sparseMs,
    minimum_reuse_for_strict_saving:minimum,dense_modeled_total_ms:denseTotal,sparse_modeled_total_ms:sparseTotal,
    gpu_latency_ms:null,model_accuracy:null,actual_backend_packed_bytes:null};
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
      p[control.name]=control.name==="backend"?control.value:Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic logical packing and explicitly assumed rates",
      ...estimateBudget(p),production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {
        "output_features":"Weight rows M / 输出维度","input_features":"Weight columns K / 输入维度",
        "tokens":"Dense operand columns N / Token 数","reuse":"Reuse count / 压缩后复用次数",
        "pack_ms":"Assumed one-time pack ms / 假定压缩成本",
        "dense_effective_tflops":"Assumed dense TFLOP/s / 假定稠密有效算力",
        "sparse_effective_tflops":"Assumed sparse dense-equivalent TFLOP/s / 假定稀疏等效算力",
        "launch_us":"Assumed launch µs / 假定启动时间",
    }
    controls = '<label for="backend">FP16/BF16 backend / 后端<select id="backend" name="backend"><option value="cutlass">CUTLASS</option><option value="cusparselt">cuSPARSELt</option></select></label>'
    for key,label in labels.items():
        floating = key in ("pack_ms","dense_effective_tflops","sparse_effective_tflops","launch_us")
        minimum = 0 if key in ("pack_ms","launch_us") else 0.001 if floating else 1
        controls += f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="{minimum}" step="{0.001 if floating else 1}" value="{p[key]}"></label>'
    return page("2:4 sparse GEMM / 半结构化稀疏工作台",
                '<main><div class="eyebrow">ROUND 100 · LOGICAL 2:4 · SYNTHETIC ASSUMPTIONS</div>'
                '<h1>Compression is not pruning, and zeros are not speedup.<br>压缩不等于剪枝，零值不等于加速。</h1>'
                '<div class="panel"><b>契约：</b>逐 quartet 先明确剪枝，再保留值与索引身份；'
                '只与同一 pruned-dense 参照判定乘法正确性。转置后未必仍满足 2:4。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>仅 FP16/BF16、两字节 values、每 quartet 四个逻辑 metadata bits。'
                'Caller-side 权重 padding 与 N 按 8 对齐为显式假设；源码 sparse conversion 不自动替你 pad。'
                '物理 swizzle/实际 packed allocation 未测。两方向单独 pad/pack 的预算不含训练 mask。'
                '<p>TFLOP/s 均为原稠密等效工作量下的有效率假设，不用营销 sparse peak 再乘 2。'
                '只模型化 kernel+一次 pack，非 GPU 实测、模型精度或完整训练速度。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    b = report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Logical sparse quartet card sorter</title>'
        '<desc id="description">A card sorter retains selected values with their position cards, '
        'then sends them to a reuse tray. A transpose requires its own compatible direction.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 100 · SYNTHETIC DEMO · SEMI-STRUCTURED SPARSE GEMM</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Two retained values still need their position metadata.</text>'
        '<text x="64" y="180" font-size="24">Prune explicitly; compare against the same pruned matrix; preserve both value and index order.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">Row-valid 2:4 does not imply transpose-valid 2:4.</text>'
        +f'<text x="80" y="700" font-size="25">Aligned FP16 logical storage: {b["compressed_to_original_ratio"]*100:.2f}% '
        +f'of dense; modeled default reuse must reach {b["minimum_reuse_for_strict_saving"]} calls.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Metadata, padding, a second orientation and packing overhead all change the tradeoff.</text>'
        +'<text x="96" y="842" font-size="23">Actual backend packing, GPU latency and model accuracy remain unknown.</text>'
        +'<text x="96" y="883" font-size="21">Assumed dense-equivalent rates are not sparse peak FLOPS or measured training acceleration.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/semi_structured_sparsity_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
