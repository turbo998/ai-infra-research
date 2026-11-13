"""Self-contained RoPE layout evidence cards, H5 budgets and original SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "rope-interleaved-neox-partial-rotary-layout"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic scalar Q/K vectors · independently_reproduced=false · '
                     'Original Python CPU arithmetic timing separately measured; no native FlashInfer/CUDA execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 114 · RoPE layout contracts</span><span>{n} / 7</span></footer></section>')

    c,b = r["case"],r["budget"]
    card("HANDS-ON · POSITIONAL EMBEDDING KERNELS",
         f'A {c["rotary_dim"]}-value rotary prefix preserves a {c["head_dim"]-c["rotary_dim"]}-value tail only with the right pairing.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>Q position {c["query_position"]}, K position {c["key_position"]}</p>'
         f'<p>Head {c["head_dim"]}; rotary {c["rotary_dim"]}</p></div>'
         '<div class="panel"><h2>Structured input</h2><pre>'
         +json.dumps({"q":c["query"],"k":c["key"]},indent=1)+'</pre></div>'
         '<div class="panel"><h2>Top decisions</h2>'
         f'<p>Reference {c["complex_relative_reference"]:.6f}</p><p>Converted error {c["converted_absolute_error"]:.3e}</p>'
         '<span class="pill">Permute Q and K together</span></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; November 10, 2026 '
         'source cache/layout repair and dimension guard verified. No native bug/fix or GPU speedup reproduced.')
    deep = '<blockquote>Adjacent pairs and split-half pairs represent the same planes only after a matching Q/K coordinate permutation.</blockquote>'
    selected = (("Converted split-half",c["converted_absolute_error"],"Confirm"),
                ("Unconverted layout switch",c["wrong_layout_absolute_error"],"Send to a human"),
                ("Already-rotated key rotated again",c["double_rotation_absolute_error"],"Send to a human"))
    maximum = max(value for _,value,_ in selected)
    for label,value,action in selected:
        deep += f'<div class="panel"><span class="pill">score · absolute dot-score error</span><span class="pill">{action}</span>'
        deep += f'<h2>{label}: {value:.6g}</h2><p>Confidence: null; these are synthetic arithmetic errors, not model probabilities.</p>'
        deep += bar(value,maximum)+'</div>'
    card("SCENARIO: SAME-POSITION CHECKS CAN PASS THE WRONG FORMAT",
         f'Changing layout without conversion moves the cross-position score by {c["wrong_layout_absolute_error"]:.6f}.',
         deep,'<b>False reassurance:</b> both unconverted layouts preserve a same-position dot product, because each '
         'is orthogonal. Cross-position comparisons expose the incompatible frequency-plane assignment.')
    rows = [("Partial prefix","Even and within head dimension?",f'{c["rotary_dim"]} ≤ {c["head_dim"]}',"Confirm"),
            ("Tail","Unchanged?",str(c["tail_preserved"]),"Confirm"),
            ("Converted layout","Complex relative-angle oracle",f'{c["converted_absolute_error"]:.3e} error',"Confirm"),
            ("Same position","Detects wrong layout?","✕ Both preserve raw dot","Insufficient test"),
            ("Cross position","Unconverted layout",f'✕ {c["wrong_layout_absolute_error"]:.6f} error',"Reject"),
            ("Cached key","Apply rotation a second time?",f'✕ {c["double_rotation_absolute_error"]:.6f} error',"Track cache representation")]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Equal-position norm/dot tests do not certify relative-position semantics.</div>'
    matrix += '<div class="callout">Dimension guard, pairing, position and cache state must agree.</div></div>'
    card("WHAT I TESTED","A format flag changes coordinate planes, not merely memory traversal.",
         matrix,'<b>Original plain-RoPE model:</b> float64 scalar arithmetic, fixed theta, no YaRN/Llama3 scaling, '
         'M-RoPE, quantization, attention softmax, model weights, fused GPU schedule or native cache validation.')
    card("UNDER THE HOOD","A shared permutation preserves the relative-angle dot kernel.",
         '<pre>ADJACENT: (x0,x1), (x2,x3), (x4,x5)\n'
         'SPLIT:    (x0,x3), (x1,x4), (x2,x5)\n'
         'convert prefix: [x0,x2,x4,x1,x3,x5]\n'
         'tail stays unchanged\n'
         'per pair: a*cos - b*sin, a*sin + b*cos\n'
         'cache half-width must equal rotary_dim/2</pre>'
         '<p>↓ independently checked with complex relative-angle products</p>'
         '<pre>dot(R(m)q, R(n)k) = dot(q, R(n-m)k)\n'
         'split(Pq,m) = P * adjacent(q,m)\n'
         'split(Pk,n) = P * adjacent(k,n)\n'
         'dot(Pa,Pb) = dot(a,b)\n'
         'same-position tests collapse n-m to zero\n'
         'a post-RoPE cached key must not rotate again</pre>',
         '<b>Checkpoint boundary:</b> equivalent projection conversion permutes the matching Q/K output coordinates '
         'per head. Toggling only the runtime format flag does not convert an existing checkpoint.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>'
    for row in r["variants"]:
        delta = row["wrong_relative_error_pct"]-row["correct_relative_error_pct"]
        robust += f'<tr><td>Shared offset +{row["shared_position_shift"]}; relative error</td>'
        robust += f'<td>{row["correct_relative_error_pct"]:.6g}%</td><td>{row["wrong_relative_error_pct"]:.6f}%</td>'
        robust += f'<td>{delta:+.6f}</td></tr>'
    robust += '</table><div class="panel"><h2>Same-position controls</h2>'
    for row in c["same_position_controls"]:
        robust += f'<p>interleaved={row["interleaved"]}: {row["score"]:.6f}; raw dot {row["unrotated_score"]:.6f}</p>'
    robust += '</div>'
    card("ROBUSTNESS CHECK","Shared position shifts preserve the correct kernel, not a wrong checkpoint layout.",
         robust,'<b>Plain-RoPE invariant only.</b> Floating arithmetic is checked within sealed tolerance, not promised '
         'bitwise equal at arbitrary positions or across GPU dtypes. Perturbation errors are not quality percentages.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: scalar paired rotations · Gray: dense rotation-matrix application. Coefficients/matrices precomputed.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>{row["rotary_dim"]} rotary values</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","Plane rotations avoid dense matrix arithmetic without changing the vector.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; 25 operations per sample, '
         'median repeat means. Coefficient/matrix construction excluded; no FlashInfer kernel or GPU scheduling speedup.')
    card("MY TAKE",
         f'Default selected coefficient cache is {b["selected_cos_sin_cache_bytes"]:,} bytes, separate from Q/K payload.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Declared frequency-plane format.</p>'
         '<p>Matched Q/K coordinate conversion.</p><p>Cross-position and untouched-tail tests.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Format flag alone called checkpoint conversion.</p>'
         '<p>Same-position checks called full parity.</p><p>Double rotating post-RoPE keys.</p></div></div>'
         '<pre>READINESS: original plain scalar RoPE contracts\n'
         'SOURCE: pinned FlashInfer, Apache-2.0\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'NATIVE CUDA / FIX REPRODUCTION: not executed\n'
         'MODEL QUALITY / PEAK HBM: unknown\n'
         'PAYLOAD IS NOT EXTRA ALLOCATION</pre>',
         f'<b>Selected full Q/K output payload:</b> {b["selected_full_qk_payload_bytes"]:,} bytes; '
         f'rotated prefix {b["selected_rotated_prefix_payload_bytes"]:,}. '
         'Cache instances are an assumption; layers may share configuration-compatible coefficients.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 114 · RoPE layout and partial dimensions · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["max_positions","rotary_dim","head_dim","cache_element_bytes","cache_instances","active_rows","query_heads","key_heads","qk_element_bytes"]){
    const min=key==="active_rows"?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" requires an exact integer >= "+min);
  }
  if(p.rotary_dim%2||p.rotary_dim>p.head_dim)throw Error("rotary_dim must be even and within head_dim");
  if(p.cache_element_bytes!==4||![2,4,8].includes(p.qk_element_bytes))throw Error("pinned cache API requires float32 coefficients; unsupported selected Q/K width");
  const rows=BigInt(p.active_rows)*(BigInt(p.query_heads)+BigInt(p.key_heads)),width=BigInt(p.qk_element_bytes);
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  return {selected_cos_sin_cache_bytes:exact(BigInt(p.max_positions)*BigInt(p.rotary_dim)*BigInt(p.cache_element_bytes)*BigInt(p.cache_instances)),
    selected_full_qk_payload_bytes:exact(rows*BigInt(p.head_dim)*width),
    selected_rotated_prefix_payload_bytes:exact(rows*BigInt(p.rotary_dim)*width),
    selected_untouched_tail_payload_bytes:exact(rows*BigInt(p.head_dim-p.rotary_dim)*width),
    extra_allocation_bytes:null,native_rope_latency_ms:null,peak_hbm_bytes:null,model_quality:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const p={};
    for(const key of ["max_positions","rotary_dim","head_dim","cache_element_bytes","cache_instances","active_rows","query_heads","key_heads","qk_element_bytes"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic coefficient cache and selected Q/K scalar payload",
      ...estimateBudget(p),production_approved:false,
      note:"Pinned cached API requires float32 coefficients: rotary_dim/2 cosines plus rotary_dim/2 sines per position. Cache instances are selected assumptions, not layer count by default. Q/K payload is not extra allocation or memory traffic; inplace/fused behavior is not executed. This does not certify kernel Q/K dtype/shape support, cache-index bounds, model quality or native precision."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    controls = ""
    for key,label in (("max_positions","Cached positions / 缓存位置数"),("rotary_dim","Rotary prefix / 旋转前缀维数"),
                      ("head_dim","Head dimension / Head 维数"),("cache_element_bytes","Coefficient bytes / 系数字节"),
                      ("cache_instances","Selected cache instances / Cache 份数"),("active_rows","Active token rows / 活跃行数"),
                      ("query_heads","Query heads / Q heads"),("key_heads","Key heads / K heads"),
                      ("qk_element_bytes","Q/K bytes / QK 字节")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{r["budget"]["input"][key]}"></label>'
    return page("RoPE layouts / 旋转位置编码与 Partial Dimension 工作台",
                '<main><div class="eyebrow">ROUND 114 · PAIRING · POSITION · PARTIAL PREFIX</div>'
                '<h1>A layout flag is not a checkpoint conversion.<br>改 Layout Flag，不等于转换权重。</h1>'
                '<div class="panel">Adjacent pairs 与 split-half pairs 需同步 permute Q/K prefix 才等价。'
                'Head=8、rotary=6 的两条 tail 保持原样；same-position dot tests 单独使用不能识别错误配对。'
                '<p>已是 post-RoPE 的 cached K 不再旋转；还需验证真实 position index、dtype 与 cache configuration。</p></div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>Cache=positions×rotary_dim×selected bytes×instances；'
                '不同 layers 可共享同配置 cache，不默认乘层数。Q/K 全 output payload 不等于额外 allocation 或 traffic。'
                '<p>原创模型是 plain RoPE float64 标量算术，不执行 FlashInfer/CUDA、特殊 RoPE scaling、'
                'native cache bounds、GPU shape support、quantized checkpoint 或真实模型 quality。</p></div></main>',CALCULATOR)


def summary_svg(r):
    c = r["case"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Coordinate-pair card sorter</title>'
        '<desc id="description">An original sixty-solid index-card sorter represents matching query/key '
        'coordinate permutations before plane rotation; untouched tail cards stay outside the rotary prefix.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 114 · ROPE PAIRING · PARTIAL PREFIX AND RELATIVE POSITIONS</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Same-position dot checks can pass the wrong RoPE layout.</text>'
        '<text x="64" y="180" font-size="24">A coordinate-card sorter analogy: convert Q and K pairing together, preserve the unrotated tail.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Head {c["head_dim"]}, rotary {c["rotary_dim"]}; converted score error {c["converted_absolute_error"]:.3e}.</text>'
        +f'<text x="80" y="700" font-size="25">Wrong unconverted cross-position score error {c["wrong_layout_absolute_error"]:.6f}; tail preserved: {c["tail_preserved"]}.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Adjacent pairs and split-half pairs agree only with the matching Q/K coordinate permutation.</text>'
        +'<text x="96" y="842" font-size="23">Relative-angle oracle checks cross positions; cached post-RoPE keys must not rotate twice.</text>'
        +'<text x="96" y="883" font-size="21">November 10 source repair tracked; original scalar study is not a native FlashInfer failure/fix reproduction.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/rope_layout_partial_dimension_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
