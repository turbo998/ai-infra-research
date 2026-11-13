"""Sealed-data clipping cards and a partial-memory/noise formula workbench."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "dp-sgd-ghost-clipping-per-example-sequence-norm"


def deck(report: dict) -> str:
    b = report["budget"]
    cases = {row["id"]:row for row in report["cases"]}
    cards = []

    def card(eyebrow,title,evidence,takeaway):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7">'
                     f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                     '<p class="muted">SYNTHETIC · independently_reproduced=false · no DP guarantee, secure RNG or GPU measurement</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 99 · per-example clipping contracts</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · PRIVATE-TRAINING MECHANICS",
         'Clip <span class="accent">individual examples</span> before accumulating their gradients.',
         '<div class="grid"><div class="panel"><h2>Fixture</h2>'
         '<p>3 examples<br>2 tokens per example<br>6 model parameters</p>'
         '<span class="pill">Flat clip bound: 2</span></div><div class="panel"><h2>Sequence contract</h2>'
         '<p>Sum token outer products.</p><p>Keep all Gram cross terms.</p><p>Include bias in the global norm.</p>'
         '</div><div class="panel"><h2>Separate budget</h2>'
         f'<p>{b["input"]["physical_examples"]} physical examples<br>{b["physical_chunks"]} physical chunks<br>'
         f'{b["dense_to_gram_ratio"]:.0f}× selected-buffer ratio</p>'
         '<span class="pill">Privacy epsilon: null</span></div></div>',
         '<b>Original scalar model with independent finite differences.</b> '
         'This checks clipping arithmetic, not a private training implementation; no Opacus/GPU execution.')
    wrong = cases["diagonal-only"]
    exact = cases["sequence-gram-ghost"]
    maximum = max(wrong["individual_clipped_norms"])
    detail = '<blockquote>Two aligned token gradients reinforce each other. '
    detail += 'Dropping cross-token terms underestimates the example norm.</blockquote>'
    for row in (exact,wrong):
        value = max(row["individual_clipped_norms"])
        detail += '<div class="panel"><span class="pill">score · largest clipped-example norm</span>'
        detail += '<span class="pill">'+('Confirm' if row["matches_reference"] else 'Send to a human')+'</span>'
        detail += f'<h2>{row["id"]}</h2><p>{value:.6f} versus bound 2 · confidence: null</p>'
        detail += bar(value,maximum)+'</div>'
    card("SCENARIO: SEQUENCE CROSS TERMS",
         f'Ignoring cross terms leaves a {maximum:.4f} norm above the chosen bound 2.',
         detail,'<b>Scope:</b> the selected flat-clipping contract fails. '
         'No privacy-budget result or broad impossibility claim is inferred from this counterexample.')
    names = {
        "sequence-gram-ghost":"Full sequence Gram",
        "diagonal-only":"Only token-diagonal terms",
        "frobenius-product":"Whole-sequence norm product",
        "clip-after-example-aggregation":"Clip after summing examples",
        "independent-weight-bias-bounds":"Separate weight/bias bounds",
        "stopped-coefficient-replay":"Stopped-coefficient replay",
        "differentiate-through-coefficients":"Differentiate clip factors",
    }
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        matrix += f'<tr><td>{names[row["id"]]}</td><td>Same flat-clipped sum?</td>'
        matrix += f'<td>{"Match" if row["matches_reference"] else "Different ✕"}<br>{row["maximum_absolute_error"]:.2e}</td>'
        matrix += f'<td>{"Confirm" if row["matches_reference"] else "Review policy"}</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout"><b>Gate:</b> max update error &lt; 1e-8.</div>'
    matrix += '<div class="callout"><b>Interpretation:</b> a conservative norm bound or another clipping policy is not automatically a privacy violation.</div></div>'
    card("WHAT I TESTED","Exact Gram norms and stopped replay preserve the chosen update.",
         matrix,'<b>Separate policies stay separate.</b> Per-parameter clipping and whole-sequence norm '
         'bounds must not be labeled equivalent to one global per-example norm.')
    card("UNDER THE HOOD","Sequence ghost norms need every token-pair inner product.",
         '<pre>G_i = SUM_t outer(delta_it, activation_it)\n'
         '||G_i||² = SUM_t,u dot(delta_it, delta_iu)\n'
         '                      * dot(activation_it, activation_iu)\n'
         'bias_i = SUM_t delta_it\n'
         'global_norm_i = sqrt(||G_i||² + ||bias_i||²)</pre>'
         '<p>↓ stop coefficients before the weighted second backward</p>'
         '<pre>c_i = min(1, C / (global_norm_i + 1e-6))\n'
         'clipped_sum = SUM_i stop_gradient(c_i) * gradient_i\n'
         'noisy_sum = clipped_sum + sigma*C*z\n'
         'release = noisy_sum / expected_batch_size\n\n'
         'z here is a fixed synthetic vector, NOT Gaussian RNG.\n'
         'No accountant or privacy guarantee is provided.</pre>',
         '<b>Source boundary:</b> Opacus also accounts for accumulated_iterations in mean scaling. '
         'The toy fixes one logical release and an explicit expected denominator.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["partition_checks"]:
        robust += '<tr><td>Same logical clipped sum</td><td>[3] examples</td>'
        robust += f'<td>{row["physical_sizes"]}</td><td>Max error {row["maximum_absolute_error"]:.1e}</td></tr>'
    release = report["synthetic_release"]
    robust += '</table><div class="panel"><h2>Expected count is not actual sampled count</h2>'
    robust += f'<p>Fixture: actual {release["actual_examples"]}, explicit expected denominator {release["expected_batch_size"]}.</p>'
    robust += '<p>Replacing denominator 4 with 3 multiplies the same synthetic release by 4/3.</p>'
    robust += '<p>Independent repeated noise additions at the same final denominator increase variance.</p></div>'
    card("ROBUSTNESS CHECK","Four physical partitions retain one logical clipped sum.",
         robust,'<b>Do not clip a microbatch average instead of its examples.</b> Physical partition '
         'invariance is numerical evidence, not a validation of Poisson sampling or a DP accountant.')
    maximum = max(max(row["dense_one_pass_read_modeled_ms"],row["gram_one_pass_read_modeled_ms"])
                  for row in report["scaling"])
    timing = '<p>Modeled one-pass read: blue two sequence Gram buffers; gray materialized per-example weights.</p>'
    for row in report["scaling"]:
        timing += f'<div class="latency-row"><b>{row["input"]["tokens"]} tokens · {row["input"]["physical_examples"]} physical examples</b>'
        for key,label,baseline in (("gram_one_pass_read_modeled_ms","Two Grams",False),
                                   ("dense_one_pass_read_modeled_ms","Dense gradients",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","Long sequences can make two Gram buffers larger than dense gradients.",
         timing,'<b>Assumed 1000 GB/s + 5 µs, one read only.</b> '
         'The T² Gram shape is not always smaller than Din×Dout. Compute, replay and other storage '
         'excluded; not measured GPU latency. Median/repeats N/A.')
    card("MY TAKE","A clipping arithmetic check is not a private-training certificate.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Per-example norm parity and finite differences.</p><p>Stopped weighted replay.</p>'
         '<p>Explicit logical/physical batch boundaries.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Clipping averaged examples as a shortcut.</p><p>Dropping sequence cross terms.</p>'
         '<p>Deriving epsilon without an accountant.</p></div></div>'
         '<pre>READINESS: original numerical prototype\n'
         'SOURCE: pinned Opacus; upstream Apache-2.0 rights retained\n'
         'NOISE: fixed vector, no Gaussian or secure generator\n'
         'ACCOUNTANT / SAMPLER / PRIVATE TRAINING: not implemented\n'
         'GPU / MODEL QUALITY / PEAK HBM: not measured\n'
         'EPSILON / DELTA: null, not zero</pre>',
         '<b>Synthetic, independently_reproduced=false.</b> Established training mechanics, '
         'not a new private-model result or a reproduction of mixed-precision hardware behavior.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}'
              'td{overflow-wrap:anywhere}</style>')
    return page("Round 99 · per-example ghost clipping · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["logical_examples","physical_examples","tokens","input_dimension","output_dimension","scalar_bytes","expected_batch_size"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(p.physical_examples>p.logical_examples||![4,8].includes(p.scalar_bytes))
    throw Error("Physical examples must fit the logical batch; scalar bytes must be 4 or 8");
  if(!Number.isFinite(p.clip)||p.clip<=0)throw Error("Clip must be finite positive");
  if(!Number.isFinite(p.noise_multiplier)||p.noise_multiplier<0)throw Error("Noise multiplier must be finite nonnegative");
  if(!Number.isFinite(p.assumed_read_gbps)||p.assumed_read_gbps<=0)throw Error("Bandwidth must be finite positive");
  if(!Number.isFinite(p.setup_us)||p.setup_us<0)throw Error("Setup must be finite nonnegative");
  const dense=p.physical_examples*p.input_dimension*p.output_dimension*p.scalar_bytes;
  const gram=2*p.physical_examples*p.tokens*p.tokens*p.scalar_bytes;
  const activation=p.physical_examples*p.tokens*(p.input_dimension+p.output_dimension)*p.scalar_bytes;
  if(![dense,gram,activation].every(Number.isSafeInteger))throw Error("Byte budget exceeds exact numerical range");
  const chunks=Math.ceil(p.logical_examples/p.physical_examples);
  const noise=p.noise_multiplier*p.clip,std=noise/p.expected_batch_size,repeated=std*Math.sqrt(chunks);
  if(![noise,std,repeated].every(Number.isFinite))throw Error("Noise formula exceeds finite numerical range");
  return {physical_chunks:chunks,per_example_weight_gradient_bytes:dense,two_sequence_gram_bytes:gram,
    activation_backprop_bytes:activation,dense_to_gram_ratio:dense/gram,mean_release_noise_std:std,
    repeated_chunk_noise_std:repeated,
    dense_one_pass_read_modeled_ms:dense/(p.assumed_read_gbps*1e6)+p.setup_us/1000,
    gram_one_pass_read_modeled_ms:gram/(p.assumed_read_gbps*1e6)+p.setup_us/1000,
    privacy_epsilon:null,privacy_delta:null,peak_hbm_bytes:null,gpu_latency_ms:null};
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
    result.textContent=JSON.stringify({provenance:"synthetic partial buffers; assumed Gaussian noise scaling only",
      ...estimateBudget(p),secure_rng:false,private_training_approved:false,
      note:p.noise_multiplier===0?"Zero multiplier adds no DP noise; no privacy guarantee.":"No sampler, accountant or secure generator is implemented."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {
        "logical_examples":"Logical examples / 逻辑批大小","physical_examples":"Physical chunk examples / 物理批",
        "tokens":"Tokens per example / 每例 token","input_dimension":"Input dimension / 输入维度",
        "output_dimension":"Output dimension / 输出维度","scalar_bytes":"Scalar bytes (4/8) / 标量字节",
        "clip":"Flat clip C / 全局裁剪阈值","noise_multiplier":"Noise multiplier sigma / 噪声倍率",
        "expected_batch_size":"Expected mean denominator / 期望分母",
        "assumed_read_gbps":"Assumed one-pass GB/s / 假定读取带宽",
        "setup_us":"Assumed setup µs / 假定启动时间",
    }
    controls = ""
    for key,label in labels.items():
        floating = key in ("clip","noise_multiplier","assumed_read_gbps","setup_us")
        minimum = 0 if key in ("noise_multiplier","setup_us") else 0.001 if floating else 1
        controls += f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="{minimum}" step="{0.001 if floating else 1}" value="{p[key]}"></label>'
    return page("Ghost clipping / 逐样本裁剪工作台",
                '<main><div class="eyebrow">ROUND 99 · SYNTHETIC MECHANICS · NOT PRIVATE TRAINING</div>'
                '<h1>Keep example identity until clipping.<br>逐样本裁剪后，再累积。</h1>'
                '<div class="panel"><b>契约：</b>序列 Gram 保留 token-pair cross terms；weight/bias 合成全局 norm；'
                '二次 weighted backward 固定裁剪系数。固定噪声向量只是合成示例，不是 Gaussian/secure RNG。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>只预算一个 Linear 层物理 chunk 的逐例 weight gradients、两份 T² Gram、'
                'activation/backprop buffers，不代表 peak HBM。噪声标准差 sigma*C/E 是给定 Gaussian 假设的公式；'
                '每 chunk 独立加同强度噪声再合并，最终分母不变时 std 多 sqrt(K)。'
                '<p>没有 sampler、accountant、epsilon/delta 或私人训练认证。一次读取时间不是 GPU 执行延迟。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    b = report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Per-example gradient card sorter</title>'
        '<desc id="description">Individual example cards pass the clipping gate before entering '
        'the shared accumulation tray. Token-pair cross terms retain each example identity.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 99 · SYNTHETIC DEMO · PER-EXAMPLE GHOST CLIPPING</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Clip each example before the accumulation tray.</text>'
        '<text x="64" y="180" font-size="24">Sequence cross terms define its norm; stop clipping coefficients during weighted replay.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">Dropping token cross terms can leave a clipped norm above its chosen bound.</text>'
        +f'<text x="80" y="700" font-size="25">Separate default partial-buffer ratio: {b["dense_to_gram_ratio"]:.0f}×; '
        +'longer sequences can reverse the memory advantage.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Physical chunks preserve one logical sum; Gaussian noise scaling needs an explicit denominator.</text>'
        +'<text x="96" y="842" font-size="23">Fixed noise vector is not Gaussian RNG; epsilon and delta remain unknown, not zero.</text>'
        +'<text x="96" y="883" font-size="21">No sampler, accountant, secure generator, private training guarantee or GPU benchmark.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/ghost_clipping_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
