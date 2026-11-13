"""Reproducible Round 94 views of sealed loss and collective evidence."""

from html import escape
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import bar, page
from experiments.contrastive_gradcache_selection.render import sorter_scene


ROOT = Path(__file__).resolve().parents[2]
SLUG = "vocabulary-parallel-cross-entropy-global-logsumexp"


def deck(report: dict) -> str:
    b = report["budget"]
    cards = []

    def card(eyebrow, title, evidence, takeaway):
        number = len(cards) + 1
        cards.append(
            f'<section class="card" aria-label="Card {number} of 7"><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{title}</h1><p class="muted">SYNTHETIC DEMO · independently_reproduced=false · no GPU measurements</p>'
            f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
            f'<footer><span>Round {report["round"]} · Vocabulary-parallel loss</span><span>{number} / 7</span></footer></section>'
        )

    card("HANDS-ON · DISTRIBUTED LOSS",
         f'<span class="accent">{b["logits_storage_ratio"]:.0f} shards replace replicated full logits.</span>',
         '<div class="grid"><div class="panel"><h2>Scenarios</h2>'
         + "".join(f'<p>{escape(row["id"])}</p>' for row in report["cases"])
         + '</div><div class="panel"><h2>Contract</h2><p>Unsmoothed exact CE</p>'
         + f'<p>{report["fixture"]["vocabulary"]} real classes<br>{report["fixture"]["partitions"]} toy shards</p>'
         + '<p>Global MAX, target SUM, denominator SUM</p></div>'
         + '<div class="panel"><h2>Budget</h2>'
         + f'<p>{b["input"]["tokens"]} token rows<br>{b["input"]["vocabulary"]} classes<br>{b["input"]["partitions"]} partitions</p>'
         + '<span class="pill">Confidence: null</span><span class="pill">GPU latency: null</span></div></div>',
         '<b>Python standard-library numerical checks and separate byte/startup accounting.</b> '
         'Storage is logits only, not a peak-HBM guarantee. No trained-model score is produced.')
    target = next(row for row in report["cases"] if row["id"] == "target-on-every-shard")
    exact = report["cases"][0]
    evidence = '<blockquote>Only the rank owning the global target subtracts one. '
    evidence += 'Other ranks retain their local probabilities, not an extra local target.</blockquote>'
    for row in (exact, target):
        evidence += (
            '<div class="panel"><span class="pill">score · derivative error</span><span class="pill">'
            + ('Confirm' if row["matches_reference"] else 'Send to a human')
            + f'</span><h2>{escape(row["id"])}</h2><p>Loss {row["loss"]:.8f} · maximum gradient error '
            + f'{row["maximum_gradient_error"]:.3e} · confidence: null</p>'
            + bar(row["maximum_gradient_error"], target["maximum_gradient_error"]) + '</div>'
        )
    card("SCENARIO: GLOBAL TARGET OWNERSHIP", "The same forward loss can hide an extra target on every shard.",
         evidence, '<b>Global IDs need explicit offsets and ownership.</b> Matching loss alone does not '
         'validate a custom backward implementation.')
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        matrix += f'<tr><td>{escape(row["id"])}</td><td>Dense loss/gradient parity?</td>'
        matrix += f'<td>{"Match" if row["matches_reference"] else "Mismatch ✕"}<br>{row["maximum_gradient_error"]:.3e}</td>'
        matrix += f'<td>{"Confirm" if row["matches_reference"] else "Send to a human"}</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout"><b>Gate:</b> loss and max gradient error '
    matrix += f'below {report["reference"]["tolerance"]}.</div><div class="callout"><b>Four misses:</b> '
    matrix += 'wrong denominator, target ownership, padding and local-only objective.</div></div>'
    card("WHAT I TESTED", f'{len(report["cases"])} protocol paths; only the exact path passes.',
         matrix, '<b>Hand-designed correctness controls, not model accuracy.</b> Every coordinate also has '
         'a dense finite-difference check in the local tests.')
    card("UNDER THE HOOD", "Reduce three token-wise statistics, not the full vocabulary matrix.",
         '<pre>m = MAX_r(max(local_logits_r))\n'
         's = SUM_r(sum(exp(local_logits_r - m)))\n'
         't = SUM_r(owner_r ? local_target_logit - m : 0)\n'
         'loss = log(s) - t</pre><p>↓ local backward with the shared global denominator</p>'
         '<pre>p_r = exp(local_logits_r - m) / s\n'
         'grad_r = (p_r - owner_target_one_hot_r) * upstream\n\n'
         'Mask non-real padded classes before reductions.\n'
         'Apply token masking and mean normalization explicitly.</pre>',
         '<b>Tensor parallelism is not data parallelism.</b> Loss values are replicated across vocabulary '
         'ranks; do not average this denominator or divide by TP size again.')
    robustness = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["perturbations"]:
        robustness += f'<tr><td>{escape(row["id"])}</td><td>{exact["loss"]:.10f}</td>'
        robustness += f'<td>{row["loss"]:.10f}</td><td>{row["absolute_loss_error"]:.3e}</td></tr>'
    robustness += '</table><div class="panel"><h2>Stress boundaries</h2><p>Logits near +1000 and -1000, '
    robustness += 'target ids spanning vocabulary shards, non-real padded classes and an ignored token.</p>'
    robustness += '<p>All-ignored rows return loss=null plus an explicit skip-step status, not a fabricated zero loss.</p></div>'
    card("ROBUSTNESS CHECK", "Changing shard order and a common logit shift preserves the objective.",
         robustness, '<b>Stable global shifting avoids direct exp overflow.</b> These tests cover the original '
         'fixture, not all backend dtypes, reductions or label-smoothing paths.')
    maximum = max(row["allgather_modeled_ms"] for row in report["scaling"])
    scaling = '<p>Blue: three-statistic reductions · gray: full-logit AllGather. Assumed ring communication ms only.</p>'
    for row in report["scaling"]:
        scaling += f'<div class="latency-row"><b>{row["input"]["tokens"]} token rows</b>'
        scaling += f'<div class="latency-pair"><span>Statistics <b>{row["stats_modeled_ms"]:.5f} ms</b></span>'
        scaling += bar(row["stats_modeled_ms"], maximum) + '</div>'
        scaling += f'<div class="latency-pair"><span>Gather <b>{row["allgather_modeled_ms"]:.5f} ms</b></span>'
        scaling += bar(row["allgather_modeled_ms"], maximum, True) + '</div></div>'
    card("LATENCY", "Three startups can lose at one token despite far fewer bytes.",
         scaling, '<b>Not GPU latency.</b> Effective bandwidth and per-hop alpha are assumptions; one deterministic '
         'calculation per tier, median/repeats N/A. No training-step speedup is inferred.')
    card("MY TAKE", "Keep exact global normalization; shard columns without changing the objective.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Large-vocabulary output heads.</p><p>Existing vocabulary TP.</p>'
         '<p>Exact full-softmax objective.</p><p>Explicit offsets and valid-token weights.</p>'
         '<p>Dense-reference gradient tests.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Claiming all output-head traffic disappears.</p><p>AVG instead of denominator SUM.</p>'
         '<p>Unmasked padded classes.</p><p>Unverified smoothing or class weights.</p>'
         '<p>Sampled softmax as an identical objective.</p></div></div>'
         '<pre>READINESS: original analytical/numerical prototype\nQUALITY / GPU LATENCY: not measured\n'
         'DATA: experiments/vocab_parallel_cross_entropy_selection/report.json\n'
         'SOURCE: pinned Megatron-LM core/ops/loss/cross_entropy.py\n'
         'RIGHTS: upstream rights retained; no implementation copied</pre>',
         '<b>SYNTHETIC DEMO · independently_reproduced=false.</b> The pinned legacy path is a compatibility '
         'alias. Padding, ignored-token policy and finite differences are harness extensions, not API guarantees.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:250px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}</style>')
    return page("Round 94 · Vocabulary-parallel CE · synthetic seven cards", styles + '<main>' + "".join(cards) + '</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["tokens","vocabulary","partitions"])
    if(!Number.isSafeInteger(p[key]) || p[key]<1) throw Error(key+" must be a positive safe integer");
  if(p.partitions>p.vocabulary) throw Error("Partitions cannot exceed vocabulary");
  if(!Number.isFinite(p.bandwidth_gbps) || p.bandwidth_gbps<=0) throw Error("Bandwidth must be finite and positive");
  if(!Number.isFinite(p.alpha_us) || p.alpha_us<0) throw Error("Alpha must be finite and nonnegative");
  const width=Math.ceil(p.vocabulary/p.partitions), padded=width*p.partitions;
  const full=4*p.tokens*padded, local=4*p.tokens*width;
  if(!Number.isSafeInteger(full)) throw Error("Logits budget exceeds exact numerical range");
  const gather=full*(p.partitions-1)/p.partitions;
  const stats=3*4*p.tokens*2*(p.partitions-1)/p.partitions;
  return {padded_vocabulary:padded,padding_classes:padded-p.vocabulary,
    local_fp32_logits_bytes:local,gathered_fp32_logits_bytes:full,logits_storage_ratio:full/local,
    allgather_send_bytes_per_rank:gather,stats_send_bytes_per_rank:stats,
    forward_wire_ratio:stats ? gather/stats : null,
    allgather_modeled_ms:gather/(p.bandwidth_gbps*1e6)+(p.partitions-1)*p.alpha_us/1000,
    stats_modeled_ms:stats/(p.bandwidth_gbps*1e6)+6*(p.partitions-1)*p.alpha_us/1000,
    forward_statistics_collectives:3};
}
const form=document.getElementById("calculator"), result=document.getElementById("result"),
  error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="Invalid input; correct the highlighted field.";return;}
  try{
    const p={...BASE};
    for(const control of form.elements) if(control.name){
      if(control.value.trim()==="") throw Error(control.name+" is required");
      p[control.name]=Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic; forward CE communication budget only",
      ...estimateBudget(p),gpu_latency_ms:null,training_quality:null,production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {"tokens": "Token rows / 参与 CE 的行数", "vocabulary": "Real vocabulary / 真实词表",
              "partitions": "Vocabulary TP ranks / 词表分片数", "bandwidth_gbps": "Effective GB/s / 有效字节带宽",
              "alpha_us": "Per-hop startup µs / 单跳启动"}
    controls = "".join(
        f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
        f'min="{0 if key=="alpha_us" else 0.001 if key=="bandwidth_gbps" else 1}" '
        f'step="{0.001 if key in ("alpha_us","bandwidth_gbps") else 1}" value="{p[key]}"></label>'
        for key,label in labels.items()
    )
    return page(
        "Vocabulary-parallel CE workbench / 词表并行交叉熵工作台",
        '<main><div class="eyebrow">ROUND 94 · SYNTHETIC DEMO</div>'
        '<h1>One global denominator.<br>词表分片不应改变 softmax 目标。</h1>'
        '<p>FP32 logits/statistics accounting only, not GPU latency or end-to-end training performance.</p>'
        '<div class="panel"><b>协议：</b>全局 MAX → target SUM → exp-denominator SUM。'
        '只在 target owner 分片减一；TP 不是 DP，不能额外平均词表分母。</div>'
        f'<form id="calculator"><div class="grid">{controls}</div><p>'
        '<button type="submit">Recalculate / 重新计算</button></p></form>'
        '<p id="error" class="error" role="alert"></p>'
        '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
        '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
        '<div class="panel"><b>口径：</b>理想 Ring 的每 rank 发送字节与启动开销；'
        '三个统计归约顺序执行，不赠送 kernel fusion/overlap 收益。GB/s 不是 Gb/s。'
        '<p>未计 GEMM、hidden gradient collective、优化器、峰值临时内存或反向流量。'
        '非整除词表按等宽 padding；padding 类必须排除。P=1 的通信比值保持 null。</p></div></main>',
        "const BASE=" + json.dumps(p) + ";\n" + CALCULATOR,
    )


def summary_svg(report: dict) -> str:
    b = report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Vocabulary shards share one exact denominator</title>'
        '<desc id="description">Three physical card magazines represent the three vocabulary shards in the toy. '
        'Their maxima and sums feed a common exact loss. Synthetic, not measured GPU performance.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 94 · SYNTHETIC DEMO · VOCABULARY PARALLELISM</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Shard the vocabulary, not the meaning of cross-entropy.</text>'
        '<text x="64" y="180" font-size="24">Global MAX + target SUM + denominator SUM; subtract one only on the owning shard.</text>'
        + sorter_scene()
        + '<text x="80" y="650" font-size="26" font-weight="700">Three toy shards → one global normalizer → local exact gradients</text>'
        + f'<text x="80" y="700" font-size="25">Separate budget: {b["logits_storage_ratio"]:.0f}× smaller FP32 logits '
        + 'storage per rank; this is not a peak-HBM or training-speed claim.</text>'
        + '<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        + '<g fill="#ffffff"><text x="96" y="799" font-size="23">Global target ids need shard offsets; non-real padded classes must be masked.</text>'
        + '<text x="96" y="842" font-size="23">Ignored tokens need explicit reduction policy; an all-ignored batch is not zero loss.</text>'
        + '<text x="96" y="883" font-size="21">Startup-bound small batches can lose to gathering · hidden-gradient traffic still exists</text></g>'
        + '<text x="64" y="955" font-size="20">Data: experiments/vocab_parallel_cross_entropy_selection/report.json · independently_reproduced=false</text>'
        + '</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT / "docs" / "reports" / f"{SLUG}-selection-cards.html": deck(report),
        ROOT / "docs" / "reports" / f"{SLUG}-sizer.html": workbench(report),
        ROOT / "assets" / "svg" / f"{SLUG}-selection-faq-summary.svg": summary_svg(report),
    }
