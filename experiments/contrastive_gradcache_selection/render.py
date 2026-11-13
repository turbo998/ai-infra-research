"""Views generated only from Round 93's sealed synthetic evidence."""

from html import escape
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "contrastive-embedding-global-negatives-gradcache"


def deck(report: dict) -> str:
    g, budget = report["gradient_check"], report["budget"]
    cards = []
    f = lambda x: f"{x:.2f}"

    def card(eyebrow: str, title: str, evidence: str, takeaway: str) -> None:
        number = len(cards) + 1
        cards.append(
            f'<section class="card" aria-label="Card {number} of 7"><div class="eyebrow">{eyebrow}</div>'
            f'<h1>{title}</h1><p class="muted">SYNTHETIC DEMO · independently_reproduced=false · no trained model</p>'
            f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
            f'<footer><span>Round {report["round"]} · Contrastive gradient semantics</span><span>{number} / 7</span></footer></section>'
        )

    card(
        "HANDS-ON · EMBEDDING TRAINING",
        '<span class="accent">Accumulating gradients does not pool negatives.</span>',
        '<div class="grid"><div class="panel"><h2>Scenarios</h2>'
        + "".join(f'<p>{escape(row["id"])}</p>' for row in g["cases"])
        + '</div><div class="panel"><h2>Contract</h2>'
        + f'<p>{g["pairs"]} pairs<br>{g["workers"]} logical workers<br>one-way InfoNCE<br>shared scalar encoder</p>'
        + '<p>Gate: gradient error below the sealed finite-difference tolerance.</p></div>'
        + '<div class="panel"><h2>Workbench</h2>'
        + f'<p>{budget["local_negatives_per_query"]} local vs {budget["global_negatives_per_query"]} global negatives</p>'
        + f'<p>{f(budget["modeled_compute_overhead_pct"])}% assumed extra compute</p>'
        + '<span class="pill">Tokens: N/A</span><span class="pill">Confidence: null</span></div></div>',
        '<b>Python standard library; no GPU or API.</b> Gradient checks are numerical evidence for this toy '
        'only. Memory and compute budgets are separate analytical assumptions.',
    )
    evidence = f'<blockquote>Reference scalar gradient: {g["reference_gradient"]:.8f}. Fixed dropout masks, '
    evidence += f'temperature {g["temperature"]}; central difference epsilon {g["finite_difference_epsilon"]}.</blockquote>'
    selected = [g["cases"][i] for i in (0, 1, 3)]
    maximum = max(row["relative_gradient_error_pct"] for row in selected)
    for row in selected:
        evidence += (
            '<div class="panel"><span class="pill">score · gradient error</span><span class="pill">Confirm</span>'
            f'<h2>{escape(row["id"])}</h2><p>{row["relative_gradient_error_pct"]:.6f}% relative error · confidence: null</p>'
            + bar(row["relative_gradient_error_pct"], maximum) + '</div>'
        )
    card("SCENARIO: SAME LOSS, WRONG GRADIENT", "Detached candidates can preserve the loss but break its gradient.",
         evidence, '<b>Check derivatives, not just displayed loss.</b> The detached control drops candidate-side '
         'contributions from remote query rows. Matching scalar loss alone cannot catch this.')
    table = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in g["cases"]:
        match = row["matches_global_reference"]
        table += (
            f'<tr><td>{escape(row["id"])}</td><td>Global reference derivative?</td>'
            f'<td>{"Match" if match else "Mismatch ✕"}<br>{row["absolute_gradient_error"]:.3e}</td>'
            f'<td>{"Confirm" if match else "Send to a human"}</td></tr>'
        )
    card("WHAT I TESTED", f'{len(g["cases"])} paths. Three controls expose different failures.',
         table + '<div class="grid two"><div class="callout"><b>Threshold:</b> absolute derivative error '
         + f'&lt; {g["tolerance"]}. Not a model confidence band.</div>'
         + '<div class="callout"><b>Misses:</b> detached targets omit gradients; local negatives change '
         + 'the objective; new dropout changes the replay Jacobian.</div></div>',
         '<b>No retrieval score is inferred.</b> Exact cached replay and the global analytic path match '
         'the independent finite-difference check for these inputs only.')
    card("UNDER THE HOOD", "Cache the embedding derivative, then replay the same encoder.",
         '<pre>' + escape(json.dumps({
             "pass_1": "h = encoder(inputs); no encoder graph",
             "loss": "global InfoNCE over the pooled candidate set",
             "cache": "v = dL/dh; retain per-mini-batch RNG"
         }, indent=2)) + '</pre><p>↓ chain rule, not stale-feature training</p>'
         '<pre>for mini_batch in original_order:\n'
         '  restore(original_rng)\n'
         '  h_replay = encoder(same_inputs, same_weights)\n'
         '  surrogate = dot(h_replay, stop_gradient(v))\n'
         '  backward(surrogate)\n'
         'optimizer_step_after_all_replay()</pre>',
         '<b>Exactness has conditions.</b> Identical weights, stochastic masks, preprocessing and '
         'batch-independent encoder behavior are required; BatchNorm/stateful layers need extra care.')
    table = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>'
    for perturbation in report["perturbations"]:
        by_id = {row["id"]: row for row in perturbation["cases"]}
        exact = by_id["gradcache-exact-replay"]["relative_gradient_error_pct"]
        changed = by_id["gradcache-changed-dropout"]["relative_gradient_error_pct"]
        table += (
            f'<tr><td>Seed {perturbation["seed"]}: derivative error</td><td>same RNG: {exact:.6f}%</td>'
            f'<td>new RNG: {changed:.2f}%</td><td>{changed-exact:+.2f}</td></tr>'
        )
    card("ROBUSTNESS CHECK", "Changing replay dropout breaks the cached chain rule.",
         table + '<div class="panel"><h2>Controlled perturbation</h2><p>The global loss and cached embedding '
         'gradient stay fixed. Only replay masks change. Finite-difference reference holds the original masks fixed.</p>'
         '<p>Neither result is a recall, MRR, loss-convergence or production determinism score.</p></div>',
         '<b>RNG state is part of the replay contract.</b> These seeds illustrate failure, not a population-wide '
         'probability. Data order and mutable module buffers also matter.')
    maximum = max(row["cached_compute_ms"] for row in report["scaling"])
    scaling = '<p>Blue: cached replay · gray: monolithic · assumed compute milliseconds, not end-to-end latency.</p>'
    for row in report["scaling"]:
        scaling += (
            f'<div class="panel"><b>{row["input"]["local_batch"]} local pairs</b><p>Cached <b>'
            f'{f(row["cached_compute_ms"])} ms</b></p>' + bar(row["cached_compute_ms"], maximum)
            + f'<p>Monolithic <b>{f(row["monolithic_compute_ms"])} ms</b></p>'
            + bar(row["monolithic_compute_ms"], maximum, True) + '</div>'
        )
    card("LATENCY", f'One extra forward costs {f(budget["modeled_compute_overhead_pct"])}% in this compute budget.',
         scaling, '<b>Analytical, not measured.</b> Forward/backward cost is an input assumption; networking, '
         'collectives and scheduling are excluded. Median and repeat statistics are N/A.')
    card("MY TAKE", "Use cache replay to fit a global objective, not to promise free throughput.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Large in-batch negative objectives.</p><p>Activation-bound encoders.</p>'
         '<p>Deterministic replay controls.</p><p>Chunked score rows.</p><p>Verified distributed gradients.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Assuming DDP automatically pools negatives.</p>'
         '<p>Detached remote targets as exact training.</p><p>Mutable weights between passes.</p>'
         '<p>Uncontrolled dropout/BatchNorm.</p><p>Ignoring false negatives or quadratic logits.</p></div></div>'
         '<pre>READINESS: original analytical and numerical toy\nRETRIEVAL QUALITY: not measured\n'
         'DATA: experiments/contrastive_gradcache_selection/report.json\n'
         'REFERENCES: pinned Sentence Transformers; GradCache paper\n'
         'RIGHTS: third-party rights retained; no new license grant</pre>',
         '<b>SYNTHETIC DEMO · independently_reproduced=false.</b> Validate single-device versus distributed '
         'parameter gradients before scaling; derive normalization for the actual gather/DDP semantics.')
    return page("Round 93 · Contrastive GradCache · synthetic seven-card report", '<main>' + "".join(cards) + '</main>')


CALCULATOR = r"""
function estimateBudget(p) {
  for (const key of ["local_batch","workers","dimensions","mini_batch","score_rows","embedding_bytes"])
    if (!Number.isSafeInteger(p[key]) || p[key]<1) throw Error(key+" must be a positive safe integer");
  if(p.mini_batch>p.local_batch || p.score_rows>p.local_batch)
    throw Error("Mini-batch and score rows must not exceed local batch");
  for(const key of ["activation_mib_per_pair","forward_ms_per_pair","backward_ms_per_pair"])
    if(!Number.isFinite(p[key]) || p[key]<=0) throw Error(key+" must be finite and positive");
  const n=p.local_batch*p.workers, cache=2*n*p.dimensions*(p.embedding_bytes+4);
  const logitsDense=4*p.local_batch*n, logitsChunked=4*p.score_rows*n;
  const dense=p.local_batch*p.activation_mib_per_pair*2**20+logitsDense;
  const cached=p.mini_batch*p.activation_mib_per_pair*2**20+cache+logitsChunked;
  if(![n,cache,logitsDense,logitsChunked,dense,cached].every(Number.isFinite) ||
     Math.max(n,cache,logitsDense,logitsChunked,dense,cached)>Number.MAX_SAFE_INTEGER)
    throw Error("Budget exceeds safe numerical range");
  const mono=p.local_batch*(p.forward_ms_per_pair+p.backward_ms_per_pair);
  const replay=p.local_batch*(2*p.forward_ms_per_pair+p.backward_ms_per_pair);
  return {global_pairs:n,global_negatives_per_query:n-1,local_negatives_per_query:p.local_batch-1,
    cache_and_embedding_grad_bytes:cache,dense_score_bytes:logitsDense,chunked_score_bytes:logitsChunked,
    dense_modeled_bytes:dense,cached_modeled_bytes:cached,modeled_memory_ratio:dense/cached,
    monolithic_compute_ms:mono,cached_compute_ms:replay,modeled_compute_overhead_pct:100*(replay/mono-1)};
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
    const r=estimateBudget(p);
    result.textContent=JSON.stringify({provenance:"synthetic; not peak HBM or GPU latency",...r,
      quality:null,network_latency_ms:null,production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {
        "local_batch": "Local pairs / 每卡对数", "workers": "Workers / 进程数",
        "dimensions": "Embedding dimension / 嵌入维度", "mini_batch": "Encoder replay mini-batch / 重放微批",
        "score_rows": "Score tile rows / 分块得分行数", "activation_mib_per_pair": "Assumed activation MiB per pair / 激活预算",
        "embedding_bytes": "Embedding bytes per element / 嵌入元素字节",
        "forward_ms_per_pair": "Assumed forward ms per pair / 前向假设耗时",
        "backward_ms_per_pair": "Assumed backward ms per pair / 反向假设耗时",
    }
    controls = "".join(
        f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
        f'min="{1 if isinstance(p[key], int) else 0.001}" step="{1 if isinstance(p[key], int) else 0.001}" value="{p[key]}"></label>'
        for key, label in labels.items()
    )
    return page(
        "Global negatives and GradCache workbench / 全局负样本与缓存重放工作台",
        '<main><div class="eyebrow">ROUND 93 · SYNTHETIC DEMO</div>'
        '<h1>Pool negatives, then replay the encoder.<br>全局对比目标不等于梯度累积。</h1>'
        '<p>Original analytical budget, not GPU peak memory, end-to-end latency or retrieval quality.</p>'
        '<div class="panel"><b>三阶段：</b>无图编码 → 全局 loss 求嵌入梯度 → 相同 RNG/权重重放并反向。'
        '跨卡负样本需显式 gather；DDP 本身不自动扩大 softmax 候选池。</div>'
        f'<form id="calculator"><div class="grid">{controls}</div><p>'
        '<button type="submit">Recalculate / 重新计算</button></p></form>'
        '<p id="error" role="alert" class="error"></p>'
        '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
        '<noscript>Inline JavaScript is required; no estimate is available without it.</noscript>'
        '<div class="panel"><b>预算口径：</b>2 组全局 embeddings + FP32 嵌入梯度，FP32 score 行块；'
        '激活预算已按 query/document 对计。未计模型/优化器、通信缓冲、softmax 临时量、碎片与运行时开销。'
        '<p>更多负样本可能包含更多 false negatives；不能用数量推导 MRR/Recall 改善。'
        '计算时间只按假设的前向/反向单价核算，不是 GPU 或 NCCL 延迟。</p></div></main>',
        "const BASE=" + json.dumps(p) + ";\n" + CALCULATOR,
    )


def sorter_scene() -> str:
    """Three supported card magazines; shared original figure geometry."""
    solids = []

    def pos(x, y, z):
        return 800 + 1.4*(x-y), 450 + 0.45*(x+y) - 1.2*z

    def block(x, y, z, w, d, h, accent=False):
        top = [pos(*p) for p in ((x,y,z+h),(x+w,y,z+h),(x+w,y+d,z+h),(x,y+d,z+h))]
        low = [pos(*p) for p in ((x,y,z),(x+w,y,z),(x+w,y+d,z),(x,y+d,z))]
        shapes = []
        for vertices, fill in (
            ([top[1],top[2],low[2],low[1]], "#475569"),
            ([top[3],top[2],low[2],low[3]], "#64748b"),
            (top, "#b11f4b" if accent else "#e2e8f0"),
        ):
            points = " ".join(f"{a:.2f},{c:.2f}" for a,c in vertices)
            shapes.append(f'<polygon points="{points}" fill="{fill}" stroke="#475569" stroke-width="1"/>')
        solids.append(((z,x+y), "".join(shapes)))

    for x in (-190,170):
        for y in (-110,90):
            block(x,y,-14,22,22,8)
    block(-215,-125,-6,430,255,12)
    for x in (-180,-65,50):
        block(x,-92,6,94,185,10)
        for side in (-92,84):
            block(x,side,16,94,9,16)
        for y in range(-65,75,20):
            block(x+8,y,16,78,6,38,accent=(x==50))
            block(x+16,y+1,54,10,4,2)
    for x in (-205,195):
        for y in (-115,110):
            block(x,y,6,5,5,3)
    return "".join(markup for _,markup in sorted(solids))


def summary_svg(report: dict) -> str:
    b = report["budget"]
    scene = sorter_scene()
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Global negatives and exact GradCache replay</title>'
        '<desc id="description">An isometric index-card sorter: two input magazines and a shared embedding cache. '
        'Original synthetic budget, not measured GPU memory or latency.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 93 · SYNTHETIC DEMO · EMBEDDING TRAINING</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A shared candidate pool needs a shared gradient contract.</text>'
        '<text x="64" y="180" font-size="24">Cache embedding derivatives; replay the same weights, inputs and stochastic masks.</text>'
        + scene
        + '<text x="80" y="650" font-size="26" font-weight="700">Local magazines → pooled candidate drawer → exact encoder replay</text>'
        + f'<text x="80" y="700" font-size="25">Budget: {b["local_negatives_per_query"]} local vs '
        + f'{b["global_negatives_per_query"]} global negatives per query; no retrieval quality inference.</text>'
        + '<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        + '<g fill="#ffffff"><text x="96" y="799" font-size="23">Detached remote features: same loss can hide wrong gradients.</text>'
        + '<text x="96" y="842" font-size="23">New dropout masks on replay: cached derivatives attach to the wrong Jacobian.</text>'
        + '<text x="96" y="883" font-size="21">Gather is explicit · DDP scaling must be derived · score rows must be chunked · false negatives remain</text></g>'
        + '<text x="64" y="955" font-size="20">Data: experiments/contrastive_gradcache_selection/report.json · independently_reproduced=false</text>'
        + '</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path, str]:
    return {
        ROOT / "docs" / "reports" / f"{SLUG}-selection-cards.html": deck(report),
        ROOT / "docs" / "reports" / f"{SLUG}-sizer.html": workbench(report),
        ROOT / "assets" / "svg" / f"{SLUG}-selection-faq-summary.svg": summary_svg(report),
    }
