"""Self-contained views of GNN dependency and fanout evidence."""

from html import escape
import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT=Path(__file__).resolve().parents[2]
SLUG="gnn-neighbor-sampling-dependency-fanout"


def deck(report: dict) -> str:
    b=report["budget"]
    cases={row["id"]:row for row in report["cases"]}
    cards=[]

    def card(eyebrow,title,evidence,takeaway):
        n=len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">SYNTHETIC DEMO · independently_reproduced=false · no PyG/GPU or model-quality measurement</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 97 · GNN neighbor dependencies</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · GRAPH MINIBATCHES","Sample dependencies, preserve seed identity and message direction.",
         '<div class="grid"><div class="panel"><h2>Original graph</h2>'
         f'<p>{len(report["fixture"]["incoming_neighbors"])} nodes<br>{len(report["fixture"]["seeds"])} seeds<br>'
         f'{report["fixture"]["layers"]} mean-message layers</p><p>ReLU and self features</p></div>'
         '<div class="panel"><h2>Contract</h2><p>Incoming dependencies</p><p>Seed-first local order</p>'
         '<p>Global n_id feature mapping</p><p>Seed-only loss</p></div><div class="panel"><h2>Separate budget</h2>'
         f'<p>{b["input"]["seeds"]} seeds<br>Fanouts {escape(str(b["input"]["fanouts"]))}<br>'
         f'{b["unique_node_upper_bound"]:,} node upper bound</p><span class="pill">Actual unique count: null</span></div></div>',
         '<b>Original scalar forward/backward plus an independent exact subset enumeration.</b> '
         'Fanout bounds and feature-transfer formulas are synthetic accounting, not measured GPU memory or latency.')
    exact=cases["full-neighbor-closure"]
    wrong=cases["one-hop-for-two-layers"]
    evidence='<blockquote>Two message-passing layers depend on neighbors of neighbors. '
    evidence+='Computing two layers on a one-hop directional subgraph does not restore missing edges.</blockquote>'
    for row in (exact,wrong):
        evidence+='<div class="panel"><span class="pill">score · seed output error</span><span class="pill">'
        evidence+=('Confirm' if row["matches_reference"] else 'Send to a human')+'</span>'
        evidence+=f'<h2>{escape(row["id"])}</h2><p>{len(row["metadata"]["n_id"])} sampled nodes · '
        evidence+=f'max error {row["maximum_absolute_error"]:.3e} · confidence: null</p>'
        evidence+=bar(row["maximum_absolute_error"],wrong["maximum_absolute_error"])+'</div>'
    card("SCENARIO: DEPENDENCY CLOSURE",
         f'One missing hop creates {wrong["maximum_absolute_error"]:.4f} seed-output error.',
         evidence,'<b>Full-neighbor closure is the exact gate.</b> A bounded fanout intentionally '
         'changes the sampled operator and needs separate quality evaluation, not an automatic exactness label.')
    matrix='<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        policy=row["intentional_sampling_approximation"]
        answer='Sampling difference' if policy else 'Match' if row["matches_reference"] else 'Mismatch ✕'
        action='Evaluate quality' if policy else 'Confirm' if row["matches_reference"] else 'Review contract'
        target='Same sampled edges' if row["id"]=="sample-mean-with-full-degree" else 'Full-graph seeds'
        matrix+=f'<tr><td>{escape(row["id"])}</td><td>{target}</td><td>{answer}<br>{row["maximum_absolute_error"]:.2e}</td><td>{action}</td></tr>'
    matrix+='</table><div class="grid two"><div class="callout"><b>Gate:</b> maximum output error &lt; 1e-12.</div>'
    matrix+='<div class="callout"><b>Scope:</b> degree control compares the same sampled edges; the other rows use full-graph seeds.</div></div>'
    card("WHAT I TESTED","Exact closure passes; bounded sampling is explicitly not full-graph parity.",
         matrix,'<b>No model-accuracy score.</b> Wrong local/global indices, missing hops and changed '
         'message direction break the chosen contract even if the tensor shapes are valid.')
    card("UNDER THE HOOD","Seed-first order and n_id connect local tensors to global features.",
         '<pre>frontier = seed_global_ids\n'
         'for hop, fanout in enumerate(fanouts):\n'
         '    choose incoming dependencies of frontier\n'
         '    retain newly discovered global nodes\n'
         '    record local directed edges\n\n'
         'local_x = global_x[n_id]</pre><p>↓ apply only the intended supervision</p>'
         '<pre>h_next[i] = ReLU(Wself h[i] + Wneighbor mean(h[N(i)]) + b)\n'
         'seed_output = local_output[:batch_size]\n'
         'seed_labels = global_y[n_id[:batch_size]]\n\n'
         'Directional, bidirectional and induced are different.\n'
         'More layers do not recreate missing dependencies.</pre>',
         '<b>Source scope:</b> pinned NeighborLoader documents seed-first order, n_id, fanout/hop '
         'controls and subgraph policies. The deterministic first-neighbor toy is not a backend reproduction.')
    robustness='<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["uniform_subset_enumeration"]:
        delta=row["expected_relu_of_sample_mean"]-row["full_relu_mean"]
        robustness+=f'<tr><td>Uniform subset, fanout {row["fanout"]}</td><td>ReLU(full mean) {row["full_relu_mean"]:.3f}</td>'
        robustness+=f'<td>E[ReLU(sample mean)] {row["expected_relu_of_sample_mean"]:.3f}</td><td>{delta:.3f}</td></tr>'
    robustness+='</table><div class="panel"><h2>Separate exact example: [-2, 1, 1]</h2>'
    robustness+='<p>Every subset of a given size is enumerated with equal probability, without replacement.</p>'
    robustness+='<p>The sample linear mean is unbiased, but applying ReLU before expectation is not interchangeable.</p>'
    robustness+='<p>Changes are raw activation units, not accuracy points.</p></div>'
    card("ROBUSTNESS CHECK","An unbiased neighbor mean can become biased after ReLU.",
         robustness,'<b>Exact finite enumeration, not Monte Carlo.</b> This is a counterexample to '
         'an unbiased-prediction claim, not evidence that a trained sampled GNN has worse accuracy.')
    maximum=max(row[key] for row in report["scaling"] for key in ("subset_transfer_modeled_ms","full_transfer_modeled_ms"))
    scaling='<p>Blue: capped fanout feature-copy bound · gray: full-graph feature copy. Assumed transfer ms only.</p>'
    for row in report["scaling"]:
        scaling+=f'<div class="latency-row"><b>Fanouts {escape(str(row["input"]["fanouts"]))} · ≤{row["unique_node_upper_bound"]:,} nodes</b>'
        scaling+=f'<div class="latency-pair"><span>Subset <b>{row["subset_transfer_modeled_ms"]:.3f} ms</b></span>'
        scaling+=bar(row["subset_transfer_modeled_ms"],maximum)+'</div>'
        scaling+=f'<div class="latency-pair"><span>Full graph <b>{row["full_transfer_modeled_ms"]:.3f} ms</b></span>'
        scaling+=bar(row["full_transfer_modeled_ms"],maximum,True)+'</div></div>'
    card("LATENCY","Feature-copy bounds grow with fanout; computation remains unmeasured.",
         scaling,'<b>Assumptions:</b> 50 GB/s uncached byte transfer, 10 µs setup, no overlap. '
         'Adjacency, sampling and model compute are excluded. Deterministic formulas; median/repeats N/A.')
    card("MY TAKE","Choose fanout as a memory/quality tradeoff, not an exactness switch.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Dependency-aware graph minibatches.</p>'
         '<p>Full-neighbor reference tests.</p><p>Seed-only supervision and global IDs.</p>'
         '<p>Measured quality at each fanout.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>One hop for two directional layers.</p><p>Using local ids as global ids.</p>'
         '<p>Calling all sampled nodes seeds.</p><p>Calling a tree bound the unique count.</p></div></div>'
         '<pre>READINESS: original numerical prototype\n'
         'SOURCE: pinned PyG NeighborLoader documentation\n'
         'RIGHTS: upstream rights retained; no implementation copied\n'
         'GPU LATENCY / MODEL ACCURACY: not measured\n'
         'FINITE FANOUT: deterministic fixture, not uniform sampling\n'
         'UNIFORM ENUMERATION: separate three-value counterexample</pre>',
         '<b>Synthetic, independently_reproduced=false.</b> This is established graph minibatch '
         'architecture, not a 2026 algorithm release or a popularity claim.')
    styles=('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
            '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
            '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:11px 6px}</style>')
    return page("Round 97 · GNN neighbor dependency and fanout · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR=r"""
function estimateBudget(p){
  for(const key of ["graph_nodes","seeds","feature_dimension","scalar_bytes"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(p.seeds>p.graph_nodes||![2,4].includes(p.scalar_bytes))throw Error("Invalid seed count or scalar width");
  if(!Array.isArray(p.fanouts)||!p.fanouts.length||p.fanouts.some(n=>!Number.isSafeInteger(n)||n<1))
    throw Error("Budget fanouts must be positive integers");
  if(!Number.isFinite(p.bandwidth_gbps)||p.bandwidth_gbps<=0)throw Error("Bandwidth must be finite and positive");
  if(!Number.isFinite(p.transfer_setup_us)||p.transfer_setup_us<0)throw Error("Setup must be finite and nonnegative");
  let multiplicity=p.seeds,frontier=p.seeds;
  for(const fanout of p.fanouts){frontier*=fanout;multiplicity+=frontier;
    if(!Number.isSafeInteger(multiplicity))throw Error("Fanout bound exceeds exact numerical range");}
  const upper=Math.min(p.graph_nodes,multiplicity),subset=upper*p.feature_dimension*p.scalar_bytes;
  const full=p.graph_nodes*p.feature_dimension*p.scalar_bytes;
  if(!Number.isSafeInteger(full)||!Number.isSafeInteger(subset))throw Error("Feature bytes exceed exact range");
  return {tree_occurrence_bound:multiplicity,unique_node_upper_bound:upper,actual_unique_nodes:null,
    feature_subset_bound_bytes:subset,full_graph_feature_bytes:full,
    subset_transfer_modeled_ms:subset/(p.bandwidth_gbps*1e6)+p.transfer_setup_us/1000,
    full_transfer_modeled_ms:full/(p.bandwidth_gbps*1e6)+p.transfer_setup_us/1000};
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
      p[control.name]=control.name==="fanouts"?control.value.split(",").map(n=>Number(n.trim())):Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic; upper bounds and assumed feature transfer only",
      ...estimateBudget(p),gpu_latency_ms:null,model_accuracy:null,production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p=report["budget"]["input"]
    labels={"graph_nodes":"Total graph nodes / 全图节点","seeds":"Distinct seed nodes / 种子数",
            "feature_dimension":"Feature dimension / 特征维度","scalar_bytes":"Scalar bytes (2/4) / 标量字节",
            "fanouts":"Fanouts, comma-separated / 各跳 fanout","bandwidth_gbps":"Assumed GB/s / 假定字节带宽",
            "transfer_setup_us":"Transfer setup µs / 传输启动"}
    controls=""
    for key,label in labels.items():
        if key=="fanouts":
            controls+=f'<label for="{key}">{label}<input required type="text" id="{key}" name="{key}" value="{",".join(map(str,p[key]))}"></label>'
        else:
            controls+=f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="{0 if key=="transfer_setup_us" else 0.001 if key=="bandwidth_gbps" else 1}" step="{0.001 if key in ("transfer_setup_us","bandwidth_gbps") else 1}" value="{p[key]}"></label>'
    return page("GNN neighbor fanout / 图邻居采样工作台",
                '<main><div class="eyebrow">ROUND 97 · SYNTHETIC BOUNDS</div>'
                '<h1>Count dependencies before buying the feature-copy budget.<br>先核对依赖，再预算 fanout。</h1>'
                '<div class="panel"><b>语义：</b>双层方向性消息需要两跳依赖，首 batch_size 节点为 seed，'
                '特征必须通过 n_id 映射。有限 fanout 是质量/资源权衡，不保证等价全邻居输出。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p>'
                '<button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>上界不等于计数：</b>B×(1+s1+s1s2+...) 含重复出现；unique 上界再被全图节点数限制，'
                '真实 unique 保持 null。仅预算无缓存特征拷贝，不计 adjacency、采样、参数、激活或 GPU compute。'
                '<p>GB/s 是字节带宽；公式时间不是 PyG 实测，也不推导模型精度。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    b=report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">GNN seed and dependency magazines</title>'
        '<desc id="description">Three card magazines represent seed nodes, first-hop dependencies and '
        'second-hop dependencies. Local indices need their original global node identities.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 97 · SYNTHETIC DEMO · GRAPH MINIBATCH DEPENDENCIES</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Two message layers need their two-hop dependency closure.</text>'
        '<text x="64" y="180" font-size="24">Seeds → incoming neighbors → neighbors of neighbors; retain global identity through n_id.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">An unbiased linear sample mean does not make ReLU predictions unbiased.</text>'
        +f'<text x="80" y="700" font-size="25">Separate fanout budget: ≤{b["unique_node_upper_bound"]:,} unique nodes; '
        +'the actual deduplicated count is unknown.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Supervise only intended seeds; directional, bidirectional and induced graphs differ.</text>'
        +'<text x="96" y="842" font-size="23">Bounded fanout changes the sampled operator; quality must be evaluated separately.</text>'
        +'<text x="96" y="883" font-size="21">Feature-copy formula is not GPU latency · occurrence bounds are not exact unique counts</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/gnn_neighbor_sampling_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
