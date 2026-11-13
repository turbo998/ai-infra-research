"""Reproducible sparse embedding cards, workbench and physical summary."""

from html import escape
import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT=Path(__file__).resolve().parents[2]
SLUG="sparse-embedding-table-row-column-sharding"


def deck(report: dict) -> str:
    b=report["budget"]
    cards=[]

    def card(eyebrow,title,evidence,takeaway):
        n=len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic pooling inputs · measured local CPU timings only · independently_reproduced=false</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 96 · Sparse embedding sharding</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · SPARSE TABLES","The same pooled vector survives three ownership layouts.",
         '<div class="grid"><div class="panel"><h2>Layouts</h2><p>Table-wise</p><p>Row-wise</p>'
         '<p>Column-wise</p><p>Empty bags and duplicates</p></div><div class="panel"><h2>Original fixture</h2>'
         f'<p>{len(report["fixture"]["table"])} table rows<br>{len(report["fixture"]["table"][0])} dimensions<br>'
         f'{len(report["fixture"]["bags"])} jagged bags</p><p>Ideal unweighted mean</p></div>'
         '<div class="panel"><h2>Separate budget</h2>'
         f'<p>{b["table_wise_peak_weight_bytes"]/1e9:.1f} GB full table<br>'
         f'{b["row_wise_peak_weight_bytes"]/1e9:.1f} GB max row shard</p>'
         '<span class="pill">GPU latency: null</span><span class="pill">Quality: null</span></div></div>',
         '<b>Numerics: original Python scalar model.</b> Timing is measured on local '
         f'{escape(report["latency"]["environment"]["os"])} / {escape(report["latency"]["environment"]["architecture"])} '
         f'/ Python {escape(report["latency"]["environment"]["python"])}; serial functions, no TorchRec or GPU.')
    exact=next(row for row in report["cases"] if row["id"]=="row-ideal-mean")
    wrong=next(row for row in report["cases"] if row["id"]=="sum-of-local-means")
    evidence='<blockquote>Unequal row-shard bag counts need a global sum divided by the global count. '
    evidence+='Summing local means gives each shard the wrong weight.</blockquote>'
    for row in (exact,wrong):
        evidence+='<div class="panel"><span class="pill">score · pooled-vector error</span><span class="pill">'
        evidence+=('Confirm' if row["matches_ideal"] else 'Send to a human')+'</span>'
        evidence+=f'<h2>{escape(row["id"])}</h2><p>Maximum error {row["maximum_absolute_error"]:.3e} · confidence: null</p>'
        evidence+=bar(row["maximum_absolute_error"],wrong["maximum_absolute_error"])+'</div>'
    card("SCENARIO: JAGGED MEAN",f'Local means differ by {wrong["maximum_absolute_error"]:.3f} from the global pooled reference.',
         evidence,'<b>Duplicates count as occurrences.</b> Deduplicating IDs requires preserving their '
         'multiplicity or equivalent weights in both forward and table gradients.')
    matrix='<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        policy=row["id"]=="global-count-plus-epsilon"
        answer='Policy difference' if policy else 'Match' if row["matches_ideal"] else 'Mismatch ✕'
        action='Confirm epsilon' if policy else 'Confirm' if row["matches_ideal"] else 'Send to a human'
        matrix+=f'<tr><td>{escape(row["id"])}</td><td>Ideal mean parity?</td><td>{answer}<br>{row["maximum_absolute_error"]:.2e}</td><td>{action}</td></tr>'
    matrix+='</table><div class="grid two"><div class="callout"><b>Gate:</b> maximum error below 1e-12.</div>'
    matrix+='<div class="callout"><b>Not an upstream bug claim:</b> the pinned RW callback intentionally uses count + 1e-6.</div></div>'
    card("WHAT I TESTED","Three layouts match; epsilon normalization is a separate numerical policy.",
         matrix,'<b>Three incorrect controls and one explicit policy difference.</b> All table-gradient '
         'coordinates also have independent finite-difference tests for SUM, MEAN and weighted SUM.')
    card("UNDER THE HOOD","Row sharding reduces partial sums; column sharding reconstructs dimensions.",
         '<pre>ROW: route IDs to row owners\n'
         '     sum partial vectors across owners\n'
         '     divide once by full bag count for ideal mean\n'
         'COLUMN: deliver IDs to every column owner\n'
         '        concatenate pieces in original dimension order\n'
         'TABLE: one owner stores the whole table</pre><p>↓ pinned implementation boundaries</p>'
         '<pre>Training RW: sparse AlltoAll → pooled ReduceScatter\n'
         'TW/CW: sparse AlltoAll → pooled AlltoAll\n'
         'RW/TWRW mean callback: divisor + 1e-6\n\n'
         'subgroup members = [1,3,5,7]\n'
         'global_root = members[local_root=0]  # 1, not 0</pre>',
         '<b>Recent first-party context:</b> the November 11, 2026 fix maps subgroup-local root zero '
         'to a global rank before replicated-table initialization. Training and inference distributors differ.')
    robustness='<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["perturbations"]:
        robustness+=f'<tr><td>{escape(row["id"])}</td><td>Ideal global mean</td><td>Same vectors within tolerance</td><td>{row["maximum_absolute_error"]:.2e}</td></tr>'
    robustness+='</table><div class="panel"><h2>Other gates</h2><p>Empty bags return zero vectors.</p>'
    robustness+='<p>Repeated IDs accumulate repeated gradient contributions.</p>'
    robustness+='<p>Weighted SUM is tested; weighted MEAN is not claimed.</p>'
    robustness+='<p>Global rank zero is not a member of the chosen subgroup.</p></div>'
    card("ROBUSTNESS CHECK","Reordering occurrences and changing row-shard count preserves ideal pooling.",
         robustness,'<b>Numerical parity is not recommendation quality.</b> Hot-ID skew, optimizer state, '
         'communication codecs and actual distributed kernels are outside this scalar proof.')
    maximum=max(row[key] for row in report["latency"]["rows"] for key in ("dense_median_ms","row_sharded_median_ms"))
    timing='<p>Blue: row-sharded serial Python toy · gray: direct dense toy. Measured local CPU ms, including validation.</p>'
    for row in report["latency"]["rows"]:
        timing+=f'<div class="latency-row"><b>{row["bags"]} bags</b>'
        timing+=f'<div class="latency-pair"><span>Shard toy <b>{row["row_sharded_median_ms"]:.3f} ms</b></span>'
        timing+=bar(row["row_sharded_median_ms"],maximum)+'</div>'
        timing+=f'<div class="latency-pair"><span>Direct toy <b>{row["dense_median_ms"]:.3f} ms</b></span>'
        timing+=bar(row["dense_median_ms"],maximum,True)+'</div></div>'
    card("LATENCY","Serial ownership bookkeeping costs CPU time; it does not predict GPU scaling.",
         timing,'<b>Seven measured repeats, median; two warmups per path, alternating timing order.</b> '
         'Inputs are synthetic. No parallel workers, GPU, FBGEMM or TorchRec were executed; no training speedup is inferred.')
    card("MY TAKE","Choose ownership from table size, input replication and pooled-output pressure.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Table-wise for many suitably sized tables.</p>'
         '<p>Row-wise for a table too large for one rank.</p><p>Column-wise when dimension splitting is worthwhile.</p>'
         '<p>Global counts and explicit root mapping.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Local means as global means.</p><p>Dedup without multiplicity.</p><p>Wrong column reassembly.</p>'
         '<p>Logical tensor bytes called wire traffic.</p></div></div>'
         '<pre>READINESS: original numerical/CPU prototype\n'
         'SOURCE: pinned TorchRec sharding and mean callback\n'
         'RIGHTS: upstream BSD-3-Clause metadata; no source copied\n'
         'GPU LATENCY / RECOMMENDATION QUALITY: not measured\n'
         'CPU SAMPLES: sealed latency.json; never rebuilt by --check\n'
         'NUMERICAL CONTRACT: ideal mean distinct from count+epsilon</pre>',
         '<b>synthetic inputs · independently_reproduced=false.</b> The budget excludes optimizer state, '
         'offsets/length metadata, activations and actual collective topology.')
    styles=('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
            '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
            '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:11px 6px}</style>')
    return page("Round 96 · Sparse embedding ownership · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR=r"""
function estimateBudget(p){
  for(const [key,value]of Object.entries(p))
    if(!Number.isSafeInteger(value)||value<1)throw Error(key+" must be a positive safe integer");
  if(p.shards>Math.min(p.vocabulary,p.dimension))throw Error("Need at least one row and column per shard");
  if(![2,4].includes(p.weight_bytes)||![2,4].includes(p.output_bytes)||![4,8].includes(p.id_bytes))
    throw Error("Unsupported scalar width");
  if(p.vocabulary>2**(8*p.id_bytes-1))throw Error("Table rows exceed signed ID range");
  const full=p.vocabulary*p.dimension*p.weight_bytes,ids=p.global_bags*p.ids_per_bag*p.id_bytes;
  const pooled=p.global_bags*p.dimension*p.output_bytes;
  const result={table_wise_peak_weight_bytes:full,
    row_wise_peak_weight_bytes:Math.ceil(p.vocabulary/p.shards)*p.dimension*p.weight_bytes,
    column_wise_peak_weight_bytes:p.vocabulary*Math.ceil(p.dimension/p.shards)*p.weight_bytes,
    row_wise_logical_id_delivery_bytes:ids,column_wise_logical_id_delivery_bytes:ids*p.shards,
    row_wise_dense_partial_bytes_all_ranks:pooled*p.shards,column_wise_output_piece_bytes_all_ranks:pooled};
  if(Object.values(result).some(n=>!Number.isSafeInteger(n)))throw Error("Budget exceeds exact numerical range");
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
    result.textContent=JSON.stringify({provenance:"synthetic; logical tensor budget, not wire traffic",
      ...estimateBudget(p),gpu_latency_ms:null,recommendation_quality:null,production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p=report["budget"]["input"]
    labels={"vocabulary":"Table rows / 表行数","dimension":"Embedding dimension / 维度","shards":"Shards / 分片",
            "global_bags":"Global bags / 全局 bag 数","ids_per_bag":"IDs per bag / 每袋 ID 数",
            "weight_bytes":"Weight bytes (2/4) / 权重字节","id_bytes":"ID bytes (4/8) / 索引字节",
            "output_bytes":"Output bytes (2/4) / 输出字节"}
    controls="".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="1" step="1" value="{p[key]}"></label>' for key,label in labels.items())
    return page("Sparse embedding sharding / 稀疏嵌入分片工作台",
                '<main><div class="eyebrow">ROUND 96 · SYNTHETIC BUDGET</div>'
                '<h1>Shard ownership, not the pooling contract.<br>分片所有权，不偷换 pooling。</h1>'
                '<div class="panel"><b>三种分法：</b>TW 整表驻留一个 owner；RW 分行合并 partial；'
                'CW 分列并按原顺序拼接。均值应按完整 bag 计数；重复 ID 是重复贡献。'
                '<p>固定源码 RW/TWRW mean callback 使用 count + 1e-6，不冒充理想均值的 bitwise 等价。</p></div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p>'
                '<button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>口径：</b>单表、等宽区间分片的最大权重块；logical ID delivery 含本地投递，'
                'partial bytes 是物化量，不是实际网络发送字节。未计 offsets/lengths、优化器、激活、通信拓扑或访问倾斜。'
                '<p>封存 CPU 耗时只测原创串行 Python 函数，不随这些输入外推 GPU 延迟。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    b=report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Sparse embedding ownership choices</title>'
        '<desc id="description">Three card magazines represent table-wise, row-wise and column-wise ownership '
        'of a sparse embedding table. Global occurrence counts and dimensions must remain correct.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 96 · SPARSE EMBEDDING OWNERSHIP · ORIGINAL NUMERICAL MODEL</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Distribute the table, preserve every bag occurrence.</text>'
        '<text x="64" y="180" font-size="24">Table-wise ownership · row-wise partial reduction · column-wise dimension reconstruction</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">Global means need global counts; duplicate IDs still carry multiplicity.</text>'
        +f'<text x="80" y="700" font-size="25">Separate budget: {b["table_wise_peak_weight_bytes"]/1e9:.1f} GB full table → '
        +f'{b["row_wise_peak_weight_bytes"]/1e9:.1f} GB max row shard; optimizer state excluded.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Training RW: sparse AlltoAll then pooled ReduceScatter; inference output paths differ.</text>'
        +'<text x="96" y="842" font-size="23">Subgroup [1,3,5,7]: local root 0 is global rank 1, not global rank 0.</text>'
        +'<text x="96" y="883" font-size="21">Pinned mean callback adds epsilon · local CPU timings are not GPU scaling evidence</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/sparse_embedding_sharding_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
