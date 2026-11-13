"""Self-contained conservative pruning evidence cards, H5 budgets and original SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "parquet-sql-null-row-group-pruning"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic integer-column metadata · independently_reproduced=false · '
                     'Original Python CPU query timing separately measured; no native DuckDB/Parquet/object-store execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 112 · Conservative SQL pruning</span><span>{n} / 7</span></footer></section>')

    c,b = r["case"],r["budget"]
    card("HANDS-ON · COLUMNAR DATA INFRASTRUCTURE",
         f'{len(c["selected_groups"])} of {len(c["groups"])} groups retain <span class="accent">all {c["exact_matching_rows"]} matching rows</span>.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>{c["query"]}</p><p>{c["total_rows"]} rows · nullable integers</p></div>'
         '<div class="panel"><h2>Structured input</h2><pre>'
         +json.dumps(c["groups"],indent=1)+'</pre></div>'
         '<div class="panel"><h2>Top decisions</h2>'
         f'<p>Read groups {c["selected_groups"]}</p><p>{c["candidate_rows"]} candidate rows; '
         f'{c["exact_matching_rows"]} actual matches</p><span class="pill">WHERE keeps TRUE only</span></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; November 11, 2026 '
         'source pin. November 5 scratch-buffer changes are recent; metadata pruning and SQL null logic are longstanding.')
    deep = f'<blockquote>{c["query"]}. A mixed group [5,6,7,10] may satisfy both the inner predicate and its negation.</blockquote>'
    for label,value,action in (("Conservative pruning",c["pruned_matching_rows"],"Confirm"),
                               ("Wrong NOT of may-match",c["exact_matching_rows"]-c["wrong_boolean_not_lost_matches"],"Send to a human"),
                               ("Full row scan oracle",c["exact_matching_rows"],"Confirm")):
        deep += f'<div class="panel"><span class="pill">score · retained matching rows</span><span class="pill">{action}</span>'
        deep += f'<h2>{label}: {value}</h2><p>Calibrated confidence: null. Counts are not model probabilities.</p>'
        deep += bar(value,c["exact_matching_rows"])+'</div>'
    card("SCENARIO: NOT DOES NOT INVERT A MAY-MATCH BIT",
         f'A Boolean NOT shortcut drops {c["wrong_boolean_not_lost_matches"]} valid rows.',
         deep,'<b>Different quantifiers:</b> NOT(exists a true row) is not exists a row where NOT is true. '
         'Keep TRUE/FALSE/UNKNOWN possibilities; reject a group only when TRUE is impossible.')
    rows = [("Mixed group","NOT(x>9)",str(c["group_truth_sets"][3]),"Read"),
            ("All-null group","WHERE UNKNOWN",str(c["group_truth_sets"][2]),"Skip"),
            ("Missing stats","Any value/null remains possible","Keep all groups","Conservative fallback"),
            ("Constant bounds","min=max=7 proves no null?","✕ NULL still possible","Inspect null count"),
            ("Cross-column correlation","x=1 AND y=1","May match; actual 0","False positive allowed"),
            ("Wrong Boolean NOT","No lost qualifying rows?","✕ Loses 3","Reject shortcut")]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Missing metadata widens possibilities; it never proves absence.</div>'
    matrix += '<div class="callout">Possible TRUE is a candidate gate, not a row-level match certificate.</div></div>'
    card("WHAT I TESTED","Min/max bounds are not a nullability or correlation certificate.",
         matrix,'<b>Model boundary:</b> trusted conservative integer bounds and optional exact null counts; eq, gt, IS NULL, '
         'AND/OR/NOT only. No strings, NaN, collation, nested/repeated columns, joins, UDFs or Parquet decoder.')
    card("UNDER THE HOOD","Skip exactly when the conservative truth set excludes TRUE.",
         '<pre>METADATA: rows, min_non_null, max_non_null,\n'
         '          optional null_count\n'
         'leaf comparisons over bounds -> {T,F,U} subset\n'
         'AND / OR: SQL 3-valued Cartesian product\n'
         'NOT: T->F, F->T, U->U\n'
         'WHERE keeps TRUE; FALSE and UNKNOWN fail</pre>'
         '<p>↓ conservative abstraction, followed by retained-row filtering</p>'
         '<pre>if TRUE not in possible_truths(predicate, stats):\n'
         '    skip row group\n'
         'else:\n'
         '    evaluate actual retained rows\n'
         'DuckDB inspected gate: ALWAYS_FALSE or FALSE_OR_NULL\n'
         'No metadata -> NO_PRUNING_POSSIBLE\n'
         'Known null_count refines independent validity state</pre>',
         '<b>Source-derived architecture, original interpreter.</b> This is not a native optimizer replica or reproduced '
         'upstream defect. Metadata construction and storage reads are outside the CPU timing.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>{row["name"]}</td><td>{c["exact_matching_rows"]}/{c["exact_matching_rows"]} retained</td>'
        robust += f'<td>{row["retained_matches"]}/{c["exact_matching_rows"]}; {row["retained_match_percent"]:.0f}%</td>'
        robust += f'<td>{row["change_retention_points"]:+.0f}</td></tr>'
    robust += '</table><div class="panel"><h2>Independent nullability control</h2><p>Values [7,NULL], min=max=7; '
    robust += f'unknown null count leaves IS NULL possibilities {r["constant_bounds"]["is_null_possible_truths"]}. '
    robust += f'Incorrectly inferring no null loses {r["constant_bounds"]["wrong_inferred_no_null_lost_matches"]} qualifying row.</p></div>'
    card("ROBUSTNESS CHECK","Missing statistics increase reads, not missing matches.",
         robust,'<b>Truth-set soundness assumes accurate conservative metadata.</b> Structural field checks cannot detect a '
         'writer lying about bounds. Correlation loss admits false positives; it must never justify extra exclusion.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timings = '<p>Blue: metadata gate + retained-row scan · Gray: full row scan. Same final count.</p>'
    for row in r["latency"]["rows"]:
        timings += f'<div class="latency-row"><h2>{row["row_groups"]} groups · {row["rows"]:,} rows · {row["selected_groups"]} kept</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timings += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timings += '</div>'
    card("LATENCY","A fixed one-in-eight candidate profile separates metadata cost from full scans.",
         timings,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; three complete queries per '
         'sample, median repeat means. Precomputed stats; no metadata build, decoding, filesystem/network or native DuckDB timing.')
    card("MY TAKE",
         f'Default selected chunks plus metadata total {b["pruned_plus_metadata_bytes"]:,} modeled bytes.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Trusted conservative metadata.</p>'
         '<p>Independent null and value bounds.</p><p>Residual row predicates after pruning.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>NOT of a may-match bit.</p><p>Min=max implies NOT NULL.</p>'
         '<p>Candidate counts called exact query matches.</p></div></div>'
         '<pre>READINESS: original integer predicate contract model\n'
         'SOURCE: pinned DuckDB, MIT\n'
         'PROVENANCE: synthetic; independently_reproduced=false\n'
         'NATIVE ENGINE / PARQUET / STORAGE: not executed\n'
         'UNIFORM COMPRESSED CHUNK SIZES: selected assumptions\n'
         'ACTUAL BYTES / NATIVE LATENCY / PEAK MEMORY: unknown</pre>',
         f'<b>Metadata is not free:</b> selected allowance {b["metadata_bytes"]:,} bytes remains even if no group is read. '
         'Uniform selected-column chunks omit readahead, page indexes, Bloom filters, footer caching and request rounding.')
    styles = ('<style>.latency-row{padding:10px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:13px}th,td{padding:9px 5px}</style>')
    return page("Round 112 · Conservative Parquet pruning · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["row_groups","selected_groups","columns","projected_columns","column_chunk_bytes","metadata_bytes"]){
    const min=(key==="selected_groups"||key==="metadata_bytes")?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" requires an exact integer >= "+min);
  }
  if(p.selected_groups>p.row_groups||p.projected_columns>p.columns)
    throw Error("selected groups/projected columns exceed totals");
  const full=BigInt(p.row_groups)*BigInt(p.columns)*BigInt(p.column_chunk_bytes),
    projected=BigInt(p.row_groups)*BigInt(p.projected_columns)*BigInt(p.column_chunk_bytes),
    pruned=BigInt(p.selected_groups)*BigInt(p.projected_columns)*BigInt(p.column_chunk_bytes),m=BigInt(p.metadata_bytes);
  const exact=n=>{const value=Number(n);if(!Number.isSafeInteger(value))throw Error("budget exceeds exact integer range");return value;};
  return {unprojected_column_chunk_bytes:exact(full),projected_column_chunk_bytes:exact(projected),
    selected_column_chunk_bytes:exact(pruned),avoided_projected_chunk_bytes:exact(projected-pruned),
    unprojected_plus_metadata_bytes:exact(full+m),projected_plus_metadata_bytes:exact(projected+m),
    pruned_plus_metadata_bytes:exact(pruned+m),metadata_bytes:p.metadata_bytes,
    actual_reader_bytes:null,network_bytes:null,native_scan_latency_ms:null,peak_memory_bytes:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const p={};
    for(const key of ["row_groups","selected_groups","columns","projected_columns","column_chunk_bytes","metadata_bytes"]){
      if(form.elements[key].value.trim()==="")throw Error(key+" required");
      p[key]=Number(form.elements[key].value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic uniform compressed-column-chunk and metadata allowance",
      ...estimateBudget(p),production_approved:false,
      note:"Selected group count is an input, not a predicted selectivity. Candidate pruning does not replace row filtering. Metadata remains when zero groups qualify. Actual reads include unmodeled cache/readahead/index/request effects; no measured storage speedup."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    controls = ""
    for key,label in (("row_groups","Total row groups / 总 Row Group"),("selected_groups","Candidate groups / 候选组"),
                      ("columns","Stored columns / 存储列数"),("projected_columns","Projected columns / 读取列数"),
                      ("column_chunk_bytes","Compressed bytes/column/group / 单列单组压缩字节"),
                      ("metadata_bytes","Metadata allowance bytes / 元数据预算字节")):
        controls += f'<label for="{key}">{label}<input required type="number" step="1" id="{key}" name="{key}" value="{r["budget"]["input"][key]}"></label>'
    return page("Conservative Parquet pruning / SQL NULL 与 Row Group 剪枝工作台",
                '<main><div class="eyebrow">ROUND 112 · WHERE RETAINS TRUE ONLY</div>'
                '<h1>May match is not a match.<br>可能命中，不等于已经命中。</h1>'
                '<div class="panel">WHERE NOT(x&gt;9) 的四组样本只读取组 0/3，保留全部 6 条匹配行；'
                '错误地对 may-match 位取 NOT，会漏掉组 3 的 3 条匹配行。'
                '<p>min=max=7 不证明没有 NULL；missing statistics 应扩大可能性，不是当作零值或无数据。</p></div>'
                f'<form id="calculator"><div class="grid">{controls}</div><p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>均匀压缩 column chunk 字节与 selected groups 均为用户输入假设，'
                '不是引擎预测/实际 IO。零候选组也保留元数据预算。'
                '<p>原始整数 SQL 三值模型不实现 Parquet、DuckDB、浮点 NaN、collation、嵌套列、page index 或 Bloom filter。'
                '通过 metadata gate 的组仍需逐行过滤；不把 CPU 原型时间冒充 object-store 吞吐。</p></div></main>',CALCULATOR)


def summary_svg(r):
    c = r["case"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Conservative row-group card sorter</title>'
        '<desc id="description">An original solid sorting cabinet represents metadata candidate gates; '
        'unknown cards remain candidates and require row-level checks rather than being discarded.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 112 · PARQUET METADATA · SQL THREE-VALUED PREDICATES</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A row group is skippable only when TRUE is impossible.</text>'
        '<text x="64" y="180" font-size="24">A candidate-card sorter analogy: min/max, validity and residual row predicates are separate contracts.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Read groups {c["selected_groups"]}; keep all {c["exact_matching_rows"]} matches from {c["total_rows"]} rows.</text>'
        +f'<text x="80" y="700" font-size="25">Wrong NOT of may-match drops {c["wrong_boolean_not_lost_matches"]} matches; constant min=max does not prove NOT NULL.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">SQL NOT maps UNKNOWN to UNKNOWN; WHERE keeps TRUE, not FALSE or UNKNOWN.</text>'
        +'<text x="96" y="842" font-size="23">Missing statistics widen possibilities. Lost correlation allows extra reads, never lost matches.</text>'
        +'<text x="96" y="883" font-size="21">Original integer model; no native DuckDB, Parquet decoding, actual read bytes or object-store latency.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/parquet_predicate_pruning_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
