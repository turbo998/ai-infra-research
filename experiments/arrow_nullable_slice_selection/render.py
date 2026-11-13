"""Self-contained nullable-buffer evidence cards, H5 and original SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "arrow-nullable-list-slice-buffer-retention"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic immutable buffers · independently_reproduced=false · '
                     'Original Python CPU timings separately measured; no native Arrow/RSS/HBM observation.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 108 · Arrow-style nullable slices</span><span>{n} / 7</span></footer></section>')

    s,c,b = r["slice"],r["controls"],r["budget"]
    card("HANDS-ON · COLUMNAR AI DATA",
         '<span class="accent">Required objects</span> and valid null buffers are different contracts.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>Parent rows: {s["parent_rows"]}<br>Slice offset: {s["parent_offset"]}<br>Length: {s["length"]}</p>'
         '<span class="pill">List&lt;int32&gt; model</span></div>'
         '<div class="panel"><h2>Offsets and bitmaps</h2>'
         f'<p>{s["visible_offsets"]}</p><p>Parent validity: 0x{s["parent_validity_hex"]}</p>'
         f'<p>Child validity: 0x{s["child_validity_hex"]}</p></div>'
         '<div class="panel"><h2>Selected buffer bytes</h2>'
         f'<p>View retains {s["retained_buffer_bytes"]} B<br>Canonical copy {s["compact_buffer_bytes"]} B</p>'
         f'<span class="pill">First CPU tier: {r["latency"]["rows"][0]["primary_median_ms"]:.6f} ms</span></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; synthetic byte buffers. '
         'Pinned November 13, 2026 Arrow checks missing required objects while preserving valid nullable buffers.')
    scores = '<blockquote>A four-row slice contains a null list, an empty list, '
    scores += 'a list with a null child and a fully valid list.</blockquote>'
    correct = s["length"]-s["maximum_row_mismatches"]
    wrong = s["length"]-c["wrong_bit_row_mismatches"]
    for label,rows,matches in (("Apply row offset to validity bits",s["logical_rows"],correct),
                                ("Forget validity bit offset",c["ignore_validity_bit_offset"],wrong)):
        scores += '<div class="panel"><span class="pill">score · exact logical-row verifier</span>'
        scores += f'<span class="pill">{"Confirm" if matches == s["length"] else "Send to a human"}</span><h2>{label}</h2>'
        scores += f'<p>{json.dumps(rows)}</p><p>Matching rows {matches}/{s["length"]}; calibrated confidence: null.</p>'
        scores += bar(matches,s["length"])+'</div>'
    card("SCENARIO: BIT OFFSET IS NOT BYTE OFFSET",
         f'Ignoring one row of offset misreads {c["wrong_bit_row_mismatches"]}/{s["length"]} logical rows.',
         scores,'<b>Verifier bars are not probabilities.</b> Bit i+array_offset decides parent validity. '
         'A parent null can own nonempty hidden child storage; a valid empty list is not null.')
    rows = [
        ("Row view","Independent logical reference",f'{s["maximum_row_mismatches"]} mismatches',"Confirm"),
        ("Canonical copy","Same logical rows",f'0 mismatches; offsets {s["compact_offsets"]}',"Confirm"),
        ("Validity origin","Ignore row offset",f'✕ {c["wrong_bit_row_mismatches"]} mismatches',"Reject"),
        ("Child origin","Add parent offset twice",f'✕ {c["wrong_child_row_mismatches"]} mismatches',"Reject"),
        ("Raw child span","Same as logical flatten?",f'✕ {len(s["contiguous_child_span"])} vs {len(s["logical_flatten"])} values',"Not equivalent"),
        ("Child object","None versus nullable child","Required object rejected; child nulls preserved","Separate contracts"),
    ]
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    matrix += "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in rows)
    matrix += '</table><div class="grid two"><div class="callout">Validity determines values; hidden null payload is not data.</div>'
    matrix += '<div class="callout">Selected buffer bytes exclude Python objects and peak resident memory.</div></div>'
    card("WHAT I TESTED","Parent nulls, empty lists and child nulls survive slice and canonical copy.",
         matrix,'<b>Original byte model:</b> little-endian List&lt;int32&gt;, immutable buffers, strict bounded slices. '
         'Not pyarrow constructor, LargeList, dictionary, run-end or device-buffer parity.')
    card("UNDER THE HOOD","Apply the parent row offset once, then use absolute child offsets.",
         '<pre>physical_row = view.offset + logical_row\n'
         'parent_valid = bit(parent_bitmap, physical_row)\n'
         'start = offsets[physical_row]\n'
         'stop  = offsets[physical_row + 1]\n'
         'child j: check child validity before reading payload</pre>'
         '<p>↓ parent null skips its entire child span</p>'
         f'<pre>contiguous child span = {s["contiguous_child_span"]}\n'
         f'logical flatten = {s["logical_flatten"]}\n'
         f'raw values child length = {len(s["values_property"])}\n'
         'values property is not logical flatten\n'
         'null child payload sentinel is ignored, not imputed</pre>',
         '<b>Source distinction:</b> ListArray.flatten accounts for sliced offsets and nonempty spans behind '
         'null parents. Its .values child may still expose the backing child. Some null-free contiguous flatten paths can share buffers.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>Slice {row["start"]}:{row["start"]+row["length"]}</td><td>Logical list oracle</td>'
        robust += f'<td>View / canonical copy</td><td>{row["logical_mismatches"]} / {row["compact_mismatches"]} row errors</td></tr>'
    robust += '</table><div class="panel"><h2>Nested slicing is relative to the view</h2>'
    robust += '<p>Compose row offsets once; preserve bitmap bit origins. Empty slices and bitmap-byte crossings are checked.</p></div>'
    card("ROBUSTNESS CHECK","Composed slices keep logical rows without rebasing shared storage.",
         robust,'<b>Absence is type-specific.</b> In this List model, an absent bitmap means all valid. '
         'A missing required parent/child array is an error. This rule is not extended to Arrow Null, unions or run-end types.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timing = '<p>Blue: create a view handle · Gray: read logical rows and build validated canonical buffers.</p>'
    for row in r["latency"]["rows"]:
        timing += f'<div class="latency-row"><h2>{row["parent_rows"]} parent rows · {row["slice_rows"]} selected</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timing += f'<div class="latency-pair"><b>{row[key]:.6f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timing += '</div>'
    card("LATENCY","Constant-work view creation and actual materialization do different work.",
         timing,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats; each sample is the '
         'mean of 25 operations, chart is median across repeat means. View does not read rows; copy does. '
         'Not native Arrow speedup, throughput or resident-memory measurement.')
    card("MY TAKE",
         f'A tiny view can retain {b["view_retained_buffer_bytes"]:,} backing-buffer bytes.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Short-lived shared-buffer views.</p><p>Validity-aware ragged ingestion.</p>'
         '<p>Explicit compact copies for ownership control.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Null and empty lists conflated.</p><p>Raw child data called logical flatten.</p>'
         '<p>Zero-copy called zero retained memory.</p></div></div>'
         '<pre>READINESS: independent nullable-buffer model\n'
         'SOURCE: pinned Arrow, Apache-2.0 rights retained\n'
         'MODEL: List&lt;int32&gt;, read-only little-endian buffers\n'
         'NATIVE CONSTRUCTOR FAILURE: not reproduced\n'
         'RSS / GPU / PEAK HBM: not measured\n'
         'PROVENANCE: synthetic; independently_reproduced=false</pre>',
         f'<b>Default synthetic budget:</b> {b["input"]["slice_rows"]} selected rows, '
         f'{b["canonical_copy_buffer_bytes"]:,} canonical-copy payload bytes. '
         'Releasing one view does not free buffers still owned elsewhere; copying does not automatically reduce process RSS.')
    styles = ('<style>.latency-row{padding:12px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 108 · Nullable Arrow-style slices · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["parent_rows","child_values","slice_rows","visible_child_values"]){
    if(!Number.isSafeInteger(p[key])||p[key]<0)throw Error(key+" must be a nonnegative exact integer");
  }
  if(![4,8].includes(p.offset_bytes)||![1,2,4,8].includes(p.value_bytes))
    throw Error("offset_bytes 4/8 and value_bytes 1/2/4/8 required");
  for(const key of ["parent_bitmap","child_bitmap"]){
    if(!Number.isSafeInteger(p[key])||![0,1].includes(p[key]))throw Error(key+" must be integer 0 or 1");
  }
  if(p.slice_rows>p.parent_rows||p.visible_child_values>p.child_values)throw Error("visible counts exceed backing counts");
  if(p.slice_rows===0&&p.visible_child_values!==0)throw Error("an empty slice cannot expose child values");
  const bitmap=n=>Number((BigInt(n)+7n)/8n);
  const buffers=(rows,children)=>({offsets:(rows+1)*p.offset_bytes,values:children*p.value_bytes,
    parent_validity:bitmap(rows)*p.parent_bitmap,child_validity:bitmap(children)*p.child_bitmap});
  const backing=buffers(p.parent_rows,p.child_values),copied=buffers(p.slice_rows,p.visible_child_values);
  const result={backing_buffers:backing,canonical_copy_buffers:copied,
    view_retained_buffer_bytes:Object.values(backing).reduce((a,b)=>a+b,0),
    view_new_data_buffer_bytes:0,canonical_copy_buffer_bytes:Object.values(copied).reduce((a,b)=>a+b,0),
    max_offset_fits_selected_signed_type:p.offset_bytes===8||p.child_values<=2147483647,
    python_object_bytes:null,peak_rss_bytes:null,peak_hbm_bytes:null,native_arrow_latency_ms:null};
  const integers=[...Object.values(backing),...Object.values(copied),
    result.view_retained_buffer_bytes,result.canonical_copy_buffer_bytes];
  if(!integers.every(Number.isSafeInteger))throw Error("budget exceeds exact JavaScript integer range");
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
    result.textContent=JSON.stringify({provenance:"synthetic backing/visible counts and buffer-presence flags",
      ...estimateBudget(p),production_approved:false,
      note:"Views keep backing buffers alive. Canonical copy drops hidden null-parent spans, but this is an original policy, not a pyarrow API allocation guarantee. Counts do not prove semantic validity. Excludes Python objects, aliases held elsewhere, allocator/RSS and peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    labels = dict(parent_rows="Backing parent rows / 父数组行数",child_values="Backing child values / Child 元素数",
                  slice_rows="Selected rows / Slice 行数",visible_child_values="Visible child values / 可见 Child 数",
                  offset_bytes="Offset bytes / 偏移字节",value_bytes="Value bytes / 值字节",
                  parent_bitmap="Parent bitmap present 0/1 / 父有效位图",child_bitmap="Child bitmap present 0/1 / 子有效位图")
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'step="1" value="{p[key]}"></label>' for key,label in labels.items())
    return page("Arrow nullable slices / 可空 List Buffer 工作台",
                '<main><div class="eyebrow">ROUND 108 · IMMUTABLE BUFFERS · SYNTHETIC OWNERSHIP BUDGET</div>'
                '<h1>Zero-copy still retains its parent.<br>零拷贝不等于零驻留，null 不等于空列表。</h1>'
                '<div class="panel">Parent row offset 同时影响 offsets 和 validity bits；'
                'child offsets 已是子数组坐标，不要再加一次 parent offset。'
                'Raw child span 可以包含 null parent 背后的隐藏数据，不等于 logical flatten。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>Backing/visible counts 为 synthetic 输入，位图 flag 仅表示 buffer 是否存在。'
                '原始复制策略会丢弃 null parent 的隐藏 child spans，不是 pyarrow 默认 compact API。'
                '<p>32-bit offset 不够时需另选表示；模型本体仅 List&lt;int32&gt;，工作台字节数不认证其他类型。'
                '未测 native Arrow、RSS、实际 allocator 或 peak HBM；其他 alias 仍可保留原 buffer。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r):
    s,b = r["slice"],r["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Nullable row-card buffer sorter</title>'
        '<desc id="description">A detailed solid sorter represents parent rows with validity gates and child spans. '
        'A view keeps the installed parent storage, while canonical copying routes only logical visible values.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 108 · ARROW-STYLE LIST BUFFERS · NULLABILITY AND OWNERSHIP</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A tiny view retains storage; null is not an empty list.</text>'
        '<text x="64" y="180" font-size="24">A mechanical analogy: validity gates hide row contents without removing installed storage.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Visible offsets {s["visible_offsets"]}; compact offsets {s["compact_offsets"]} preserve the same logical rows.</text>'
        +f'<text x="80" y="700" font-size="25">Default {b["input"]["slice_rows"]}-row view retains {b["view_retained_buffer_bytes"]:,} buffer bytes; canonical copy {b["canonical_copy_buffer_bytes"]:,}.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Apply array offset to parent offsets and validity bits once; child positions are already absolute.</text>'
        +'<text x="96" y="842" font-size="23">November 13 required-object guards do not forbid valid null buffers or nullable child values.</text>'
        +'<text x="96" y="883" font-size="21">Original immutable byte model only; selected buffers are not native Arrow latency or process RSS.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/arrow_nullable_slice_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
