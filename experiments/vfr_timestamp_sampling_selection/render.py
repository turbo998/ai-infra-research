"""Self-contained VFR evidence cards, logical output budget and original SVG."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "vfr-video-timestamp-clip-sampling"


def deck(r):
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic integer-tick video metadata · independently_reproduced=false · '
                     'Original CPU lookup timings separately measured; no codec/GPU or video-model quality result.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 107 · VFR timestamp sampling</span><span>{n} / 7</span></footer></section>')

    tl,s,d,p,b = r["timeline"],r["sampling"],r["duplicates"],r["padding"],r["budget"]
    card("HANDS-ON · VIDEO DATA PIPELINES",
         '<span class="accent">Stable decoder APIs</span> still need presentation-time semantics.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         f'<p>{len(tl["pts"])} synthetic frames<br>PTS {tl["pts"]}</p>'
         f'<p>Half-open end: {tl["end"]} ticks</p></div>'
         '<div class="panel"><h2>Structured input</h2>'
         f'<p>Header FPS: {s["average_fps_numerator"]}/{s["average_fps_denominator"]}</p>'
         f'<p>{s["denominator"]} timestamp requests</p><p>Ticks/s: {tl["ticks_per_second"]}</p></div>'
         '<div class="panel"><h2>Decisions</h2>'
         f'<p>{s["approximate_mismatches"]}/{s["denominator"]} average-FPS mismatches</p>'
         f'<span class="pill">First CPU tier: {r["latency"]["rows"][0]["primary_median_ms"]:.4f} ms</span>'
         '<p>Model tokens: not measured</p></div></div>',
         f'<b>Provenance:</b> {r["latency"]["hardware"]}, Python {r["latency"]["python"]}; synthetic PTS. '
         'Pinned November 9, 2026 source graduates low-level APIs from BETA to stable, not the invention of VFR lookup.')
    index = s["request_ticks"].index(1040)
    scores = '<blockquote>At timestamp tick 1040, the previous frame ends and frame index 1 begins.</blockquote>'
    for label,selected in (("Exact presentation-time lookup",s["exact_indices"][index]),
                           ("Average-FPS estimate",s["approximate_indices"][index])):
        match = int(selected == s["exact_indices"][index])
        scores += '<div class="panel"><span class="pill">score · binary identity verifier</span>'
        scores += f'<span class="pill">{"Confirm" if match else "Send to a human"}</span><h2>{label}</h2>'
        scores += f'<p>Selected index {selected}; verifier {match}/1. Calibrated confidence: null.</p>'
        scores += bar(match,1)+'</div>'
    card("SCENARIO: HALF-OPEN FRAME OWNERSHIP",
         'The exact boundary belongs to the new frame—not its predecessor.',
         scores,'<b>Verifier bars are not probabilities.</b> At the exact next PTS, ownership changes. '
         'Our strictly increasing integer PTS model avoids float-rounding ambiguity; native timestamps use seconds.')
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for tick,exact,approximate in zip(s["request_ticks"],s["exact_indices"],s["approximate_indices"]):
        correct = exact == approximate
        matrix += f'<tr><td>VFR</td><td>Tick {tick}; exact {exact}</td>'
        matrix += f'<td>FPS estimate {approximate}{" ✕" if not correct else ""}</td>'
        matrix += f'<td>{"Confirm" if correct else "Use exact PTS"}</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout">Mismatch denominator is nine synthetic requests.</div>'
    matrix += '<div class="callout">This is not a measured video-model accuracy rate or a native decoder failure.</div></div>'
    card("WHAT I TESTED",
         f'Average FPS selects a different frame on {s["approximate_mismatches"]} of {s["denominator"]} VFR requests.',
         matrix,'<b>Approximate is a different contract.</b> The source documents metadata-based approximate '
         'indexing and exact scanning/custom mappings. An estimate disagreement is not a violation of the approximate API.')
    card("UNDER THE HOOD","Deduplicating target identities must still restore every output slot.",
         '<pre>exact: find i with PTS[i] &lt;= t &lt; next_PTS[i]\n'
         'approx: floor((t - begin) * header_average_FPS)\n'
         'stream guard: begin &lt;= t &lt; end\n'
         'sort unique target indices for lookup/decode requests</pre>'
         '<p>↓ scatter back using the original request order</p>'
         f'<pre>requests = {d["request_ticks"]}\n'
         f'target indices = {d["restored_slots"]}\n'
         f'unique identities = {d["unique_targets"]}\n'
         f'output-to-unique = {d["output_to_unique"]}\n'
         f'output slots = {d["output_slot_count"]}; target identities = {d["unique_target_count"]}\n'
         'actual decoded frame count = null</pre>',
         '<b>Unique target count is not actual decode work.</b> Keyframe seeks, codec dependencies and pre-roll '
         'may decode other frames. The source copies duplicate output slots; it does not promise a unique-only result tensor.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["variants"]:
        robust += f'<tr><td>{row["frames"]} frame timeline</td><td>Interval oracle</td><td>Binary search</td><td>{row["index_mismatches"]} mismatches</td></tr>'
    robust += f'<tr><td>Uniform CFR control</td><td>Exact lookup</td><td>Average FPS</td><td>{s["cfr_mismatches"]}/{s["cfr_requests"]} mismatches</td></tr></table>'
    robust += '<div class="panel"><h2>Wrap cycles the valid clip timestamp list</h2>'
    robust += f'<p>Source-style wrap: {p["wrap"]["frame_indices"]}</p>'
    robust += f'<p>Wrong whole-stream modulo: {p["wrong_whole_stream_modulo_indices"]}</p></div>'
    card("ROBUSTNESS CHECK","Clip wrap is not modulo over the whole video duration.",
         robust,'<b>Padding changes sampling semantics.</b> repeat_last repeats the last valid requested timestamp, '
         'not necessarily the video’s final frame. error rejects overflow; neither invalid times nor missing data are silently clamped.')
    maximum = max(max(row["primary_median_ms"],row["baseline_median_ms"]) for row in r["latency"]["rows"])
    timing = '<p>Blue: binary metadata lookup · Gray: linear interval oracle. Each tier has 128 requests.</p>'
    for row in r["latency"]["rows"]:
        timing += f'<div class="latency-row"><h2>{row["frames"]:,} synthetic PTS entries</h2>'
        for key,baseline in (("primary_median_ms",False),("baseline_median_ms",True)):
            timing += f'<div class="latency-pair"><b>{row[key]:.4f} ms</b>{bar(row[key],maximum,baseline)}</div>'
        timing += '</div>'
    card("LATENCY","Binary metadata lookup does not measure video decoding.",
         timing,'<b>Measured original Python CPU:</b> two warmups, seven alternating repeats, median; noisy shared-host samples retained. '
         'PTS metadata already exists; linear reference also constructs interval tuples. File scanning, I/O, codec, seeks and tensors excluded.')
    card("MY TAKE",
         f'{b["output_slots"]} output slots still occupy {b["output_tensor_bytes"]:,} logical bytes.',
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>PTS-aware video sampling.</p><p>Explicit clip-tail policy.</p>'
         '<p>Preserve duplicates and request order.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Average FPS called exact VFR time.</p><p>Unique targets called total decoded frames.</p>'
         '<p>Metadata lookup called NVDEC throughput.</p></div></div>'
         '<pre>READINESS: original integer-tick sampling\n'
         'SOURCE: pinned TorchCodec, BSD-3-Clause\n'
         'MODEL: unique increasing PTS, declared next-PTS holds\n'
         'CODEC / GPU / MODEL QUALITY: not measured\n'
         'OUTPUT PAYLOAD: unpadded logical tensor, not peak HBM\n'
         'PROVENANCE: synthetic; independently_reproduced=false</pre>',
         '<b>Memory boundary:</b> even if several requests share targets, a materialized output retains every slot. '
         'Unique-target payload is a hypothetical separate buffer, not guaranteed allocator or decoder savings.')
    styles = ('<style>.latency-row{padding:12px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:180px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 107 · VFR timestamp sampling · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["clips","frames_per_clip","height","width","channels"]){
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive exact integer");
  }
  if(!Number.isSafeInteger(p.unique_targets)||p.unique_targets<1)throw Error("unique_targets must be positive for a nonempty output");
  if(![1,2,4].includes(p.scalar_bytes))throw Error("scalar_bytes must be 1, 2 or 4");
  const slots=p.clips*p.frames_per_clip;
  if(p.unique_targets>slots)throw Error("unique_targets cannot exceed requested output slots");
  const frame=p.height*p.width*p.channels*p.scalar_bytes;
  const result={output_slots:slots,unique_target_count:p.unique_targets,
    repeated_output_slots:slots-p.unique_targets,one_frame_bytes:frame,output_tensor_bytes:slots*frame,
    unique_target_tensor_payload_bytes:p.unique_targets*frame,pts_and_duration_float64_bytes:slots*16,
    output_index_int64_bytes:slots*8,actual_decoded_frame_count:null,decoder_latency_ms:null,
    peak_hbm_bytes:null,model_quality:null};
  if(!Object.values(result).filter(v=>typeof v==="number").every(Number.isSafeInteger))
    throw Error("budget exceeds exact JavaScript integer range");
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
    result.textContent=JSON.stringify({provenance:"synthetic clip dimensions and unique-target input",
      ...estimateBudget(p),production_approved:false,
      note:"Unique target count is not actual codec-decoded frames. All output slots remain materialized; unique payload is a hypothetical separate buffer. Excludes codec surfaces, seek pre-roll, pitch, conversion, allocator and peak HBM."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r):
    p = r["budget"]["input"]
    labels = dict(clips="Clips / 片段数",frames_per_clip="Frames per clip / 每段帧数",
                  unique_targets="Synthetic unique targets / 假设唯一目标帧数",
                  height="Output height / 输出高",width="Output width / 输出宽",
                  channels="Channels / 通道",scalar_bytes="Bytes per scalar / 每标量字节")
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'step="1" value="{p[key]}"></label>' for key,label in labels.items())
    return page("VFR video / 视频时间戳采样工作台",
                '<main><div class="eyebrow">ROUND 107 · PRESENTATION TIME · SYNTHETIC OUTPUT PAYLOAD</div>'
                '<h1>Sample presentation time, not assumed FPS.<br>按 PTS 取帧，保留每一个输出位置。</h1>'
                '<div class="panel">Synthetic VFR 的平均 FPS 对九次请求选错五次。'
                'Wrap 循环有效 clip timestamp list，不是把时间对整段视频取模。'
                'Unique targets 不等于实际 codec decode 数，也不会自动缩小 materialized output。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>唯一目标帧数是 synthetic 输入，不从 FPS 猜测。'
                '同帧请求可以共享查找/目标，但输出仍有完整 clip × frame slots。'
                '<p>仅为 unpadded logical tensor 和元数据 bytes，不含 codec surfaces、seek pre-roll、pitch、'
                '色彩转换、allocator 或 peak HBM。未测实际 TorchCodec、NVDEC 或视频模型精度。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r):
    s,d = r["sampling"],r["duplicates"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Presentation-time frame-card sorter</title>'
        '<desc id="description">A detailed solid sorter orders frame cards by presentation timestamp. '
        'Shared target cards route back into all original output slots instead of deleting repeated requests.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 107 · VFR VIDEO · PRESENTATION TIME AND CLIP POLICIES</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">Presentation-time ownership is not average-FPS arithmetic.</text>'
        '<text x="64" y="180" font-size="24">A mechanical analogy: sort frame cards by PTS, then restore every requested output slot.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Synthetic VFR: average FPS differs on {s["approximate_mismatches"]}/{s["denominator"]} requests; exact half-open boundaries are explicit.</text>'
        +f'<text x="80" y="700" font-size="25">{d["unique_target_count"]} unique targets restore {d["output_slot_count"]} slots in request order: {d["restored_slots"]}.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Clip wrap cycles valid requested timestamps; repeat_last need not select the final video frame.</text>'
        +'<text x="96" y="842" font-size="23">Unique targets do not count codec dependency/pre-roll decoding or shrink materialized output.</text>'
        +'<text x="96" y="883" font-size="21">November 2026 stable API capability; original metadata model only, no codec/GPU benchmark.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/vfr_timestamp_sampling_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r):
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
