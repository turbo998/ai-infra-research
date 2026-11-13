"""Self-contained views of sealed streaming-state correctness checks."""

from html import escape
import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "streaming-codec-convolution-state-stride-overlap-add"


def deck(report: dict) -> str:
    b = report["budget"]
    cases = {row["id"]: row for row in report["cases"]}
    cards = []

    def card(eyebrow, title, evidence, takeaway):
        number = len(cards) + 1
        cards.append(f'<section class="card" aria-label="Card {number} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">SYNTHETIC DEMO · independently_reproduced=false · no GPU/audio-quality measurement</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 95 · Streaming codec state</span><span>{number} / 7</span></footer></section>')

    card("HANDS-ON · STREAMING STATE", "Three persistent buffers preserve one continuous convolution.",
         '<div class="grid"><div class="panel"><h2>Boundaries</h2><p>Irregular transport packets</p>'
         '<p>Stride-aligned kernel calls</p><p>One EOS policy</p><p>New request reset</p></div>'
         '<div class="panel"><h2>State</h2><p>Pending input</p><p>Previous causal context</p>'
         '<p>Bias-free transpose overlap</p></div><div class="panel"><h2>Original toy</h2>'
         f'<p>{len(report["fixture"]["input"])} scalar samples<br>{len(report["fixture"]["weights"])} taps<br>'
         f'Stride {report["fixture"]["stride"]}<br>{report["fixture"]["eos_padding_samples"]} EOS zeros</p>'
         '<span class="pill">Confidence: null</span></div></div>',
         f'<b>Separate one-layer-pair budget: {b["total_state_bound_bytes"]/1024:.0f} KiB for '
         f'{b["input"]["streams"]} hypothetical streams.</b> This excludes the rest of the codec, '
         'transformer KV, weights, activations and allocator overhead.')
    exact = cases["transpose-bias-free-carry"]
    wrong = cases["transpose-carry-includes-bias"]
    evidence = '<blockquote>The next convolution-transpose chunk already adds its bias. '
    evidence += 'Carrying a biased tail adds that bias again at the seam.</blockquote>'
    for row in (exact, wrong):
        evidence += '<div class="panel"><span class="pill">score · boundary error</span><span class="pill">'
        evidence += ('Confirm' if row["matches_reference"] else 'Send to a human') + '</span>'
        evidence += f'<h2>{escape(row["id"])}</h2><p>{row["output_samples"]} output samples · max error '
        evidence += f'{row["maximum_absolute_error"]:.3e} · confidence: null</p>'
        evidence += bar(row["maximum_absolute_error"], wrong["maximum_absolute_error"]) + '</div>'
    card("SCENARIO: TRANSPOSE OVERLAP", f'A carried bias adds {wrong["maximum_absolute_error"]:.2f} error at chunk seams.',
         evidence, '<b>Carry contributions, not a second output bias.</b> Bias removal is required even '
         'though the streamed and offline output lengths match.')
    table = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        error = f'{row["maximum_absolute_error"]:.2e}' if row["same_length"] else 'Shape mismatch'
        table += f'<tr><td>{escape(row["id"])}</td><td>Offline output parity?</td>'
        table += f'<td>{"Match" if row["matches_reference"] else "✕"}<br>{error}</td>'
        table += f'<td>{"Confirm" if row["matches_reference"] else "Send to a human"}</td></tr>'
    table += '</table><div class="grid two"><div class="callout"><b>Gate:</b> same length and '
    table += f'error &lt; {report["fixture"]["tolerance"]}.</div><div class="callout"><b>Four misses:</b> '
    table += 'reset, dropped samples, duplicate bias, stale state.</div></div>'
    card("WHAT I TESTED", "Seven state-machine paths expose four boundary failures.",
         table, '<b>Correctness cases, not model accuracy.</b> A length mismatch has unknown numeric '
         'error; it is never zip-truncated into a zero-error success.')
    card("UNDER THE HOOD", "Align stride frames before calling the kernel; pad only once at EOS.",
         '<pre>effective_kernel = (K - 1) * dilation + 1\n'
         'history = effective_kernel - stride\n'
         'pending += transport_packet\n'
         'aligned = floor(len(pending) / stride) * stride\n'
         'emit conv(previous + pending[:aligned])\n'
         'retain incomplete input, not a new padded chunk</pre><p>↓ transpose reconstruction</p>'
         '<pre>raw = local_transpose(frames) + bias\n'
         'raw[:K-S] += old_partial\n'
         'new_partial = raw[-(K-S):] - bias\n'
         'emit raw[:len(frames) * S]\n\n'
         'New stream: clear ingress AND history AND partial.</pre>',
         '<b>Alignment matters:</b> this source uses effective-kernel minus stride history, '
         'not a universal K−1 convention. K=S is a no-overlap special case in the implementation.')
    robustness = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["perturbations"]:
        label = "17 one-sample packets" if len(row["id"]) > 30 else row["id"]
        robustness += f'<tr><td>{escape(label)}</td><td>{row["reference_samples"]} outputs</td>'
        robustness += f'<td>{row["output_samples"]} outputs</td><td>{row["maximum_absolute_error"]:.2e}</td></tr>'
    robustness += '</table><div class="panel"><h2>Lifecycle gates</h2>'
    robustness += '<p>EOS pads one unfinished stride frame, or rejects it explicitly.</p>'
    robustness += '<p>Reused slots reset; interleaved streams keep independent buffers.</p>'
    robustness += '<p>A finished stream rejects more packets. Transpose final tail stays trimmed.</p></div>'
    card("ROBUSTNESS CHECK", "One packet or 17 packets reconstruct the same five output frames.",
         robustness, '<b>Chunk partition invariance is an actual test.</b> It does not prove equivalence '
         'for replicate padding, arbitrary normalizers, resampling, RVQ or a full neural codec.')
    maximum = max(row["baseline_collection_bound_ms"] for row in report["scaling"])
    scaling = '<p>Blue: chunk collection bound · gray: one assumed full utterance. No computation is timed.</p>'
    for row in report["scaling"]:
        scaling += f'<div class="latency-row"><b>{row["input"]["chunk_samples"]} stage samples</b>'
        scaling += f'<div class="latency-pair"><span>Chunk <b>{row["chunk_collection_bound_ms"]:.0f} ms</b></span>'
        scaling += bar(row["chunk_collection_bound_ms"], maximum) + '</div>'
        scaling += f'<div class="latency-pair"><span>Full input <b>{row["baseline_collection_bound_ms"]:.0f} ms</b></span>'
        scaling += bar(row["baseline_collection_bound_ms"], maximum, True) + '</div></div>'
    card("LATENCY", "Collection bounds grow from 10 to 80 ms with the chunk.",
         scaling, '<b>Assumed continuous 24 kHz stage input; not service latency.</b> Network, compute, '
         'queueing and playout are excluded. Deterministic formulas; median/repeats N/A.')
    card("MY TAKE", "Keep state per stream; use packet boundaries only for transport.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Explicit chunk adapters.</p>'
         '<p>Known padding/alignment convention.</p><p>Offline-vs-streamed waveform checks.</p>'
         '<p>Complete state reset on reuse.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Resetting history at every packet.</p><p>Padding every packet independently.</p>'
         '<p>Emitting a trimmed transpose tail.</p><p>Claiming toy parity implies audio quality.</p></div></div>'
         '<pre>READINESS: original scalar prototype\n'
         'SOURCE: pinned Kyutai Moshi convolution modules\n'
         'RIGHTS: upstream Apache-2.0 metadata; no implementation copied\n'
         'SCOPE: constant causal padding; one layer pair\n'
         'GPU LATENCY / AUDIO QUALITY: not measured\n'
         'ADAPTER EOS POLICY: explicit harness extension</pre>',
         '<b>Synthetic, independently_reproduced=false.</b> Established streaming mechanics, not '
         'a new-model launch. Source contract rejects non-stride-aligned kernel calls.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:11px 6px}</style>')
    return page("Round 95 · Streaming codec state · seven synthetic cards", styles + '<main>' + "".join(cards) + '</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const [key,value] of Object.entries(p))
    if(!Number.isSafeInteger(value)||value<1) throw Error(key+" must be a positive safe integer");
  if(p.kernel<p.stride) throw Error("Kernel must cover stride");
  if(![2,4].includes(p.scalar_bytes)) throw Error("Scalar bytes must be 2 or 4");
  if(p.chunk_samples%p.stride) throw Error("Processing chunk must align to stride");
  const effective=(p.kernel-1)*p.dilation+1, factor=p.streams*p.channels*p.scalar_bytes;
  const conv=factor*(effective-p.stride), partial=factor*(p.kernel-p.stride), pending=factor*(p.stride-1);
  if(!Number.isSafeInteger(conv+partial+pending)) throw Error("State exceeds exact numerical range");
  return {conv_history_bytes:conv,transpose_partial_bytes:partial,ingress_pending_bound_bytes:pending,
    total_state_bound_bytes:conv+partial+pending,chunk_collection_bound_ms:1000*p.chunk_samples/p.stage_rate_hz,
    baseline_collection_bound_ms:1000*p.baseline_samples/p.stage_rate_hz};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),
  error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct the invalid input.";return;}
  try{
    const p={...BASE};
    for(const control of form.elements)if(control.name){
      if(control.value.trim()==="")throw Error(control.name+" is required");
      p[control.name]=Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic; one illustrative layer pair",
      ...estimateBudget(p),gpu_latency_ms:null,audio_quality:null,production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    labels = {"streams": "Concurrent streams / 流数", "channels": "Per-stage channels / 通道数",
              "kernel": "Kernel taps / 卷积核", "stride": "Stride / 步幅", "dilation": "Causal dilation / 膨胀",
              "scalar_bytes": "Scalar bytes (2 or 4) / 标量字节", "stage_rate_hz": "Stage samples/s / 本层采样率",
              "chunk_samples": "Aligned processing chunk / 对齐处理块", "baseline_samples": "Full-input baseline / 整段基线"}
    p = report["budget"]["input"]
    controls = "".join(
        f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="1" step="1" value="{p[key]}"></label>'
        for key,label in labels.items()
    )
    return page("Streaming codec state / 流式卷积状态工作台",
                '<main><div class="eyebrow">ROUND 95 · SYNTHETIC DEMO</div>'
                '<h1>Packets are not convolution boundaries.<br>传输分块不等于状态重置。</h1>'
                '<p>One hypothetical causal/transpose layer pair, not a complete codec footprint.</p>'
                '<div class="panel"><b>三本账：</b>pending 输入、effective-kernel−stride 历史、K−S 无 bias overlap。'
                '<p>真实 kernel 调用必须 stride 对齐；EOS 仅统一补一次，或显式拒绝残帧。</p></div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>输入/输出通道数在这个示例相同；dilation 只作用于 causal 分支。'
                '采样率是被建模层的 stage rate，不能给下采样层机械套用原始 PCM rate。'
                '<p>未计其他 codec 层、RVQ、Transformer cache、权重、激活、分配器。'
                'collection bound 不是 GPU、网络或完整语音服务延迟。</p></div></main>',
                "const BASE=" + json.dumps(p) + ";\n" + CALCULATOR)


def summary_svg(report: dict) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Three separate streaming buffers</title>'
        '<desc id="description">A three-magazine card sorter represents pending ingress, previous convolution '
        'history and bias-free transpose overlap. Stream lifecycle, not GPU speed, is the subject.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 95 · SYNTHETIC DEMO · STREAMING CODEC STATE</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A packet boundary must not erase the signal history.</text>'
        '<text x="64" y="180" font-size="24">Three magazines: pending input, previous context and bias-free overlap contributions.</text>'
        + sorter_scene()
        + '<text x="80" y="650" font-size="26" font-weight="700">Align to stride → preserve history → subtract carry bias → reset at request reuse</text>'
        + f'<text x="80" y="700" font-size="25">Toy: {len(report["fixture"]["input"])} samples plus '
        + f'{report["fixture"]["eos_padding_samples"]} EOS zeros produce five exact frames.</text>'
        + '<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        + '<g fill="#ffffff"><text x="96" y="799" font-size="23">Source kernel requires positive stride-aligned chunks; arbitrary ingress is an adapter policy.</text>'
        + '<text x="96" y="842" font-size="23">Never flush the transpose tail if the causal reference trims it.</text>'
        + '<text x="96" y="883" font-size="21">One-layer state budget is not full codec memory · collection wait is not service latency</text></g>'
        + '<text x="64" y="955" font-size="20">Data: experiments/streaming_codec_state_selection/report.json · independently_reproduced=false</text>'
        + '</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT / "docs" / "reports" / f"{SLUG}-selection-cards.html": deck(report),
        ROOT / "docs" / "reports" / f"{SLUG}-sizer.html": workbench(report),
        ROOT / "assets" / "svg" / f"{SLUG}-selection-faq-summary.svg": summary_svg(report),
    }
