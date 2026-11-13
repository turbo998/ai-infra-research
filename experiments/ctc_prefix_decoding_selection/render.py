"""Self-contained CTC probability evidence with sealed CPU timing samples."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "ctc-prefix-beam-blank-streaming-state"


def deck(report: dict) -> str:
    cases = {row["id"]:row for row in report["cases"]}
    cards = []

    def card(eyebrow,title,evidence,takeaway):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7">'
                     f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                     '<p class="muted">SYNTHETIC INPUTS · independently_reproduced=false · CPU timing separately measured; no ASR-quality claim</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 101 · CTC blank and prefix state</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · CTC DECODING",
         'The best <span class="accent">frame path</span> need not yield the most probable transcript.',
         '<div class="grid"><div class="panel"><h2>Fixture</h2>'
         '<p>2 frames<br>Blank + a + b<br>Exact gate: &lt; 1e-12</p>'
         '<span class="pill">No language model</span></div><div class="panel"><h2>Structured input</h2>'
         '<pre>[0.45, 0.35, 0.20]\n[0.45, 0.35, 0.20]</pre>'
         '<p>Blank, a, b probabilities.</p></div><div class="panel"><h2>Top result</h2>'
         f'<p>Greedy empty path: {report["greedy"]["path_probability"]:.4f}</p>'
         f'<p>Summed transcript a: {report["best_transcript"]["probability"]:.4f}</p>'
         '<span class="pill">WER / confidence: null</span></div></div>',
         '<b>Original standard-library CPU functions.</b> Only the timing subsection is measured '
         'local execution; frame probabilities are synthetic. No speech model, GPU or native decoder is run.')
    scores = '<blockquote>Collapse consecutive equal labels before removing blank. '
    scores += 'Then sum all frame paths yielding the same transcript.</blockquote>'
    for label,value,action in (("Greedy best path: empty",report["greedy"]["path_probability"],"Send to a human"),
                               ("Summed transcript: a",report["best_transcript"]["probability"],"Confirm")):
        scores += f'<div class="panel"><span class="pill">score · acoustic event probability</span>'
        scores += f'<span class="pill">{action}</span><h2>{label}</h2><p>{value:.4f} · calibrated confidence: null</p>'
        scores += bar(value,1.0)+'</div>'
    card("SCENARIO: PATH VERSUS TRANSCRIPT",
         'Transcript a has probability 0.4375 despite a blank-greedy winner.',
         scores,'<b>Different questions:</b> the best path is not a bug when that is the objective. '
         'It must not be relabeled as the summed transcript posterior.')
    names = {
        "exact-prefix-logsum":"Exact log-sum prefix",
        "best-path-per-transcript":"Viterbi per transcript",
        "remove-blank-before-repeat-collapse":"Remove blank first",
        "merge-blank-nonblank-repeat-ends":"Merge repeat-end states",
        "carried-prefix-state":"Carry streaming state",
        "reset-and-concatenate-chunks":"Reset and concatenate",
        "beam-one-pruning":"Prune to beam one",
    }
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        answer = "Search/objective difference" if row["intentional_search_policy"] else "Match" if row["matches_reference"] else "Mismatch ✕"
        matrix += f'<tr><td>{names[row["id"]]}</td><td>{row["fixture"]}</td>'
        matrix += f'<td>{answer}<br>{row["maximum_probability_or_score_error"]:.2e}</td><td>'
        matrix += ('Evaluate policy' if row["intentional_search_policy"] else 'Confirm' if row["matches_reference"] else 'Review recurrence')+'</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout"><b>Gate:</b> exact summed transcript scores.</div>'
    matrix += '<div class="callout"><b>Failure:</b> merging repeat-end states creates total score mass 1.65.</div></div>'
    card("WHAT I TESTED","Exact prefix scores match enumeration; blank-state shortcuts fail.",
         matrix,'<b>Per-row fixtures are explicit.</b> Search pruning and Viterbi scoring answer different '
         'questions; wrong collapse order or state loss breaks the selected acoustic CTC contract.')
    card("UNDER THE HOOD","Blank and nonblank ends cannot be merged before repeated-label updates.",
         '<pre>prefix state: (log_pb, log_pnb)\n'
         'pb = paths ending in blank\n'
         'pnb = paths ending in a nonblank label\n\n'
         'blank step: new_pb(prefix) += total(prefix) * P(blank)\n'
         'same label, no emission: new_pnb(prefix) += pnb * P(label)</pre>'
         '<p>↓ a repeated emitted label requires a separating blank</p>'
         '<pre>repeat extension:\n'
         '    new_pnb(prefix + label) += pb * P(label)\n'
         'different-label extension:\n'
         '    new_pnb(prefix + label) += (pb+pnb) * P(label)\n\n'
         'Actual implementation uses logadd, not linear multiplication.\n'
         'LM, hotword, BPE and timestamp states are excluded.</pre>',
         '<b>Source scope:</b> inspected pyctcdecode merges log scores and retains last-character/partial '
         'state; its streaming API also carries LM caches, processed frames and finalization flags.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["partition_checks"]:
        robust += f'<tr><td>Same acoustic posterior</td><td>One 3-frame chunk</td><td>{row["sizes"]}</td>'
        robust += f'<td>Max error {row["maximum_error"]:.2e}</td></tr>'
    underflow = report["underflow"]
    robust += '</table><div class="panel"><h2>Positive events can underflow in linear arithmetic</h2>'
    robust += f'<p>{underflow["frames"]} binary frames, target a: log probability {underflow["log_probability"]:.6f}.</p>'
    robust += '<p>Linear float value is zero by underflow, not an impossible transcript or zero confidence.</p>'
    robust += '<p>A separate fixed-target forward automaton agrees with its closed form.</p></div>'
    card("ROBUSTNESS CHECK","Four chunkings preserve the posterior when prefix state is retained.",
         robust,'<b>Resetting a chunk is not EOS.</b> Independent chunk collapse plus concatenation can '
         'invent aa in two frames, even though acoustic CTC needs a separating third blank frame.')
    latency = report["latency"]
    maximum = max(row["path_enumeration_median_ms"] for row in latency["rows"])
    timing = '<p>Measured original CPU medians: blue exact prefix DP, gray exact path enumeration.</p>'
    for row in latency["rows"]:
        timing += f'<div class="latency-row"><b>{row["frames"]} frames · {3**row["frames"]:,} unmerged paths</b>'
        for key,label,baseline in (("prefix_dp_median_ms","Prefix DP",False),
                                   ("path_enumeration_median_ms","Enumerate paths",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","Nine frames already enumerate 19,683 paths in the exact baseline.",
         timing,'<b>Windows ARM64, Python '+latency["environment"]["python"]+'.</b> '
         'Two warmups, seven alternating repeats; validation included. Not native pyctcdecode/NeMo, '
         'GPU inference or production ASR latency.')
    card("MY TAKE","Keep acoustic score semantics separate from search and ASR quality.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Blank/repeat recurrence reviews.</p><p>Exact tiny-lattice references.</p>'
         '<p>Streaming state and log-space checks.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Greedy paths called transcript posteriors.</p><p>Concatenating reset chunks.</p>'
         '<p>Toy probability called calibrated confidence.</p></div></div>'
         '<pre>READINESS: original acoustic CTC prototype\n'
         'SOURCE: pinned pyctcdecode; Apache-2.0 rights retained\n'
         'CPU TIMING: measured original exact functions only\n'
         'LM / HOTWORDS / BPE / ALIGNMENT: not implemented\n'
         'ASR WER / CER / GPU LATENCY: not measured\n'
         'PROVENANCE: synthetic inputs, independently_reproduced=false</pre>',
         '<b>Established decoder mechanics.</b> Contemporary NeMo integration adds scoring/pruning '
         'options; those are not reproduced or validated by this pure-acoustic experiment.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}'
              'td{overflow-wrap:anywhere}</style>')
    return page("Round 101 · CTC prefix state · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["frames","labels","beam_width","token_index_bytes"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(![2,4,8].includes(p.token_index_bytes)||!Number.isFinite(p.frame_ms)||p.frame_ms<=0)
    throw Error("Index bytes must be 2/4/8 and frame duration finite positive");
  if(p.labels>=2**(8*p.token_index_bytes))throw Error("Labels exceed selected unsigned index capacity");
  let exact=1;
  for(let i=0;i<p.frames;i++){
    if(exact>Math.floor(Number.MAX_SAFE_INTEGER/(p.labels+1))){exact=null;break;}
    exact*=p.labels+1;
  }
  const updates=p.beam_width*(p.labels+2),logs=p.beam_width*16;
  const payload=p.beam_width*p.frames*p.token_index_bytes;
  if(![updates,logs,payload].every(Number.isSafeInteger))throw Error("Payload bound exceeds exact numerical range");
  const logPaths=p.frames*Math.log10(p.labels+1),span=p.frames*p.frame_ms;
  if(![logPaths,span].every(Number.isFinite))throw Error("Span/path logarithm exceeds finite numerical range");
  return {path_count_if_js_safe:exact,log10_unmerged_path_count:logPaths,
    per_frame_contribution_bound:updates,two_log_scores_bytes:logs,prefix_token_payload_upper_bytes:payload,
    audio_span_ms:span,gpu_latency_ms:null,asr_wer:null,calibrated_confidence:null};
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
    result.textContent=JSON.stringify({provenance:"synthetic beam-cap and logical payload formulas",
      ...estimateBudget(p),production_approved:false,
      note:"Null exact path count means outside JS safe-integer display range; log10 remains a formula, not a measured state count."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {"frames":"Acoustic frames / 声学帧数","labels":"Labels excluding blank / 非 Blank 标签数",
              "beam_width":"Pruned beam cap / 搜索上限","token_index_bytes":"Prefix index bytes (2/4/8) / 索引字节",
              "frame_ms":"Audio span per frame ms / 每帧音频跨度"}
    controls = ""
    for key,label in labels.items():
        controls += f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="{0.001 if key=="frame_ms" else 1}" step="{0.001 if key=="frame_ms" else 1}" value="{p[key]}"></label>'
    return page("CTC prefix decoding / Blank 与流式前缀工作台",
                '<main><div class="eyebrow">ROUND 101 · ACOUSTIC CTC · SYNTHETIC BOUNDS</div>'
                '<h1>Keep blank and nonblank histories separate.<br>Blank 与 Nonblank 末状态不能提前合并。</h1>'
                '<div class="panel"><b>例子：</b>两帧均为 [blank=.45,a=.35,b=.20]，greedy 为空串，'
                '但 a 汇总全部合法路径后概率 .4375。不是 WER 或校准 confidence。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>(labels+1)^frames 是未合并路径数，不是实际 retained states；'
                'beam×(labels+2) 为每帧 contribution 上界，prefix payload 不含 Python 对象、LM、timestamps 或模型。'
                '<p>每帧音频跨度不是计算延迟。有限 beam 是近似策略，不能从字节上界推导识别质量或 GPU 性能。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">CTC blank and nonblank card sorter</title>'
        '<desc id="description">Separate blank-ending and label-ending card trays preserve a prefix '
        'until equivalent frame paths can be summed. Streaming carries both trays forward.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 101 · SYNTHETIC INPUTS · CTC PREFIX STATE</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A blank-ending prefix is not a label-ending prefix.</text>'
        '<text x="64" y="180" font-size="24">Repeated emitted labels need a separating blank; sum equivalent paths in log space.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Two-frame toy: transcript a = {report["best_transcript"]["probability"]:.4f}; '
        +f'greedy blank path = {report["greedy"]["path_probability"]:.4f}.</text>'
        +'<text x="80" y="700" font-size="25">Resetting chunk state and concatenating can invent an impossible repeated label.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Exact prefix DP and path enumeration agree; finite beam search intentionally drops mass.</text>'
        +'<text x="96" y="842" font-size="23">Measured timing is local CPU scalar code, not native decoder or production ASR latency.</text>'
        +'<text x="96" y="883" font-size="21">Acoustic probability is not calibrated confidence; WER, CER and GPU latency remain unknown.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/ctc_prefix_decoding_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
