import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "mel-filterbank-cast-order-axis-frame-contract"
JS = r"""
function estimate(p){
  for(const key of Object.keys(DEFAULT)){
    const min=["samples","center"].includes(key)?0:1;
    if(!Number.isSafeInteger(p[key])||p[key]<min)throw Error(key+" requires bounded exact integer >= "+min);
  }
  if(![0,1].includes(p.center)||![2,4,8].includes(p.element_bytes))throw Error("center=0/1 and output width=2/4/8 required");
  if(p.fft_length%2||p.frame_length>p.fft_length)throw Error("select even FFT no smaller than frame");
  if(!p.center&&p.samples<p.frame_length)throw Error("uncentered count needs a full frame");
  if(p.center&&p.samples<2)throw Error("original reflect selection needs at least two samples");
  const padded=p.samples+2*Math.floor(p.frame_length/2)*p.center;
  const frames=1+Math.floor((padded-p.frame_length)/p.hop_length),bins=p.fft_length/2+1;
  const out={frame_count:frames,frequency_bins:bins,
    selected_filter_float32_bytes:bins*p.mel_filters*4,
    selected_power_spectrum_bytes:p.batch*frames*bins*p.element_bytes,
    selected_mel_output_bytes:p.batch*frames*p.mel_filters*p.element_bytes};
  if(![padded,...Object.values(out)].every(Number.isSafeInteger))throw Error("result exceeds exact arithmetic range");
  return out;
}
"""


def workbench(r):
    fields = "".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                     for k,v in r["default_budget"]["inputs"].items())
    body = f"""<h1>Pin the frontend contract before swapping its backend</h1>
    <p>Round 116 · Original frame-count/payload arithmetic, not native audio or ASR execution.</p>
    <p>center 0/1 uses frame_length//2 padding on each side. Original budget selects even FFT length and positive full frames.</p>
    <form id="form"><div class="grid">{fields}</div><button>Check frame and axis budget</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Filter axes, HTK/Slaney, normalization, cast order, power/log units and actual backend support must be validated separately.</p>"""
    script = f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");
 error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required integer controls");
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  out.textContent=JSON.stringify({inputs:p,outputs:estimate(p),
   note:"Filter coefficients fixed to source float32. Power-spectrum/mel widths are selected accounting assumptions, not native intermediate precision. Logical payloads are not simultaneous allocations, complex STFT workspace, device traffic or peak HBM. No waveform decoding or feature parity is certified."},null,2);
 }catch(e){error.textContent=e.message;}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Mel frontend frame and axis budget",body,script)


def cards(r):
    c = r["case"]
    total,different = c["matrix_entries"],c["cast_order_different_entries"]
    agreement = 100*(total-different)/total
    timings = r["latency"]["rows"]
    maximum = max(v["dense_median_ms"] for v in timings)
    charts = "".join(f"<div class='metric'><b>{v['frames']} frames · 40 filters × 257 bins</b>"
                     f"<p>Nonzero weights {v['nonzero_median_ms']:.6f} ms</p>{bar(v['nonzero_median_ms'],maximum)}"
                     f"<p>Dense scalar {v['dense_median_ms']:.6f} ms</p>{bar(v['dense_median_ms'],maximum,True)}</div>"
                     for v in timings)
    variants = "".join(f"<tr><td>frame {v['frame_length']}, center {v['center']}</td><td>16,000 samples / hop 160</td>"
                       f"<td>{v['outputs']['frame_count']} frames</td><td>Pin padding</td></tr>" for v in r["variants"])
    contract = html.escape(json.dumps({"sample_rate":16000,"fft":512,"mel":40,"scale":"slaney",
                                       "normalization":"slaney","dtype":"float32","axes":"mel × frequency"},indent=2))
    sections = [
        ("HANDS-ON · AUDIO FRONTEND","An audio dependency swap still needs a numerical feature contract",
         "<div class='grid'><div><h3>Scenarios</h3><p>Cast order · filter axis · scale · frame parity</p></div>"
         "<div><h3>Structured input</h3><p>16 kHz · FFT 512<br>40 mel × 257 bins</p></div>"
         f"<div><h3>Top decisions</h3><p>{different}/{total} weights differ after a cast-order change.</p><span class='pill'>Original CPU projection only</span></div></div>"
         "<div class='dark'>November 12 Transformers audio dependency migration verified. Synthetic scalar study, independently_reproduced=false; no native filter parity, ASR accuracy or GPU benchmark.</div>"),
        ("SCENARIO: CAST ORDER",f"Same float32 shape, but {different} scalar weights change by rounding order",
         "<blockquote>round32(round32(triangle) × area factor) versus round32(triangle × area factor)</blockquote>"
         f"<div class='metric'><span class='pill'>score</span><h3>Cast before area normalization</h3><p>Reference original scalar bank: {total} entries</p>{bar(total,total)}</div>"
         f"<div class='metric'><span class='pill'>score</span><h3>✕ Normalize in float64, cast only at the end</h3><p>{different} changed entries; max {c['cast_order_max_ulp']} ULP</p>{bar(different,total)}</div>"
         "<p>Observed original scalar weights, not native backend bit-exactness or calibrated confidence. Small numeric changes do not establish an ASR quality regression.</p>"),
        ("WHAT I TESTED","400/401-sample centered frames produce 101/100 frames at the same hop",
         f"<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>{variants}"
         f"<tr><td>Power four</td><td>Power to dB</td><td>{c['power_four_db']:.6f} dB</td><td>Use 10 log10</td></tr>"
         f"<tr><td>✕ Amplitude factor on power</td><td>20 log10(4)</td><td>{c['wrong_twenty_log_power_four']:.6f} dB</td><td>Reject unit mismatch</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: distinguish frame/window/FFT lengths and matrix axes.</div><div class='metric'>Failure analysis: matching dimensions alone does not pin scale, normalization, rounding or logarithm units.</div></div>"),
        ("UNDER THE HOOD","Transpose the source filter bank; do not silently relabel its axes",
         f"<pre>{contract}</pre><p>↓ Original deterministic projection</p>"
         "<pre>Source helper: [frequency, mel]\nSource model bridge: contiguous transpose\nOriginal bank: [mel, frequency]\n\nfeature[t, m] = sum_k bank[m, k] * power[t, k]\n\nframes = 1 + floor((L + 2*floor(frame/2)*center\n                    - frame) / hop)</pre>"
         "<p>Pinned source explicitly selects Slaney scale/area normalization and float32 before normalization. Original scalar transcendental/reduction behavior is not a native parity certificate.</p>"),
        ("ROBUSTNESS CHECK",f"Late casting lowers exact weight agreement to {agreement:.3f}% in the scalar fixture",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Exact weight agreement</td><td>{total}/{total}</td><td>{total-different}/{total}</td><td>{agreement-100:.3f}</td></tr>"
         "<tr><td>Frame permutation / projected values</td><td>matching order</td><td>inverse permutation agrees</td><td>0</td></tr></table>"
         "<p>HTK and Slaney have equal matrix shapes but different frequency spacing. The independent piecewise triangle oracle and sparse/dense projection test guard semantics, not just sizes.</p>"
         "<div class='dark'>The November 12 change removes librosa dependencies and preserves selected model filter contracts; it is not evidence that every decoder/resampler backend is interchangeable.</div>"),
        ("LATENCY","Skipping zero filter weights changes Python work, not the native audio pipeline",
         charts+"<p>Original Python projection only: two warmups, seven alternating repeats, three operations/sample; median repeat means. Filter construction/compression, decode, resample, STFT and native/GPU execution excluded.</p>"),
        ("MY TAKE","Pin the feature contract before measuring backend speed",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Audio frontend migration audits, explicit filter axes/scales/dtypes, frame and payload accounting.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Claiming native bit parity, ASR quality, GPU throughput, exact allocator peaks or arbitrary model defaults.</p></div></div>"
         "<div class='dark'>Apache-2.0 inspected sources; original scalar formulas, timing and artwork. Native filter bit-exactness, backend latency, ASR quality and peak HBM remain null. Browser visual check not performed.</div>"),
    ]
    return page("Mel frontend contract — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Mel frontend cabinet: retain axes, scale, rounding and frame contracts</title>
    <desc>Original static 60-solid magazine sorter as a frequency-to-mel weighting cabinet.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 116 · AUDIO FRONTEND</text>
    <text x="88" y="158" font-size="36" font-weight="700">Same feature shape does not preserve the numerical contract</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Frequency bins → explicit filter orientation → mel features</text>
    <text x="88" y="758" font-size="25">Pin HTK/Slaney, area normalization and float32 cast order</text>
    <text x="88" y="806" font-size="25">Separate frame/window/FFT lengths, centering and power/log units</text>
    <text x="88" y="880" font-size="18">Synthetic original analogy · November 12 dependency migration · no native parity / ASR / GPU claim</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
