import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="moss-audio-profiling-dummy-sample-rate-budget"
JS=r"""
function integer(x,lo=0,hi=10000000){return Number.isInteger(x)&&x>=lo&&x<=hi;}
function plan(p,strategy="rate_aligned"){
 if(!p||Object.keys(p).sort().join(",")!=="audio_count,hop,mel_dim,mel_sr,override_samples,seq_len,text_tokens,time_markers")throw Error("Exact original sizing fields required");
 if(![8000,16000,22050,24000,44100,48000].includes(p.mel_sr)||!integer(p.hop,1,1024)||!integer(p.mel_dim,1,256)||!integer(p.audio_count,0,64)||!integer(p.seq_len,1,131072)||!integer(p.text_tokens,0,131072)||typeof p.time_markers!=="boolean")throw Error("Bounded sizing profile required");
 if(p.override_samples!==null&&!integer(p.override_samples))throw Error("Override must be null or nonnegative samples");
 if(!["rate_aligned","legacy_fixed"].includes(strategy))throw Error("Known sizing strategy required");
 const maximum=p.mel_sr*30,budgetFrames=Math.ceil(maximum/p.hop),budget=Math.ceil(budgetFrames/8);
 let samples=strategy==="rate_aligned"?maximum:16000;const warnings=[];
 if(p.override_samples){if(p.override_samples>samples)warnings.push("override_above_selected_max_ignored");samples=Math.min(samples,p.override_samples);}
 const frames=Math.ceil(samples/p.hop),count=Math.ceil(frames/8);let digits=0;
 if(p.time_markers){const fullSeconds=Math.floor(count*p.hop*8/p.mel_sr);for(let s=2;s<=fullSeconds;s+=2)digits+=String(s).length;}
 const n=p.audio_count,span=n*(count+digits+2);
 return {max_audio_seconds:30,selected_samples:n?samples:0,declared_raw_mel_frames_per_item:n?frames:0,
 declared_audio_budget:n?{audio:budget}:{},declared_audio_tokens_per_item:n?count:0,total_audio_tokens:n*count,
 marker_digit_tokens_per_item:n?digits:0,audio_span_tokens:span,context_ledger_fits:span+p.text_tokens<=p.seq_len,
 budget_shortfall_per_item:n?Math.max(0,budget-count):0,declared_unique_numpy_waveform_bytes:n?samples*8:0,
 declared_float32_mel_batch_bytes:n*p.mel_dim*frames*4,warnings,native_extracted_frames:null,native_peak_bytes:null,native_audio_tokenizer_or_encoder_verified:null};
}
"""


def workbench(r):
    body="<h1>Size audio profiling inputs to the declared duration budget,not a fixed sample count</h1><p>Original integer sizing.No waveforms,NumPy,STFT,tokenizer,Torch or GPUprofiling.Native raw-feature length and peak memory remain unknown.</p>"
    body+=f'<form id="form"><label>Original sample/feature/context profile<textarea id="input" rows="11" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Compare declared dummy coverage</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
    body+="<p>Frames are declared ceil-budget witnesses,not extracted feature lengths.Audio token budget excludes BOS/EOS/marker digits.Context ledger does not truncate the dummy.Shared waveform bytes and dense mel budgets are not allocator peak.</p>"
    return page("Audio profiling budget workbench",body,JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),legacy_fixed_samples:plan(p,"legacy_fixed")},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["legacy"];marked=r["matrix"][1]["outputs"]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],"Declared sizing",str(x["outputs"]["total_audio_tokens"])+"audio tokens",
        "✕ review shortfall/context" if x["outputs"]["budget_shortfall_per_item"] or not x["outputs"]["context_ledger_fits"] else "Confirm native profile"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["fused_median_ms"],x["three_stage_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['items']}original frame lengths</b><p>Fused ceiling{x['fused_median_ms']:.6f}ms</p>{bar(x['fused_median_ms'],mx)}<p>Three scalar stages{x['three_stage_median_ms']:.6f}ms</p>{bar(x['three_stage_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · AUDIO PROFILING","MOSS-Audio dummy length now follows mel sample rate times its30second budget",
         "<div class='grid'><div>Scenarios<br>Default audio<br>Overrides/markers<br>Multiple items</div><div>Input<br>Sample rate/hop<br>Declared frames<br>Context separately</div>"
         f"<div>Decision<br>{a['selected_samples']}samples<br>{a['declared_audio_tokens_per_item']}declared pads<br>Native peak unknown</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14source;CPUinteger sizing separately measured on Python{r['latency']['python']}/{r['latency']['machine']}.No waveform/feature/model execution.</div>"),
        ("SCENARIO: DEFAULT AUDIO",f"At16kHz/hop160,the original30second profile declares{a['declared_audio_tokens_per_item']}pads versus{b['declared_audio_tokens_per_item']}for16000samples",
         "<blockquote>Declared raw frames3000versus100;three stride2ceiling stages give375versus13.The real source test checks default processing coverage,but was not run here.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm native profiling · confidence unknown</span><p>Legacy declared shortfall{b['budget_shortfall_per_item']}pads;default aligned sample count480000.</p></div>"
         f"<div class='metric'><span class='pill'>score · original declared audio token count</span><p>Aligned{a['declared_audio_tokens_per_item']}</p>{bar(a['declared_audio_tokens_per_item'],a['declared_audio_tokens_per_item'])}<p>Legacy{b['declared_audio_tokens_per_item']}</p>{bar(b['declared_audio_tokens_per_item'],a['declared_audio_tokens_per_item'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original profiles distinguish budget coverage,overrides,markers and context",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Short override can intentionally underfill maximum profiling budget.Above-max clamps with an explicit warning.</div><div class='callout'>seq_len does not shrink this source's max-per-item calculation;our context ledger flags oversubscription separately.</div></div>"),
        ("UNDER THE HOOD","Three ceil-halving length stages equal one ceil-divide-by-eight on integer frame counts",
         "<pre>Maximum samples=mel_sr×30\nBudget frames=ceil(samples/hop)\nAudio tokens=ceil(ceil(ceil(frames/2)/2)/2)\nEquivalent integer ceil(frames/8)\nDummy length follows same sample-rate budget</pre><p>↓ Distinct allocations and tokens</p>"
         "<pre>Shared NumPy dummy waveform reference\nFloat32 dense mel batch per audio item\nAudio-token count excludes BOS/EOS/digits\nActual extracted frames remain unmeasured\nNative peak memory is not dense tensor budget</pre>"),
        ("ROBUSTNESS CHECK",f"Time markers keep{marked['total_audio_tokens']}audio tokens but expand original full span to{marked['audio_span_tokens']}",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Markers</td><td>{a['audio_span_tokens']}span tokens</td><td>{marked['audio_span_tokens']}span tokens</td><td>{marked['marker_digit_tokens_per_item']}digit tokens</td></tr>"
         "<tr><td>Short override</td><td>480000samples</td><td>16000samples</td><td>Maximum budget unchanged</td></tr>"
         "<tr><td>Three items</td><td>One shared waveform</td><td>Three mel batch rows</td><td>Not three waveform buffers</td></tr>"
         "<tr><td>No audio</td><td>Audio budget mapping</td><td>Empty mapping</td><td>No item allocation</td></tr></table>"),
        ("LATENCY","Original fused scalar sizing excludes waveform extraction,torch kernels and GPUprofiling",timing+
         "<p>2warmups,7alternating repeats,3batches/sample;median repeat means.Precreated positive frame lengths.Native source uses Torch lengths,not these Python scalar loops.</p>"),
        ("MY TAKE","Use a budget-sized default dummy,but verify real feature/token coverage and peak allocation",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Shared processor configuration,sample-rate-aware duration,explicit overrides,independent context/span checks and native profiler evidence.</div><div class='bad'><h3>⛔ Not for</h3>Fixed16000samples for every rate,short override as worst-case coverage,confusing audio pads/full span,or dense budgets as peak memory.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.No native waveform/STFT/Torch/encoder/peak/visual acceptance.Ceil-budget frame model is not universal extracted-feature rounding.</div>")]
    return page("Audio profiling — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["legacy"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Audio budget gauge cabinet</title><desc>Original sixty-solid cabinet;depth of the audio compartment represents declared profiling coverage,not GPUpeak allocation.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG169 · AUDIO BUDGET GAUGE</text><text x="88" y="158" font-size="34">A30second budget needs sample-rate-aligned dummy length</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original default{a['selected_samples']}samples /{a['declared_raw_mel_frames_per_item']}declared frames /{a['total_audio_tokens']}audio tokens</text><text x="88" y="760" font-size="24">Legacy16000samples gives{b['total_audio_tokens']}declared tokens / shortfall{b['budget_shortfall_per_item']}</text><text x="88" y="810" font-size="24">Audio budget,full prompt span,shared waveform and dense mel batch are distinct</text><text x="88" y="880" font-size="18">ActualNov14source · original integer sizing · waveform/STFT/model/GPUpeak/visual acceptance unknown</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
