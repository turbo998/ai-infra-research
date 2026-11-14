import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="polyphase-rational-resampling-phase-antialias"
JS=r"""
function gcd(a,b){while(b){[a,b]=[b,a%b]}return a}
function budget(p){
 if(!Object.values(p).every(Number.isSafeInteger)||p.input_samples<1||p.up<1||p.down<1||![1,2,4,8].includes(p.element_bytes))throw Error("Positive exact rate/count and selected byte width required");
 const g=gcd(p.up,p.down),u=p.up/g,d=p.down/g;
 const count=(BigInt(p.input_samples)*BigInt(u)+BigInt(d)-1n)/BigInt(d);
 const out={reduced_up:u,reduced_down:d,output_samples:Number(count),input_payload_bytes:p.input_samples*p.element_bytes,output_payload_bytes:Number(count)*p.element_bytes};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(x,up,down,phase=0){
 if(x.length<1||x.length>64||!x.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000000)||!Number.isSafeInteger(up)||!Number.isSafeInteger(down)||up<1||down<1||up>64||down>64)throw Error("Teaching controls: 1..64 integer samples, magnitude<=1000000, rates1..64");
 const g=gcd(up,down),u=up/g,d=down/g;if(u===1&&d===1)return x;
 const taps=[1,2,3,2,1],out=[];
 for(let m=0;m<Math.ceil(x.length*u/d);m++){
  const c=m*d+2+phase;let numerator=0;
  for(let k=0;k<x.length;k++){const j=c-k*u;if(j>=0&&j<5)numerator+=x[k]*taps[j]*u}
  if(!Number.isSafeInteger(numerator))throw Error("Exact numerator range exceeded");out.push(numerator/9);
 }return out;
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Rational resampling rate and phase workbench",
        f"""<h1>Reduced rate, centered FIR and anti-aliasing are different contracts</h1>
        <p>Round125 · Original custom odd symmetric five-tap FIR [1,2,3,2,1]/9.
        No native SciPy default Kaiser filter or audio-quality reproduction.</p>
        <form id="form"><h2>Independent payload budget</h2><div class="grid">{fields}</div>
        <h2>Teaching FIR simulation (not the 44.1k/48k budget)</h2>
        <label>Integer input, comma separated<input id="samples" value="1,2,3,4,5" required></label>
        <label>Teaching up<input id="up_toy" value="3" type="number" step="1" required></label>
        <label>Teaching down<input id="down_toy" value="2" type="number" step="1" required></label>
        <button>Inspect rate, phase and output</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Zero extension only. Selecting another rate does not redesign the toy FIR; it may be unsuitable
        for that conversion. The one-phase-shift path is an intentional wrong-centering control.
        Float display is not arbitrary-precision Fraction output.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)])),raw=document.getElementById("samples").value.split(",");
 if(raw.some(s=>!s.trim()))throw Error("Missing sample");
 const x=raw.map(Number),u=Number(document.getElementById("up_toy").value),d=Number(document.getElementById("down_toy").value);
 out.textContent=JSON.stringify({budget:budget(p),centered:simulate(x,u,d),wrong_one_phase_shift:simulate(x,u,d,1),
 note:"Selected fixed custom filter, not default native FIR or audio quality"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["dense_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['input_samples']} input samples · U3/D2</b>"
                  f"<p>Nonzero-phase gather {v['polyphase_median_ms']:.6f} ms</p>{bar(v['polyphase_median_ms'],mx)}"
                  f"<p>Dense zero insertion {v['dense_median_ms']:.6f} ms</p>{bar(v['dense_median_ms'],mx,True)}</div>"
                  for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · AUDIO RATE CONTRACT","44,100→48,000 sample budgeting reduces to U160/D147, not two huge FIR rates",
         "<div class='grid'><div><h3>Scenarios</h3><p>Rate reduction<br>FIR center/scale<br>Anti-aliasing</p></div>"
         "<div><h3>Structured input</h3><p>Selected44,100 input samples<br>up48,000/down44,100<br>float32 width</p></div>"
         "<div><h3>Top decisions</h3><p>48,000 output samples<br>Design at reduced upsampled rate</p><span class='pill'>Native latency: unknown</span></div></div>"
         f"<div class='dark'>Established SciPy resample_poly semantics; latest tree update not a new resampler launch. Original CPU Python{r['latency']['python']} / {r['latency']['machine']}; synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: ANTI-ALIAS CONTROL","A three-tap filter removes this alternating fixture's interior alias energy: 7→0",
         f"<blockquote>Input alternates +1/-1; naive every-second-sample output is constant +1. Selected [1/4,1/2,1/4] FIR, U1/D2, zero extension.</blockquote>"
         f"<div class='metric'><span class='pill'>score · synthetic squared amplitude, not confidence</span><h3>Naive interior energy {c['interior_naive_energy']} over {c['interior_samples']} samples</h3>{bar(c['interior_naive_energy'],c['interior_naive_energy'],True)}<p>✕ Confirm filter contract</p></div>"
         f"<div class='metric'><span class='pill'>score · synthetic squared amplitude, not confidence</span><h3>Filtered interior energy {c['interior_filtered_energy']}</h3>{bar(float(c['interior_filtered_energy']),c['interior_naive_energy'])}<p>Act automatically only within selected test fixture</p></div>"
         "<p>Excludes first boundary sample. Not broadband stopband attenuation or audio/model quality.</p>"),
        ("WHAT I TESTED","Correct rate alone does not establish phase, gain, boundary or spectral behavior",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>48k/44.1k</td><td>Reduced ratio?</td><td>160/147</td><td>Reduce gcd</td></tr>"
         "<tr><td>5 samples,U3/D2</td><td>Output count?</td><td>8, ceiling not floor</td><td>Exact integer math</td></tr>"
         f"<tr><td>Wrong phase</td><td>Coordinates agree?</td><td>✕ {c['phase_shift_matching_coordinates']}/{c['output_coordinates']}</td><td>Fix center</td></tr>"
         f"<tr><td>Missing up gain</td><td>Coordinates agree?</td><td>✕ {c['unscaled_matching_coordinates']}/{c['output_coordinates']}</td><td>Scale selected FIR</td></tr>"
         "<tr><td>Constant input</td><td>Boundary remains1?</td><td>✕ First3/4 with zero extension</td><td>Choose pad contract</td></tr></table>"
         "<div class='two grid'><div class='callout'>Fixed toy FIR is not redesigned for arbitrary H5 ratios.</div><div class='callout'>No universal first-sample equality for arbitrary custom FIRs.</div></div>"),
        ("UNDER THE HOOD","Centered FIR output samples only nonzero upsampled positions",
         "<pre>U = up/gcd(up,down)\nD = down/gcd(up,down)\nM = ceil(N*U/D)\ncenter = m*D + half_filter_length\n\ny[m] = sum_k x[k] * U*h[center-k*U]\nonly in-range filter/source indices count</pre>"
         "<p>↓ Independent original oracle</p><pre>insert zeros between input samples\nfull dense convolution with U*h\nsample at m*D + half_filter_length\nzero outside finite support\n\nU=D=1 → independent identity-copy rule</pre>"),
        ("ROBUSTNESS CHECK","A one-grid phase shift changes every coordinate of the five-sample U3/D2 fixture",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>First output</td><td>{c['centered_output'][0]}</td><td>{c['one_phase_shift_output'][0]}</td><td>Grid phase +1</td></tr>"
         f"<tr><td>Last output</td><td>{c['centered_output'][-1]}</td><td>{c['one_phase_shift_output'][-1]}</td><td>Grid phase +1</td></tr>"
         f"<tr><td>Matching coordinates</td><td>{c['output_coordinates']}/{c['output_coordinates']}</td><td>{c['phase_shift_matching_coordinates']}/{c['output_coordinates']}</td><td>Not probability points</td></tr></table>"
         "<p>Exact Fraction model/dense oracle; intentional wrong phase and omitted gain controls. Zero boundary policy explicitly selected.</p>"),
        ("LATENCY","Original nonzero-phase gather avoids dense zero-filled FIR work for the fixed U3/D2 toy",
         chart+"<p>Two warmups, seven alternating repeats, three operations/sample; median repeat means. Original Python float paths, same values checked; no native SciPy/GPU/filter-design timing.</p>"),
        ("MY TAKE","Declare rate, FIR, phase and boundary together before replacing audio preprocessing",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Exact count/phase/gain audits, selected anti-alias controls and independent polyphase values.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Blind decimation, default-filter quality claims, arbitrary-rate fixed-toy FIR or chunk-independent streaming correctness.</p></div></div>"
         "<div class='dark'>BSD-3-Clause SciPy source, original equations/oracle/art. Audio quality/native latency/GPU memory unknown. Browser inspection not performed.</div>"),
    ]
    return page("Rational polyphase resampling — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Polyphase sample-ticket sorter: select phase without computing inserted zeros</title>
<desc>Original sixty-solid sorting cabinet metaphor for rational grid, FIR phase and boundary policy.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG125 · RATIONAL POLYPHASE CONTRACT</text>
<text x="88" y="158" font-size="36" font-weight="700">A rate ratio is not a phase or anti-alias guarantee</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Reduce up/down by gcd; output count uses ceiling, not floor</text>
<text x="88" y="758" font-size="25">Centered FIR needs the selected phase and upsampling gain</text>
<text x="88" y="806" font-size="25">Low-pass and boundary extension are part of preprocessing semantics</text>
<text x="88" y="880" font-size="18">Original exact toy FIR / CPU models · no native default filter / audio quality / GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
