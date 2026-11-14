import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="resize-integer-size-half-pixel-align-corners"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==4||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![1,2,4,8].includes(p.element_bytes))throw Error("Positive exact shape and selected byte width required");
 const n=BigInt(p.input_side),m=BigInt(p.output_side),c=BigInt(p.channels),b=BigInt(p.element_bytes);
 const out={selected_input_payload_bytes:Number(n*n*c*b),selected_output_payload_bytes:Number(m*m*c*b),selected_two_axis_float64_coordinate_bytes:Number(2n*m*8n)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(x,m,mode){
 if(x.length<1||x.length>64||!x.every(v=>Number.isFinite(v)&&Math.abs(v)<=1000000)||!Number.isSafeInteger(m)||m<1||m>64||!["half_pixel","align_corners","asymmetric","pytorch_half_pixel"].includes(mode))throw Error("Teaching controls:1..64 finite values,magnitude<=1000000,output1..64,supported mode");
 const n=x.length,s=m/n,coords=Array.from({length:m},(_,j)=>mode==="align_corners"?(m===1?0:j*(n-1)/(m-1)):mode==="pytorch_half_pixel"&&m===1?0:mode==="asymmetric"?j/s:(j+.5)/s-.5);
 const values=coords.map(raw=>{const p=Math.min(Math.max(raw,0),n-1),i=Math.floor(p),t=p-i;return x[i]*(1-t)+x[Math.min(i+1,n-1)]*t});
 return {coordinates:coords,values};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Resize coordinate-contract workbench",
        f"""<h1>Same output shape does not establish the same source sampling grid</h1>
        <p>Round126 · Original sizes-mode 1-D linear interpolation with clamped edges, not native ONNX.
        No antialias, ROI, cubic or vision-model quality reproduction.</p>
        <form id="form"><h2>Independent square-image payload budget</h2><div class="grid">{fields}</div>
        <h2>Teaching sizes-mode axis</h2><label>Values<input id="samples" value="0,10,20" required></label>
        <label>Output axis length<input id="length" value="5" type="number" step="1" required></label>
        <label>Coordinate mode<select id="mode"><option>half_pixel</option><option>align_corners</option><option>asymmetric</option><option>pytorch_half_pixel</option></select></label>
        <button>Inspect source grid</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Separate sealed Python scales-mode fixture reproduces the old unrounded-denominator mistake.
        This interactive toy derives scale from integer sizes; original Fraction oracle remains authoritative for exact values.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)])),raw=document.getElementById("samples").value.split(",");
 if(raw.some(s=>!s.trim()))throw Error("Missing sample");
 out.textContent=JSON.stringify({result:simulate(raw.map(Number),Number(document.getElementById("length").value),document.getElementById("mode").value),budget:budget(p),
 note:"Selected integer sizes and clamped linear interpolation only; no native quality/performance claim"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["recompute_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>Axis{v['input_width']}→{v['output_width']} · eight channels</b>"
                  f"<p>Cached grid {v['cached_median_ms']:.6f} ms</p>{bar(v['cached_median_ms'],mx)}"
                  f"<p>Per-channel grid {v['recompute_median_ms']:.6f} ms</p>{bar(v['recompute_median_ms'],mx,True)}</div>"
                  for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · VISION COORDINATES","Integer output length2 restores both corners; unrounded length12/5 misses the last source pixel",
         "<div class='grid'><div><h3>Scenarios</h3><p>Scales versus sizes<br>Singleton output<br>Nearest tie modes</p></div>"
         "<div><h3>Input</h3><p>[1,2,3,4]<br>Scale3/5<br>align_corners</p></div>"
         "<div><h3>Top decision</h3><p>Use realized integer length<br>Not scale×input−1</p><span class='pill'>Native latency: unknown</span></div></div>"
         f"<div class='dark'>Sep30ONNXreference fix in Nov12pinned tree; established geometry,not new November theory. Original CPU Python{r['latency']['python']} / {r['latency']['machine']}; synthetic,independently_reproduced=false.</div>"),
        ("SCENARIO: CORNER RETENTION","Correct align_corners retains2/2 corner values; wrong denominator retains1/2",
         f"<blockquote>Input{c['input']}, scale{c['scale']}, integer output{c['integer_output_length']}. Fixed{c['fixed_output']} versus wrong{c['wrong_output']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · deterministic coordinate equality,not confidence</span><h3>Fixed2/2</h3>{bar(c['integer_output_length'],c['integer_output_length'])}<p>Act automatically after contract verification</p></div>"
         f"<div class='metric'><span class='pill'>score · deterministic coordinate equality,not confidence</span><h3>Wrong{c['matching_coordinates']}/2</h3>{bar(c['matching_coordinates'],c['integer_output_length'],True)}<p>✕ Confirm integer denominator</p></div>"),
        ("WHAT I TESTED","Singleton half_pixel and pytorch_half_pixel sample different positions even at the same shape",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>4→1,half_pixel</td><td>Source center?</td><td>{c['singleton_half_pixel']}</td><td>Use selected grid</td></tr>"
         f"<tr><td>4→1,pytorch_half_pixel</td><td>Special position?</td><td>{c['singleton_pytorch_half_pixel']}</td><td>Do not rename blindly</td></tr>"
         "<tr><td>Nearest half tie</td><td>Same rounding?</td><td>✕ Floor/ceil preferences differ</td><td>Specify mode</td></tr>"
         "<tr><td>Clamped edge</td><td>ROI extrapolation?</td><td>✕ Not equivalent</td><td>Separate policy</td></tr>"
         "<tr><td>Downsampling</td><td>Linear means antialias?</td><td>✕ Not guaranteed</td><td>Separate filter</td></tr></table>"
         "<div class='two grid'><div class='callout'>Model excludes zero-length outputs,cubic,ROI,antialias and quantized casting.</div><div class='callout'>Native source/float kernels not executed; exact Fraction geometry plus independent hat-basis oracle.</div></div>"),
        ("UNDER THE HOOD","Scale-based extent floors first; align_corners uses the resulting output length",
         "<pre>M = floor(N*selected_scale)\nalign_corners:\n  M==1 → 0\n  else j*(N-1)/(M-1)\nhalf_pixel:\n  (j+1/2)/scale - 1/2\npytorch_half_pixel:\n  M==1 → 0,else half_pixel</pre>"
         "<p>↓ Original independent value oracle</p><pre>Clamp source coordinate to[0,N-1]\nSum every input value times its triangular hat basis\nCompare with two-neighbor linear interpolation\n\nnearest ties are separate rounding policy\nnot a coordinate transformation</pre>"),
        ("ROBUSTNESS CHECK","3→5 mode changes alter interior values without changing output shape",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>half_pixel→align_corners</td><td>{c['sizes_3_to_5_outputs']['half_pixel']}</td><td>{c['sizes_3_to_5_outputs']['align_corners']}</td><td>Grid values,not probability</td></tr>"
         f"<tr><td>Nearest tie floor→ceil</td><td>{c['nearest_prefer_floor']}</td><td>{c['nearest_prefer_ceil']}</td><td>Middle0→10</td></tr></table>"
         "<p>Exact synthetic coordinates; changing resize mode is a preprocessing change,not merely an optimization.</p>"),
        ("LATENCY","Original cached coordinate grids avoid per-channel grid reconstruction across eight channels",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Initial cached plan excluded. Original1-DPythonfloat paths,not2-D/nativeONNX/GPU/quality.</p>"),
        ("MY TAKE","Record realized shape,scale,coordinate mode and rounding before replacing a vision resize",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Cross-export coordinate audits,singleton and tie controls,selected image payload budgeting.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Shape-only equivalence,blind mode replacement,antialias/ROI claims or native vision-model accuracy.</p></div></div>"
         "<div class='dark'>Apache-2.0ONNXsource;original model/oracle/art. Native latency/model quality/HBM unknown. Browser visual inspection not performed.</div>"),
    ]
    return page("Resize coordinate contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Resize grid ticket archive: integer extent,coordinate mode and nearest tie are separate</title>
<desc>Original sixty-solid sorting cabinet analogy for sampling coordinates and policy labels.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG126 · VISION RESIZE COORDINATES</text>
<text x="88" y="158" font-size="36" font-weight="700">Same shape is not the same source-pixel grid</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Align corners to realized integer output size,not unrounded scale product</text>
<text x="88" y="758" font-size="25">Singleton half-pixel and framework-specific modes can differ</text>
<text x="88" y="806" font-size="25">Coordinate mapping,nearest ties and antialias filtering are distinct</text>
<text x="88" y="880" font-size="18">Original Fraction geometry / hat-basis oracle / CPU timing · no native quality or GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
