import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="quantize-linear-rne-zero-point-axis"
JS=r"""
function estimate(p){
 if(!Object.values(p).every(Number.isSafeInteger)||p.rows<1||p.columns<1||![-2,-1,0,1].includes(p.axis))throw Error("positive exact dimensions and rank-two axis required");
 const n=[p.rows,p.columns][(p.axis+2)%2];
 const out={selected_float32_input_bytes:p.rows*p.columns*4,selected_one_byte_output_bytes:p.rows*p.columns,
 selected_float32_scale_bytes:n*4,selected_one_byte_zero_point_bytes:n};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");
 return out;
}
function ratio(text){
 const parts=text.trim().split("/");
 if(parts.length>2||!parts.every(s=>/^-?[0-9]+$/.test(s)))throw Error("exact input requires integer or numerator/denominator");
 let n=BigInt(parts[0]),d=parts.length===2?BigInt(parts[1]):1n;
 if(d===0n)throw Error("nonzero denominator required");
 if(d<0n){n=-n;d=-d}
 if(n>2147483647n||n< -2147483647n||d>2147483647n)throw Error("bounded exact ratio required");
 return [n,d];
}
function rne(n,d){
 const negative=n<0n,a=negative?-n:n;let q=a/d,r=a%d;
 if(2n*r>d||(2n*r===d&&q%2n===1n))q++;
 return negative?-q:q;
}
function calculate(matrix,scales,zeros,axis,dtype){
 if(!matrix.length||!matrix[0].length||matrix.length*matrix[0].length>64||matrix.some(row=>row.length!==matrix[0].length))throw Error("nonempty rectangular matrix, at most 64 values required");
 if(![-2,-1,0,1].includes(axis)||!["int8","uint8"].includes(dtype))throw Error("explicit rank-two axis and int8/uint8 required");
 axis=(axis+2)%2;
 const width=[matrix.length,matrix[0].length][axis],low=dtype==="int8"?-128:0,high=dtype==="int8"?127:255;
 if(scales.length!==width||zeros.length!==width||zeros.some(z=>!Number.isSafeInteger(z)||z<low||z>high))throw Error("per-axis zero points must match dimension and dtype");
 const s=scales.map(ratio);if(s.some(([n])=>n<=0n))throw Error("positive scale required");
 return matrix.map((row,r)=>row.map((text,c)=>{
  const [n,d]=ratio(text),i=axis===0?r:c,[sn,sd]=s[i];
  const rounded=rne(n*sd,d*sn);
  if(rounded>2147483647n||rounded< -2147483647n)throw Error("rounded quotient outside original bounded int32 scope");
  return Math.max(low,Math.min(high,Number(rounded)+zeros[i]));
 }));
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>Round first; add the integer zero point second</h1>
    <p>Round 121 · Original exact-ratio per-axis int8/uint8 model. No native ONNX runtime/division precision/GPU reproduction.</p>
    <form id="form"><label>Matrix rows separated by semicolons
    <input id="matrix" value="1/2,3/2;-1/2,-3/2" required></label>
    <label>Per-axis scales<input id="scales" value="1,1/2" required></label>
    <label>Integer zero points<input id="zeros" value="1,-1" required></label>
    <label>Output dtype<select id="dtype"><option>int8</option><option>uint8</option></select></label>
    <div class="grid">{fields}</div><button>Inspect exact per-axis quantization</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Axis selects both toy metadata and independent logical memory budget. BigInt exact inputs do not model native division rounding.
    No scalar-scale/blocked/FP6/FP8/INT4 coverage. Metadata bytes assume float32 scales and one-byte zero points.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  const matrix=document.getElementById("matrix").value.split(";").map(row=>row.split(",").map(s=>s.trim()));
  const scales=document.getElementById("scales").value.split(",").map(s=>s.trim());
  const parts=document.getElementById("zeros").value.split(",").map(s=>s.trim());
  if(parts.some(s=>!/^(-?[0-9]+)$/.test(s)))throw Error("integer zero points required");
  const result=calculate(matrix,scales,parts.map(Number),p.axis,document.getElementById("dtype").value);
  out.textContent=JSON.stringify({result,budget:estimate(p),note:"Integer zero point does not commute with ties-to-even at half ties. Logical bytes are not peak HBM or model quality."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Exact integer QuantizeLinear contract workbench",body,script)


def cards(r):
    c,b=r["case"],r["boundary"]
    maximum=max(max(v["cached_median_ms"],v["division_median_ms"]) for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>{v['elements']} exact-ratio values</b>"
                   f"<p>Cached reciprocal {v['cached_median_ms']:.6f} ms</p>{bar(v['cached_median_ms'],maximum)}"
                   f"<p>Per-element division {v['division_median_ms']:.6f} ms</p>{bar(v['division_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · INTEGER QUANTIZATION","Moving an odd zero point inside ties-to-even changes the integer result",
         "<div class='grid'><div><h3>Scenarios</h3><p>Half ties<br>Per-axis shape<br>Saturation</p></div>"
         "<div><h3>Structured input</h3><p>Exact 2×2 matrix<br>scales [1,1/2]<br>zero points [1,-1]</p></div>"
         "<div><h3>Top decisions</h3><p>Round then add<br>Pin metadata axis</p><span class='pill'>Original exact-ratio model</span></div></div>"
         "<div class='dark'>Pinned ONNX reference rounds quotient, adds zero point, then clips for integer outputs. Existing integer law, not a November launch. Synthetic, independently_reproduced=false.</div>"),
        ("SCENARIO: HALF TIE","At x/scale=1/2 and zero point 1, the two expressions return 1 and 2",
         f"<blockquote>{b['odd_zero_point_counterexample']}</blockquote>"
         f"<div class='metric'><span class='pill'>score · integer output agreement</span><h3>Declared order: {c['correct_matches']}/{c['elements']}</h3>{bar(c['correct_matches'],c['elements'])}</div>"
         f"<div class='metric'><span class='pill'>score · negative control</span><h3>Zero point inside rounding: {c['wrong_rounding_matches']}/{c['elements']}</h3>{bar(c['wrong_rounding_matches'],c['elements'],True)}</div>"
         "<p>Not model-quality scores or confidence. RNE parity is why odd integer translation fails at a half tie.</p>"),
        ("WHAT I TESTED","Matching tensor shapes does not prove matching per-axis quantization",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Correct axis</td><td>2×2 square fixture</td><td>{c['correct']}</td><td>Act automatically</td></tr>"
         f"<tr><td>✕ Wrong axis</td><td>Still shape-valid</td><td>{c['wrong_axis']}</td><td>Check metadata</td></tr>"
         f"<tr><td>Signed half ties</td><td>{b['half_ties']}</td><td>{b['rounded']}</td><td>Use even integer</td></tr>"
         f"<tr><td>Saturation</td><td>Outside int8 range</td><td>{b['lower_saturation']}, {b['upper_saturation']}</td><td>Clip, not wrap</td></tr>"
         "<tr><td>Source dtype</td><td>Omitted output metadata</td><td>Default can be uint8</td><td>Confirm dtype</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: exact-ratio arithmetic is not native float-division precision.</div><div class='metric'>Failure analysis: per-tensor, per-axis and blocked metadata are different contracts.</div></div>"),
        ("UNDER THE HOOD","Integer order is quotient → nearest even → zero point → saturation",
         "<pre>original exact-ratio model:\n  ratio = x / positive_scale\n  q = nearest_even(ratio)\n  q = q + integer_zero_point\n  q = clamp(q, dtype_min, dtype_max)\n\nper-axis index follows selected dimension</pre>"
         "<p>↓ Independent original verifier</p><pre>compare distances to floor(ratio) and ceil(ratio)\nif equal: choose even integer\n\nat +1/2: round=0\n0+1=1, but round(1/2+1)=2\n\nbounded int32 quotient; no native overflow emulation</pre>"),
        ("ROBUSTNESS CHECK","Changing axis in a square matrix preserves shape but only half the selected outputs",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Axis 1 → axis 0</td><td>{c['correct_matches']}/{c['elements']}</td><td>{c['wrong_axis_matches']}/{c['elements']}</td><td>{100*(c['wrong_axis_matches']-c['correct_matches'])/c['elements']:.0f}</td></tr>"
         f"<tr><td>Round → add versus add → round</td><td>{c['correct_matches']}/{c['elements']}</td><td>{c['wrong_rounding_matches']}/{c['elements']}</td><td>{100*(c['wrong_rounding_matches']-c['correct_matches'])/c['elements']:.0f}</td></tr></table>"
         f"<p>Correct unsaturated reconstruction max exact error {c['max_exact_reconstruction_error']}; not lossless or model-quality evidence. Negative axis -1 denotes axis 1.</p>"),
        ("LATENCY","Original Fraction reciprocal/multiply and divide paths produce identical integers",
         charts+"<p>Reciprocal construction included; exact small ratios, not native floating reciprocal equivalence. Two warmups, seven alternating repeats, three operations/sample; median repeat means. ONNX runtime, float division precision, layouts and GPU excluded.</p>"),
        ("MY TAKE","Pin dtype, zero-point placement and metadata axis before evaluating a quantized kernel",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Integer Q/DQ contract checks, half ties, saturation and per-axis metadata audits.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Native bit-exactness, floating reciprocal rewrites, FP6/FP8/INT4 coverage, model quality or GPU throughput.</p></div></div>"
         "<div class='dark'>Apache-2.0 inspected ONNX source; original rational/BigInt models and CPU timing. August 29 float6 reference change separately dated, not evaluated here. Native latency, model quality and peak HBM null. Browser inspection not performed.</div>"),
    ]
    return page("Integer QuantizeLinear contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Integer quantization ticket sorter: round, offset and saturate in declared order</title>
    <desc>Figure-tier original 60-solid magazine sorter maps exact-ratio tickets through nearest-even, zero-point and saturation stages.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 121 · INTEGER QUANTIZATION CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">Rounding and odd zero-point translation do not commute</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Quotient → nearest even → integer zero point → saturation</text>
    <text x="88" y="758" font-size="25">A shape-valid square matrix can still use the wrong metadata axis</text>
    <text x="88" y="806" font-size="25">Exact-ratio proof is not native floating division or reciprocal parity</text>
    <text x="88" y="880" font-size="18">Original Fraction / BigInt model · int8 / uint8 only · no native runtime / GPU / quality reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
