import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="onnx-finite-fp4-fp6-cast-rne-nan-policy"
JS=r"""
const specs={FLOAT4E2M1:[2,1,1],FLOAT6E2M3:[2,3,1],FLOAT6E3M2:[3,2,3]};
function abs(v){return v<0n?-v:v;}
function gcd(a,b){a=abs(a);b=abs(b);while(b){const t=a%b;a=b;b=t;}return a;}
function ratio(n,d){const g=gcd(n,d);return {numerator:(n/g).toString(),denominator:(d/g).toString()};}
function text(ticks,negative=false){const whole=Math.floor(ticks/16),r=ticks%16;return (negative?"-":"")+whole+(r?"."+String(r*625).padStart(4,"0").replace(/0+$/,""):"");}
function decode(raw,fmt){
 if(!Object.hasOwn(specs,fmt))throw Error("Supported finite format required");const [e,m,bias]=specs[fmt],sign=2**(e+m);
 if(!Number.isInteger(raw)||raw<0||raw>=2*sign)throw Error("Bounded exact raw code required");
 const negative=Boolean(raw&sign),code=raw&(sign-1),exp=code>>m,mant=code&(2**m-1);
 const ticks=(exp?2**m+mant:mant)*2**((exp?exp:1)-bias-m)*16;
 return {raw_code:raw,value:text(ticks,negative),magnitude_ticks:ticks,negative,negative_zero:negative&&ticks===0};
}
function book(fmt){const [e,m,bias]=specs[fmt],sign=2**(e+m),ticks=Array.from({length:sign},(_,i)=>decode(i,fmt).magnitude_ticks);
 return {format:fmt,sign_bit:sign,total_bits:1+e+m,exponent_bits:e,mantissa_bits:m,bias,raw_patterns:2*sign,nan_encodings:0,infinity_encodings:0,
 positive_ticks:ticks,minimum_subnormal:text(ticks[1]),maximum:text(ticks.at(-1))};}
function parse(token){
 if(typeof token!=="string"||token.length<1||token.length>32)throw Error("Bounded decimal/special token string required");
 const lower=token.toLowerCase();if(lower==="nan")return {kind:"nan",n:0n,d:1n,negative:false};
 if(["inf","+inf","-inf"].includes(lower))return {kind:"inf",n:0n,d:1n,negative:lower.startsWith("-")};
 if(!/^[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d{1,2})?$/.test(token))throw Error("Original decimal or explicit nan/inf token required");
 const negative=token.startsWith("-"),unsigned=token.replace(/^[+-]/,"").toLowerCase(),parts=unsigned.split("e"),exponent=parts.length===2?Number(parts[1]):0;
 const digits=parts[0].split("."),scale=(digits[1]||"").length-exponent;let n=BigInt(digits[0]+(digits[1]||"")),d=1n;
 if(scale>=0)d=10n**BigInt(scale);else n*=10n**BigInt(-scale);
 if(n>d*1000000n)throw Error("Original UI finite magnitude bound exceeded");const g=gcd(n,d);n/=g;d/=g;if(negative)n=-n;
 return {kind:"finite",n,d,negative};
}
function convert(token,fmt){
 const q=parse(token),table=book(fmt);
 if(q.kind==="nan")return {input:token,format:fmt,status:"blocked_nan_policy",raw_code:null,value:null,negative_zero:null,absolute_error_ratio:null,
 policy_error:"NaN cast is unspecified;original policy requires explicit handling"};
 const ticks=table.positive_ticks;let code=ticks.length-1;
 if(q.kind==="finite"){
  const target=abs(q.n)*16n;let lo=0,hi=ticks.length;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(BigInt(ticks[mid])*q.d<target)lo=mid+1;else hi=mid;}
  const candidates=[Math.max(0,lo-1),Math.min(lo,ticks.length-1)];
  code=candidates[0];for(const c of candidates){const a=abs(BigInt(ticks[c])*q.d-target),b=abs(BigInt(ticks[code])*q.d-target);
   if(a<b||(a===b&&((c&1)<(code&1)||((c&1)===(code&1)&&c<code))))code=c;}
 }
 const raw=code|(q.negative?table.sign_bit:0),decoded=decode(raw,fmt),signedTick=BigInt(decoded.magnitude_ticks)*(q.negative?-1n:1n);
 const error=q.kind==="inf"?null:abs(q.n*16n-signedTick*q.d);
 let status="documented_infinity_saturation";if(q.kind==="finite"){
  if(abs(q.n)*16n>BigInt(ticks.at(-1))*q.d)status="finite_saturation";
  else if(decoded.magnitude_ticks===0&&q.n!==0n)status="underflow_to_signed_zero";
  else status=error===0n?"exact":"rne_rounded";
 }
 return {input:token,format:fmt,status,raw_code:raw,value:decoded.value,negative_zero:decoded.negative_zero,
 absolute_error_ratio:error===null?null:ratio(error,16n*q.d),policy_error:null};
}
function evaluate(p){
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(["formats","tokens"]))throw Error("Exact formats/tokens input required");
 if(!Array.isArray(p.formats)||p.formats.length<1||p.formats.length>3||new Set(p.formats).size!==p.formats.length||p.formats.some(v=>!Object.hasOwn(specs,v)))throw Error("Unique supported finite formats required");
 if(!Array.isArray(p.tokens)||p.tokens.length<1||p.tokens.length>32)throw Error("One to thirty-two input tokens required");p.tokens.forEach(parse);
 const rows=p.formats.flatMap(fmt=>p.tokens.map(t=>convert(t,fmt))),summary=p.formats.map(fmt=>{
  const entries=rows.filter(v=>v.format===fmt&&v.absolute_error_ratio!==null);let n=0n,d=1n;
  entries.forEach(v=>{const a=BigInt(v.absolute_error_ratio.numerator),b=BigInt(v.absolute_error_ratio.denominator);n=n*b+a*d;d*=b;const g=gcd(n,d);n/=g;d/=g;});
  return {format:fmt,finite_error_observations:entries.length,mean_absolute_error_ratio:entries.length?ratio(n,d*BigInt(entries.length)):null};
 });
 return {rows,summary,blocked_nan_rows:rows.filter(v=>v.status==="blocked_nan_policy").length,
 scope:"original exact rational inputs and saturating RNE policy;no float32 pre-rounding,scales,calibration,ONNX/reference/backend/GPU casts"};
}
"""


def workbench(r):
    return page("Finite FP4/FP6 RNE and NaN policy workbench",
        f"""<h1>NaN cast is unspecified—not a fabricated zero or maximum code</h1>
        <p>Round148 · Original exact decimal/rational model with reject-NaN policy and documented saturating finite formats.No native ONNX casts.</p>
        <form id="form"><label>Original formats/token JSON<textarea id="input" rows="8" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <button>Evaluate exact RNE,signed zero and explicit NaN gate</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Strings preserve decimal intent and signed zero;JS BigInt avoids rational pre-rounding.NaN rows are explicitly blocked.
        Native NaN behavior remains unknown;this is not a backend guarantee of exceptions.No float32 pre-rounding,scales,calibration,model-quality or GPU evaluation.</p>""",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const result=evaluate(JSON.parse(document.getElementById("input").value));
 out.textContent=JSON.stringify(result,null,2);if(result.blocked_nan_rows)error.textContent=result.blocked_nan_rows+"NaN rows blocked by original policy;native result unspecified";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    rows=r["case"]["outputs"]["rows"]
    matrix="".join("<tr>"+"".join(f"<td style='font-size:14px;padding:7px'>{v}</td>" for v in (
        row["format"].replace("FLOAT","F"),row["input"],"✕ NaN policy blocked" if row["raw_code"] is None else row["value"],
        "Explicit handling" if row["raw_code"] is None else "Confirm native path"))+"</tr>" for row in rows)
    selected=[row for row in rows if row["input"]=="0.75"]
    errors=[int(row["absolute_error_ratio"]["numerator"])/int(row["absolute_error_ratio"]["denominator"]) for row in selected]
    error_cards="".join(f"<div class='metric'><span class='pill'>score · exact original error,not confidence</span><h3>{row['format']}:0.75→{row['value']}</h3>"
                        f"<p>Absolute error{error:g}</p>{bar(error,max(errors))}</div>" for row,error in zip(selected,errors))
    mx=max(max(v["bisect_median_ms"],v["scan_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['positive_tick_queries']}positive tick queries</b><p>Bisected candidates {v['bisect_median_ms']:.6f}ms</p>{bar(v['bisect_median_ms'],mx)}"
                   f"<p>32candidate scan {v['scan_median_ms']:.6f}ms</p>{bar(v['scan_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · FINITE FLOAT POLICY","All16FP4and64FP6bit patterns are finite;NaN casting has no prescribed encoding",
         "<div class='grid'><div><h3>Scenarios</h3><p>RNE midpoint<br>Signed zero<br>Saturation versus NaN</p></div><div><h3>Input</h3><p>Exact original decimals<br>Three finite formats<br>No scales/calibration</p></div>"
         "<div><h3>Decisions</h3><p>Preserve sign of zero<br>Gate NaN explicitly<br>Probe native Cast behavior</p><span class='pill'>Native NaN result:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Original CPU Python{r['latency']['python']} / {r['latency']['machine']}.Nov13BJTdocumentation clarification,not new native conversion.No ONNX/GPU executed.</div>"),
        ("SCENARIO: MIDPOINT0.75","RNE maps0.75to1inFP4;bothFP6formats represent it exactly",
         "<blockquote>FP4neighbors0.5(code1)and1(code2):equal distance,even code2wins.</blockquote>"+error_cards),
        ("WHAT I TESTED","Twelve primary format/input cases keep three NaN rows outside error denominators",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Two finite-error observations per format;Inf/NaN are excluded from finite-error metrics,not zero.</div><div class='callout'>✕ means original policy block,not measured backend exception.</div></div>"),
        ("UNDER THE HOOD","Exact dyadic codebooks use RNE even-code ties and saturating finite extrema",
         "<pre>FLOAT4E2M1:sign1,exp2,mant1,bias1\nFLOAT6E2M3:sign1,exp2,mant3,bias1\nFLOAT6E3M2:sign1,exp3,mant2,bias3\nSubnormal exponent:1−bias\nNormal:2^(exp−bias)*(1+mant/2^m)</pre>"
         "<p>↓ Original exact rational verifier</p><pre>Generate all finite magnitude codes\nFind adjacent nearest candidates\nEqual distance → even low code bit\nPreserve sign on zero/underflow\nSaturate finite overflow and infinities\nNaN → explicit policy block;no fabricated code</pre>"),
        ("ROBUSTNESS CHECK","Negative zero preserves raw sign codes8/32;finite25does not imply a NaN mapping",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Zero sign</td><td>+0 code0</td><td>−0 code8/32</td><td>Sign preserved</td></tr>"
         "<tr><td>FP4midpoint</td><td>+0.25→+0</td><td>−0.25→−0</td><td>Signed underflow</td></tr>"
         "<tr><td>Finite25</td><td>FP4→6</td><td>FP6→7.5/24</td><td>Format-dependent range</td></tr>"
         "<tr><td>NaN</td><td>No prescribed code</td><td>Original policy rejects</td><td>Native remains unknown</td></tr></table>"),
        ("LATENCY","Original nearest-code lookup is not native Cast or low-bit GPU performance",
         timing+"<p>Prebuilt positive integer ticks,FP6E3M2codebook;decimal parsing,sign/NaN/Inf handling,table construction and native casts excluded.Two warmups,seven alternating repeats,three operations/sample;median repeat means.</p>"),
        ("MY TAKE","FP6E3M2extends finite range to28;explicit exceptional policy remains mandatory",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Bitfield/range reasoning,exact synthetic RNE,signed-zero handling,underflow/saturation monitoring and explicit NaN routing.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Mapping NaN to0or maximum by assumption,model-accuracy claims,block scales,NVFP4/MXFP kernels or native backend guarantees.</p></div></div>"
         "<div class='dark'>Apache-2.0source rights preserved,no implementation mirror.Original exact inputs omit float32 pre-rounding;native NaN,GPU Cast,runtime peak and model quality unknown.Visual browser inspection absent.</div>"),
    ]
    return page("Finite FP4/FP6 Cast policy — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Finite-format selector cabinet with an explicit exceptional-input gate</title>
<desc>Original sixty-solid cabinet metaphor;finite-code trays have zero-sign detents and separate saturation/NaN policy decisions.Not GPU hardware or native Cast behavior.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG148 · FINITE CODES DO NOT DEFINE A NaN CAST RESULT</text>
<text x="88" y="158" font-size="34" font-weight="700">RNE,signed zero,saturation and NaN policy are separate contracts</text>{sorter_scene()}
<text x="88" y="710" font-size="24">FP4E2M1maximum6;FP6E2M3maximum7.5;FP6E3M2maximum28</text>
<text x="88" y="760" font-size="24">Midpoint0.75→1inFP4;negative zero uses raw sign codes8or32</text>
<text x="88" y="810" font-size="24">NaN result is unspecified;unknown is not a fabricated zero,error coordinate or maximum</text>
<text x="88" y="880" font-size="18">Original exact rational saturating policy · no float32 pre-rounding,ONNX/reference/GPU Cast,scales or quality evaluation</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
