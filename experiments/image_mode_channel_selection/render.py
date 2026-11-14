import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="onnx-image-decoder-source-mode-rgb-bgr-contract"
JS=r"""
const channels={"1":1,L:1,LA:2,P:1,RGB:3,RGBA:4};
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error("Bounded exact integer required");return v;}
function reverse(values,c=3){const out=[];for(let i=0;i<values.length;i+=c)out.push(...values.slice(i,i+c).reverse());return out;}
function score(v){let result=0;for(let i=0;i<v.length;i+=3)result+=2*v[i]+v[i+1]-v[i+2];return result;}
function rgb(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="height,mode,palette,pixel_format,pixels,width")throw Error("Exact original fixture required");
 if(typeof p.mode!=="string"||!Object.hasOwn(channels,p.mode)||!["RGB","BGR"].includes(p.pixel_format))throw Error("Supported original source mode/target required");
 const n=integer(p.height,1,16)*integer(p.width,1,16),c=channels[p.mode];
 if(!Array.isArray(p.pixels)||p.pixels.length!==n*c)throw Error("Original flat pixel count mismatch");
 for(const v of p.pixels){if(p.mode==="1"){if(typeof v!=="boolean")throw Error("Bilevel boolean fixture required");}else integer(v,0,255);}
 if(p.mode==="P"){
  if(!Array.isArray(p.palette)||p.palette.length<1||p.palette.length>256)throw Error("Original RGB palette required");
  for(const colour of p.palette){if(!Array.isArray(colour)||colour.length!==3)throw Error("Three-channel palette entries required");for(const v of colour)integer(v,0,255);}
  if(p.pixels.some(v=>v>=p.palette.length))throw Error("Original palette index outside bounds");
 }else if(p.palette!==null)throw Error("Palette applies only to original P fixtures");
 const result=[];for(let i=0;i<n;i++){const v=p.pixels.slice(i*c,(i+1)*c);
  if(p.mode==="1")result.push(...Array(3).fill(v[0]?255:0));
  else if(["L","LA"].includes(p.mode))result.push(v[0],v[0],v[0]);
  else if(p.mode==="P")result.push(...p.palette[v[0]]);else result.push(...v.slice(0,3));}
 return result;
}
function evaluate(p){
 const normalized=rgb(p),h=p.height,w=p.width,c=channels[p.mode],out=p.pixel_format==="RGB"?normalized:reverse(normalized);
 const naive_shape=c===1?[h,w]:[h,w,c],naive_dtype=p.mode==="1"?"bool":"uint8";let naive=[...p.pixels],naive_error=null;
 if(p.pixel_format==="BGR"){if(c===1){naive=null;naive_error="original naive raw fixture has no channel axis to reverse";}else naive=reverse(naive,c);}
 return {shape:[h,w,3],dtype:"uint8",pixels:out,logical_output_bytes:out.length,source_mode_value_count:p.pixels.length,canonical_rgb:normalized,
 toy_rgb_score:score(normalized),toy_score_if_output_mislabeled_rgb:score(out),naive_shape,naive_dtype,naive_pixels:naive,naive_error,
 naive_matches_contract:JSON.stringify(naive_shape)===JSON.stringify([h,w,3])&&naive_dtype==="uint8"&&JSON.stringify(naive)===JSON.stringify(out)};
}
"""


def workbench(r):
    return page("Image source-mode normalization workbench",
        f"<h1>Normalize source mode to RGB before any BGR channel reversal</h1><p>Round150 · Original decoded logical fixtures;no compressed-image/Pillow/ONNX execution.</p>"
        f'<form id="form"><label>Source mode,dimensions,pixels,palette and target<textarea id="input" rows="8" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Normalize and compare unsafe raw-channel reversal</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Original alpha is discarded,not composited;palette resolves to RGB;bilevel becomes0/255 uint8.Grayscale target,codecs,colour profiles and native strides are outside scope.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const v=evaluate(JSON.parse(document.getElementById("input").value));
 out.textContent=JSON.stringify(v,null,2);if(!v.naive_matches_contract)error.textContent=v.naive_error||"Original naive path violates RGB/BGR shape,value or dtype contract";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"]
    matrix="".join("<tr>"+"".join(f"<td style='font-size:14px;padding:6px'>{html.escape(str(v))}</td>" for v in (
        row["inputs"]["mode"],row["inputs"]["pixel_format"],"[1,2,3] uint8",
        "Pass raw RGB" if row["outputs"]["naive_matches_contract"] else "✕ Normalize first"))+"</tr>" for row in r["matrix"])
    delta=abs(c["toy_score_if_output_mislabeled_rgb"]-c["toy_rgb_score"])
    mx=max(max(v["strided_median_ms"],v["scalar_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['pixels']}pre-normalized pixels</b><p>Strided copies {v['strided_median_ms']:.6f}ms</p>{bar(v['strided_median_ms'],mx)}"
                   f"<p>Scalar loop {v['scalar_median_ms']:.6f}ms</p>{bar(v['scalar_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · IMAGE CONTRACT","Six source modes need RGB normalization;ten of twelve original naive paths violate the output contract",
         "<div class='grid'><div><h3>Modes</h3><p>Bilevel/grayscale<br>Palette<br>Alpha-bearing</p></div><div><h3>Input</h3><p>Decoded logical pixels<br>RGB/BGR target<br>HWC uint8 required</p></div>"
         "<div><h3>Decision</h3><p>Resolve mode first<br>Discard alpha,not composite<br>Then reorder colour</p><span class='pill'>Codec latency:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Original CPU Python{r['latency']['python']}/{r['latency']['machine']}reorder measured.Nov13BJTreference correction,opset20pre-existing.No compressed images decoded.</div>"),
        ("SCENARIO: RGBA TO BGR",f"Two RGBApixels become six BGRbytes;mislabeling BGR changes the original toy score by{delta}",
         "<blockquote>RGBA[9,70,220,0]→RGB[9,70,220]→BGR[220,70,9];alpha0does not mean composite onto black.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · source-mode gate</span><h3>8logical values→6uint8 output bytes</h3><p>Confirm alpha policy,then reorder;not native peak memory.</p></div>"
         f"<div class='metric'><span class='pill'>score · original toy units,not confidence</span><h3>Canonical RGB score{c['toy_rgb_score']};mislabelled score{c['toy_score_if_output_mislabeled_rgb']}</h3>{bar(delta,max(abs(c['toy_rgb_score']),abs(c['toy_score_if_output_mislabeled_rgb'])))}</div>"),
        ("WHAT I TESTED","Twelve mode/order cases all produce[1,2,3] uint8;only raw RGB fixtures need no normalization",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>10/12original naive misses include absent channel axes,alpha positions,palette indices and bool dtype.</div>"
         "<div class='callout'>No native codec/container or runtime conformance claim.</div></div>"),
        ("UNDER THE HOOD","RGB mode conversion precedes BGR reversal;static rank inference does not guarantee contiguous storage",
         "<pre>pixel_format:RGB(default) or BGR\nSource:1/L/LA/P/RGB/RGBA\nLogical output:H×W×3,uint8\nEncoded source stream is not modeled</pre><p>↓ Original logical verifier</p>"
         "<pre>Bilevel → three0/255 channels\nL/LA → replicate intensity,discard alpha\nP → resolve RGB palette\nRGBA → retain RGB,discard alpha\nBGR → reverse only three colour channels\nProbe native strides separately</pre>"),
        ("ROBUSTNESS CHECK","Changing two alpha values preserves six colour bytes;two BGR swaps recover original RGB",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>RGBAalpha</td><td>0/255</td><td>255/0</td><td>0colour bytes changed</td></tr>"
         "<tr><td>Channel order</td><td>RGB</td><td>Swap twice</td><td>Exact recovery</td></tr>"
         "<tr><td>Bilevel</td><td>bool source</td><td>uint8 RGB</td><td>0/255,not0/1</td></tr>"
         "<tr><td>Palette</td><td>Indices0/1</td><td>RGB colours</td><td>Do not reverse indices</td></tr></table>"),
        ("LATENCY","Original copied RGB-to-BGR reorder excludes source-mode conversion,image decoding and GPU upload",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Normalized input is prepared before timing.</p>"),
        ("MY TAKE","A three-channel uint8 contract is not a native codec,alpha-compositing or contiguous-layout guarantee",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Mode/shape/dtype gates,explicit alpha policy,palette resolution and RGB/BGR consumer alignment.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Raw channel reversal on arbitrary modes,codec support/throughput,colour management,Grayscale conformance or model-quality claims.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.Native image decode,encoded immutability,contiguity,peak and quality unknown.Original copied output is not reference-view stride certification.No visual browser inspection.</div>"),
    ]
    return page("Image source-mode normalization — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Source-mode normalization cabinet before a three-channel order selector</title><desc>Original sixty-solid cabinet metaphor;grayscale,palette and alpha-bearing logical modes become RGB before colour-channel reversal.Not a physical decoder or compressed-codec test.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG150 · NORMALIZE MODE BEFORE PERMUTING COLOUR CHANNELS</text>
<text x="88" y="158" font-size="34" font-weight="700">Six source modes do not inherently share a three-channel uint8 representation</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Twelve original mode/order cases;ten unsafe raw paths violate shape/value/dtype contracts</text>
<text x="88" y="760" font-size="24">Two RGBApixels:8logical values → 6BGRbytes;alpha discarded,not composited</text>
<text x="88" y="810" font-size="24">RGB/BGR order is independent of native array contiguity,compressed codec or GPU upload</text>
<text x="88" y="880" font-size="18">Original post-decode logical fixtures · no Pillow/NumPy/ONNX,image opening,Grayscale,colour profile or quality evaluation</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
