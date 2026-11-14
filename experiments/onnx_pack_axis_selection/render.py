import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="onnx-pack-unpack-axis-radix-word-contract"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error("Bounded exact integer required");return v;}
function params(p){
 let r=p.radix,g=p.group_size;
 if(p.bits!==null){integer(p.bits,1,8);if(r!==null||g!==null)throw Error("bits excludes radix/group_size");r=2**p.bits;g=1;}
 else{integer(r,2,256);integer(g,1,62);}
 const limit=BigInt(r)**BigInt(g);if(limit>(1n<<63n)-1n)throw Error("Radix power must fit positive INT64");
 return [r,g,(limit-1n).toString(2).length,limit];
}
function validate(p){
 const fields=["shape","values","axis","bits","radix","group_size","word_bits"];
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(fields.sort()))throw Error("Exact original rank-two code tensor and metadata required");
 if(!Array.isArray(p.shape)||p.shape.length!==2)throw Error("Rank-two shape required");p.shape.forEach(v=>integer(v,0,16));
 if(p.shape[0]*p.shape[1]>128||!Array.isArray(p.values)||p.values.length!==p.shape[0]*p.shape[1])throw Error("Matched at-most128code row-major input required");
 integer(p.axis,-2,1);if(![8,32].includes(p.word_bits))throw Error("UINT8 or UINT32 storage width required");
 const info=params(p);p.values.forEach(v=>integer(v,0,info[0]-1));return info;
}
function required(n,g,w,b){return Math.ceil(Math.ceil(n/g)*w/b);}
function lanes(shape,values,axis){
 return Array.from({length:shape[1-axis]},(_,other)=>Array.from({length:shape[axis]},(_,i)=>values[axis===1?other*shape[1]+i:i*shape[1]+other]));
}
function assemble(lines,shape,axis){
 const values=Array(shape[0]*shape[1]).fill(0);lines.forEach((line,other)=>line.forEach((v,i)=>{values[axis===1?other*shape[1]+i:i*shape[1]+other]=v;}));return values;
}
function packLane(values,r,g,w,b){
 let buffer=0n,held=0;const words=[],mask=(1n<<BigInt(b))-1n;
 for(let start=0;start<values.length;start+=g){
  let code=0n;for(let i=Math.min(values.length,start+g)-1;i>=start;i--)code=code*BigInt(r)+BigInt(values[i]);
  buffer|=code<<BigInt(held);held+=w;
  while(held>=b){words.push(Number(buffer&mask));buffer>>=BigInt(b);held-=b;}
 }
 if(held)words.push(Number(buffer));return words;
}
function unpackLane(words,count,r,g,w,limit,b){
 let buffer=0n,held=0,cursor=0;const values=[],mask=(1n<<BigInt(w))-1n;
 for(let group=0;group<Math.ceil(count/g);group++){
  while(held<w){buffer|=BigInt(words[cursor++])<<BigInt(held);held+=b;}
  let code=buffer&mask;buffer>>=BigInt(w);held-=w;if(code>=limit)throw Error("Original strict validator rejects unused radix group index");
  const size=Math.min(g,count-values.length);for(let j=0;j<size;j++){values.push(Number(code%BigInt(r)));code/=BigInt(r);}
 }return values;
}
function pack(p){
 const [r,g,w,limit]=validate(p),axis=(p.axis+2)%2,shape=p.shape.slice();shape[axis]=required(shape[axis],g,w,p.word_bits);
 const words=assemble(lanes(p.shape,p.values,axis).map(v=>packLane(v,r,g,w,p.word_bits)),shape,axis),bytes=[];
 words.forEach(word=>{let v=BigInt(word);for(let i=0;i<p.word_bits/8;i++){bytes.push(Number(v&255n));v>>=8n;}});
 return {shape,words,word_bits:p.word_bits,raw_little_endian_bytes:bytes,
 parameters:{radix:r,group_size:g,group_width:w,radix_power_decimal:limit.toString()},selected_storage_bytes:words.length*p.word_bits/8};
}
function unpack(packet,p,count){
 const [r,g,w,limit]=validate(p),axis=(p.axis+2)%2;integer(count,0,16);
 if(!packet||!Array.isArray(packet.shape)||packet.shape.length!==2||!Array.isArray(packet.words)||packet.word_bits!==p.word_bits)throw Error("Packed tensor/storage metadata required");
 packet.shape.forEach(v=>integer(v,0,128));const shape=packet.shape;
 if(shape[1-axis]!==p.shape[1-axis]||shape[axis]!==required(count,g,w,p.word_bits)||packet.words.length!==shape[0]*shape[1])throw Error("Exact packed selected-axis word count required");
 packet.words.forEach(v=>integer(v,0,2**p.word_bits-1));const output=shape.slice();output[axis]=count;
 if(output[0]*output[1]>128)throw Error("Original UI decoded code bound exceeded");
 return {shape:output,values:assemble(lanes(shape,packet.words,axis).map(v=>unpackLane(v,count,r,g,w,limit,p.word_bits)),output,axis)};
}
"""


def workbench(r):
    return page("ONNX Pack/Unpack axis and radix workbench",
        f"""<h1>Each non-axis lane starts a new word;flat packing is a different tensor</h1>
        <p>Round147 · Original strict rank-two UINT8-code model—not ONNX/reference/function/ORT evaluation.</p>
        <form id="form"><label>Original packing JSON<textarea id="input" rows="8" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <label>Logical Unpack count along axis<input id="count" type="number" value="3" min="0" max="16" step="1" required></label>
        <button>Pack words and decode declared count</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>word_bits8/32maps to UINT8/UINT32 storage.Scalar count is original JSON metadata,not an executed INT64tensor.
        Large radix powers remain exact BigInt/decimal strings.High padding bits and unused tail digits are ignored by decode.
        Invalid inputs are unspecified by the new ONNX schema;this workbench intentionally adds strict errors—not a function-lowering guarantee.
        No signed-code quantization,INT4dtype conversion or GPU/backend support certification.</p>""",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Valid metadata/count required");const p=JSON.parse(document.getElementById("input").value),packet=pack(p);
 out.textContent=JSON.stringify({packed:packet,decoded:unpack(packet,p,Number(document.getElementById("count").value))},null,2);}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];t=r["ternary"]
    mx=max(max(v["reservoir_median_ms"],v["bitwise_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['codes']}original3-bit codes</b><p>Reservoir {v['reservoir_median_ms']:.6f}ms</p>{bar(v['reservoir_median_ms'],mx)}"
                   f"<p>Individual bit writes {v['bitwise_median_ms']:.6f}ms</p>{bar(v['bitwise_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · LOW-BIT WORDS","New opset29Pack/Unpack preserve independent axis lanes—not one flattened bitstream",
         "<div class='grid'><div><h3>Scenarios</h3><p>Word crossings<br>Radix groups<br>Logical count and padding</p></div><div><h3>Input</h3><p>UINT8 integer codes<br>UINT8/UINT32 words<br>Original rank-two model</p></div>"
         "<div><h3>Decisions</h3><p>Keep axis/count metadata<br>LSB-first words<br>Probe backend support</p><span class='pill'>Native ORT support:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Original CPU Python{r['latency']['python']} / {r['latency']['machine']}.Feature dated Nov13BJT,reviewed Nov14.No ONNX/ORT/GPU execution.</div>"),
        ("SCENARIO: TWO3-CODE LANES","Two3-code lanes require4bytes at bits3;flattening a different tensor uses3bytes",
         f"<blockquote>Shape[2,3],axis1,codes{c['inputs']['values']} → words{c['packed']['words']};each lane restarts at word bit0.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact selected bytes</span><h3>Independent lane payload:{c['packed']['selected_storage_bytes']}B</h3>{bar(c['packed']['selected_storage_bytes'],c['packed']['selected_storage_bytes'])}<p>Confirm axis/shape,not compression confidence</p></div>"
         f"<div class='metric'><span class='pill'>choice · deterministic shape</span><h3>Axis0has shape{c['axis0_packed']['shape']}and{c['axis0_packed']['selected_storage_bytes']}B</h3><p>Different layout with a valid inverse—not the same packed array</p></div>"),
        ("WHAT I TESTED","Six contract cases distinguish packing layout,radix validity and count metadata",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Flatten all codes</td><td>Same tensor words?</td><td>✕3Bvs4B</td><td>Keep independent lanes</td></tr>"
         "<tr><td>Change axis</td><td>Same packed layout?</td><td>✕New shape/words</td><td>Store axis metadata</td></tr>"
         "<tr><td>Count3→4</td><td>Word count proves length?</td><td>✕Both require2words/lane</td><td>Supply original count</td></tr>"
         "<tr><td>Radix3group5</td><td>Independent2bits needed?</td><td>Five digits in1B</td><td>Distinguish codecs</td></tr>"
         "<tr><td>UINT32bit31</td><td>Signed JS shifts safe?</td><td>✕Use exact integer math</td><td>BigInt original model</td></tr>"
         "<tr><td>Invalid ternary index</td><td>Schema guarantees throw?</td><td>✕Invalid unspecified</td><td>Original strict preflight</td></tr></table>"
         "<div class='grid two'><div class='callout'>Exact radix/word/count contracts;not model confidence or numerical quantization.</div><div class='callout'>Function decompositions need not diagnose invalid inputs.Native support unknown.</div></div>"),
        ("UNDER THE HOOD","A group index needs W=ceil(log2(radix^group_size))bits,then crosses words contiguously",
         "<pre>Original metadata:\nUINT8 codes;storage bits8or32\nbits spelling=(2^bits,group1)\nGrouped index=Σdigit_j*radix^j\nWidth=bit_length(radix^group_size−1)\nPer-lane words=ceil(ceil(N/G)*W/word_bits)</pre>"
         "<p>↓ Original bounded-reservoir verifier</p><pre>Append group index low bits first\nFlush complete storage words\nZero-pad final word/group on Pack\nUnpack requires scalar logical count\nIgnore high/tail padding after count\nReject invalid radix states in this strict model</pre>"),
        ("ROBUSTNESS CHECK","Flipping unused high padding leaves six decoded values unchanged—not integrity protection",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Padding</td><td>{c['packed']['words']}</td><td>{c['padding_flipped_words']}</td><td>Same6values</td></tr>"
         "<tr><td>Axis1</td><td>1</td><td>−1</td><td>Equivalent rank-two axis</td></tr>"
         "<tr><td>Count</td><td>3</td><td>4</td><td>Extra zero code per lane</td></tr>"
         "<tr><td>Ternary states</td><td>0..242</td><td>243..255</td><td>13invalid group states</td></tr></table>"
         "<p>Count controls trimming;word count alone cannot recover original logical length.Unused padding is not a checksum.</p>"),
        ("LATENCY","Original streaming versus individual-bit packing does not measure ONNX or GPU throughput",
         timing+"<p>Single lane,bits3,prebuilt codes;grouping/output construction included.Input/parameter prep and matrix validation excluded.Two warmups,seven alternating repeats,three operations/sample;median repeat means.Timer sizes exceed UI bound.</p>"),
        ("MY TAKE","Five ternary digits fit one byte;operator availability still needs opset/backend validation",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Unsigned code-stream representation,per-axis word boundaries,exact UINT32 arithmetic,logical length and base-radix alternatives.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Numerical quantization,INT4dtype conversion,signed-code semantics,assuming padding validates integrity,or claiming all runtimes support opset29.</p></div></div>"
         f"<div class='dark'>Original ternary payload:{t['packed']['selected_storage_bytes']}B versus independent2-bit:{t['two_bit_alternative']['selected_storage_bytes']}B.Apache-2.0source rights preserved;no mirror or native evaluation.Visual browser inspection absent.</div>"),
    ]
    return page("ONNX Pack/Unpack word contracts — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Code sorting loom with independent word trays</title>
<desc>Original sixty-solid cabinet metaphor:each lane begins a new word tray;radix group tickets and logical-count tags are independent from the storage-word shape.Not ONNX or GPU hardware.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG147 · INDEPENDENT LANES, CONTIGUOUS GROUP BITS</text>
<text x="88" y="158" font-size="35" font-weight="700">Packing axis and logical count survive beyond the word array</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Each lane restarts at word bit0;group indices cross word boundaries LSB-first</text>
<text x="88" y="760" font-size="24">Two3-code lanes at bits3use4B;radix3group5fits five codes in1B</text>
<text x="88" y="810" font-size="24">Word count does not prove original length;high padding is ignored,not integrity</text>
<text x="88" y="880" font-size="18">Original strict UINT8-code model · no numerical quantization,ONNX/reference/function/ORT/GPU execution or browser capture</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
