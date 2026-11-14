import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="onnx-string-tensor-utf8-opaque-bytes-nul-contract"
JS=r"""
function bounded(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error("Bounded exact integer required");return v;}
function dimensions(dims){if(!Array.isArray(dims)||dims.length>2)throw Error("Original rank0..2 required");return dims.reduce((a,v)=>a*bounded(v,0,16),1);}
function bytes(record){
 if(!record||typeof record!=="object"||Array.isArray(record))throw Error("Typed original record required");
 const keys=Object.keys(record).sort().join(",");
 if(record.kind==="bytes"&&keys==="hex,kind"){
  if(typeof record.hex!=="string"||record.hex.length>1024||! /^(?:[0-9a-fA-F]{2})*$/.test(record.hex))throw Error("Bounded full hex-byte pairs required");
  return Uint8Array.from(record.hex.match(/../g)||[],s=>parseInt(s,16));
 }
 if(record.kind==="text"&&keys==="kind,text"){
  if(typeof record.text!=="string"||[...record.text].length>512)throw Error("Bounded Unicode scalar text required");
  for(const c of record.text){const q=c.codePointAt(0);if(q>=0xd800&&q<=0xdfff)throw Error("Unpaired surrogate cannot be UTF-8 encoded");}
  const v=new TextEncoder().encode(record.text);if(v.length>512)throw Error("Original UTF-8 byte bound exceeded");return v;
 }
 throw Error("Exact typed original fields required");
}
function hex(v){return Array.from(v,b=>b.toString(16).padStart(2,"0")).join("");}
function varint(v){bounded(v,0,4096);const a=[];while(v>=128){a.push((v&127)|128);v=Math.floor(v/128);}a.push(v);return a;}
function encode(dims,values){
 if(!Array.isArray(values)||values.length!==dimensions(dims)||values.some(v=>!(v instanceof Uint8Array)))throw Error("Byte values/shape product mismatch");
 if(values.reduce((a,v)=>a+v.length,0)>4096)throw Error("Original total payload bound exceeded");
 const out=[];for(const d of dims)out.push(8,...varint(d));out.push(16,8);
 for(const v of values)out.push(50,...varint(v.length),...v);return Uint8Array.from(out);
}
function decode(wire){
 if(!(wire instanceof Uint8Array)||wire.length>8192)throw Error("Bounded wire bytes required");
 let pos=0,dtype=null;const dims=[],values=[];
 function read(){let result=0;for(const shift of [0,7]){if(pos>=wire.length)throw Error("Truncated original varint");
  const b=wire[pos++];result|=(b&127)<<shift;if(b<128)return bounded(result,0,4096);}throw Error("Varint subset exceeded");}
 while(pos<wire.length){const tag=read();if(tag===8)dims.push(read());else if(tag===16){if(dtype!==null)throw Error("Duplicate datatype outside subset");dtype=read();}
  else if(tag===50){const n=read();if(pos+n>wire.length)throw Error("Truncated original bytes field");values.push(wire.slice(pos,pos+n));pos+=n;}
  else throw Error("Unsupported original field/wire type");}
 if(dtype!==8||values.length!==dimensions(dims))throw Error("Original STRING type/shape mismatch");
 if(values.reduce((a,v)=>a+v.length,0)>4096)throw Error("Original total payload bound exceeded");return {dims,values};
}
function inspect(v){try{return {utf8_text:new TextDecoder("utf-8",{fatal:true,ignoreBOM:true}).decode(v),utf8_status:"valid_utf8_without_normalization"};}
 catch(e){if(!(e instanceof TypeError))throw e;return {utf8_text:null,utf8_status:"invalid_utf8_bytes_preserved"};}}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="dims,raw,values"||typeof p.raw!=="boolean")throw Error("Exact dims/values/raw required");
 if(p.raw)throw Error("STRING uses repeated string_data;raw_data forbidden");const count=dimensions(p.dims);
 if(!Array.isArray(p.values)||p.values.length!==count)throw Error("Flat typed records/shape product mismatch");
 const values=p.values.map(bytes),wire=encode(p.dims,values),restored=decode(wire),naive=values.map(v=>{let n=v.length;while(n&&v[n-1]===0)n--;return v.slice(0,n);});
 const rows=values.map((v,i)=>({index:i,kind:p.values[i].kind,hex:hex(v),byte_length:v.length,restored_hex:hex(restored.values[i]),
 naive_trim_hex:hex(naive[i]),lost_bytes:v.length-naive[i].length,...inspect(v)}));
 const payload=values.reduce((a,v)=>a+v.length,0);
 return {dims:restored.dims,element_count:count,rows,preserved_elements:rows.filter(v=>v.hex===v.restored_hex).length,
 naive_corrupted_elements:rows.filter(v=>v.hex!==v.naive_trim_hex).length,payload_bytes:payload,naive_payload_bytes:naive.reduce((a,v)=>a+v.length,0),
 wire_hex:hex(wire),wire_bytes:wire.length,framing_bytes:wire.length-payload,naive_wire_bytes:encode(p.dims,naive).length,
 original_distinct_values:new Set(values.map(hex)).size,naive_distinct_values:new Set(naive.map(hex)).size};
}
"""


def workbench(r):
    initial=html.escape(json.dumps(r["case"]["inputs"],ensure_ascii=False))
    return page("STRING tensor UTF-8 and byte-preservation workbench",
                f"<h1>Opaque bytes and trailing NULs are values,not removable padding</h1><p>Round149 · Original flat typed records and bounded wire subset;no native ONNX/NumPy/protobuf execution.</p>"
                f'<form id="form"><label>Dims/raw and typed text or hex bytes<textarea id="input" rows="9" required>{initial}</textarea></label>'
                '<button>Compare exact preservation with naive trailing-NUL trim</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
                "<p>Scalar dims[] needs one element;zero-sized dimensions need none.Empty strings remain fields.Invalid UTF-8 bytes remain intact,with a visible diagnostic;raw=true is rejected."
                "The trim baseline is an original mistake model,not native historical NumPy reproduction.</p>",
                JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const v=evaluate(JSON.parse(document.getElementById("input").value));
 out.textContent=JSON.stringify(v,null,2);const invalid=v.rows.filter(r=>r.utf8_text===null).length;
 if(invalid)error.textContent=invalid+"opaque byte values are not valid UTF-8;bytes preserved,not silently replaced";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"];rob=r["robustness"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        row["index"],row["kind"]+":"+row["hex"],row["restored_hex"] or "(empty)",
        "✕ Trim loses"+str(row["lost_bytes"])+"B" if row["lost_bytes"] else "Preserve empty field"))+"</tr>" for row in c["rows"])
    selected=c["rows"][:3]
    decisions="".join(f"<div class='metric'><span class='pill'>score · original bytes,not confidence</span>"
                     f"<h3>Value{row['index']}:preserve{row['byte_length']}B;naive trim loses{row['lost_bytes']}B</h3>"
                     f"<p>Confirm byte contract;invalid UTF-8 is not invalid byte storage.</p>{bar(row['lost_bytes'],max(v['byte_length'] for v in selected))}</div>" for row in selected)
    mx=max(max(v["join_median_ms"],v["concat_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['fragments']}prebuilt fragments/{v['assembled_bytes']}B</b><p>Join {v['join_median_ms']:.6f}ms</p>{bar(v['join_median_ms'],mx)}"
                   f"<p>Repeated concat {v['concat_median_ms']:.6f}ms</p>{bar(v['concat_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · BYTE CONTRACT",f"Four original values preserve{c['payload_bytes']}B;naive NUL trimming corrupts{c['naive_corrupted_elements']}/4",
         "<div class='grid'><div><h3>Scenarios</h3><p>Trailing NUL<br>Opaque invalid UTF-8<br>Scalar versus zero tensor</p></div>"
         "<div><h3>Structured input</h3><p>dims[2,2]<br>Typed text/hex records<br>raw=false</p></div>"
         f"<div><h3>Decisions</h3><p>UTF-8 encode text only<br>Retain every byte<br>Use string_data</p><span class='pill'>Original subset:{c['wire_bytes']}B;native unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Nov13BJT helper correction,not new format.Original Python{r['latency']['python']}/{r['latency']['machine']}CPU fragment assembly separately measured.</div>"),
        ("SCENARIO: TERMINAL NUL","Three of four values lose terminal zeros under the original naive trim baseline",decisions),
        ("WHAT I TESTED",f"Four rows retain{c['payload_bytes']}payload bytes plus{c['framing_bytes']}original framing bytes",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Three trim misses;invalid UTF-8 inspection is visible without altering stored bytes.</div>"
         "<div class='callout'>STRING raw_data blocked;native legacy outputs remain unknown.</div></div>"),
        ("UNDER THE HOOD","Field6length delimiters preserve NULs and empty strings;STRING raw_data is forbidden",
         "<pre>"+html.escape(json.dumps(r["case"]["inputs"],ensure_ascii=False,indent=2))+"</pre><p>↓ Original bounded wire verifier</p>"
         "<pre>dims field1,varint;data_type field2=8\nstring_data field6:length + exact bytes\nstr → UTF-8;bytes → unchanged\n"+c["wire_hex"]+
         "\nDecode original subset;check type and shape product</pre><p>Not a general TensorProto parser or native serialization size.</p>"),
        ("ROBUSTNESS CHECK","Three distinct terminal-NUL values collapse to one;scalar-empty and zero-sized both cost4B",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>73/7300/730000</td><td>{rob['collision']['original_distinct_values']}distinct</td><td>{rob['collision']['naive_distinct_values']}after trim</td><td>2lost identities</td></tr>"
         f"<tr><td>Empty field</td><td>Scalar:{rob['scalar_empty']['element_count']}element</td><td>Zero dim:{rob['zero_dimension']['element_count']}</td><td>Same4B,different bytes</td></tr>"
         "<tr><td>Composed/decomposed text</td><td>é:2UTF-8bytes</td><td>e+accent:3bytes</td><td>No normalization</td></tr></table>"),
        ("LATENCY","Original prebuilt-fragment assembly excludes native serialization,UTF-8 conversion and network",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Input fragments are prepared before timing.</p>"),
        ("MY TAKE","Exact byte identity is a prerequisite for reproducible artifacts,not a text-decoding success test",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Explicit UTF-8 text versus opaque bytes,NUL/empty-value retention,shape-count gates and artifact integrity checks.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Assuming all STRING bytes decode as UTF-8,trimming zeros or declaring native backend/protobuf/network performance.</p></div></div>"
         "<div class='dark'>Apache-2.0source rights retained,no implementation mirror.Original subset omits names/metadata/unknown fields.Native helper/protobuf/legacy NumPy outputs unknown;no browser visual inspection.</div>"),
    ]
    return page("Byte-preserving STRING tensors — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    c=r["case"]["outputs"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Byte-preserving envelope cabinet with text and opaque-byte lanes</title><desc>Original sixty-solid cabinet metaphor;length delimiters retain terminal NULs and empty fields.Not physical hardware or native serializer behavior.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG149 · STRING VALUES ARE NOT NULL-TERMINATED C STRINGS</text>
<text x="88" y="158" font-size="34" font-weight="700">Encode text as UTF-8;retain opaque bytes and every terminal zero</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Four original values:{c['payload_bytes']}Bpayload + {c['framing_bytes']}Bsubset framing = {c['wire_bytes']}B</text>
<text x="88" y="760" font-size="24">Naive trim corrupts3/4values;three terminal-NUL identities collapse to one</text>
<text x="88" y="810" font-size="24">Scalar empty string and zero-sized tensor differ in count,despite equal4Bsubset size</text>
<text x="88" y="880" font-size="18">Original bounded wire/typed-record model · no native ONNX,NumPy,protobuf runtime,network or GPU certification</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
