import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="hicache-logical-kv-dtype-storage-namespace"
JS=r"""
const formats={float8_e4m3fn:[4,3,7,1,true],float8_e5m2:[5,2,15,1,false],float16:[5,10,15,2,false],bfloat16:[8,7,127,2,false]};
function token(v,nullable=false){if(nullable&&v===null)return;if(typeof v!=="string"||!/^[A-Za-z0-9_.:/-]{1,64}$/.test(v))throw Error("Bounded ASCII metadata token required");}
function abs(v){return v<0n?-v:v;}
function ratio(n,d){let a=abs(n),b=d;while(b){const t=a%b;a=b;b=t;}return {numerator:(n/a).toString(),denominator:(d/a).toString()};}
function data(hex){return Uint8Array.from(hex.match(/../g)||[],s=>parseInt(s,16));}
function probe(payload,dtype){
 const [e,m,bias,w,fn]=formats[dtype];if(payload.length<w)return {status:"no_complete_scalar",ratio:null,negative_zero:null};
 let raw=payload[0];if(w===2)raw+=payload[1]*256;const negative=Boolean(raw>>>(e+m)),exp=(raw>>>m)&(2**e-1),mant=raw&(2**m-1);
 if(exp===2**e-1&&(!fn||mant===2**m-1))return {status:fn||mant?"nan":"infinity",ratio:null,negative_zero:null};
 const power=(exp||1)-bias-m;let n=BigInt(exp?2**m+mant:mant),d=1n;if(power>=0)n*=2n**BigInt(power);else d=2n**BigInt(-power);if(negative)n=-n;
 return {status:"finite",ratio:ratio(n,d),negative_zero:negative&&n===0n};
}
function key(p,dtype){return JSON.stringify(["original-v1",p.model_id,p.storage_tag,dtype,p.extra_backend_tag,p.page_id]);}
function read(cache,k,request){
 if(!cache.has(k))return {status:"miss",payload_hex:null,stored_dtype_writer_witness:null,dtype_writer_witness_matches:null,
 payload_writer_witness_matches:null,reader_probe:null,finite_probe_absolute_error:null};
 const stored=cache.get(k),same=stored.logical_dtype===request.logical_dtype,payload=data(stored.payload_hex),expected=data(request.payload_hex);
 const actual=probe(payload,request.logical_dtype),wanted=probe(expected,request.logical_dtype);let error=null;
 if(actual.ratio!==null&&wanted.ratio!==null){const a=actual.ratio,b=wanted.ratio;
  error=ratio(abs(BigInt(a.numerator)*BigInt(b.denominator)-BigInt(b.numerator)*BigInt(a.denominator)),BigInt(a.denominator)*BigInt(b.denominator));}
 return {status:payload.length!==expected.length?"length_mismatch_by_writer_witness":same?"hit":"foreign_hit_by_writer_witness",
 payload_hex:stored.payload_hex,stored_dtype_writer_witness:stored.logical_dtype,dtype_writer_witness_matches:same,
 payload_writer_witness_matches:stored.payload_hex===request.payload_hex,reader_probe:actual,finite_probe_absolute_error:error};
}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="extra_backend_tag,model_id,page_id,storage_tag,writes")throw Error("Exact original fields required");
 token(p.model_id,true);token(p.storage_tag,true);token(p.extra_backend_tag,true);token(p.page_id);
 if(!Array.isArray(p.writes)||p.writes.length<1||p.writes.length>4)throw Error("Original1..4unique dtype writes required");
 const seen=new Set(),writes=[];for(const w of p.writes){
  if(!w||typeof w!=="object"||Array.isArray(w)||Object.keys(w).sort().join(",")!=="logical_dtype,payload_hex,physical_dtype")throw Error("Exact typed write fields required");
  if(typeof w.logical_dtype!=="string"||!Object.hasOwn(formats,w.logical_dtype)||seen.has(w.logical_dtype))throw Error("Resolved unique logical dtype required,no backing fallback");
  if(!["uint8","float16","bfloat16"].includes(w.physical_dtype))throw Error("Original physical label required");
  if(typeof w.payload_hex!=="string"||! /^(?:[0-9a-fA-F]{2}){1,32}$/.test(w.payload_hex))throw Error("Bounded complete hex-byte pairs required");
  if(data(w.payload_hex).length%formats[w.logical_dtype][3])throw Error("Complete scalar words required");
  seen.add(w.logical_dtype);writes.push({...w,payload_hex:w.payload_hex.toLowerCase()});
 }
 const legacy=new Map(),physical=new Map(),typed=new Map();
 for(const w of writes){legacy.set(key(p,null),w);physical.set(key(p,"backing:"+w.physical_dtype),w);typed.set(key(p,"logical:"+w.logical_dtype),w);}
 const rows=writes.map(w=>({logical_dtype:w.logical_dtype,physical_dtype:w.physical_dtype,typed_key:key(p,"logical:"+w.logical_dtype),
 expected_probe_writer_witness:probe(data(w.payload_hex),w.logical_dtype),legacy:read(legacy,key(p,null),w),
 physical_only:read(physical,key(p,"backing:"+w.physical_dtype),w),typed:read(typed,key(p,"logical:"+w.logical_dtype),w)}));
 const stored=cache=>[...cache.values()].reduce((a,w)=>a+w.payload_hex.length/2,0),first=writes[0],cold=read(new Map(),key(p,"logical:"+first.logical_dtype),first);
 return {rows,legacy_objects:legacy.size,physical_only_objects:physical.size,typed_objects:typed.size,
 legacy_foreign_reads:rows.filter(r=>!r.legacy.dtype_writer_witness_matches).length,physical_only_foreign_reads:rows.filter(r=>!r.physical_only.dtype_writer_witness_matches).length,
 typed_foreign_reads:rows.filter(r=>!r.typed.dtype_writer_witness_matches).length,
 selected_stored_payload_bytes:{legacy:stored(legacy),physical_only:stored(physical),typed:stored(typed)},
 migration:{legacy_exists:true,typed_before_repopulate:cold,fallback_used:false,typed_after_repopulate:read(new Map([[key(p,"logical:"+first.logical_dtype),first]]),key(p,"logical:"+first.logical_dtype),first)},
 native_sglang_key_compatibility:null,native_kv_accuracy:null};
}
"""


def workbench(r):
    return page("Logical KV dtype object-cache namespace workbench",
        f"<h1>Two FP8logical formats can share uint8 backing without sharing object keys</h1><p>Round153 · Original canonical JSON cache and exact scalar probes.No SGLang/Mooncake or native KVcast.</p>"
        f'<form id="form"><label>Namespace metadata and typed raw-value writes<textarea id="input" rows="10" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare legacy,physical-only and logical-dtype identities</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Writer witnesses are original oracle metadata,not self-describing legacy payload headers.Resolved logical dtype is required;auto is rejected."
        "Cold typed miss never falls back to an untyped object.Other layout tags and native key schemes remain separate.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const r=evaluate(JSON.parse(document.getElementById("input").value));out.textContent=JSON.stringify(r,null,2);
 if(r.physical_only_foreign_reads)error.textContent=r.physical_only_foreign_reads+"physical-only reads conflict with writer dtype witnesses;logical typed keys isolate them";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"]
    error=c["rows"][0]["physical_only"]["finite_probe_absolute_error"]
    wanted=c["rows"][0]["expected_probe_writer_witness"]["ratio"]
    error_bar=bar(int(error["numerator"])/int(error["denominator"]),abs(int(wanted["numerator"])/int(wanted["denominator"]))) if error is not None and wanted is not None else "<p>No finite scalar error coordinate.</p>"
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        row["logical_dtype"],row["physical_dtype"],row["typed"]["payload_hex"],
        "✕ Physical-only foreign hit" if not row["physical_only"]["dtype_writer_witness_matches"] else "Confirm native layout"))+"</tr>" for row in c["rows"])
    mx=max(max(v["hoisted_median_ms"],v["rebuilt_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['pages']}original page keys</b><p>Hoisted prefix {v['hoisted_median_ms']:.6f}ms</p>{bar(v['hoisted_median_ms'],mx)}"
                   f"<p>Full descriptor {v['rebuilt_median_ms']:.6f}ms</p>{bar(v['rebuilt_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · KV NAMESPACE","Four logical formats need four original objects;physical-only identity merges the two FP8formats",
         "<div class='grid'><div><h3>Scenarios</h3><p>FP8E4M3/E5M2<br>BF16/FP16<br>Cold migration</p></div>"
         "<div><h3>Input</h3><p>Same page ID<br>Raw2Bteaching payloads<br>Logical dtype+format tags</p></div>"
         "<div><h3>Decision</h3><p>Resolve logical dtype<br>Keep layout tags<br>No untyped fallback</p><span class='pill'>Native hit-rate/quality:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.ActualNov14HiCache correction.Original Python{r['latency']['python']}/{r['latency']['machine']}key assembly measured;not source msgspec/Mooncake.</div>"),
        ("SCENARIO: SHARED UINT8","Byte0x3chas value1.5inE4M3FNand1inE5M2;physical dtype alone cannot identify interpretation",
         "<blockquote>Logical E4M3FN/E5M2can both useuint8backing.A physical-only key allows one writer to overwrite the other.</blockquote>"
         "<div class='metric'><span class='pill'>choice · require resolved logical metadata</span><h3>Logical keys retain both FP8objects</h3><p>Same raw bytes can mean different values;equal checksums do not define dtype.</p></div>"
         "<div class='metric'><span class='pill'>score · original scalar error,not model accuracy</span><h3>E4M3writer1.5getsphysical-only value2</h3>"+error_bar+"<p>Absolute error1/2in this original witness fixture only.</p></div>"),
        ("WHAT I TESTED","Legacy,physical-only and logical namespaces retain1,3and4original objects",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         f"<div class='grid two'><div class='callout'>Writer-witness foreign reads:{c['legacy_foreign_reads']}legacy,{c['physical_only_foreign_reads']}physical-only,{c['typed_foreign_reads']}logical.</div>"
         "<div class='callout'>Typed migration is cold miss,not untyped success fallback.</div></div>"),
        ("UNDER THE HOOD","Resolved token-view dtype joins the existing storage-format namespace;other layouts still need tags",
         "<pre>Source facts:\nmodel name + storage_format_tag\n+ dtype_{resolved logical dtype}\nuint8 backing is not logical FP8dtype\nDCP/non-DCP retain independent suffix rules</pre><p>↓ Original canonical JSON model</p>"
         "<pre>[schema,model,storage_tag,\n logical_dtype,extra_tag,page_id]\nWrite/read exact original bytes\nCompare with separate writer witnesses\nCold typed miss → no legacy fallback\nNot native msgspec digest or key compatibility</pre>"),
        ("ROBUSTNESS CHECK","Cold typed lookup has no payload until repopulation;missing model still retains dtype identity",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Migration</td><td>Legacy object exists</td><td>Typed namespace absent</td><td>Miss,payload null</td></tr>"
         "<tr><td>Repopulation</td><td>Typed miss</td><td>Fresh typed write</td><td>Exact hit</td></tr>"
         "<tr><td>Storage/layout tags</td><td>Default</td><td>Envelope/layout-v2</td><td>Different keys</td></tr>"
         "<tr><td>Model absent</td><td>Model prefix</td><td>null model</td><td>Logical dtype retained</td></tr></table>"),
        ("LATENCY","Original JSONnamespace assembly is not storage latency,msgspec hashing or native cache performance",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Page/input setup,payload probes,hashes,native storage and GPUexcluded.</p>"),
        ("MY TAKE","Logical dtype prevents one class of cache aliasing;it does not fully certify layout,tenant or model identity",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Typed persistent KVidentity,format/extra tags,explicit cold migration and byte-versus-interpretation checks.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Inferring FP8format fromuint8,falling back to old untyped pages,claiming native KVquality/hit-rate or complete key compatibility.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.Writer witnesses are not on-wire headers.SGLang/Mooncake/key compatibility,hit-rate,latency and quality unknown.No visual browser inspection.</div>"),
    ]
    return page("Logical KV dtype identity — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Typed persistent KVobject cabinet with logical dtype and layout latches</title><desc>Original sixty-solid cabinet metaphor;physical uint8 does not identify logical FP8 interpretation.Cold typed migration does not silently read untyped objects.Not native object storage or GPU cache hardware.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG153 · PHYSICAL BYTE TYPE DOES NOT DEFINE LOGICAL KV FORMAT</text>
<text x="88" y="158" font-size="34" font-weight="700">Resolve logical dtype,retain format tags and keep cold migration explicit</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Four original formats:legacy1object,physical-only3objects,logical4objects</text>
<text x="88" y="760" font-size="24">FP8byte3CmeansE4M3FN1.5orE5M2 1;uint8backing alone is ambiguous</text>
<text x="88" y="810" font-size="24">Typed miss has null payload;repopulate rather than read old untyped pages</text>
<text x="88" y="880" font-size="18">ActualNov14HiCache correction · original JSONnamespace/scalar probes,no SGLang/Mooncake/native KVquality or latency</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
