import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page, bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "rounded-silu-bit-pattern-lut-capture-cache"
JS = r"""
const DTYPES=["float16","bfloat16"],TABLE_BYTES=131072;
const buffer=new ArrayBuffer(4),view=new DataView(buffer);
function raw32(x){view.setFloat32(0,x,true);return view.getUint32(0,true);}
function from32(x){view.setUint32(0,x,true);return view.getFloat32(0,true);}
function decode(w,d){
 if(!DTYPES.includes(d)||!Number.isInteger(w)||w<0||w>65535)throw Error("uint16 word and16-bit dtype required");
 if(d==="bfloat16")return from32(w*65536);
 const sign=w&32768?-1:1,e=(w>>>10)&31,m=w&1023;
 if(e===31)return m?NaN:sign*Infinity;
 return sign*(e?Math.pow(2,e-15)*(1+m/1024):Math.pow(2,-24)*m);
}
function rshift(m,s){const q=Math.floor(m/Math.pow(2,s)),r=m-q*Math.pow(2,s),half=Math.pow(2,s-1);return q+(r>half||r===half&&q%2===1?1:0);}
function encode(x,d){
 if(!DTYPES.includes(d)||Number.isNaN(x))throw Error("NaN payload parity is outside original prototype");
 const r=raw32(x);
 if(d==="bfloat16")return ((r+32767+((r>>>16)&1))>>>16)&65535;
 const sign=(r>>>16)&32768,e=((r>>>23)&255)-127,m=r&8388607;
 if(e===128||e>15)return sign|31744;
 if(e<-25)return sign;
 if(e<-14)return sign|rshift(m+8388608,-e-1);
 return sign|((e+15)*1024+rshift(m,13));
}
function finite(w,d){const x=decode(w,d);if(!Number.isFinite(x))throw Error("Finite input words only;native NaN/Inf parity unknown");return x;}
function activation(x){const exp=-x>90?Infinity:Math.fround(Math.exp(-x));return Math.fround(x/Math.fround(1+exp));}
function lane(g,u,d){const x=finite(g,d),up=finite(u,d),a=activation(x),w=encode(a,d);
 return {gate_word:g,up_word:u,activation_word:w,rounded_output_word:encode(Math.fround(decode(w,d)*up),d),single_round_output_word:encode(Math.fround(a*up),d)};}
function sparseTable(gates,d){return new Map(gates.map(g=>[g,encode(activation(finite(g,d)),d)]));}
function lookup(g,u,d,t){finite(g,d);const up=finite(u,d);if(!t.has(g))throw Error("Absent sparse fixture word;no default/zero activation");return encode(Math.fround(decode(t.get(g),d)*up),d);}
function route(p){
 const fields="cached_keys,capturing,clamp,cuda,device,dtype,hip,major,operation,shared_optin_bytes,tokens";
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!==fields)throw Error("Exact original dispatch fields required");
 if(![...DTYPES,"float32"].includes(p.dtype)||!["silu","gelu","gelu_tanh"].includes(p.operation))throw Error("Known dtype/activation required");
 for(const k of ["cuda","hip","capturing","clamp"])if(typeof p[k]!=="boolean")throw Error("Explicit booleans required");
 for(const k of ["device","major","shared_optin_bytes","tokens"])if(!Number.isInteger(p[k])||p[k]<0||p[k]>4294967295)throw Error("Bounded nonnegative integers required");
 if(!Array.isArray(p.cached_keys)||p.cached_keys.some(k=>!Array.isArray(k)||k.length!==2||!DTYPES.includes(k[0])||!Number.isInteger(k[1])||k[1]<0||k[1]>4294967295))throw Error("Exact dtype/device cache keys required");
 if(new Set(p.cached_keys.map(k=>JSON.stringify(k))).size!==p.cached_keys.length)throw Error("Duplicate cache keys");
 const key=[p.dtype,p.device],eligible=p.operation==="silu"&&!p.clamp&&DTYPES.includes(p.dtype)&&p.cuda,
 cached=eligible&&p.cached_keys.some(k=>k[0]===key[0]&&k[1]===key[1]),
 supported=!p.hip&&p.major===10&&p.shared_optin_bytes>=TABLE_BYTES;
 let action,reason;
 if(!eligible){action="fallback";reason="activation_clamp_dtype_or_device";}
 else if(cached){action="reuse";reason="exact_dtype_device_key_before_capture_or_property_probe";}
 else if(!supported){action="fallback";reason="uncached_hardware_gate";}
 else if(p.capturing){action="fallback";reason="cold_capture_must_not_allocate";}
 else{action="build";reason="uncaptured_supported_cache_miss";}
 const after=p.cached_keys.map(k=>[...k]);if(action==="build")after.push(key);
 const selected=["build","reuse"].includes(action);
 return {action,reason,table_key:eligible?key:null,cached_keys_after:after,new_logical_table_bytes:action==="build"?TABLE_BYTES:0,
 table_selected:selected,activation_launch:p.tokens>0,shared_bytes_per_table_cta:selected&&p.tokens>0?TABLE_BYTES:0,native_driver_or_kernel_execution_verified:null};
}
"""


def workbench(r):
    p = {"dtype": "float16", "gate_words": [v["gate_word"] for v in r["matrix"] if v["dtype"] == "float16"],
         "up_word": r["case"]["up_word"]}
    return page("Rounded SiLU word/cache workbench",
        "<h1>Preserve16-bit identity and round activation before multiplying</h1>"
        "<p>Round158 · Two independent original models:finite word arithmetic and declared cache/dispatch state."
        "No CUDA/libdevice/graphs or browser visual acceptance.</p>"
        f'<form id="form"><label>Finite arithmetic words<textarea id="numeric" rows="5" required>{html.escape(json.dumps(p))}</textarea></label>'
        f'<label>Independent cache/dispatch ledger<textarea id="input" rows="7" required>{html.escape(json.dumps(r["dispatch_matrix"][0]["inputs"]))}</textarea></label>'
        '<button>Compare rounding and cache phases</button></form><p id="error" role="alert"></p>'
        '<pre id="result" aria-live="polite"></pre>'
        "<p>NaN/Inf input payloads are deliberately rejected by this finite prototype,not by the native full-domain LUT."
        "Sparse fixture construction is not a native65536-entry allocation or shared-memory copy."
        "Empty uncaptured warmup may still build a table;warm exact-key cache is reusable during capture.</p>",
        JS + r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");
 out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Both inputs required");
 const p=JSON.parse(document.getElementById("numeric").value),cfg=JSON.parse(document.getElementById("input").value);
 if(!p||Object.keys(p).sort().join(",")!=="dtype,gate_words,up_word"||!DTYPES.includes(p.dtype)||!Array.isArray(p.gate_words)||!p.gate_words.length||p.gate_words.length>128)throw Error("Explicit16-bit dtype,1..128gate words and finite up word required");
 const t=sparseTable(p.gate_words,p.dtype),rows=p.gate_words.map(g=>({...lane(g,p.up_word,p.dtype),lookup_output_word:lookup(g,p.up_word,p.dtype,t)}));
 out.textContent=JSON.stringify({arithmetic:{rows,single_round_misses:rows.filter(r=>r.rounded_output_word!==r.single_round_output_word).length,lookup_misses:rows.filter(r=>r.rounded_output_word!==r.lookup_output_word).length},
 independent_dispatch:route(cfg),native_libdevice_byte_parity:null},null,2);
 }catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    s, c = r["summary"], r["case"]
    matrix = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["dtype"] + "/" + f'{x["gate_word"]:04x}', "round before multiply",
        f'{x["lookup_output_word"]:04x}', "✕ single-round" if x["rounded_output_word"] != x["single_round_output_word"] else "Same fixture bytes"))
        + "</tr>" for x in r["matrix"])
    mx = max(max(v["lookup_median_ms"], v["direct_median_ms"]) for v in r["latency"]["rows"])
    timing = "".join(f"<div class='metric'><b>{v['lanes']}original finite lanes</b><p>Sparse lookup{v['lookup_median_ms']:.6f}ms</p>{bar(v['lookup_median_ms'], mx)}"
                    f"<p>Direct rounded arithmetic{v['direct_median_ms']:.6f}ms</p>{bar(v['direct_median_ms'], mx, True)}</div>" for v in r["latency"]["rows"])
    dispatch = {x["name"]: x["outputs"] for x in r["dispatch_matrix"]}
    w = r["overflow_witness"]
    sections = [
        ("HANDS-ON · ROUNDED ACTIVATION", f"Single-final-round misses{s['single_round_misses']}/{s['lanes']}original lanes;bit-index lookup misses{s['lookup_misses']}",
         "<div class='grid'><div><h3>Scenarios</h3><p>Intermediate rounding<br>Signed zero keys<br>Cold/warm capture</p><span class='pill'>Threshold:exact fixture words</span></div>"
         "<div><h3>Input</h3><p>16-bit gate/up words<br>dtype/device cache keys<br>SM10/optin ledger</p></div>"
         "<div><h3>Decision</h3><p>Retain rounding stage<br>Do not normalize keys<br>Warm before capture</p><span class='pill'>Native parity/latency unknown</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14LUTaddition.Original Python{r['latency']['python']}/{r['latency']['machine']}finite model;libm-exp is not CUDA__nv_expf.</div>"),
        ("SCENARIO: NORMAL HALF WORD", f"Original gate{c['gate_word']:04x}produces{c['rounded_output_word']:04x},not single-round{c['single_round_output_word']:04x}",
         "<blockquote>SiLUfirst rounds toT;that rounded value is multiplied inFP32and rounded toT again.LUTs must preserve that intermediate cast,not merely implement a mathematically similar sigmoid.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · confirm native byte parity</span><h3>Gate{c['gate_word']:04x},up{c['up_word']:04x},activation{c['activation_word']:04x}</h3><p>Original CPUwords only;no fabricated model confidence.</p></div>"
         f"<div class='metric'><span class='pill'>score · exact finite fixture words</span><h3>LUTmatches{s['lanes']-s['lookup_misses']}/{s['lanes']}</h3>{bar(s['lanes']-s['lookup_misses'],s['lanes'])}"
         f"<h3>Single-round matches{s['lanes']-s['single_round_misses']}/{s['lanes']}</h3>{bar(s['lanes']-s['single_round_misses'],s['lanes'])}</div>"),
        ("WHAT I TESTED", f"Two dtypes×{s['original_sparse_entries_per_dtype']}finite gate words expose rounding,zero and overflow distinctions",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>" + matrix + "</table>"
         "<div class='grid two'><div class='callout'>Original full-dtype arithmetic is finite/sparse;native test enumerates all65536gate words,not every gate/up pair.</div>"
         "<div class='callout'>FP32exp overflow and NaN/Inf behavior cannot be certified by a float64stable sigmoid.</div></div>"),
        ("UNDER THE HOOD", f"Native contract indexes65536raw words and copies{s['native_logical_table_bytes_from_contract']}bytes per table CTA",
         "<pre>Original contract model:\nuint16 gate → decode T → FP32SiLU\nround activation toT → FP32multiply\nround output toT\nKey is raw word,not numeric value\nSparse CPUfixture ≠ full native LUT</pre><p>↓ Independent dispatch ledger</p>"
         "<pre>SiLU,unclamped,CUDA,float16/bfloat16\nExact(dtype,device)cache hit first\nMiss:major10,optin≥128KiB\nCold capture → fallback,no table allocation\nWarm capture → reuse existing table\nPer-CTA shared bytes ≠ extra HBMbytes</pre>"),
        ("ROBUSTNESS CHECK", "Empty warmup may build128KiB;exact warm cache can serve capture without allocating",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Cold capture</td><td>{dispatch['cold_supported']['action']}/{dispatch['cold_supported']['new_logical_table_bytes']}B</td><td>{dispatch['cold_capture']['action']}/{dispatch['cold_capture']['new_logical_table_bytes']}B</td><td>Allocation excluded</td></tr>"
         f"<tr><td>Exact cache</td><td>{dispatch['warm_capture']['action']}</td><td>{dispatch['wrong_dtype_cache']['action']}</td><td>dtype must match</td></tr>"
         f"<tr><td>Empty warmup</td><td>{dispatch['empty_warmup']['new_logical_table_bytes']}Bbuild</td><td>No activation launch</td><td>Not zero allocation</td></tr>"
         f"<tr><td>±0float keys</td><td>2raw keys</td><td>{s['zero_key_misses']}/{s['zero_key_cases']}word misses</td><td>Signed identity lost</td></tr>"
         f"<tr><td>BF16gate−89</td><td>{w['original_fp32_divide_activation_word']:04x}</td><td>Stable float64:{w['original_high_precision_stable_activation_word']:04x}</td><td>Native bytes unknown</td></tr></table>"),
        ("LATENCY", "CPUeight-key lookup excludes full-table construction and every GPUshared-copy/launch cost",
         timing + "<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Finite validation in both paths.No CUDA/libdevice,occupancy,graphs or model-throughput measurement.</p>"),
        ("MY TAKE", "Select a rounded bit-pattern LUT only within its dtype,device,capture and shared-memory contract",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Intermediate-cast fidelity,signed word identity,exact-key warmup and explicit SM10/128KiBdispatch evidence.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Float-value keys,single-final-round fusion,cold-capture allocation,clamped/other activations,or CPUtimings sold as native speedups.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained;original implementation.Native infinity/NaN payloads,all gate/up pairs,GPUlatency and allocator peak unknown.No browser visual inspection.</div>"),
    ]
    return page("Rounded SiLU bit-table — seven cards", "".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i, (label, title, body) in enumerate(sections, 1)))


def figure(r):
    s = r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Rounded activation word-index cabinet</title><desc>Original sixty-solid card-index cabinet metaphor.Raw16-bit keys retain signed identity;cached rounded values are reused before capture.No CUDAor visual acceptance evidence.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG158 · A CARD-INDEX CABINET RETAINS WORD IDENTITY</text>
<text x="88" y="158" font-size="34" font-weight="700">Round first,index exact bits,warm exact dtype/device cache</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original single-round misses{s['single_round_misses']}/{s['lanes']};bit-key lookup misses{s['lookup_misses']}</text>
<text x="88" y="760" font-size="24">Native65536×2Btable;128KiBshared copy per CTAis not additional HBM</text>
<text x="88" y="810" font-size="24">Empty warmup can build;warm capture reuses;cold capture does not allocate</text>
<text x="88" y="880" font-size="18">ActualNov14addition · finite CPUlibm model,not CUDA__nv_expf/full-domain parity or GPUlatency</text></g></svg>
"""


def outputs(r):
    return {ROOT / f"docs/reports/{SLUG}-selection-cards.html": cards(r),
            ROOT / f"docs/reports/{SLUG}-sizer.html": workbench(r),
            ROOT / f"assets/svg/{SLUG}-selection-faq-summary.svg": figure(r)}
