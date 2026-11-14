import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="multimodal-graph-cumulative-offset-padding"
JS=r"""
const CUMULATIVE=["cu_seqlens","cu_window_seqlens"];
function split(src,layout){return layout==="packed"?[src.slice(0,src.length/2),src.slice(src.length/2)]:[src];}
function validate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="capacity,key,layout,override,previous,shared,src")throw Error("Exact original buffer fields required");
 if(typeof p.key!=="string"||!p.key||!["single","packed","generic","scalar"].includes(p.layout)||![null,"zeros","repeat","sections","prefix_only"].includes(p.override)||typeof p.shared!=="boolean")throw Error("Known explicit key/layout/override/shared required");
 if(!Number.isInteger(p.capacity)||p.capacity<1||p.capacity>4096||!Array.isArray(p.previous)||p.previous.length!==p.capacity||p.previous.some(v=>!Number.isInteger(v)||v<0||v>1073741824))throw Error("Bounded capacity and complete previous buffer required");
 if(p.src===null)return;
 if(p.layout==="scalar"){if(!Number.isInteger(p.src)||p.src<0||p.src>1073741824||p.capacity!==1)throw Error("Scalar needs one slot");return;}
 if(!Array.isArray(p.src)||p.src.length>4096||p.src.some(v=>!Number.isInteger(v)||v<0||v>1073741824))throw Error("Bounded integer vector required");
 if(["single","packed"].includes(p.layout)){
 if(!p.src.length)throw Error("Cumulative source needs its zero offset");
 if(p.layout==="packed"&&(p.src.length%2||p.capacity%2))throw Error("Packed equal-sized sections required");
 for(const s of split(p.src,p.layout))if(s[0]!==0||s.some((v,i)=>i&&s[i-1]>v))throw Error("Each cumulative section starts0and is nondecreasing");}
}
function expected(src,n,layout){
 if(layout==="scalar")return [src];
 if(layout==="generic")return [...src,...Array(n-src.length).fill(0)];
 const width=layout==="packed"?n/2:n;return split(src,layout).flatMap(s=>[...s,...Array(width-s.length).fill(s[s.length-1])]);
}
function copyBuffer(dst,src,rule){
 const written=Array(dst.length).fill(false),put=(i,v)=>{dst[i]=v;written[i]=true;};
 if(rule==="sections"){
 if(!src.length||src.length%2||dst.length%2)throw Error("Section override needs nonempty even vectors");
 const width=dst.length/2;split(src,"packed").forEach((s,j)=>{for(let i=0;i<width;i++)put(j*width+i,i<s.length?s[i]:s[s.length-1]);});
 }else{if(rule==="repeat"&&!src.length)throw Error("No empty source terminal");
 for(let i=0;i<dst.length;i++){if(i<src.length)put(i,src[i]);else if(rule!=="prefix_only")put(i,rule==="repeat"?src[src.length-1]:0);}}
 return written;
}
function evaluate(p,old=false){
 validate(p);const src=p.src,buf=[...p.previous];
 if(src===null){if(p.shared)throw Error("Shared capture buffer requires replay values");
 return {status:"captured_constant_retained",buffer:buf,rule:null,words_written:0,full_buffer_written:false,word_misses:null,native_graph_verified:null};}
 const n=p.layout==="scalar"?1:src.length;
 if(n>p.capacity)return {status:"eager_required",buffer:buf,rule:null,words_written:0,full_buffer_written:false,word_misses:null,native_graph_verified:null};
 let rule,written;
 if(p.layout==="scalar"){rule="scalar_copy";buf[0]=src;written=[true];}
 else{rule=p.override||(!old&&CUMULATIVE.includes(p.key)?"repeat":"zeros");written=copyBuffer(buf,src,rule);}
 const target=expected(src,p.capacity,p.layout),misses=buf.filter((v,i)=>v!==target[i]).length,full=written.every(Boolean);
 return {status:misses===0&&full?"valid_policy":"invalid_padding_policy",buffer:buf,rule,words_written:written.filter(Boolean).length,full_buffer_written:full,word_misses:misses,expected_buffer:target,logical_int32_buffer_bytes:4*buf.length,native_graph_verified:null};
}
"""


def workbench(r):
    return page("Cumulative graph-buffer padding workbench",
        "<h1>Repeat terminal offsets for empty padding;packed Q/Ksections need independent tails</h1>"
        "<p>Round160 · Original1-D metadata/overwrite model,not CUDAgraphs or FlashInferattention."
        "Overrides are bounded original presets,not executable callbacks.</p>"
        f'<form id="form"><label>Source layout,previous capture buffer and explicit override<textarea id="input" rows="9" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Assess legacy/default padding</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Scalar copies bypass padding callbacks.Shared buffers require replay values.Oversize replay requires eager fallback,not truncation."
        "A partially written callback is explicitly invalid;native graph/allocator evidence remains unknown.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const p=JSON.parse(document.getElementById("input").value),after=evaluate(p);
 out.textContent=JSON.stringify({before:evaluate(p,true),after},null,2);
 if(after.status!=="valid_policy"&&after.status!=="captured_constant_retained")error.textContent=after.status+": "+(after.status==="eager_required"?"no graph replay claimed":after.word_misses+"mismatched words;full overwrite="+after.full_buffer_written);
 }catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    s=r["summary"];c=r["case"];q=r["packed_case"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],x["after"]["rule"],x["after"]["buffer"],
        "✕ old metadata" if x["before"]["status"]!="valid_policy" else "Confirm native scope"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["direct_median_ms"],x["redundant_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['words']}original words</b><p>Overwrite{x['direct_median_ms']:.6f}ms</p>{bar(x['direct_median_ms'],mx)}"
                  f"<p>Redundant zero+overwrite{x['redundant_median_ms']:.6f}ms</p>{bar(x['redundant_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · GRAPH METADATA",f"Original invalid padding cases{s['before_invalid_cases']}→{s['after_invalid_cases']};offset tails encode empty sequences",
         "<div class='grid'><div><h3>Scenarios</h3><p>Cumulative tails<br>Packed Q/Ksections<br>Shared overwrite</p></div>"
         "<div><h3>Input</h3><p>Actual source offsets<br>Fixed capture capacity<br>Override priority</p></div>"
         "<div><h3>Decision</h3><p>Repeat terminal<br>Separate packed tails<br>No stale slots</p><span class='pill'>Threshold:0word misses</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14multimodal graph correction.CPUlist copy onPython{r['latency']['python']}/{r['latency']['machine']};no GPU/attention measurement.</div>"),
        ("SCENARIO: SINGLE OFFSETS",f"Original zero-tail misses{s['primary_before_word_misses']}/{s['primary_words']}words;logical buffer remains{s['primary_logical_bytes']}B",
         f"<blockquote>Source{c['inputs']['src']}becomes{c['after']['buffer']}.Repeated terminal offsets create zero-length padded sequences,not a descending reset to0.</blockquote>"
         "<div class='metric'><span class='pill'>choice · confirm native attention contract</span><h3>Sequence lengths3,5,0,0</h3><p>Metadata proof,not attention-output accuracy.</p></div>"
         f"<div class='metric'><span class='pill'>score · original word equality</span><h3>Before{s['primary_words']-s['primary_before_word_misses']}/{s['primary_words']}</h3>{bar(s['primary_words']-s['primary_before_word_misses'],s['primary_words'])}"
         f"<h3>After{s['primary_words']}/{s['primary_words']}</h3>{bar(s['primary_words'],s['primary_words'])}</div>"),
        ("WHAT I TESTED","Single,packed,ordinary and scalar buffers obey different copy semantics",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Packed FlashInferQ/Kstreams repeat their own terminal offsets.</div><div class='callout'>Explicit config overrides precede cumulative defaults;scalar values bypass callbacks.</div></div>"),
        ("UNDER THE HOOD","Two cumulative key defaults cannot replace a packed backend's section-aware override",
         "<pre>Original rule dispatch:\nScalar → copy,bypass callbacks\nConfig override → first priority\ncu_seqlens/cu_window_seqlens → repeat\nOther vector keys → zero tail\nShared missing values → error\nOversize replay → eager required</pre><p>↓ Packed section contract</p>"
         f"<pre>Source Q[0,3,8] K[0,4,9]\nFlat repeat:{q['before']['buffer']}\nSection repeat:{q['after']['buffer']}\nEach section starts0;each padded delta0\nFull buffer written;pointer retained\nNo native graph or FlashInferexecuted</pre>"),
        ("ROBUSTNESS CHECK","Unwritten shared tails retain previous-budget data even when the source prefix is correct",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Tail policy</td><td>Repeat terminal</td><td>Zero-fill</td><td>Descending offsets</td></tr>"
         "<tr><td>Packed streams</td><td>Independent tails</td><td>Flat repeat</td><td>Q/Kboundary misplaced</td></tr>"
         "<tr><td>Override</td><td>Write full buffer</td><td>Prefix-only</td><td>Invalid stale slots</td></tr>"
         "<tr><td>Missing source</td><td>Shared buffer</td><td>Static nonshared</td><td>Error versus capture constant</td></tr>"
         "<tr><td>Source capacity</td><td>Fits</td><td>Oversize</td><td>Eager required,no truncation</td></tr></table>"),
        ("LATENCY","CPUredundant zeroing is measured separately from native graph replay and attention",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Preallocated lists and identical final buffers.No Torch/H2D/CUDA/FlashInfer/model-throughput costs.</p>"),
        ("MY TAKE","Preserve cumulative layout,whole-buffer writes and explicit backend overrides across shared budgets",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Empty padding intervals,independent Q/Ksections,override precedence,scalar bypass and shared replay values.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Zeroing cumulative tails,flat padding packed offsets,partial callbacks,empty offset vectors,or CPUcopy times sold as GPUthroughput.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,original1-D model.Native graph correctness,FlashInferattention,allocator peak and browser visual acceptance unknown.</div>"),
    ]
    return page("Cumulative graph buffers — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    s=r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Cumulative-offset overwrite cabinet</title><desc>Original sixty-solid indexed-buffer cabinet metaphor.Padded offsets repeat terminal values;packed streams use separate sections.Shared buffer writes must cover every slot.No GPUor visual acceptance evidence.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG160 · EMPTY SEQUENCES REPEAT THE TERMINAL OFFSET</text>
<text x="88" y="158" font-size="34" font-weight="700">Keep Q/Ksections separate and overwrite every shared slot</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original invalid cases{s['before_invalid_cases']}to{s['after_invalid_cases']};single buffer stays{s['primary_logical_bytes']}B</text>
<text x="88" y="760" font-size="24">Scalar bypass / explicit override / cumulative repeat / ordinary zero tail</text>
<text x="88" y="810" font-size="24">Shared missing source is an error;oversize requires eager execution,not truncation</text>
<text x="88" y="880" font-size="18">ActualNov14correction · original metadata model,not CUDAgraph or FlashInferattention verification</text></g></svg>
"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
