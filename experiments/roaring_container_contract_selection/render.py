import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="roaring-array-bitset-run-container-payload"
JS=r"""
function describe(values,allow_runs=true){
 if(typeof allow_runs!=="boolean"||!Array.isArray(values)||values.length>65536||values.some(v=>!Number.isInteger(v)||v<0||v>=2**32))throw Error("At most65536 unsigned32bit exact item IDs and explicit run flag required");
 const unique=[...new Set(values)].sort((a,b)=>a-b),groups=new Map();
 for(const v of unique){const key=Math.floor(v/65536);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(v%65536);}
 const containers=[];
 for(const [key,low] of groups){const cardinality=low.length;let runs=0,previous=-2;for(const value of low){if(value!==previous+1)runs++;previous=value;}
 let kind=cardinality<=4096?"array":"bitset",payload_bytes=kind==="array"?2*cardinality:8192;
 if(allow_runs&&2+4*runs<payload_bytes){kind="run";payload_bytes=2+4*runs;}
 containers.push({key,cardinality,kind,runs,payload_bytes});}
 return {unique_count:unique.length,containers,container_payload_bytes:containers.reduce((sum,c)=>sum+c.payload_bytes,0),uint32_identity_payload_bytes:4*unique.length,flat_bitmap_through_max_id_bytes:unique.length?Math.ceil((unique.at(-1)+1)/8):0,note:"container payload only;cookies/keys/cardinality/offset tables/allocator excluded;original strict run-saving policy"};
}
function simulate(A,B,allow_runs=true){
 const a=describe(A,allow_runs),b=describe(B,allow_runs),right=new Set(B),intersection=[...new Set(A)].filter(v=>right.has(v)).sort((x,y)=>x-y);
 return {A:a,B:b,intersection,intersection_count:intersection.length,union_count:new Set([...A,...B]).size,note:"Exact original set operations;not CRoaring or ANN/ACL execution. No whole-file header or native-memory claim."};
}
"""


def workbench(r):
    a,b=r["case"]["A_ids"],r["case"]["B_ids"]
    return page("Roaring array,bitset and run payload workbench",
        f"""<h1>4096 is an array boundary—not a universal compression verdict</h1>
        <p>Round137 · Original unsigned32bit identity grouping and selected container payload budget.</p>
        <form id="form"><label>Set A IDs JSON<textarea id="A" rows="5" required>{json.dumps(a)}</textarea></label>
        <label>Set B IDs JSON<textarea id="B" rows="5" required>{json.dumps(b)}</textarea></label>
        <label><input id="runs" type="checkbox" checked> Choose run payload only when strictly smaller</label>
        <button>Evaluate exact identity sets</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Low16bit values belong to their high16bit container key. The same low value in different containers is a different ID.
        Payload excludes cookie/descriptive/offset headers and allocator overhead. No production authorization or revocation-latency claim.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Both bounded unsigned32bit ID arrays required");
 out.textContent=JSON.stringify(simulate(JSON.parse(document.getElementById("A").value),JSON.parse(document.getElementById("B").value),document.getElementById("runs").checked),null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(row["sorted_merge_median_ms"] for row in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{row['items_per_set']} IDs per set</b><p>Prebuilt integer AND+bit_count {row['integer_and_median_ms']:.6f}ms</p>{bar(row['integer_and_median_ms'],mx)}"
                   f"<p>Sorted intersection list+len {row['sorted_merge_median_ms']:.6f}ms</p>{bar(row['sorted_merge_median_ms'],mx,True)}</div>" for row in r["latency"]["rows"])
    boundaries="".join(f"<tr><td>{v['name']}</td><td>{v['without_runs']['containers'][0]['kind']}/{v['without_runs']['container_payload_bytes']}B</td><td>{v['with_runs']['containers'][0]['kind']}/{v['with_runs']['container_payload_bytes']}B</td><td>Exact selected payload</td></tr>" for v in r["boundaries"])
    sections=[
        ("HANDS-ON · EXACT ID FILTERS","103 fixture identities use12B of selected container payload versus206B without runs",
         "<div class='grid'><div><h3>Scenarios</h3><p>Array/bitset boundary<br>Contiguous run payload<br>High-key identity</p></div>"
         "<div><h3>Input</h3><p>Unsigned32bit IDs<br>16bit key / 16bit low<br>Sorted exact sets</p></div>"
         "<div><h3>Decisions</h3><p>MAX low range65535<br>Non-run threshold4096<br>Headers remain separate</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Format-spec facts;original payload encoder/decoder and interval oracle. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}. Not CRoaring/native memory.</div>"),
        ("SCENARIO: TWO-HIGH-KEY SETS","A103IDs and B102IDs intersect at52identities,not by matching low16bits globally",
         f"<blockquote>A uses{c['A']['container_payload_bytes']}B:three isolated low IDs plusone100-ID run. Exact intersection{c['intersection_count']};union{c['union_count']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · selected byte count,not confidence</span><h3>Run-enabled A{c['A']['container_payload_bytes']}B</h3>{bar(c['A']['container_payload_bytes'],c['A_without_runs']['container_payload_bytes'])}<p>Array6B+run6B;full-file headers excluded</p></div>"
         f"<div class='metric'><span class='pill'>score · selected byte count,not confidence</span><h3>No-run A{c['A_without_runs']['container_payload_bytes']}B</h3>{bar(c['A_without_runs']['container_payload_bytes'],c['A_without_runs']['container_payload_bytes'],True)}<p>Two array payloads,not allocator memory</p></div>"),
        ("WHAT I TESTED","4097 isolated lows choose bitset;4097 contiguous lows can instead use a6B run payload",
         "<table><tr><th>SCENARIO</th><th>WITHOUT RUNS</th><th>WITH RUNS</th><th>ACTION</th></tr>"+boundaries+"</table>"
         "<div class='grid two'><div class='callout'>Non-run cardinality≤4096 determines array;larger means bitset.</div><div class='callout'>Run structure—not count alone—can change selected payload.</div></div>"),
        ("UNDER THE HOOD","Run pairs store length-minus-one;whole-file portable headers are not these payload bytes",
         "<pre>Original unsigned32bit identity:\nkey = id &gt;&gt; 16\nlow = id &amp; 65535\nnon-run: c&lt;=4096 ? array(2*c) : bitset(8192)\nrun payload: 2+4*r\nOriginal strict-saving policy may choose run</pre>"
         "<p>↓ Original payload decoder with explicit metadata</p><pre>Array:sorted unique uint16LE lows\nBitset:8192 bytes,LSB indexed\nRun:uint16LE run count\n    pairs(start,length_minus_one)\nRebuild identity: (key&lt;&lt;16) | low\nVerify cardinality,bounds,payload length\nWhole-file cookies/keys/offsets not emitted</pre>"),
        ("ROBUSTNESS CHECK","Reversed and duplicated inputs preserve exact sets;dropping high keys creates a false match",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Inputorder/duplicates</td><td>Canonical set</td><td>Same payload summary</td><td>Exact equality</td></tr>"
         "<tr><td>SwapA/Bintersection</td><td>52 IDs</td><td>52 same IDs</td><td>0 IDs</td></tr>"
         "<tr><td>ID7vs65543</td><td>Empty intersection</td><td>✕ low-only finds7</td><td>1 false match</td></tr></table>"),
        ("LATENCY","Python integer cardinality skips list materialization;this comparison is not native Roaring speed",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. C-backed integer AND+bit_count versus original Python sorted intersection list+len,unequal work. Construction/encoding/decoding/SIMD/native dispatch/ANN/ACL excluded.</p>"),
        ("MY TAKE","Choose exact bitmap representation by density and runs,then validate freshness and identity mapping separately",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Exact ID-set intersection,metadata filtering and explicit payload/encoding boundaries.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Assumed native memory,universal SIMD speedup,approximate cardinality semantics or automatic ACL freshness/security guarantees.</p></div></div>"
         "<div class='dark'>Pinned Apache-2.0 format-spec;original payload model and oracle.Native64-bit rank/select capability tracked but not executed;visual browser inspection not performed.</div>"),
    ]
    return page("Roaring container contracts — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Exact-ID filing cabinet:array,bitset and run payloads</title>
<desc>Original sixty-solid cabinet metaphor:high-key drawers contain low16bit identity sets;alternative payloads preserve exact membership.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG137 · EXACT ID CONTAINER PAYLOADS</text>
<text x="88" y="158" font-size="36" font-weight="700">High-key drawers prevent low-bit identity confusion</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Array:2c bytes · bitset:8192 bytes · run:2+4r bytes</text>
<text x="88" y="758" font-size="25">4096 non-run values use array;4097 use bitset</text>
<text x="88" y="806" font-size="25">Runs may change selected payload;headers and allocator stay separate</text>
<text x="88" y="880" font-size="18">Original payload codec/interval oracle · no full-format/native CRoaring/SIMD or ACL freshness certification</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
