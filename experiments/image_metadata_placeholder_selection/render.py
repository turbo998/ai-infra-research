import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="kimi-k3-metadata-only-image-original-dimensions-placeholder"
JS=r"""
function integer(x,max=8192){return Number.isInteger(x)&&x>=1&&x<=max;}
function plan(p,memo=true,proposal="original"){
 if(!p||Object.keys(p).sort().join(",")!=="allow_missing,embedding_shape,fixed_tokens,merge,patch,patch_budget,side_patches,sizes")throw Error("Exactly the original dimension/configuration fields are required");
 if(!Array.isArray(p.sizes)||p.sizes.length<1||p.sizes.length>64||p.sizes.some(s=>!Array.isArray(s)||s.length!==2||s.some(x=>!integer(x))))throw Error("1..64 positive integer width,height pairs required");
 if(!integer(p.patch,128)||!integer(p.merge,8)||!integer(p.patch_budget,4096)||!integer(p.side_patches,64)||p.side_patches%p.merge!==0)throw Error("Positive bounded merge-aligned resize profile required");
 if(p.fixed_tokens!==null&&!integer(p.fixed_tokens,1024))throw Error("Fixed output is null or positive bounded count");
 if(typeof p.allow_missing!=="boolean")throw Error("Explicit deployment boolean required");
 const shape=p.embedding_shape;
 if(shape!==null&&(!Array.isArray(shape)||shape.length!==2||!integer(shape[0],262144)||!integer(shape[1],128)))throw Error("Null or positive embedding row,column shape required");
 if(shape===null&&!p.allow_missing)throw Error("Embedding values required by deployment");
 if(!["original","padded_headers","collapse_occurrences"].includes(proposal))throw Error("Known proposal required");
 const cache=new Map(),cap=p.side_patches*p.patch,factor=p.patch*p.merge;
 const rows=p.sizes.map(([w,h])=>{const key=w+","+h;if(memo&&cache.has(key))return cache.get(key);
 const density=Math.max(1,Math.floor(w/p.patch))*Math.max(1,Math.floor(h/p.patch));
 const scale=Math.min(1,Math.sqrt(p.patch_budget/density),cap/w,cap/h);
 const nw=Math.min(cap,Math.max(1,Math.floor(w*scale))),nh=Math.min(cap,Math.max(1,Math.floor(h*scale)));
 const pw=Math.ceil(nw/factor)*factor,ph=Math.ceil(nh/factor)*factor;
 const v={resized:[nw,nh],padded:[pw,ph],tokens:p.fixed_tokens===null?(pw/factor)*(ph/factor):p.fixed_tokens};
 if(memo)cache.set(key,v);return v;});
 const total=rows.reduce((s,x)=>s+x.tokens,0);
 if(shape!==null&&shape[0]!==total)throw Error("Embedding rows do not sum to derived counts");
 let indices=rows.map((_,i)=>i);
 if(proposal==="collapse_occurrences"){const seen=new Set();indices=indices.filter(i=>{const k=p.sizes[i].join(",");if(seen.has(k))return false;seen.add(k);return true;});}
 let offset=0;const ranges=[],headers=indices.map(i=>{ranges.push([offset,offset+rows[i].tokens]);offset+=rows[i].tokens;
 return {occurrence:i,original_size:proposal==="padded_headers"?rows[i].padded:p.sizes[i],pad_tokens:rows[i].tokens};});
 const expected=p.sizes.map((s,i)=>({occurrence:i,original_size:s,pad_tokens:rows[i].tokens}));
 return {mode:shape===null?"metadata_only_deferred":"declared_embedding_rows",resize_rows:rows,headers,planned_embedding_ranges:ranges,
 total_pad_tokens:total,retained_pad_tokens:offset,placeholder_misses:expected.filter(x=>!headers.some(y=>JSON.stringify(x)===JSON.stringify(y))).length,
 occurrences:rows.length,retained_occurrences:headers.length,
 compact_descriptor_bytes:JSON.stringify(p.sizes.map(s=>({image_sizes:s}))).length,
 declared_uncompressed_rgb_bytes:p.sizes.reduce((s,[w,h])=>s+w*h*3,0),
 native_tokenizer_token_ids:null,native_embedding_execution:null,native_transport_bytes:null};
}
"""


def workbench(r):
    body="<h1>Metadata can reconstruct ordered image placeholders,not execute missing embeddings</h1><p>Original bounded dimensions/configuration,not checkpoint defaults or a tensor ABI.Deployment permission is not a client override.Prompt header dimensions stay original.</p>"
    body+=f'<form id="form"><label>Original profile<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Plan occurrence headers and embedding ranges</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
    body+="<p>Compact descriptor and declared uncompressed RGB budgets exclude framing,IDs,compression and embeddings.They are not native network traffic.Header records are abstract,not tokenizer token IDs.</p>"
    return page("Image metadata placeholder workbench",body,JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),wrong_padded_headers:plan(p,true,"padded_headers"),wrong_occurrence_collapse:plan(p,true,"collapse_occurrences")},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["padded_headers"];c=r["case"]["collapsed"]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],"Original contract",("✕ "+x["outputs"]["error"]) if "error" in x["outputs"] else str(x["outputs"]["total_pad_tokens"])+"pads",
        "Reject invalid input" if "error" in x["outputs"] else "Confirm EC/tensor readiness"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["memo_median_ms"],x["direct_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['occurrences']}ordered occurrences</b><p>Scoped geometry memo{x['memo_median_ms']:.6f}ms</p>{bar(x['memo_median_ms'],mx)}<p>Direct geometry{x['direct_median_ms']:.6f}ms</p>{bar(x['direct_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · IMAGE METADATA","Metadata-only image dictionaries still require original dimensions and derived token counts",
         "<div class='grid'><div>Cases<br>Metadata-only<br>Flat rows<br>Repeated occurrences</div><div>Input<br>Positive W,H<br>Resize policy<br>Missing-values gate</div>"
         f"<div>Decision<br>{a['occurrences']}occurrences<br>{a['total_pad_tokens']}planned pad tokens<br>Native latency unknown</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false;actualNov14vLLMsource pin.Original CPUgeometry separately measured on Python{r['latency']['python']}/{r['latency']['machine']}.No model/tokenizer/PIL/Torch.</div>"),
        ("SCENARIO: REPEATED IMAGE",f"Three original occurrences require{a['total_pad_tokens']}pad tokens,not{c['retained_pad_tokens']}after a wrong size-only collapse",
         "<blockquote>Original sizes65×33,9×7,65×33;declared patch8/merge2/side16 profile is not checkpoint configuration.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm EC/tensor readiness · confidence unknown</span><p>{a['retained_occurrences']}ordered placeholders;missing tensor execution remains deferred.</p></div>"
         f"<div class='metric'><span class='pill'>score · original token cardinality</span><p>Correct{a['total_pad_tokens']}</p>{bar(a['total_pad_tokens'],a['total_pad_tokens'])}<p>Collapsed{c['retained_pad_tokens']}</p>{bar(c['retained_pad_tokens'],a['total_pad_tokens'])}</div>"),
        ("WHAT I TESTED",f"Six original cases separate metadata admission,flat-row sums and resize-derived cardinality",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Dimensions remain required even when values are optional.Missing-values permission belongs to the deployment.</div><div class='callout'>A declared flat tensor needs sum(counts)rows;the planner does not verify hidden width,device or native ABI.</div></div>"),
        ("UNDER THE HOOD","Resize policy derives counts while prompt headers preserve original width and height",
         "<pre>Original dimension packet\nimage_sizes:positive [width,height]\nDeployment:values required or optional\nDerive resize/padding and per-image counts\nSplit supplied flat rows by those counts</pre><p>↓ Separate contracts</p>"
         "<pre>Header uses ORIGINAL dimensions\nPad repetition uses derived token count\nKeep every occurrence in request order\nMetadata-only does not supply embeddings\nNative tokenizer IDs remain unknown</pre>"),
        ("ROBUSTNESS CHECK",f"Padded headers miss{b['placeholder_misses']}/{a['occurrences']}original descriptors;occurrence collapse loses{c['placeholder_misses']}",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Header dimensions</td><td>{a['placeholder_misses']}misses</td><td>{b['placeholder_misses']}padded-header misses</td><td>Original dimensions required</td></tr>"
         f"<tr><td>Repeated occurrence</td><td>{a['retained_occurrences']}kept</td><td>{c['retained_occurrences']}kept</td><td>One lost</td></tr>"
         "<tr><td>Fixed count4</td><td>Variable15/1/15</td><td>4/4/4</td><td>Policy affects counts</td></tr>"
         "<tr><td>Geometry memo</td><td>Direct ordered rows</td><td>Same ordered rows</td><td>Not image/embedding deduplication</td></tr></table>"),
        ("LATENCY","Scoped size-geometry memo preserves occurrence outputs;CPU timings exclude all image encoding",timing+
         "<p>2warmups,7alternating repeats,3batches/sample;median repeat means.Precreated three-size sequence.Per-call cache is bound to one configuration.No native vLLM/encoder/EC/GPU/tokenizer.</p>"),
        ("MY TAKE","Use metadata-only planning for an EC consumer,not as a replacement for embedding delivery",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Original dimensions,shared policy,ordered placeholders,explicit consumer permission and independently verified tensor delivery.</div><div class='bad'><h3>⛔ Not for</h3>Client-controlled permission,trusting counts,merging image occurrences by size,claiming encoder/network speed from CPUgeometry.</div></div>"
         f"<div class='dark'>Original compact descriptors{a['compact_descriptor_bytes']}B versus declared uncompressed RGBbudget{a['declared_uncompressed_rgb_bytes']}B;not a native transport measurement.Apache-2.0rights retained.Native execution/visual acceptance unknown.</div>")]
    return page("Image metadata — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Ordered image descriptor cabinet</title><desc>Original sixty-solid cabinet;individual drawers represent ordered image occurrences,not deduplicated pixel buffers.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG166 · ORDERED DESCRIPTOR CABINET</text><text x="88" y="158" font-size="34">Original dimensions remain;embedding values may be deferred</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original 3occurrences /{a['total_pad_tokens']}pad tokens / exclusive row ranges[0,15),[15,16),[16,31)</text><text x="88" y="760" font-size="24">Original W,H headers + policy-derived counts + ordered repeated occurrences</text><text x="88" y="810" font-size="24">Metadata-only admission is not native image encoder or embedding execution</text><text x="88" y="880" font-size="18">ActualNov14source · original bounded CPUmetadata model · tokenizer/GPU/transport/visual acceptance unknown</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
