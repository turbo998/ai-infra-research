import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="scatternd-mean-negative-alias-slice-contract"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==6||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>=0)||p.rows<1||p.width<1||![0,1,2].includes(p.index_depth)||![4,8].includes(p.index_element_bytes)||![2,4,8].includes(p.value_element_bytes))throw Error("Invalid exact dimensions/depth/byte widths");
 const r=BigInt(p.rows),w=BigInt(p.width),m=BigInt(p.updates),k=BigInt(p.index_depth),ix=BigInt(p.index_element_bytes),e=BigInt(p.value_element_bytes),slice=k===0n?r*w:k===1n?w:1n;
 const out={selected_data_payload_bytes:Number(r*w*e),selected_updates_payload_bytes:Number(m*slice*e),selected_indices_payload_bytes:Number(m*k*ix),selected_dense_int64_sum_count_payload_bytes:Number(r*w*16n)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(shape,data,indices,updates,k,reduction,opset){
 const ints=a=>Array.isArray(a)&&a.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000000);
 if(!ints(shape)||shape.length!==2||shape.some(v=>v<1||v>32)||!ints(data)||!ints(updates)||!Array.isArray(indices)||indices.length>64||![0,1,2].includes(k)||!["none","add","mean"].includes(reduction))throw Error("Bounded 2D integer data,depth0/1/2 and none/add/mean required");
 if(reduction==="mean"&&opset!==29)throw Error("Pinned mean requires opset29");
 const total=shape[0]*shape[1],width=k===0?total:k===1?shape[1]:1;
 if(data.length!==total||updates.length!==indices.length*width)throw Error("Data/updates shape mismatch");
 const seen=new Set(),canon=indices.map(row=>{if(!ints(row)||row.length!==k||row.some((v,i)=>v < -shape[i]||v>=shape[i]))throw Error("Index tuple outside shape bounds");
 const d=row.map((v,i)=>(v+shape[i])%shape[i]),key=JSON.stringify(d);if(reduction==="none"&&seen.has(key))throw Error("none forbids duplicate normalized locations");seen.add(key);return d;});
 const output=data.slice(),sums=data.slice(),counts=data.map(()=>1);
 canon.forEach((dest,row)=>{let prefix=0;dest.forEach((v,i)=>prefix=prefix*shape[i]+v);
 for(let c=0;c<width;c++){const target=prefix*width+c,value=updates[row*width+c];if(reduction==="mean"){sums[target]+=value;counts[target]++;output[target]=Math.floor(sums[target]/counts[target]);}else if(reduction==="add")output[target]+=value;else output[target]=value;}});
 return {canonical_destinations:canon,output_flat:output,note:"Exact bounded integers;mean original included once,floor division. Not native floating/backend parity."};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("ScatterND mean and negative-alias contract workbench",
        f"""<h1>Scatter mean includes the original value once—not pairwise repeated averaging</h1>
        <p>Round131 · Original bounded integer model,not native ONNX/FP/GPU. Opset29 mean contract verified.</p>
        <form id="form"><label>2D shape JSON<input id="shape" value="[2,2]" required></label>
        <label>Flat integer data JSON<input id="data" value="[10,-4,30,40]" required></label>
        <label>Index tuples JSON<input id="indices" value="[[0],[-2]]" required></label>
        <label>Flat integer updates JSON<input id="updates_flat" value="[2,-3,8,-2]" required></label>
        <label>Tuple depth<input id="depth" type="number" value="1" min="0" max="2" step="1" required></label>
        <label>Reduction<select id="reduction"><option>mean</option><option>add</option><option>none</option></select></label>
        <label>Selected opset<input id="opset" value="29" type="number" step="1" required></label>
        <h2>Independent logical payload budget</h2><div class="grid">{fields}</div><button>Normalize and reduce</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Negative index aliases must be normalized before duplicate checks. Mean denominator includes original once.
        None rejects duplicates,includingaliases. k0 addresses the whole tensor. No int dtype overflow/FP simulation.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)])),read=id=>JSON.parse(document.getElementById(id).value);
 out.textContent=JSON.stringify({result:simulate(read("shape"),read("data"),read("indices"),read("updates_flat"),Number(document.getElementById("depth").value),document.getElementById("reduction").value,Number(document.getElementById("opset").value)),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["repeated_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['updates']} updates ·64destinations</b><p>Grouped sum/count {v['grouped_median_ms']:.6f}ms</p>{bar(v['grouped_median_ms'],mx)}"
                  f"<p>Destination-first rescan {v['repeated_median_ms']:.6f}ms</p>{bar(v['repeated_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SCATTER MEAN","Two aliased updates share one destination;opset29 mean includes its original value once",
         "<div class='grid'><div><h3>Scenarios</h3><p>Negative aliases<br>Duplicate reduction<br>Integer floor</p></div>"
         "<div><h3>Structured input</h3><p>shape[2,2]<br>indices[[0],[-2]]<br>original[10,-4]</p></div>"
         "<div><h3>Decisions</h3><p>Normalize before grouping<br>Keep sum/count<br>Gate opset29</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Verified Nov12mean addition in ONNXopset29. Synthetic,independently_reproduced=false;original CPU Python{r['latency']['python']} / {r['latency']['machine']};no native/GPU reproduction.</div>"),
        ("SCENARIO: ALIASED MEAN","Original10 plus updates2/8 gives floor20/3=6—not update-only5 or repeated-pair7",
         f"<blockquote>Original row[10,-4];updates[2,-3]and[8,-2];correct flat output{c['correct_integer_mean']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · slice count,not confidence</span><h3>Selected updates{c['selected_total_updates']}</h3>{bar(c['selected_total_updates'],2)}<p>Group same destination before final division</p></div>"
         f"<div class='metric'><span class='pill'>score · slice count,not confidence</span><h3>Unique normalized slice{c['selected_unique_slice_destinations']}</h3>{bar(c['selected_unique_slice_destinations'],2,True)}<p>Confirm both index0 and-2 hitrow0</p></div>"),
        ("WHAT I TESTED","Integer means round toward negative infinity;duplicate none is not last-write portability",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Negativealias</td><td>0versus-2,size2</td><td>Same destination</td><td>Normalize</td></tr>"
         "<tr><td>Mean</td><td>Include original?</td><td>Once</td><td>Sum/count</td></tr>"
         "<tr><td>Integer</td><td>-5/2</td><td>-3;✕ trunc-2</td><td>Floor</td></tr>"
         "<tr><td>None</td><td>Duplicatealiases?</td><td>✕ Invalid contract</td><td>Reject explicitly</td></tr>"
         "<tr><td>k0</td><td>Tuple addresses?</td><td>Whole tensor</td><td>Validate updates shape</td></tr>"
         "<tr><td>Opset28</td><td>Mean supported?</td><td>✕ No29contract</td><td>No silent downgrade</td></tr></table>"
         "<div class='grid two'><div class='callout'>Integer exact model;FP rounding/overflow/backends untested.</div><div class='callout'>Reference traversal overwrite is not a portable duplicate-none promise.</div></div>"),
        ("UNDER THE HOOD","Normalize aliases,expand trailing slices,then divide grouped sums only once",
         "<pre>updates.shape = indices.shape[:-1]\n                + data.shape[k:]\ncanonical_index = index + dimension if negative\nk0 = whole tensor\nnone + repeated canonical location = invalid</pre>"
         "<p>↓ Original integer model and independent oracle</p><pre>sum[d] = original[d] + all updates[d]\ncount[d] = 1 + number of updates[d]\noutput[d] = floor(sum[d]/count[d])\nuntouched[d] = original[d]\n\nIndependent:enumerate destination coordinates,\nthen select every matching update tuple</pre>"),
        ("ROBUSTNESS CHECK","Reversing two updates preserves correct6 but changes repeated-pair output7→5",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>REVERSED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Correct integer means</td><td>{c['correct_integer_mean']}</td><td>{c['correct_reversed_mean']}</td><td>Exact equality</td></tr>"
         f"<tr><td>✕ Repeated-pair averaging</td><td>{c['wrong_repeated_pair_mean']}</td><td>{c['wrong_reversed_pair_mean']}</td><td>Not probability points</td></tr>"
         "<tr><td>FP bitwise invariance</td><td>Not measured</td><td>Not measured</td><td>Unknown,not0</td></tr></table>"),
        ("LATENCY","Original grouped integer sum/count avoids rescanning all updates per destination",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Setup sum/count included;nativeONNX/parallel atomics/FP/GPU excluded.</p>"),
        ("MY TAKE","Gate mean exports on opset29 and validate aliases before relying on backend traversal",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Export contracts,integer mean fixtures,negative-index audits,slice shapes and deliberate duplicate rejection.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>FP bitwise determinism,native atomics speed,integer overflow emulation or generic backend/opset29 support claims.</p></div></div>"
         "<div class='dark'>Apache-2.0ONNXsources;original arithmetic/oracle/art.NativeONNX latency/HBM/quality unknown;browser inspection not performed.</div>"),
    ]
    return page("ScatterND mean contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Scatter mean ticket sorter:normalize,group and divide</title>
<desc>Original sixty-solid sorting cabinet metaphor for aliased destination tickets and shared sum/count reduction.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG131 · SCATTER MEAN</text>
<text x="88" y="158" font-size="36" font-weight="700">Two aliased tickets share one original contribution</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Normalize negative indices before duplicate checks and grouping</text>
<text x="88" y="758" font-size="25">Mean = floor((original + all updates)/(1 + update count)) for integers</text>
<text x="88" y="806" font-size="25">None forbids duplicate destinations;opset29 adds mean and int32 indices</text>
<text x="88" y="880" font-size="18">Original exact integer/slice model and CPU timing · no native ONNX/atomic/FP/GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
