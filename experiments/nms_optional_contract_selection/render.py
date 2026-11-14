import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="nms-optional-scalars-per-class-iou-boundary"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==5||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![1,2,4,8].includes(p.element_bytes))throw Error("Positive exact dimensions/cap and selected byte width required");
 const b=BigInt(p.batches),c=BigInt(p.classes),n=BigInt(p.boxes),k=BigInt(Math.min(p.boxes,p.max_per_class)),e=BigInt(p.element_bytes);
 const out={box_payload_bytes:Number(b*n*4n*e),score_payload_bytes:Number(b*c*n*e),selected_index_upper_bound_rows:Number(b*c*k),selected_int64_index_upper_bound_bytes:Number(b*c*k*3n*8n)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(boxes,scores,cap,threshold,scoreThreshold){
 if(!Array.isArray(boxes)||boxes.length<1||boxes.length>32||!boxes.every(b=>Array.isArray(b)&&b.length===4&&b.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000))||
 !Array.isArray(scores)||scores.length<1||scores.length>4||!scores.every(s=>Array.isArray(s)&&s.length===boxes.length&&s.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000))||
 !Number.isInteger(cap)||cap<0||cap>32||!Number.isFinite(threshold)||threshold<0||threshold>1||scoreThreshold!==null&&(!Number.isSafeInteger(scoreThreshold)||Math.abs(scoreThreshold)>1000))throw Error("Bounded exact boxes/scores,cap0..32,IoU0..1 required");
 const rects=boxes.map(([y1,x1,y2,x2])=>[Math.min(y1,y2),Math.min(x1,x2),Math.max(y1,y2),Math.max(x1,x2)]);
 const iou=(a,b)=>{const area=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
 const union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area;return union>0?area/union:0};
 const selected=[];scores.forEach((row,c)=>{const ids=row.map((_,i)=>i).filter(i=>scoreThreshold===null||row[i]>scoreThreshold).sort((a,b)=>row[b]-row[a]||a-b),chosen=[];
 for(const i of ids){if(chosen.length>=cap)break;if(!chosen.some(j=>iou(rects[i],rects[j])>threshold)){chosen.push(i);selected.push([0,c,i])}}});
 return {selected_indices:selected,output_shape:[selected.length,3]};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("NMS optional-input and per-class boundary workbench",
        f"""<h1>Omitted score threshold is not explicit zero in the pinned reference</h1>
        <p>Round128 · Original bounded class-aware NMS,corner boxes [y1,x1,y2,x2],one batch.
        No native ONNX/GPU/detection-quality reproduction.</p><form id="form">
        <label>Boxes JSON<input id="toy_boxes" value="[[0,0,2,2],[0,1,2,3],[0,0,2,2],[10,10,12,12]]" required></label>
        <label>Per-class integer scores JSON<input id="scores" value="[[90,80,70,-10],[10,20,95,0]]" required></label>
        <label>Maximum per class (blank means omitted/default0)<input id="cap" value="2" type="number" step="1"></label>
        <label>IoU numerator<input id="num" value="1" type="number" step="1" required></label>
        <label>IoU denominator<input id="den" value="3" type="number" step="1" required></label>
        <label>Score threshold (blank means omitted)<input id="score" value="" type="number" step="1"></label>
        <h2>Independent selected payload bound</h2><div class="grid">{fields}</div><button>Inspect selected indices</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>IoU suppression is strict &gt;,explicit score predicate is strict &gt; in the pinned reference.
        Tie index policy is a selected deterministic convention,not a universal backend guarantee.
        Output contains [batch,class,box] indices,not box coordinates.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
 const numerator=Number(document.getElementById("num").value),denominator=Number(document.getElementById("den").value);
 if(!Number.isSafeInteger(numerator)||!Number.isSafeInteger(denominator)||numerator<0||denominator<1||denominator>1000||numerator>denominator)throw Error("Exact bounded IoU ratio required");
 const capText=document.getElementById("cap").value,scoreText=document.getElementById("score").value;
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("toy_boxes").value),JSON.parse(document.getElementById("scores").value),capText.trim()?Math.max(0,Number(capText)):0,numerator/denominator,scoreText.trim()?Number(scoreText):null),budget:budget(p),
 note:"Original exact-bounded geometry model,pinned reference predicate;native backends not executed"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["normalize_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['boxes']} disjoint boxes</b><p>Cached canonical coordinates {v['cached_median_ms']:.6f}ms</p>{bar(v['cached_median_ms'],mx)}"
                  f"<p>Normalize per pair {v['normalize_median_ms']:.6f}ms</p>{bar(v['normalize_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · VISION POSTPROCESS","Omitting the per-class cap gives empty[0,3]; a cap is not an unlimited default",
         "<div class='grid'><div><h3>Scenarios</h3><p>Optional scalar shape<br>Score omission<br>Per-class IoU gate</p></div>"
         "<div><h3>Input</h3><p>Four boxes,two classes<br>IoU boundary1/3<br>Cap2per class</p></div>"
         "<div><h3>Top decisions</h3><p>Keep absent distinct from zero<br>Return index triples</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Oct28ONNXreference scalar/omitted-input fix,not new NMS algorithm. Original CPU Python{r['latency']['python']} / {r['latency']['machine']};synthetic,independently_reproduced=false.</div>"),
        ("SCENARIO: OMITTED SCORE","Omitted score threshold retains3 signed-score boxes; explicit zero retains1",
         f"<blockquote>Disjoint scores[-1/10,0,1/10],cap3,IoU1/2. Omitted:{c['signed_omitted_score_selected']};explicit0:{c['signed_explicit_zero_score_selected']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · deterministic selected count,not confidence</span><h3>Absent score threshold3/3</h3>{bar(len(c['signed_omitted_score_selected']),3)}<p>Confirm absent-input meaning</p></div>"
         f"<div class='metric'><span class='pill'>score · deterministic selected count,not confidence</span><h3>Explicit zero1/3</h3>{bar(len(c['signed_explicit_zero_score_selected']),3,True)}<p>Act automatically only with selected predicate</p></div>"),
        ("WHAT I TESTED","IoU equal to1/3 survives strict suppression; cap1 applies separately to both classes",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Omitted cap</td><td>Unlimited?</td><td>✕ Empty[0,3]</td><td>Supply cap</td></tr>"
         "<tr><td>Scalar/vector cap</td><td>Equivalent wrapper?</td><td>Selected0-D/one-itemmodel</td><td>Normalize explicitly</td></tr>"
         "<tr><td>IoU exact boundary</td><td>Suppress on equality?</td><td>No,strict&gt;</td><td>Keep threshold contract</td></tr>"
         "<tr><td>Cap1,two classes</td><td>Global cap1?</td><td>✕ Two index triples</td><td>Per-class scope</td></tr>"
         "<tr><td>Output</td><td>Boxes or indices?</td><td>[batch,class,box]</td><td>Gather separately</td></tr></table>"
         "<div class='grid two'><div class='callout'>Pinned reference strict score comparison and lower-index tie convention are identified explicitly.</div><div class='callout'>No universal native/GPU tie-order or detector-quality certificate.</div></div>"),
        ("UNDER THE HOOD","Candidate-removal and selected-list greedy algorithms agree under the selected exact IoU policy",
         "<pre>Per class:\nfilter only if score threshold present\nrank score descending,index ascending\nselect highest eligible box\nsuppress only IoU &gt; chosen threshold\nstop at cap for this class\nappend [batch,class,box] index</pre>"
         "<p>↓ Original independent selection oracle</p><pre>Maintain remaining candidate set\npick best,remove overlapping remaining boxes\ncompare to selected-list checking implementation\n\nexact integer-cell area oracle checks IoU\nzero-area boxes use selected IoU0 policy</pre>"),
        ("ROBUSTNESS CHECK","Replacing&gt;with&gt;=changes selected box identities at the exact IoU boundary",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>IoU boundary</td><td>{c['strict_iou_selected']}</td><td>{c['wrong_equal_iou_selected']}</td><td>Index identities,not probability</td></tr>"
         f"<tr><td>Cap1 per class</td><td>Two classes</td><td>{c['cap1_selected']}</td><td>Not global top1</td></tr></table>"
         "<p>Corner versus center format and inverted corner coordinates separately tested. Shape[0,3] is preserved even when selection is empty.</p>"),
        ("LATENCY","Original cached box canonicalization avoids repeated corner normalization,not native NMS work",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Disjoint boxes,all selected. Cached setup,sorting,score predicates,native ONNX/GPU/quality excluded.</p>"),
        ("MY TAKE","Keep optional-input presence,class scope and index output explicit across detector exports",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Exact threshold/default audits,box-format checks and per-class selected-index contracts.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Unlimited omitted cap,absent-to-zero score fallback,universal tie ordering or GPU/detection-quality claims.</p></div></div>"
         "<div class='dark'>Apache-2.0ONNXreference;original model/oracles/art. Native latency/quality/HBM unknown;browser visual inspection not performed.</div>"),
    ]
    return page("NMS optional-input contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>NMS class-ticket sorter: optional presence and strict threshold boundaries</title>
<desc>Original sixty-solid sorting cabinet analogy for per-class queues and selected index tickets.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG128 · NMS OPTIONAL INPUTS</text>
<text x="88" y="158" font-size="36" font-weight="700">Absent score threshold is not explicit zero</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Omitted output cap means zero;retain empty index shape[0,3]</text>
<text x="88" y="758" font-size="25">Suppress only strict IoU excess;caps and selections are per class</text>
<text x="88" y="806" font-size="25">Output index triples,not coordinates;native tie policies need validation</text>
<text x="88" y="880" font-size="18">Original exact geometry / independent selection oracle / CPU timing · no native GPU or detection-quality reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
