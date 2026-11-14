import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="non-square-causal-mask-cache-position-polarity"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==5||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![2,4,8].includes(p.additive_element_bytes))throw Error("Positive exact dimensions and selected additive width required");
 const b=BigInt(p.batch),h=BigInt(p.heads),q=BigInt(p.queries),k=BigInt(p.keys),e=BigInt(p.additive_element_bytes);
 const out={selected_shared_2d_bool_mask_bytes:Number(q*k),selected_bh_bool_mask_if_materialized_bytes:Number(b*h*q*k),selected_shared_2d_additive_mask_bytes:Number(q*k*e),selected_int64_prefix_limits_bytes:Number(q*8n)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(values,queryPositions,keyPositions){
 const bounded=a=>Array.isArray(a)&&a.length>0&&a.length<=64&&a.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000000);
 if(!bounded(values)||!bounded(queryPositions)||!bounded(keyPositions)||values.length!==keyPositions.length)throw Error("1..64 exact bounded values/positions,matching key axis required");
 const q=queryPositions.length,k=keyPositions.length;
 const actual=queryPositions.map(p=>keyPositions.map(t=>t<=p)),upper=Array.from({length:q},(_,i)=>Array.from({length:k},(_,j)=>j<=i)),lower=Array.from({length:q},(_,i)=>Array.from({length:k},(_,j)=>j<=i+k-q));
 const attend=mask=>mask.map(row=>{const selected=values.filter((_,i)=>row[i]);return selected.length?{kind:"value",value:selected.reduce((a,b)=>a+b,0)/selected.length}:{kind:"no_allowed_keys",value:null}});
 return {actual_allow:actual,upper_left_allow:upper,lower_right_allow:lower,actual_outputs:attend(actual),upper_left_outputs:attend(upper),lower_right_outputs:attend(lower),
 wrong_mha_polarity_outputs:attend(actual.map(row=>row.map(v=>!v)))};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Non-square causal mask and position workbench",
        f"""<h1>Causality follows query/key positions,not only the rectangle shape</h1>
        <p>Round130 · Original scalar uniform-logit attention means. No QK matmul,softmax,native SDPA/GPU or model-quality reproduction.</p>
        <form id="form"><label>Scalar values JSON<input id="values" value="[1,2,4,8,16]" required></label>
        <label>Query positions JSON<input id="query_positions" value="[3,4]" required></label>
        <label>Key positions JSON<input id="key_positions" value="[0,1,2,3,4]" required></label>
        <h2>Independent selected mask payload budget</h2><div class="grid">{fields}</div><button>Compare causal conventions</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>SDPA boolean True means allow;MHA boolean True means block. Lower-right matches a contiguous suffix
        only under its positional assumption. Empty allowed rows stay tagged null in this original model,
        not silently zeroed or treated as a native backend result.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("values").value),JSON.parse(document.getElementById("query_positions").value),JSON.parse(document.getElementById("key_positions").value)),budget:budget(p),
 note:"Synthetic uniform-logit means only;not native softmax or all-masked backend semantics"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["dense_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['keys']} keys ·8queries</b><p>Rebuilt prefix sums {v['prefix_median_ms']:.6f}ms</p>{bar(v['prefix_median_ms'],mx)}"
                  f"<p>Dense coordinate filtering {v['dense_median_ms']:.6f}ms</p>{bar(v['dense_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · CAUSAL POSITIONS","A one-query,five-key upper-left causal mask keeps1key;the suffix-aligned mask keeps5",
         "<div class='grid'><div><h3>Scenarios</h3><p>Non-square alignment<br>Boolean polarity<br>Actual cache positions</p></div>"
         "<div><h3>Input</h3><p>Values[1,2,4,8,16]<br>Queries at3,4<br>Keys at0..4</p></div>"
         "<div><h3>Top decisions</h3><p>Actual key≤query position<br>SDPATrueallow<br>MHA Trueblock</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Established PyTorch SDPA/bias contracts in Nov14pinned tree,not new regional-AOTI semantics. Original CPU Python{r['latency']['python']} / {r['latency']['machine']};synthetic,independently_reproduced=false.</div>"),
        ("SCENARIO: RECTANGULAR CAUSALITY","Two suffix queries should retain4/5 keys;upper-left retains1/2 instead",
         f"<blockquote>Correct uniform means{[v['value'] for v in c['correct_uniform_outputs']]};upper-left{[v['value'] for v in c['upper_left_uniform_outputs']]}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · allowed-key count,not confidence</span><h3>Correct first query4/5</h3>{bar(c['lower_right_allowed_counts'][0],5)}<p>Act automatically only with matching positions</p></div>"
         f"<div class='metric'><span class='pill'>score · allowed-key count,not confidence</span><h3>Upper-left first query1/5</h3>{bar(c['upper_left_allowed_counts'][0],5,True)}<p>✕ Confirm non-square alignment</p></div>"),
        ("WHAT I TESTED","Boolean polarity can expose a future key and leave another query with no allowed keys",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>SDPA bool</td><td>True means?</td><td>Allow</td><td>API-specific mask</td></tr>"
         "<tr><td>MHA bool</td><td>Same True meaning?</td><td>✕ Block</td><td>Invert deliberately</td></tr>"
         f"<tr><td>Wrong polarity,row0</td><td>Uniform mean?</td><td>{c['wrong_polarity_uniform_outputs'][0]['value']}</td><td>Future-key leak control</td></tr>"
         "<tr><td>Wrong polarity,row1</td><td>Empty row equals0?</td><td>✕ Tagged null</td><td>Inspect backend separately</td></tr>"
         "<tr><td>Gapped keys</td><td>Shape gives position?</td><td>✕ Needs actual positions</td><td>Explicit predicate</td></tr></table>"
         "<div class='grid two'><div class='callout'>Uniform zero logits only;not general attention/model-quality evidence.</div><div class='callout'>Tagged empty rows are original policy,not a native NaN/zero guarantee.</div></div>"),
        ("UNDER THE HOOD","Lower-right is an index-offset convenience for a contiguous query suffix",
         "<pre>Upper-left:allow j &lt;= i\nLower-right:allow j &lt;= i+(S-L)\nActual causal predicate:\n  key_position[j] &lt;= query_position[i]\n\nSDPA boolean True:participate\nMHA boolean True:block</pre>"
         "<p>↓ Original independent value oracle</p><pre>Sort key identities by actual position\naccumulate eligible scalar values until query position\nuniform-logit output = eligible sum/count\nempty row = explicit tagged null\n\nNo QKmatmul/softmax/dropout/GQA reproduction</pre>"),
        ("ROBUSTNESS CHECK","Keeping shape2×5 while changing key positions makes lower-right admit future positions",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Key positions</td><td>{c['key_positions']}</td><td>{c['gapped_key_positions']}</td><td>Actual coordinates</td></tr>"
         f"<tr><td>Correct means</td><td>{[v['value'] for v in c['correct_uniform_outputs']]}</td><td>{[v['value'] for v in c['gapped_correct_uniform_outputs']]}</td><td>Not probability points</td></tr>"
         f"<tr><td>✕ Shape-only lower-right</td><td>Contiguous suffix valid</td><td>{[v['value'] for v in c['gapped_wrong_lower_right_outputs']]}</td><td>Wrong positional assumption</td></tr></table>"),
        ("LATENCY","Original prefix means avoid repeated scalar filtering only for the uniform-logit prefix toy",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Prefix sums rebuilt each operation;QKmatmul/softmax/general attention/nativeSDPA/GPU/cache movement excluded.</p>"),
        ("MY TAKE","Record query/key positions and mask API semantics before changing causal-attention calls",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Rectangular mask audits,cache suffix assumptions,boolean conversions and tagged empty-row controls.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Universal is_causal alignment,shape-only cache causality,prefix-mean speed as SDPA speed or native all-masked claims.</p></div></div>"
         "<div class='dark'>BSD-stylePyTorchsource;original exact model/oracle/art.NativeSDPA latency/all-masked behavior/quality/HBM unknown;browser inspection not performed.</div>"),
    ]
    return page("Non-square causal-mask contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Causal-position ticket sorter:query/key positions and mask polarity</title>
<desc>Original sixty-solid sorting cabinet metaphor for allowed-key tickets and query position limits.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG130 · NON-SQUARE CAUSAL MASKS</text>
<text x="88" y="158" font-size="36" font-weight="700">Causality follows positions,not rectangle shape</text>{sorter_scene()}
<text x="88" y="710" font-size="25">SDPA non-square is_causal is upper-left;lower-right assumes a suffix</text>
<text x="88" y="758" font-size="25">SDPA True allows;MHA True blocks—the same bits can invert policy</text>
<text x="88" y="806" font-size="25">Gapped/reordered keys need actual positions;empty rows stay explicit</text>
<text x="88" y="880" font-size="18">Original exact mask / uniform-logit scalar means / CPU timing · no native attention or GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
