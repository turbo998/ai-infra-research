import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="ordered-affine-prefix-scan-segment-reset"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==5||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![1,2,4,8].includes(p.element_bytes))throw Error("Positive exact dimensions and selected byte width required");
 const b=BigInt(p.batch),l=BigInt(p.length),c=BigInt(p.channels),s=BigInt(p.state_width),e=BigInt(p.element_bytes);
 const out={selected_last_state_bytes:Number(b*c*s*e),selected_materialized_all_states_bytes:Number(b*l*c*s*e),selected_one_affine_prefix_pair_bytes:Number(2n*b*c*s*e)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(values,initial,resetIndex){
 if(values.length<1||values.length>32||!values.every(p=>Array.isArray(p)&&p.length===2&&p.every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=16))||!Number.isSafeInteger(initial)||Math.abs(initial)>1000000||!Number.isInteger(resetIndex)||resetIndex< -1||resetIndex>=values.length)throw Error("Bounded exact integer pairs,initial and reset index required");
 const pairs=values.map(([a,b],i)=>[i===resetIndex?0:a,b]);
 const compose=(old,later)=>{const a=later[0]*old[0],product=later[0]*old[1],b=product+later[1];if(!Number.isSafeInteger(a)||!Number.isSafeInteger(product)||!Number.isSafeInteger(b))throw Error("Exact composition range exceeded");return[a,b]};
 let work=pairs.map(p=>p.slice());for(let step=1;step<work.length;step*=2){const prev=work.map(p=>p.slice());for(let i=step;i<work.length;i++)work[i]=compose(prev[i-step],prev[i])}
 let state=initial;const serial=pairs.map(([a,b])=>{state=a*state+b;if(!Number.isSafeInteger(state))throw Error("Exact state range exceeded");return state});
 const scan=work.map(([a,b])=>{const product=a*initial;if(!Number.isSafeInteger(product)||!Number.isSafeInteger(product+b))throw Error("Exact state range exceeded");return product+b});
 return {serial,scan,prefix_pairs:work,reset_before_index:resetIndex};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Ordered affine scan and zero-reset workbench",
        f"""<h1>Associative composition preserves sequence order; reset is an explicit affine map</h1>
        <p>Round127 · Original scalar exact integer toy,not native Mamba or GPU parallel execution.
        Recurrence h[t]=a[t]*h[t-1]+b[t].</p><form id="form">
        <label>Pairs as JSON<input id="pairs" value="[[2,1],[3,4],[4,2],[5,3]]" required></label>
        <label>Initial state<input id="initial" type="number" value="7" step="1" required></label>
        <label>Zero-reset before index (-1 means none)<input id="reset" type="number" value="2" step="1" required></label>
        <h2>Independent selected state payload budget</h2><div class="grid">{fields}</div><button>Compare serial and prefix scan</button>
        </form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Identity padding is(1,0),not(0,0). Zero reset replaces selected a with0,an explicit original
        packed-sequence extension rather than a claimed native reset API. Exact integer range is guarded.
        Floating-point parenthesization can differ even when the real-number algebra is associative.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const pairs=JSON.parse(document.getElementById("pairs").value);if(!Array.isArray(pairs))throw Error("Pairs must be an array");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(pairs,Number(document.getElementById("initial").value),Number(document.getElementById("reset").value)),budget:budget(p),
 note:"Original scalar algebra only; no native selectivescan/GPU/quality reproduction"},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["tree_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>Length{v['length']}</b><p>Serial recurrence {v['sequential_median_ms']:.6f}ms</p>{bar(v['sequential_median_ms'],mx)}"
                  f"<p>Python Hillis-Steele {v['tree_median_ms']:.6f}ms</p>{bar(v['tree_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SSM AFFINE SCAN","Ordered prefix composition matches all4 recurrence states; reversing the operator matches only1",
         "<div class='grid'><div><h3>Scenarios</h3><p>Order and associativity<br>Reset and padding<br>Chunk carry</p></div>"
         "<div><h3>Input</h3><p>Pairs(2,1),(3,4),(4,2),(5,3)<br>Initial7</p></div>"
         "<div><h3>Top decisions</h3><p>Preserve temporal order<br>Reset with a=0<br>Pad with identity</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Established selective-scan affine operator,not new November launch. Original CPU Python{r['latency']['python']} / {r['latency']['machine']};synthetic,independently_reproduced=false.</div>"),
        ("SCENARIO: ORDERED COMPOSITION","Correct scan yields[15,49,198,993]; reversed composition changes three states",
         f"<blockquote>Correct{c['scan_states']} versus reverse{c['reversed_composition_states']};independent product-expansion oracle matches correct recurrence.</blockquote>"
         f"<div class='metric'><span class='pill'>score · deterministic state equality,not confidence</span><h3>Correct{c['states']}/{c['states']}</h3>{bar(c['states'],c['states'])}<p>Act automatically within exact algebra</p></div>"
         f"<div class='metric'><span class='pill'>score · deterministic state equality,not confidence</span><h3>Wrong order{c['reversed_matching_states']}/{c['states']}</h3>{bar(c['reversed_matching_states'],c['states'],True)}<p>✕ Confirm composition direction</p></div>"),
        ("WHAT I TESTED","Reset,padding and chunk carry encode different state-lifetime decisions",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Reset before index2</td><td>Final state?</td><td>{c['reset_final_state']},not{c['no_reset_leaked_final_state']}</td><td>Reset a=0</td></tr>"
         f"<tr><td>Identity padding</td><td>Final state retained?</td><td>{c['identity_padded_last_state']}</td><td>Use(1,0)</td></tr>"
         f"<tr><td>Zero pair padding</td><td>Harmless?</td><td>✕ Final{c['zero_pair_padded_last_state']}</td><td>Not identity</td></tr>"
         "<tr><td>Chunk boundary</td><td>Restart from initial?</td><td>✕ Loses previous prefix</td><td>Carry state</td></tr>"
         "<tr><td>Float arithmetic</td><td>Bitwise identical?</td><td>Not established</td><td>Separate tolerance audit</td></tr></table>"
         "<div class='grid two'><div class='callout'>Zero-reset packing is our explicit extension,not claim of native reset API.</div><div class='callout'>Scalar affine algebra is not full Mamba discretization/gating/output/backward.</div></div>"),
        ("UNDER THE HOOD","Affine maps form an ordered associative monoid,not a commutative sum",
         "<pre>earlier:(a,b), later:(c,d)\ncombined map:x → c*(a*x+b)+d\npair:(c*a,c*b+d)\nidentity:(1,0)\n\nzero reset before element:(0,b)\nprefix(a,b) applied to h0:a*h0+b</pre>"
         "<p>↓ Original independent oracle</p><pre>h[t] = h0*product(a[0:t+1])\n + sum_j b[j]*product(a[j+1:t+1])\n\nHillis-Steele reads previous round buffer\nnever mix updated and old prefixes</pre>"),
        ("ROBUSTNESS CHECK","Chunk carry preserves993; restarting each two-element chunk produces153",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Chunked states</td><td>{c['carried_chunk_states']}</td><td>{c['wrong_chunk_restart_states']}</td><td>State values,not probability</td></tr>"
         f"<tr><td>Zero reset before2</td><td>{c['serial_states']}</td><td>{c['zero_reset_states']}</td><td>Packed segment boundary</td></tr></table>"
         "<p>Exact Fraction states,nonzero initial7. Native floating accumulation order/GPU output not reproduced.</p>"),
        ("LATENCY","Original serial Python recurrence is faster than Python tree scan; theoretical parallel span is not CPU wall time",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Python tree performs extra rounds/copies and is not GPU parallelism. Native Mamba/discretization/backward excluded.</p>"),
        ("MY TAKE","Preserve order,identity and segment boundaries before optimizing selective recurrence",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Affine operator audits,chunk carries,zero-reset packed-state and padding controls.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Commutative reordering,zero-pair identity,bitwise floating parity or native Mamba/GPU speed claims.</p></div></div>"
         "<div class='dark'>Apache-2.0Mamba source;original scalar model/oracle/art. JulyMamba3Blackwell fix not executed;not the same kernel as classic scan header. Native quality/latency/HBM unknown;browser inspection not performed.</div>"),
    ]
    return page("Ordered affine prefix scan — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Ordered affine-state ticket sorter: composition,reset and identity padding</title>
<desc>Original sixty-solid sorting cabinet metaphor for ordered prefix carry and segment ownership.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG127 · ORDERED AFFINE PREFIX SCAN</text>
<text x="88" y="158" font-size="36" font-weight="700">Associative does not mean reorderable</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Compose earlier then later;apply full prefix to selected initial state</text>
<text x="88" y="758" font-size="25">Zero reset uses a=0;identity padding uses pair(1,0)</text>
<text x="88" y="806" font-size="25">Carry state across chunks;packed segments need explicit ownership</text>
<text x="88" y="880" font-size="18">Original Fraction algebra / expanded-product oracle / CPU timing · no native Mamba or GPU reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
