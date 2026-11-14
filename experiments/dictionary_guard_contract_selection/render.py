import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="dynamo-dictionary-order-type-length-kwargs-guards"
JS=r"""
const programs=["named","order","unexpected","duplicate","type","constructor"];
function integer(v){if(!Number.isInteger(v)||Math.abs(v)>1000000)throw Error("Bounded exact scalar integer required");return v;}
function validate(prog,c){
 if(!c||typeof c!=="object"||Array.isArray(c)||Object.keys(c).sort().join(",")!=="base,mapping_type,pairs"||!["dict","OrderedDict"].includes(c.mapping_type))throw Error("Exact original mapping record required");
 integer(c.base);if(!Array.isArray(c.pairs)||c.pairs.length>16)throw Error("Maximum16pairs required");
 const keys=[];for(const p of c.pairs){if(!Array.isArray(p)||p.length!==2||typeof p[0]!=="string"||!/^[a-z][a-z0-9_]{0,15}$/.test(p[0]))throw Error("Bounded ASCII keyword/value pairs required");integer(p[1]);keys.push(p[0]);}
 if(new Set(keys).size!==keys.length)throw Error("Original storage keys must be unique");
 if(["named","type","duplicate"].includes(prog)&&keys.some(k=>!["alpha","beta"].includes(k)))throw Error("Original named keys alpha/beta required");
 if(prog==="unexpected"&&keys.some(k=>["alpha","beta"].includes(k)))throw Error("Only unexpected names required");return keys;
}
function result(prog,c,plannedKeys=null,plannedType=null){
 const keys=plannedKeys===null?c.pairs.map(p=>p[0]):plannedKeys,v=Object.fromEntries(c.pairs);
 let value=null,order=null,error_kind=null,error_key=null,mapping_type=null;
 if(["unexpected","duplicate"].includes(prog)&&keys.length){error_kind=prog;error_key=keys[0];}
 else if(["order","constructor"].includes(prog)){value=c.base+c.pairs.reduce((a,p)=>a+p[1],0);order=[...keys];}
 else if(prog==="duplicate")value=c.base+120;else value=c.base+10*(v.alpha||0)+(v.beta||0);
 if(prog==="type")mapping_type=plannedType===null?c.mapping_type:plannedType;
 return {value,order,error_kind,error_key,mapping_type};
}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="calls,program"||!programs.includes(p.program))throw Error("Supported original program/calls required");
 if(!Array.isArray(p.calls)||p.calls.length<1||p.calls.length>16)throw Error("Original1..16calls required");
 const observes=["order","constructor","unexpected","duplicate"].includes(p.program),guarded=new Map(),naive=new Map(),rows=[];
 for(const [i,c] of p.calls.entries()){
  const keys=validate(p.program,c),set=[...keys].sort(),g=JSON.stringify([c.mapping_type,keys.length,observes?keys:set]),n=JSON.stringify(set);
  const expected=result(p.program,c),gh=guarded.has(g),nh=naive.has(n);if(!gh)guarded.set(g,[keys,c.mapping_type]);if(!nh)naive.set(n,[keys,c.mapping_type]);
  const correct=result(p.program,c,...guarded.get(g)),bad=result(p.program,c,...naive.get(n));
  rows.push({call:i,inputs:c,expected,guarded:correct,naive:bad,guarded_cache_hit:gh,naive_cache_hit:nh,naive_matches:JSON.stringify(bad)===JSON.stringify(expected)});
 }
 return {program:p.program,observes_order:observes,rows,guarded_plans:guarded.size,naive_plans:naive.size,
 guarded_misses:rows.filter(r=>JSON.stringify(r.guarded)!==JSON.stringify(r.expected)).length,naive_misses:rows.filter(r=>!r.naive_matches).length,native_torch_frame_count:null};
}
"""


def workbench(r):
    return page("Dictionary semantic-guard cache workbench",
        f"<h1>A key set does not preserve observed order,type or exception identity</h1><p>Round152 · Original dynamic-scalar plan cache,not torch.compile.Native CPython-only observations are separately sealed.</p>"
        f'<form id="form"><label>Program and ordered mapping-call records<textarea id="input" rows="9" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare selective semantic guards with key-set-only cache</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Program named binds alpha/beta by name.Order/constructor and modeled binding errors observe insertion order.Type observes actual mapping kind."
        "Values are dynamic in this original model;native Dynamo Python-value/slot/identity guards are not modeled.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const r=evaluate(JSON.parse(document.getElementById("input").value));out.textContent=JSON.stringify(r,null,2);
 if(r.naive_misses)error.textContent=r.naive_misses+"original key-set-only cache results violate observed semantics";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def compact(v):
    return v["error_kind"]+":"+v["error_key"] if v["error_kind"] else str(v["order"] if v["order"] is not None else v["mapping_type"] or v["value"])


def cards(r):
    s=r["summary"];matrix="".join("<tr>"+"".join(f"<td style='font-size:14px;padding:6px'>{html.escape(str(v))}</td>" for v in (
        case["program"],row["call"],compact(row["expected"]),
        "Confirm guard scope" if row["naive_matches"] else "✕ Key-set-only stale"))+"</tr>" for case in r["matrix"] for row in case["rows"])
    mx=max(max(v["tuple_median_ms"],v["scalar_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['keys']}same-order keys</b><p>Tuple comparison {v['tuple_median_ms']:.6f}ms</p>{bar(v['tuple_median_ms'],mx)}"
                   f"<p>Scalar zip {v['scalar_median_ms']:.6f}ms</p>{bar(v['scalar_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SEMANTIC GUARDS",f"Key-set-only caching misses{s['naive_misses']}/{s['calls']}original calls;semantic guards miss none",
         "<div class='grid'><div><h3>Scenarios</h3><p>Fixed-name binding<br>Observed insertion order<br>Type/error identity</p></div>"
         "<div><h3>Input</h3><p>Original ordered pairs<br>Six small programs<br>Dynamic scalar values</p></div>"
         f"<div><h3>Decisions</h3><p>Guard what is observed<br>Use real type/storage length<br>Honor mapping protocol</p><span class='pill'>{s['guarded_plans']}model plans;native frames unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.ActualNov14Dynamo correction.CPython-only oracle and original Python{r['latency']['python']}/{r['latency']['machine']}guard timer separately measured.No Torch compilation.</div>"),
        ("SCENARIO: REVERSED KEYWORDS","Reversing alpha/beta preserves named value28,but changes observed kwargs order",
         "<blockquote>Named binding:base3+10×alpha2+beta5=28,regardless insertion order.Order-observing code returns different key sequences.</blockquote>"
         "<div class='metric'><span class='pill'>choice · confirm observable contract</span><h3>Named case:one model plan</h3><p>Do not demand order for valid fixed-name binding.</p></div>"
         "<div class='metric'><span class='pill'>score · original misses,not confidence</span><h3>Order case:key-set cache misses1/2calls</h3>"+bar(1,2)+"</div>"),
        ("WHAT I TESTED",f"Twelve original calls use{s['guarded_plans']}guarded versus{s['naive_plans']}naive plans;fewer plans alone is not correctness",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Binding exceptions expose first unexpected/duplicate keyword;order still matters.</div>"
         "<div class='callout'>Model plan counts are not native Dynamo frame counts.</div></div>"),
        ("UNDER THE HOOD","Structural dict length bypasses user __len__;actual type bypasses spoofed __class__",
         "<pre>Original guard key:\n(mapping kind,storage length,\n observed ordered keys OR key set)\nNative CPython observations:\n len override77;dict.__len__2\n __class__spoof:actual type still dict subclass</pre>"
         "<p>↓ Original guard-plan verifier</p><pre>Validate unique typed pairs\nSelect order guard for observed order/errors\nRetain actual mapping-kind gate\nUse runtime scalar values\nCustom kwargs:keys()+getitem protocol\nNo native compiler/value/slot guards simulated</pre>"),
        ("ROBUSTNESS CHECK","Custom mapping kwargs computes73instead of a storage shortcut28;key order changes exception identity",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Custom mapping</td><td>Storage alpha2/beta5</td><td>Protocol alpha7</td><td>28→73CPython</td></tr>"
         "<tr><td>Length</td><td>User77</td><td>Structural2</td><td>Different contracts</td></tr>"
         "<tr><td>Unexpected key</td><td>omega first</td><td>sigma first</td><td>Error key changes</td></tr>"
         "<tr><td>Mapping type</td><td>dict</td><td>OrderedDict</td><td>Observed kind changes</td></tr></table>"),
        ("LATENCY","Original same-order tuple and scalar guards are not Dynamo compile or C++guard latency",
         timing+"<p>Both check length and ordered equality.2warmups,7alternating repeats,3operations/sample;median repeat means.Expected/input keys prepared before timing.</p>"),
        ("MY TAKE","Selective guards preserve observable behavior without forcing insertion-order guards on every named call",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Compiler/cache contract reasoning,ordered kwargs and errors,actual type,storage length and native mapping protocol checks.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Declaring native recompilation counts from the model,omitting Python value/slot/identity guards or bypassing custom mapping callbacks.</p></div></div>"
         "<div class='dark'>BSD-3-Clause rights retained,no implementation mirror.Native CPython-only facts are not Torch compilation.Native frames/compile latency/GPU unknown.Original figure;no visual browser inspection.</div>"),
    ]
    return page("Dictionary semantic guards — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    s=r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Dictionary semantic-guard cabinet with separate order,type and storage-length latches</title><desc>Original sixty-solid cabinet metaphor;observable order and binding errors need different guards from fixed-name binding.Native CPython facts are separate from the synthetic cache plan.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG152 · SAME KEY SET DOES NOT IMPLY SAME OBSERVABLE SEMANTICS</text>
<text x="88" y="158" font-size="34" font-weight="700">Guard insertion order when observed;retain actual type and structural length</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original12calls:key-set-only misses{s['naive_misses']};semantic guards miss{s['guarded_misses']}</text>
<text x="88" y="760" font-size="24">CPython kwargs custom mapping73versus storage shortcut28;user len77versus storage2</text>
<text x="88" y="810" font-size="24">Named target can ignore insertion order;kwargs/error/type observation cannot</text>
<text x="88" y="880" font-size="18">ActualNov14Dynamo correction · original dynamic-scalar cache/CPython oracle,no Torch compilation or native frame counts</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
