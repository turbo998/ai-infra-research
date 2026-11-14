import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="rocm-minimax-xcd-balanced-tail-one-round-placement"
JS=r"""
function integer(x,lo,hi){return Number.isInteger(x)&&x>=lo&&x<=hi;}
function plan(p,strategy="compare"){
 if(!p||Object.keys(p).sort().join(",")!=="chiplets,cus,explicit_splits,kv_heads,query_rows,requests,topk")throw Error("Exact original fields required");
 const limits={requests:[0,64],query_rows:[1,32],chiplets:[0,16],cus:[1,4096],topk:[1,32],kv_heads:[1,4]};
 if(Object.entries(limits).some(([k,[lo,hi]])=>!integer(p[k],lo,hi))||(p.chiplets&&p.cus%p.chiplets!==0))throw Error("Bounded equal-CU topology required");
 if(p.explicit_splits!==null&&!integer(p.explicit_splits,1,4))throw Error("Null or1..4override required");
 if(!["compare","request_only"].includes(strategy))throw Error("Known strategy required");
 const base={status:"outside_tuned_profile",selected_splits:null,placement:null,valid_rows_per_chiplet:null,grid_rows:null,launched_workgroups:null,valid_workgroups:null,padding_workgroups:null,max_valid_workgroups_per_chiplet:null,one_round:null,native_followup:null,native_gpu_verified:null,native_workspace_bytes:null};
 if(p.requests===0)return {...base,status:"no_work",grid_rows:0,launched_workgroups:0,valid_workgroups:0,padding_workgroups:0};
 if(p.chiplets<=1||p.topk!==16||p.query_rows!==8||p.kv_heads!==1)return base;
 function own(kind){const n=p.requests,q=p.query_rows,x=p.chiplets;
 if(kind==="request")return [Array.from({length:x},(_,i)=>q*(Math.floor(n/x)+(i<n%x?1:0))),Math.ceil(n/x)*x*q];
 const full=Math.floor(n/x)*q,tail=(n%x)*q,per=Math.ceil(tail/x);
 return [Array.from({length:x},(_,i)=>full+Math.max(0,Math.min(per,tail-i*per))),(full+per)*x];}
 const capacity=p.cus/p.chiplets,candidates=[];
 for(const kind of strategy==="compare"?["request","balanced"]:["request"]){const [counts,grid]=own(kind);
 for(let s=1;s<=4;s++)if(Math.max(...counts)*s<=capacity)candidates.push([s,kind,counts,grid]);}
 let selected;
 if(p.explicit_splits!==null){const [counts,grid]=own("request");selected=[p.explicit_splits,"request",counts,grid];}
 else if(candidates.length){candidates.sort((a,b)=>a[0]-b[0]+(a[0]===b[0]?(a[1]==="request"?1:0)-(b[1]==="request"?1:0):0));selected=candidates[candidates.length-1];}
 else return {...base,status:"outside_one_round",native_followup:p.requests<=48?"occupancy_2_candidate":"plain_cost_model"};
 const [s,kind,counts,grid]=selected,valid=p.requests*p.query_rows*s;
 return {...base,status:p.explicit_splits===null?"one_round_candidate":"explicit_override",selected_splits:s,placement:kind,valid_rows_per_chiplet:counts,grid_rows:grid,launched_workgroups:grid*s,valid_workgroups:valid,padding_workgroups:grid*s-valid,max_valid_workgroups_per_chiplet:Math.max(...counts)*s,one_round:Math.max(...counts)*s<=capacity};
}
"""


def workbench(r):
    body="<h1>Balance the incomplete request tail without confusing valid rows with padded launches</h1><p>Original bounded row ownership and one-round search.Native FlyDSL,NVFP4/MXFP8,occupancy2and cost-model branches are not executed.No GPUperformance/bitwise claim.</p>"
    body+=f'<form id="form"><label>Declared topology/profile<textarea id="input" rows="10" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Inspect row ownership and candidates</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
    body+="<p>Request placement wins equal-split ties for locality.Explicit split overrides use request placement and may violate one-round capacity.Source-reported GPUtimes are not benchmark coordinates here.</p>"
    return page("XCDtail placement workbench",body,JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),request_only_candidate:plan(p,"request_only")},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["request_only"]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],"One-round scope",str(x["outputs"]["selected_splits"])+" / "+str(x["outputs"]["placement"]),
        "✕ scope/override review" if x["outputs"]["one_round"] is not True else "Confirm native kernel"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["analytic_median_ms"],x["enumerated_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['rows']}original rows</b><p>Analytic ownership{x['analytic_median_ms']:.6f}ms</p>{bar(x['analytic_median_ms'],mx)}<p>Grid enumeration{x['enumerated_median_ms']:.6f}ms</p>{bar(x['enumerated_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · CHIPLET PLACEMENT","Whole request groups keep locality;an incomplete tail can gain balanced row ownership",
         "<div class='grid'><div>Scenarios<br>Full groups/tail<br>One-round splits<br>Unsupported/override</div><div>Input<br>10requests×8rows<br>8chiplets×32CUs<br>Topk16/oneKVhead</div>"
         f"<div>Decision<br>{a['selected_splits']}balanced splits<br>{a['grid_rows']}grid rows<br>Native GPUunknown</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14source;original CPUownership separately measured on Python{r['latency']['python']}/{r['latency']['machine']}.No AMDGPU/FlyDSL.</div>"),
        ("SCENARIO: TEN REQUESTS",f"Original balanced tail enables{a['selected_splits']}one-round splits versus{b['selected_splits']}request-only splits",
         "<blockquote>Declared8chiplets/256CUs,80valid query rows.Balanced owns10rows/chiplet;request-only busiest owns16.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm actual kernel/locality · confidence unknown</span><p>Balanced{a['launched_workgroups']}launches/{a['padding_workgroups']}padding;request-only{b['launched_workgroups']}launches/{b['padding_workgroups']}padding.</p></div>"
         f"<div class='metric'><span class='pill'>score · valid workgroups on busiest chiplet</span><p>Balanced{a['max_valid_workgroups_per_chiplet']}</p>{bar(a['max_valid_workgroups_per_chiplet'],32)}<p>Request-only{b['max_valid_workgroups_per_chiplet']}</p>{bar(b['max_valid_workgroups_per_chiplet'],32)}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original profiles distinguish eligibility,one-round boundary and explicit overrides",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Tuned source gate:positive XCDs,topk16,8query rows and one KVhead.</div><div class='callout'>Beyond one round,native occupancy2/cost-model branches remain outside this experiment;unknown selection stays null.</div></div>"),
        ("UNDER THE HOOD","Compare feasible split counts first;request placement wins equal-count ties",
         "<pre>Preserve whole groups of XCD requests\nSpread only residual rows in contiguous runs\nCompute valid rows per chiplet\nTest splits1..4against CUcapacity\nPrefer request locality on equal split count</pre><p>↓ Launch accounting</p>"
         "<pre>Grid may include masked padding rows\nLaunched WG=grid rows×splits\nValid WG=request rows×splits\nNo one-round plan:scope boundary,not zero\nNative occupancy/register/latency unknown</pre>"),
        ("ROBUSTNESS CHECK","Explicit4splits on request placement exceed the original32CUone-round capacity",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>10requests</td><td>3balanced</td><td>2request-only</td><td>Tail distribution matters</td></tr>"
         "<tr><td>8requests tie</td><td>4request</td><td>4balanced feasible</td><td>Prefer locality</td></tr>"
         "<tr><td>Explicit4</td><td>30valid WGs/chiplet</td><td>64on busiest</td><td>Not one round</td></tr>"
         "<tr><td>33requests</td><td>One-round scope</td><td>Native occupancy2candidate</td><td>Not reproduced</td></tr></table>"),
        ("LATENCY","Original analytic row tally excludes GPUlaunch,loads,split-combine and occupancy",timing+
         "<p>2warmups,7alternating repeats,3batches/sample;median repeat means.Analytic8bin counts versus original per-grid-row enumeration.No native latency coordinates.</p>"),
        ("MY TAKE","Choose placement with locality,tail balance and padded-work accounting,then verify the actual GPU",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Explicit topology,source eligibility,bijective row mapping,feasible split choices and independent native correctness/performance evidence.</div><div class='bad'><h3>⛔ Not for</h3>Assuming all KVhead/query profiles tuned,launch count equals useful work,one-round proxy proves speed,or extrapolating CPUtally to MI355X.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Native source comments report GPUranges but no unsealed reported timings are archived as coordinates.Native output/occupancy/workspace/visual acceptance unknown.</div>")]
    return page("XCDtail placement — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["request_only"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Chiplet tail balancing cabinet</title><desc>Original sixty-solid cabinet;whole drawers retain request grouping while residual rows spread across chiplet bins.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG170 · CHIPLET TAIL BALANCING CABINET</text><text x="88" y="158" font-size="34">Balance residual rows while keeping whole request groups local</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original80rows / balanced3splits /{a['launched_workgroups']}launches /{a['padding_workgroups']}padding WGs</text><text x="88" y="760" font-size="24">Request-only2splits /{b['launched_workgroups']}launches /{b['padding_workgroups']}padding WGs</text><text x="88" y="810" font-size="24">One-round capacity and valid-row ownership are not measured GPUlatency</text><text x="88" y="880" font-size="18">ActualNov14source · original integer mapping · no AMDGPU/FlyDSL/NVFP4/MXFP8/visual acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
