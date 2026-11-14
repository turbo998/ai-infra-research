import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page, bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "grouped-moe-padding-mask-coverage-sentinel"
JS = r"""
class LegacyCoverageError extends Error{}
function validate(p){
 const fields="aiter_fused,consumer_skip_sentinel,context,env_skip,expert_coefficients,groups,ids,model_optin,padding,topk,values,weights";
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!==fields)throw Error("Exact original route fields required");
 for(const k of ["env_skip","context","aiter_fused","model_optin","consumer_skip_sentinel"])if(typeof p[k]!=="boolean")throw Error("Explicit booleans required");
 for(const k of ["groups","topk"])if(!Number.isInteger(p[k])||p[k]<1||p[k]>32)throw Error("groups/topk1..32required");
 if(!Array.isArray(p.expert_coefficients)||!p.expert_coefficients.length||p.expert_coefficients.length>32||p.expert_coefficients.some(v=>!Number.isInteger(v)||v<1||v>64))throw Error("Bounded positive expert coefficients required");
 if(p.topk>p.expert_coefficients.length)throw Error("topk exceeds domain");
 if(!Array.isArray(p.values)||p.values.length>128||p.values.some(v=>!Number.isInteger(v)||v<0||v>1024))throw Error("Bounded row values required");
 for(const k of ["ids","weights"])if(!Array.isArray(p[k])||p[k].length!==p.values.length||p[k].some(r=>!Array.isArray(r)||r.length!==p.topk))throw Error("Nbytopkarrays required");
 if(p.ids.some(r=>r.some(v=>!Number.isInteger(v)||v<0||v>=p.expert_coefficients.length)||new Set(r).size!==r.length))throw Error("Unique nonnegative precomputed ids required");
 if(p.weights.some(r=>r.some(v=>!Number.isInteger(v)||v<0||v>16)))throw Error("Bounded weight numerators required");
 if(p.padding!==null&&(!Array.isArray(p.padding)||p.padding.length>1024||p.padding.some(v=>typeof v!=="boolean")))throw Error("Null or explicit boolean mask required");
}
function dispatch(values,ids,weights,coefficients){
 let calls=0;const result=values.map((v,i)=>{let total=0;ids[i].forEach((e,j)=>{
 if(e===-1){if(weights[i][j]!==0)throw Error("Skip sentinel requires zero weight");return;}
 if(!Number.isInteger(e)||e<0||e>=coefficients.length)throw Error("Unsupported expert id;no negative index fallback");
 calls++;total+=v*coefficients[e]*weights[i][j];});return total;});return [result,calls];
}
function route(p,old=false){
 validate(p);const n=p.values.length,e=p.expert_coefficients.length,grouped=e>p.groups&&e%p.groups===0,
 eligible=old?p.model_optin:p.model_optin||!p.aiter_fused;let mask=null,reason;
 if(!grouped)reason="ungrouped_early_return";
 else if(!eligible)reason="router_mask_not_enabled";
 else if(!p.env_skip)reason="environment_off";
 else if(!p.context)reason="no_forward_context";
 else if(p.padding===null)reason="no_padding_witness";
 else if(p.padding.length<n){if(old)throw new LegacyCoverageError("Legacy undercovered mask unexecuted;native outcome unknown");reason="mask_undercoverage_return_none";}
 else{mask=p.padding.slice(0,n);reason="covered_ordered_row_witness";}
 if(mask!==null&&mask.some(Boolean)&&!p.consumer_skip_sentinel)throw Error("Original consumer lacks -1skip capability");
 const ids=p.ids.map((r,i)=>r.map(v=>mask!==null&&mask[i]?-1:v)),weights=p.weights.map((r,i)=>r.map(v=>mask!==null&&mask[i]?0:v)),
 [values,calls]=dispatch(p.values,ids,weights,p.expert_coefficients);
 return {router_masks_padding_property:eligible,valid_grouping:grouped,mask_applied:mask!==null,reason,ids,weight_numerators:weights,
 output_numerators:values,original_scalar_expert_calls:calls,original_weight_unit:10,logical_int32_fp32_route_bytes:n*p.topk*8,native_topk_logits_or_expert_execution_verified:null};
}
"""


def workbench(r):
    return page("Grouped MoE padding/coverage ledger",
        "<h1>Padding eligibility,complete row coverage and -1consumer support are different checks</h1>"
        "<p>Round159 · Original precomputed routes and integer weight numerators/unit10.No logits,grouped topk,GEMM,EPLB,AITERorDeepEPexecution.</p>"
        f'<form id="form"><label>Ordered precomputed routes,context mask and capability witnesses<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare mask policy and scalar dispatch</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Native environment default1already existed.This change makes valid grouped non-AITERrouting consult it without model opt-in."
        "AITERstill needs caller opt-in;invalid grouping returns earlier.Short local masks are not applied to wider all-gathered rows."
        "Sentinel skip does not shrink the route arrays or certify GPUwork savings.</p>",
        JS + r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const p=JSON.parse(document.getElementById("input").value),after=route(p);
 let before;try{before=route(p,true);}catch(e){if(!(e instanceof LegacyCoverageError))throw e;
 before={status:"original_legacy_undercoverage_unexecuted",native_outcome:null};error.textContent=e.message;}
 out.textContent=JSON.stringify({before,after},null,2);}catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    s = r["summary"]
    matrix = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"], x["after"]["reason"], x["after"]["original_scalar_expert_calls"],
        "✕ old padding work" if x["before"]["ids"] != x["after"]["ids"] else "Confirm scope"))
        + "</tr>" for x in r["matrix"])
    mx = max(max(v["skip_median_ms"], v["zero_weight_median_ms"]) for v in r["latency"]["rows"])
    timing = "".join(f"<div class='metric'><b>{v['rows']}original rows</b><p>Sentinel skip{v['skip_median_ms']:.6f}ms</p>{bar(v['skip_median_ms'],mx)}"
                    f"<p>Zero-weight compute{v['zero_weight_median_ms']:.6f}ms</p>{bar(v['zero_weight_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections = [
        ("HANDS-ON · MOE ROUTING", f"Original expert calls{s['before_expert_calls']}→{s['after_expert_calls']};route payload stays{s['logical_route_bytes']}B",
         "<div class='grid'><div><h3>Scenarios</h3><p>Default non-AITER<br>AITERopt-in<br>Local/global mask coverage</p></div>"
         "<div><h3>Input</h3><p>Precomputed IDs/weights<br>Ordered row witnesses<br>Environment and grouping</p></div>"
         "<div><h3>Decision</h3><p>−1with zero weight<br>Ignore incomplete mask<br>Preserve live outputs</p><span class='pill'>Threshold:0live-output misses</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14grouped-routing correction.CPUscalar dispatch onPython{r['latency']['python']}/{r['latency']['machine']},not native topk or GEMM.</div>"),
        ("SCENARIO: DEFAULT GROUPED", f"Four padded rows lose eight original assignments;live-output misses{s['live_output_misses']}/{s['live_rows']}",
         "<blockquote>masks_padding = model_optin OR NOT AITER_fused.Then require the existing env switch,forward context and a mask covering every routed row.</blockquote>"
         "<div class='metric'><span class='pill'>choice · confirm skip-capable consumer</span><h3>Padded slots carry id−1and weight0</h3><p>Weight0alone is not an instruction to omit expert computation.</p></div>"
         f"<div class='metric'><span class='pill'>score · original assignment counts,not GPUflops</span><h3>Before{s['before_expert_calls']}</h3>{bar(s['before_expert_calls'],s['before_expert_calls'])}"
         f"<h3>After{s['after_expert_calls']}</h3>{bar(s['after_expert_calls'],s['before_expert_calls'])}</div>"),
        ("WHAT I TESTED", f"{s['changed_cases']}/{s['cases']}original cases change;AITER,environment and grouping remain independent",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Invalid grouping returns through ungrouped routing before this mask branch.</div><div class='callout'>Covered length is necessary,not proof that mask ordering matches all-gathered rows.</div></div>"),
        ("UNDER THE HOOD", "Non-AITERgrouped routing consults the existing env switch;short context masks returnNone",
         "<pre>Original branch model:\nvalid_grouping first\noptin OR NOT AITER_fused\nexisting env + forward context\nmask length ≥ router row count\nprefix mask only with covered row order\npadding → ids−1,weights0</pre><p>↓ Original scalar consumer</p>"
         "<pre>id−1with weight0 → skip\nid−1without consumer support → error\nzero weight with valid id → still compute\nlive rows remain unchanged\nroute arrays retain N×topkshape\nNo logits/EPLB/GEMMs executed</pre>"),
        ("ROBUSTNESS CHECK", "A two-row context mask cannot describe eight routed rows;no fabricated suffix padding is applied",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Covered mask</td><td>8row witnesses</td><td>2row witnesses</td><td>Mask returnsNone</td></tr>"
         "<tr><td>AITER</td><td>No model opt-in</td><td>Model opt-in</td><td>Eligibility changes</td></tr>"
         "<tr><td>Grouping</td><td>4experts/2groups</td><td>4experts/4groups</td><td>Early ungrouped return</td></tr>"
         "<tr><td>Consumer</td><td>−1skip supported</td><td>Unsupported</td><td>Original explicit error</td></tr>"
         "<tr><td>Weights</td><td>−1and0</td><td>Valid id and0</td><td>Same outputs,extra scalar calls</td></tr></table>"),
        ("LATENCY", "CPUscalar sentinel skipping is not a measured GEMM,DeepEPtraffic or graph-throughput improvement",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Both paths return identical integer numerators.Setup,topk,GEMMs,EPLB,CUDA/AITER/DeepEPandDPallgather excluded.</p>"),
        ("MY TAKE", "Validate row-domain coverage and sentinel semantics without equating masked routes with freed memory",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Default grouped policy,ordered complete masks,AITERopt-in,zero-weight versus skip semantics and unchanged live outputs.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Partial local masks on global rows,negative-index wrap,blanket ROCmclaims,ungrouped-path coverage,or assignment counts sold as native FLOPs.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,original route ledger.No native grouped topk,expert accuracy,DeepEPv2configuration,GPUlatency or allocator peak verified.No browser visual inspection.</div>"),
    ]
    return page("Grouped MoE padding — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    s=r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Grouped MoE route-sentinel gate</title><desc>Original sixty-solid lane-gate cabinet metaphor.Mask eligibility,row coverage and skip-capable consumption are separate.No native expert or visual acceptance evidence.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG159 · A ROUTE GATE IS NOT A MEMORY DELETION</text>
<text x="88" y="158" font-size="34" font-weight="700">Require complete row witnesses and a true skip sentinel</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original expert calls{s['before_expert_calls']}to{s['after_expert_calls']};live-output misses{s['live_output_misses']}/{s['live_rows']}</text>
<text x="88" y="760" font-size="24">Route payload stays{s['logical_route_bytes']}B;weight0alone does not omit expert calls</text>
<text x="88" y="810" font-size="24">AITERopt-in,env switch,valid grouping and complete mask remain independent</text>
<text x="88" y="880" font-size="18">ActualNov14correction · precomputed route ledger,not native topk/GEMM/DeepEPor GPUthroughput</text></g></svg>
"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
