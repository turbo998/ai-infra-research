import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="vllm-context-kv-only-draft-metadata-layout"
JS=r"""
const CG={NEVER:0,UNIFORM_BATCH:1,ALWAYS:2};
function bounded(v,lo,hi){if(!Number.isSafeInteger(v)||v<lo||v>hi)throw Error("Bounded original integer required");}
function name(v){if(typeof v!=="string"||!/^[A-Za-z0-9_.-]{1,32}$/.test(v))throw Error("Bounded ASCIIgroup/layer name required");}
function supports(g,k){return g.fixed.includes(k)||g.multiples.some(m=>k%m===0);}
function choose(manager,groups){for(let k=manager;k>=1;k--)if(manager%k===0&&groups.every(g=>supports(g,k)))return k;throw Error("No common divisible kernel across all layout groups");}
function plan(p,k,skip){
 const width=Math.ceil(p.max_len/k),stride=p.manager_block/k;let owner=null,workspace=0;
 const rows=p.groups.map(g=>{const count=g.layers.every(n=>skip.has(n))?0:p.ubatches;if(count&&owner===null&&g.owns_workspace){owner=g.name;workspace=g.workspace_bytes;}
 return {group:g.name,layers:g.layers,builders:count,kernel_block:k,offset_entries:stride>1?stride:0,
 original_unpadded_table_bytes:stride>1?count*p.max_seqs*width*4:0,original_offset_bytes:stride>1?stride*4:0,cg:g.cg};});
 const active=rows.filter(g=>g.builders),sum=field=>rows.reduce((a,g)=>a+g[field],0);
 let cg="ALWAYS";for(const g of active)if(CG[g.cg]<CG[cg])cg=g.cg;
 return {groups:rows,builder_count:sum("builders"),metadata_layer_names:active.flatMap(g=>g.layers),attention_metadata_cg_support:cg,
 selected_workspace_owner:owner,selected_workspace_bytes:workspace,original_unpadded_table_bytes:sum("original_unpadded_table_bytes"),original_offset_bytes:sum("original_offset_bytes")};
}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="address,allocated_kernel,blocks,groups,manager_block,max_len,max_seqs,no_metadata_layers,ubatches")throw Error("Exact original fields required");
 for(const [f,lo,hi] of [["manager_block",1,1024],["blocks",1,64],["max_len",1,65536],["max_seqs",1,64],["ubatches",1,4]])bounded(p[f],lo,hi);
 if(!Array.isArray(p.groups)||p.groups.length<1||p.groups.length>16)throw Error("Original1..16groups required");
 const names=new Set(),layers=new Set();
 for(const g of p.groups){
  if(!g||typeof g!=="object"||Array.isArray(g)||Object.keys(g).sort().join(",")!=="cg,fixed,layers,multiples,name,owns_workspace,workspace_bytes")throw Error("Exact group fields required");
  name(g.name);if(names.has(g.name))throw Error("Unique group names required");names.add(g.name);
  if(!Array.isArray(g.layers)||g.layers.length<1||g.layers.length>4)throw Error("Original1..4layers required");
  for(const v of g.layers){name(v);if(layers.has(v))throw Error("Unique original layers required");layers.add(v);}
  for(const f of ["fixed","multiples"]){if(!Array.isArray(g[f])||g[f].length>16)throw Error("Bounded support declarations required");for(const v of g[f])bounded(v,1,1024);}
  if(!g.fixed.length&&!g.multiples.length)throw Error("Explicit kernel support required");
  if(typeof g.cg!=="string"||!Object.hasOwn(CG,g.cg)||typeof g.owns_workspace!=="boolean")throw Error("Original attention-only CG/owner metadata required");
  bounded(g.workspace_bytes,0,1048576);
 }
 const skip=p.no_metadata_layers;if(!Array.isArray(skip)||skip.some(v=>typeof v!=="string"||!layers.has(v))||new Set(skip).size!==skip.length)throw Error("Unique known skipped layers required");
 const common=choose(p.manager_block,p.groups);let k=p.allocated_kernel;
 if(k===null)k=common;else{bounded(k,1,p.manager_block);if(p.manager_block%k||!p.groups.every(g=>supports(g,k)))throw Error("Allocated kernel must remain valid for every layout group");}
 if(!p.address||typeof p.address!=="object"||Array.isArray(p.address)||Object.keys(p.address).sort().join(",")!=="block,position")throw Error("Exact address fields required");
 bounded(p.address.block,0,p.blocks-1);bounded(p.address.position,0,p.manager_block-1);
 const stride=p.manager_block/k,mapped={kernel_block_id:p.address.block*stride+Math.floor(p.address.position/k),position:p.address.position%k};
 const naive=choose(p.manager_block,p.groups.filter(g=>!g.layers.every(v=>skip.includes(v))));
 return {largest_common_kernel:common,allocated_kernel_used:k,stride,naive_remove_metadata_free_groups_kernel:naive,
 naive_kernel_valid_for_all_layout_groups:p.groups.every(g=>supports(g,naive)),
 same_linear_token_address:p.address.block*p.manager_block+p.address.position===mapped.kernel_block_id*k+mapped.position,
 mapped_address:mapped,before:plan(p,k,new Set()),after:plan(p,k,new Set(skip)),
 original_kv_payload_bytes:p.blocks*p.manager_block*layers.size*1*64*2*2,
 geometry:"original unpadded per-layer KV;1head,64dimensions,2byte dtype,K+V;not packed/native allocator",
 native_global_cudagraph_readiness:null,native_workspace_peak_bytes:null};
}
"""


def workbench(r):
    return page("Builder-free draft metadata and retained KVlayout planner",
        "<h1>Skip context-only draft metadata,not its KVlayout constraints</h1><p>Round155 · Original plain per-layer ledger,not vLLM/Torch/CUDAgraphs.</p>"
        f'<form id="form"><label>Groups,allocated layout,skipped layers and logical token address<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare builder and layout plans</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Only fully skipped groups get zero builders.Mixed active/skipped groups keep them.Attention-onlyALWAYS is not global graph readiness."
        "Unpadded table/payload ledger is original;native padding/packing/allocator unknown.Zero metadata builders still retain layout offsets.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");out.textContent=JSON.stringify(evaluate(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"];b=c["before"];a=c["after"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        row["name"],row["outputs"]["allocated_kernel_used"],row["outputs"]["after"]["builder_count"],
        "Confirm attention-only CG="+row["outputs"]["after"]["attention_metadata_cg_support"]))+"</tr>" for row in r["matrix"])
    mx=max(max(v["gated_median_ms"],v["eager_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['candidate_objects']}candidates,{v['active_objects']}retained</b><p>Gated{v['gated_median_ms']:.6f}ms</p>{bar(v['gated_median_ms'],mx)}"
                   f"<p>Eager{v['eager_median_ms']:.6f}ms</p>{bar(v['eager_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    zero=r["robustness"][0]["outputs"];small=r["robustness"][1]["outputs"]
    sections=[
        ("HANDS-ON · BUILDER/LAYOUT",f"Original builders fall{b['builder_count']}→{a['builder_count']};allocated kernel block remains{c['allocated_kernel_used']}",
         "<div class='grid'><div><h3>Scenarios</h3><p>Context-only draft<br>Mixed groups<br>Existing kernel layout</p></div>"
         "<div><h3>Input</h3><p>Manager256/kernel64<br>Target+draft constraints<br>Two ubatch builders</p></div>"
         "<div><h3>Decision</h3><p>Skip fully inactive metadata<br>Preserve cache views<br>Graph scope separate</p><span class='pill'>Native peak/quality unknown</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14vLLM correction.Original Python{r['latency']['python']}/{r['latency']['machine']}dummy constructor timing only.</div>"),
        ("SCENARIO: DRAFT METADATA",f"Original selected workspace changes{b['selected_workspace_bytes']}B→{a['selected_workspace_bytes']}B;layout offsets remain{a['original_offset_bytes']}logical bytes",
         "<blockquote>Context-K/V-only drafter does not need its own attention metadata,but still writes/reads KVusing allocated kernel blocks.</blockquote>"
         "<div class='metric'><span class='pill'>choice · confirm full-group skip</span><h3>Zero draft builders,no getter-owned draft workspace</h3></div>"
         f"<div class='metric'><span class='pill'>score · original selected workspace,not native peak</span><h3>Before{b['selected_workspace_bytes']}B</h3>{bar(b['selected_workspace_bytes'],b['selected_workspace_bytes'])}"
         f"<h3>After{a['selected_workspace_bytes']}B</h3>{bar(a['selected_workspace_bytes'],b['selected_workspace_bytes'])}<p>4096B is synthetic metadata,not an upstream workspace measurement.</p></div>"),
        ("WHAT I TESTED","Four original plans separate metadata participation from cache/kernel compatibility",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK:KERNEL</th><th>MODEL ANSWER:BUILDERS</th><th>ACTION</th></tr>"+matrix+"</table>"
         f"<div class='grid two'><div class='callout'>✕ Naively removing metadata-free layout groups selects{c['naive_remove_metadata_free_groups_kernel']},unsupported by draft.</div>"
         f"<div class='callout'>Original KVpayload remains{c['original_kv_payload_bytes']}B;builder-free is not KV-free.</div></div>"),
        ("UNDER THE HOOD","Fully skipped groups get zero builders while all groups retain layout constraints",
         "<pre>Source phases:\nDiscover attention groups\nSelect/reuse kernel block layout\nCreate0builders for fully skipped groups\nFilter workspace/CG/metadata traversal\nKeep KVview remapping/offsets</pre><p>↓ Original verifier</p>"
         "<pre>Skipped group ≠ removed layout group\nMixed group keeps active builders\nmap managerBlock/position to kernelBlock/offset\nCompare exact linear token address\nNo native graph capture or allocation execution</pre>"),
        ("ROBUSTNESS CHECK",f"An owned zero-byte workspace differs from no owner;provided{small['allocated_kernel_used']}-token layout must stay allocated",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Selected workspace</td><td>Owner{b['selected_workspace_owner']},bytes{b['selected_workspace_bytes']}</td><td>Owner{zero['after']['selected_workspace_owner']},bytes{zero['after']['selected_workspace_bytes']}</td><td>No owner is different</td></tr>"
         f"<tr><td>Existing layout</td><td>Kernel{c['allocated_kernel_used']}</td><td>Provided{small['allocated_kernel_used']}</td><td>Do not reselect draft-only{r['robustness'][1]['drafter_only_recomputed_kernel']}</td></tr>"
         "<tr><td>Mixed group</td><td>All draft-only layers skipped</td><td>Active target also present</td><td>Builders stay</td></tr></table>"),
        ("LATENCY","Original gated constructors omit inactive dummy state,not native KVallocation or GPUlatency",
         timing+"<p>256Bprivate original zero-state/object,25%retained;2warmups,7alternating repeats,3operations/sample.Median repeat means.Shared native workspace,KVallocation,vLLM/Torch/CUDAgraphs/GPUexcluded.</p>"),
        ("MY TAKE","Builder-free metadata does not make cache layout,model-state graph constraints or memory disappear",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Context-only draft metadata gating,shared workspace traversal,existing kernel-layout reuse and mixed-group safeguards.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Removing inactive layers from KVlayout,assuming0builders means0allocations,or treating attention-onlyALWAYS as global CUDAgraph readiness.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.Original unpadded geometry excludes packed cache/padding/allocator.Native vLLM/graph/peak/quality unknown.No browser visual inspection.</div>"),
    ]
    return page("Draft metadata/layout separation — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    c=r["case"]["outputs"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Builder-free metadata with retained cache-layout cabinet</title><desc>Original sixty-solid cabinet metaphor;context-only draft layers can have zero metadata builders without disappearing from kernel layout.Native vLLM,global CUDAgraphs and peak memory are not verified.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG155 · METADATA-FREE IS NOT KV-LAYOUT-FREE</text>
<text x="88" y="158" font-size="34" font-weight="700">Skip builders,not the allocated cache's kernel-block constraints</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original builders{c['before']['builder_count']}to{c['after']['builder_count']};kernel stays{c['allocated_kernel_used']},stride{c['stride']}</text>
<text x="88" y="760" font-size="24">Manager block3/position250 maps kernel{c['mapped_address']['kernel_block_id']}/position{c['mapped_address']['position']}</text>
<text x="88" y="810" font-size="24">Logical KVpayload{c['original_kv_payload_bytes']}Bunchanged;offsets{c['after']['original_offset_bytes']}Bstill present</text>
<text x="88" y="880" font-size="18">ActualNov14vLLM correction · original unpadded metadata ledger,no native peak/graph/quality or GPUlatency</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
