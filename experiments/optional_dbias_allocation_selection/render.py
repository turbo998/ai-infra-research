import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="rocm-ck-flash-backward-optional-dbias-allocation"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error("Bounded exact integer required");return v;}
function bytes(v){if(typeof v!=="string"||!/^\d{1,19}$/.test(v))throw Error("Decimal byte-count string required");const n=BigInt(v);if(n>2n**63n-1n)throw Error("Byte metadata bound exceeded");return n;}
function evaluate(p){
 const keys="bias_requires_grad,budget_bytes,causal,dtype,grad_bias_provided,k_lengths,kv_heads,layout,other_live_bytes,provided_grad_bias_bytes,q_lengths,query_heads";
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!==keys)throw Error("Exact original ledger fields required");
 if(!["fixed","varlen"].includes(p.layout)||!["fp16","bf16"].includes(p.dtype))throw Error("Original layout/16-bit dtype required");
 const qs=p.q_lengths,ks=p.k_lengths;if(!Array.isArray(qs)||!Array.isArray(ks)||qs.length!==ks.length||qs.length>64)throw Error("Matching length lists required");
 for(const v of [...qs,...ks])integer(v,0,2**20);
 if(p.layout==="fixed"&&(new Set(qs).size>1||new Set(ks).size>1))throw Error("Fixed lengths must be uniform");
 const h=integer(p.query_heads,1,128),kv=integer(p.kv_heads,1,128);if(h%kv)throw Error("Query heads must be divisible by KV heads");
 for(const f of ["causal","bias_requires_grad","grad_bias_provided"])if(typeof p[f]!=="boolean")throw Error("Exact boolean flags required");
 if(p.bias_requires_grad&&!p.grad_bias_provided)throw Error("Requested bias gradient requires provided grad_bias");
 let provided=0n;if(p.grad_bias_provided)provided=bytes(p.provided_grad_bias_bytes);else if(p.provided_grad_bias_bytes!==null)throw Error("Absent grad requires null byte metadata");
 const other=bytes(p.other_live_bytes),budget=bytes(p.budget_bytes),shape=[qs.length,h,Math.max(0,...qs),Math.max(0,...ks)];
 const dense=BigInt(shape[0])*BigInt(shape[2])*BigInt(shape[3]),actual=qs.reduce((a,q,i)=>a+BigInt(q)*BigInt(ks[i]),0n);
 const denseBytes=dense*BigInt(h)*2n,oldExtra=p.bias_requires_grad?0n:denseBytes,base=other+provided;
 return {placeholder_shape:shape,dense_placeholder_bytes:denseBytes.toString(),ragged_rectangular_pair_positions:actual.toString(),
 max_padded_pair_positions:dense.toString(),padding_pair_positions:(dense-actual).toString(),old_fresh_return_region_bytes:oldExtra.toString(),
 new_fresh_return_region_bytes:"0",removed_unused_bytes:oldExtra.toString(),provided_caller_owned_bytes:provided.toString(),
 old_selected_live_bytes:(base+oldExtra).toString(),new_selected_live_bytes:base.toString(),old_selected_fits_budget:base+oldExtra<=budget,new_selected_fits_budget:base<=budget,
 new_dbias_return:p.bias_requires_grad?"provided_grad_bias":"undefined_tensor",kernel_has_dbias:p.bias_requires_grad,
 kernel_dbias_pointer:p.bias_requires_grad?"provided":"null",native_other_workspace_bytes:null,native_peak_bytes:null,native_oom_prediction:null};
}
"""


def workbench(r):
    return page("Optional dbias allocation byte ledger",
        f"<h1>Do not materialize an unused quadratic bias-gradient placeholder</h1><p>Round151 · Original exact byte subledger;no actual GPU allocation or CK execution.</p>"
        f'<form id="form"><label>Fixed/varlen lengths,gradient flags and decimal byte budgets<textarea id="input" rows="10" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Evaluate exact optional-output allocation policy</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>BigInt preserves byte counts above2^53.Other-live metadata excludes caller-owned grad_bias,which is added even when unrequested."
        "Budget results cover only the selected virtual ledger;native workspace,peak and OOMremain unknown.An undefined tensor is not a zero-valued dense gradient.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const r=evaluate(JSON.parse(document.getElementById("input").value));out.textContent=JSON.stringify(r,null,2);
 if(!r.new_selected_fits_budget)error.textContent="Selected virtual ledger exceeds budget;removing unused placeholder does not free caller-owned buffers";}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"];gib=2**30
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        row["scenario"],row["inputs"]["bias_requires_grad"],row["outputs"]["new_dbias_return"],
        "Confirm native workspace" if row["outputs"]["new_selected_fits_budget"] else "✕ Selected budget exceeded"))+"</tr>" for row in r["matrix"])
    mx=max(max(v["gated_median_ms"],v["eager_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['toy_bytes']//1024}KiBzero-initialized CPUtoy</b><p>Gate {v['gated_median_ms']:.6f}ms</p>{bar(v['gated_median_ms'],mx)}"
                   f"<p>Discarded bytearray {v['eager_median_ms']:.6f}ms</p>{bar(v['eager_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    old=int(c["old_selected_live_bytes"])/gib;new=int(c["new_selected_live_bytes"])/gib
    sections=[
        ("HANDS-ON · OPTIONAL GRADIENT","The original ragged case requests a64GiBunused max-length placeholder;the new return region requests none",
         "<div class='grid'><div><h3>Scenarios</h3><p>Fixed/varlen<br>Gradient requested or absent<br>Caller-owned buffer</p></div>"
         "<div><h3>Structured input</h3><p>4sequences,max16K<br>32query/8KVheads<br>bf16,no bias gradient</p></div>"
         "<div><h3>Decision</h3><p>Keep unused output undefined<br>Preserve requested gradient<br>Account for other workspace</p><span class='pill'>Native peak/latency:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Actual Nov14ROCm CK correction.Original Python{r['latency']['python']}/{r['latency']['machine']}CPUtoy separately measured;no native benchmarks reproduced.</div>"),
        ("SCENARIO: RAGGED LONG CONTEXT",f"Removing64GiBchanges the selected virtual ledger from{old:g}to{new:g}GiB—not a native peak prediction",
         "<blockquote>4×32×16384×16384×2bytes=68719476736bytes.Unrequested has_dbias=false and pointer=null.</blockquote>"
         f"<div class='metric'><span class='pill'>score · virtual GiB,not confidence</span><h3>Old selected ledger{old:g}GiB</h3>{bar(old,old)}</div>"
         f"<div class='metric'><span class='pill'>choice · confirm actual allocator/workspace</span><h3>New selected ledger{new:g}GiB</h3>{bar(new,old)}<p>Other6GiBmetadata assumed;8GiBbudget is not a GPU success guarantee.</p></div>"),
        ("WHAT I TESTED","Six policy cases distinguish omitted fresh allocation from retained caller-owned requested gradients",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Requested gradient must be provided;undefined is not a zero tensor.</div>"
         "<div class='callout'>Subledger excludes unknown native workspace and other allocations.</div></div>"),
        ("UNDER THE HOOD","Varlen placeholder uses batch×query-heads×maxQ×maxK;causal masking does not shrink it",
         "<pre>Dense placeholder shape:[B,Hq,maxQ,maxK]\n16-bit q dtype:2bytes/element\nRagged rectangles:sum(q_i*k_i),not totalQ*totalK\nKV heads do not replace query heads</pre><p>↓ Original policy and source contract</p>"
         "<pre>bias_requires_grad=false:\n  kernel pointer=null;has_dbias=false\n  old return region:allocate unused dense tensor\n  new:undefined tensor,no fresh buffer\nbias_requires_grad=true:\n  require provided grad_bias;return it\n  caller-owned memory remains live</pre>"),
        ("ROBUSTNESS CHECK","Halving KVheads or reversing paired ragged lengths leaves the64GiBplaceholder unchanged",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>KVheads</td><td>8</td><td>4</td><td>0placeholder bytes</td></tr>"
         "<tr><td>Length order</td><td>Paired sequences</td><td>Reverse pairs</td><td>Same maxima/sum</td></tr>"
         "<tr><td>Mask</td><td>Causal</td><td>Noncausal</td><td>Same old dense shape</td></tr>"
         "<tr><td>Empty batch</td><td>Zero-byte old tensor</td><td>Undefined new output</td><td>Not interchangeable states</td></tr></table>"),
        ("LATENCY","CPUbranch versus discarded zero-filled bytearray is not ROCm at::empty or backward latency",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.CPUbytearray initializes bytes;native at::empty does not imply zero initialization.Maximum toy512KiB.</p>"),
        ("MY TAKE","Omit unrequested optional outputs without dropping requested gradients or pretending other scratch vanished",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Optional-output contracts,exact long-context memory accounting,max-length padding and request/provided-buffer gates.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Assuming every backward now fits memory,freeing caller buffers by changing a flag,or claiming native accuracy/latency from CPUtoy.</p></div></div>"
         "<div class='dark'>BSD-3-Clause source rights retained,no implementation mirror.Native gradients,ROCm/CK execution,allocator peak,workspace and OOMprediction unknown.Original figure,not MI350 hardware.No visual browser inspection.</div>"),
    ]
    return page("Optional dbias allocation — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Optional-gradient allocation cabinet with an explicit request latch</title><desc>Original sixty-solid cabinet metaphor;an unrequested bias-gradient slot stays undefined while required gradients and unknown scratch remain separate.Not native GPU hardware or measured allocator peak.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG151 · UNREQUESTED OPTIONAL OUTPUT IS NOT A DENSE ZERO TENSOR</text>
<text x="88" y="158" font-size="34" font-weight="700">Remove unused quadratic dbias allocation;keep requested gradients and scratch distinct</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original4×32×16384×16384×2B = 64GiBplaceholder;query heads,not KVheads</text>
<text x="88" y="760" font-size="24">No request:has_dbias=false,pointer=null,new return undefined—not initialized zeros</text>
<text x="88" y="810" font-size="24">Selected70→6GiBledger is virtual metadata;native peak/workspace/OOM remain unknown</text>
<text x="88" y="880" font-size="18">Actual Nov14CK fixed/varlen correction · original exact ledger/CPUtoy,no ROCm/GPU allocator or backward verification</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
