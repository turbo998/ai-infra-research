import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="multimodal-preprocessing-placement-feature-transport"
JS=r"""
const TRANSPORTS=["cpu","cuda_ipc","cuda_vmm"];
function bounded(v,lo,hi){if(!Number.isSafeInteger(v)||v<lo||v>hi)throw Error("Bounded original integer required");}
function placement(p){
 const cpu=p.platform==="cpu"||p.rl_cpu,accelerator=["cuda_alike","xpu","npu"].includes(p.platform);
 const contends=!cpu&&!p.fast_disabled&&p.hf_image_processor&&accelerator;
 let device=null;if(cpu)device="cpu";else if(p.platform==="xpu")device="xpu";else if(p.platform==="npu")device=p.processor_class==="Glm4vProcessor"?null:"npu";
 else if(p.platform==="cuda_alike")device=p.device_type+":"+p.base_gpu_id;
 const auto=contends?1:p.declared_workers===null?2:p.declared_workers,selected=p.skip_pool?1:p.explicit_workers!==null?p.explicit_workers:auto;
 return {candidate_fast_device:device,preprocessing_contends:contends,auto_processor_workers:auto,selected_processor_workers:selected,
 worker_reason:p.skip_pool?"synchronous skip-pool":p.explicit_workers!==null?"explicit override":"auto placement",candidate_device_not_actual_execution:true};
}
function transport(p){
 const mode=p.transport,keep=["cuda_ipc","cuda_vmm"].includes(mode),share=Math.floor(p.budget_bytes/p.tokenizer_workers),reserve=share*p.tokenizer_workers,active=mode==="cuda_ipc"&&!p.skip_pool;
 return {keep_features_on_device_policy:keep,cuda_ipc_policy:mode==="cuda_ipc",ipc_pool_configured:active,per_tokenizer_budget_bytes:share,aggregate_budget_split_bytes:reserve,
 unused_budget_bytes:p.budget_bytes-reserve,active_ipc_budget_split_bytes:active?reserve:0,naive_full_budget_per_tokenizer_bytes:p.budget_bytes*p.tokenizer_workers,
 cpu_executor_start_method_policy:mode==="cuda_vmm"?"spawn":"fork"};
}
function evaluate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="base_gpu_id,budget_bytes,declared_workers,device_type,explicit_workers,fast_disabled,hf_image_processor,platform,processor_class,rl_cpu,skip_pool,tokenizer_workers,transport")throw Error("Exact original fields required");
 if(typeof p.platform!=="string"||!["cpu","xpu","npu","cuda_alike","other"].includes(p.platform))throw Error("Original platform category required");
 for(const f of ["device_type","processor_class"])if(typeof p[f]!=="string"||!/^[A-Za-z0-9_]{1,32}$/.test(p[f]))throw Error("Bounded class/device label required");
 for(const f of ["rl_cpu","fast_disabled","hf_image_processor","skip_pool"])if(typeof p[f]!=="boolean")throw Error("Explicit boolean required");
 bounded(p.base_gpu_id,0,63);bounded(p.budget_bytes,1,1073741824);bounded(p.tokenizer_workers,1,64);
 for(const f of ["declared_workers","explicit_workers"])if(p[f]!==null)bounded(p[f],1,64);
 if(typeof p.transport!=="string"||!TRANSPORTS.includes(p.transport))throw Error("Known transport required;typos are not silently CPUfallback");
 const a=placement(p),b=transport(p),naive=b.keep_features_on_device_policy?1:p.declared_workers===null?2:p.declared_workers;
 return {...a,...b,naive_transport_only_auto_workers:naive,naive_auto_matches_placement:naive===a.auto_processor_workers,
 native_processor_execution:null,native_ipc_pool_allocation:null,native_feature_tensor_device:null,native_zero_share_pool_acceptance:null};
}
"""


def workbench(r):
    return page("Multimodal placement and transport policy workbench",
        "<h1>Where preprocessing runs is independent of where features travel</h1><p>Round156 · Declared policy projection,no actual processor/thread/IPC/GPUexecution.</p>"
        f'<form id="form"><label>Placement,processor threads,transport and tokenizer processes<textarea id="input" rows="11" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Resolve independent policies</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Candidate device is not actual execution.None means leave unset,not CPU.Override differs fromauto cap.Tokenizer processes split IPCbudget,not preprocessing thread count."
        "Configured zero share does not certify driver/pool support.Fork/spawn labels are source policy,not Windows execution.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");out.textContent=JSON.stringify(evaluate(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"]["outputs"];s=r["summary"];pool=r["matrix"][1]["outputs"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"]+"/"+x["inputs"]["transport"],x["outputs"]["candidate_fast_device"],
        x["outputs"]["auto_processor_workers"],
        "✕ Transport-only="+str(x["outputs"]["naive_transport_only_auto_workers"]) if not x["outputs"]["naive_auto_matches_placement"] else "Confirm runtime"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["hoisted_median_ms"],x["repeated_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['projections']}policy projections</b><p>Hoisted{v['hoisted_median_ms']:.6f}ms</p>{bar(v['hoisted_median_ms'],mx)}"
                   f"<p>Repeated{v['repeated_median_ms']:.6f}ms</p>{bar(v['repeated_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · PLACEMENT/TRANSPORT",f"Transport-only worker selection misses{s['naive_transport_only_auto_misses']}/{s['cases']}original policy cases",
         "<div class='grid'><div><h3>Scenarios</h3><p>GPUpreprocess→CPUfeatures<br>CPU/RL→GPUtransport<br>NPUdevice unset</p></div>"
         "<div><h3>Input</h3><p>Device/type witnesses<br>Auto/explicit thread count<br>Tokenizer pool processes</p></div>"
         "<div><h3>Decision</h3><p>Separate placement/transport<br>Respect manual override<br>Do not conflate workers</p><span class='pill'>Native throughput/peak unknown</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14module refactor,not new policy invention.Original Python{r['latency']['python']}/{r['latency']['machine']}CPUprojection only.</div>"),
        ("SCENARIO: GPU TO CPU",f"CPUfeature transport still has GPUpreprocessing:auto{c['auto_processor_workers']},not transport-only{c['naive_transport_only_auto_workers']}",
         "<blockquote>Fast processor shares the scheduler accelerator even when final features move toCPU.Transport destination does not determine preprocessing contention.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · confirm actual processor path</span><h3>Candidate{c['candidate_fast_device']};auto threads{c['auto_processor_workers']}</h3></div>"
         f"<div class='metric'><span class='pill'>score · original pool-budget arithmetic</span><h3>IPCsplit{pool['aggregate_budget_split_bytes']}Bversus naive{pool['naive_full_budget_per_tokenizer_bytes']}B</h3>"
         +bar(pool["aggregate_budget_split_bytes"],pool["naive_full_budget_per_tokenizer_bytes"])
         +f"<p>{pool['per_tokenizer_budget_bytes']}Bper tokenizer process;unused{pool['unused_budget_bytes']}B.No native reservation measured.</p></div>"),
        ("WHAT I TESTED",f"Six placement scenarios crossed with three transports produce{s['cases']}independent choices",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK:DEVICE</th><th>MODEL ANSWER:THREADS</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>NPUGlm4vdeviceNone leaves unset;it does not imply CPUcontention.</div><div class='callout'>Explicit thread override is not the automatic worker cap.</div></div>"),
        ("UNDER THE HOOD","Preprocessing threads,tokenizer IPCpools and CPUexecutor processes are different resource roles",
         "<pre>Placement:\nCPU/RL → cpu\nXPU → xpu\nNPUGlm4v → unset;other NPU → npu\nCUDA-alike → instance device_type:base_gpu_id\nAuto cap follows accelerator contention</pre><p>↓ Independent transport policy</p>"
         "<pre>CPU:does not keep feature outputs on device\nIPC/VMM:keep-device policy,not execution proof\nIPCbudget divided by tokenizer processes\nVMMCPUexecutor policy=spawn;others=fork\nNo tensor copy/lease/driver execution</pre>"),
        ("ROBUSTNESS CHECK","Manual4can override GPUauto1;zero per-process share remains arithmetic,not pool readiness",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Manual override</td><td>{c['selected_processor_workers']}auto</td><td>{r['robustness'][0]['outputs']['selected_processor_workers']}explicit</td><td>Explicit,not auto clamp</td></tr>"
         f"<tr><td>Instance device</td><td>{c['candidate_fast_device']}</td><td>{r['robustness'][2]['outputs']['candidate_fast_device']}</td><td>Base IDper instance</td></tr>"
         f"<tr><td>Small pool budget</td><td>{pool['per_tokenizer_budget_bytes']}Bshare</td><td>{r['robustness'][3]['outputs']['per_tokenizer_budget_bytes']}Bshare</td><td>Native acceptance unknown</td></tr>"
         "<tr><td>skip_pool</td><td>Normal processor</td><td>Synchronous1/no IPCpool</td><td>No actual wrap executed</td></tr></table>"),
        ("LATENCY","Original bulk policy projection does not measure preprocessing,IPC/VMMor native resolver frequency",
         timing+"<p>2warmups,7alternating repeats,3operations/sample;median repeat means.Immutable placement reused versus recomputed.Actual workers,hash/copy,driver/SGLang/GPUexcluded.</p>"),
        ("MY TAKE","Separate resource policies without inferring execution support or hardware speed from configuration",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Auto versus manual threads,instance device identity,transport independence and tokenizer-pool budget caps.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Transport implies preprocess device,None implies CPU,per-thread full IPCbudget,or treating policy projections as native throughput.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.Refactor preserves existing policies.Upstream performance comments not reproduced.Native tensors,IPC/VMM,zero-share support/peak unknown.No browser visual inspection.</div>"),
    ]
    return page("Multimodal placement/transport — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    c=r["case"]["outputs"];s=r["summary"];pool=r["matrix"][1]["outputs"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Separate preprocessing placement and feature transport cabinet</title><desc>Original sixty-solid cabinet metaphor with independent placement,thread and transport policies.Not native device execution or allocator measurement.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG156 · PREPROCESSING PLACEMENT DOES NOT FOLLOW FEATURE DESTINATION</text>
<text x="88" y="158" font-size="34" font-weight="700">Resolve contention,thread overrides and tokenizer-pool budgets separately</text>{sorter_scene()}
<text x="88" y="710" font-size="24">GPUcandidate{c['candidate_fast_device']}with CPUtransport:auto threads{c['auto_processor_workers']},not{c['naive_transport_only_auto_workers']}</text>
<text x="88" y="760" font-size="24">Original transport-only choices miss{s['naive_transport_only_auto_misses']}/{s['cases']};NPUunset is not CPU</text>
<text x="88" y="810" font-size="24">IPCbudget split{pool['aggregate_budget_split_bytes']}B,per tokenizer{pool['per_tokenizer_budget_bytes']}B,unused{pool['unused_budget_bytes']}B</text>
<text x="88" y="880" font-size="18">ActualNov14module refactor · original policy projection,no SGLang workers/IPC/GPUthroughput or peak measurement</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
