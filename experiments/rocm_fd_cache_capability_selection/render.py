import html
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="rocm-extensible-kv-fd-export-registration-capabilities"
JS=r"""
const BITS={REGISTERS:1,EXPORTS_FD:2,EXPORTS_FABRIC:4,LEGACY_IPC:8};
class PolicyError extends Error{constructor(code,reason,stage="configuration",enabled=null){super(reason);this.code=code;this.stage=stage;this.enabled=enabled;}}
function validate(p){
 if(!p||typeof p!=="object"||Array.isArray(p)||Object.keys(p).sort().join(",")!=="blocks_override,capabilities,cuda_alike,custom_pool,driver_probe,elastic_ep,imex,memory_override,requested,rocm")throw Error("Exact original fields required");
 for(const f of ["cuda_alike","rocm","memory_override","blocks_override","elastic_ep","custom_pool","imex"])if(typeof p[f]!=="boolean")throw Error("Explicit booleans required");
 if(p.requested!==null&&typeof p.requested!=="boolean")throw Error("requested null/true/false required");
 if(!Array.isArray(p.capabilities)||p.capabilities.some(v=>typeof v!=="string"||!Object.hasOwn(BITS,v))||new Set(p.capabilities).size!==p.capabilities.length)throw Error("Unique known capabilities required");
 if(typeof p.driver_probe!=="string"||!["unknown","supported","unsupported"].includes(p.driver_probe))throw Error("Explicit driver witness required");
}
function resolve(p,old=false){
 validate(p);const caps=new Set(p.capabilities),fd=caps.has("EXPORTS_FD"),single=["REGISTERS","EXPORTS_FD","EXPORTS_FABRIC"].some(v=>caps.has(v));
 const gates=[[!p.cuda_alike,"NON_CUDA_ALIKE"],[p.memory_override,"MEMORY_OVERRIDE"],[p.blocks_override,"BLOCKS_OVERRIDE"],[p.elastic_ep,"ELASTIC_EP"],
 [caps.has("LEGACY_IPC"),"LEGACY_IPC"],[p.custom_pool,"CUSTOM_POOL"],[p.rocm&&(old?single:caps.has("REGISTERS")),"ROCM_REGISTRATION"]];
 const gate=gates.find(v=>v[0]),unsupported=gate?gate[1]:null;
 if(fd&&(p.requested===false||unsupported!==null))throw new PolicyError("FD_REQUIRES_EXTENSIBLE",unsupported||"explicitly disabled");
 const fabric=caps.has("EXPORTS_FABRIC")&&!p.imex;let enabled;
 if(p.requested===null)enabled=unsupported===null&&(fd||!fabric);
 else{if(p.requested&&unsupported!==null)throw new PolicyError("UNSUPPORTED_EXPLICIT_REQUEST",unsupported);enabled=p.requested;}
 const warnings=enabled&&fabric?["NO_IMEX_LOCAL_TRANSFER_TCP_WARNING"]:[],reason=unsupported||(!enabled&&p.requested===null&&fabric?"NO_IMEX_AUTO_OFF":null);
 let final,action;if(!enabled){final=false;action="ordinary_allocator_without_probe";}
 else if(p.driver_probe==="unsupported"){if(fd)throw new PolicyError("FD_DRIVER_REQUIRED","no ordinary-allocator fallback for FDimporter","runtime",true);
 final=false;action="model_runtime_fallback_ordinary_allocator";}
 else{final=true;action=p.driver_probe==="supported"?"model_probe_passed":"driver_probe_not_observed";}
 return {status:enabled?"configured_enabled":"configured_disabled",configuration_enabled:enabled,reason_code:reason,
 needs_single_physical_allocation_per_segment:single,warnings,runtime_enabled_in_model:final,runtime_action:action,native_driver_probe_verified:null,native_startup_verified:null};
}
function capture(p,old=false){try{return resolve(p,old);}catch(e){if(!(e instanceof PolicyError))throw e;
 return {status:"rejected",configuration_enabled:e.enabled,reason_code:e.code,detail:e.message,stage:e.stage,runtime_enabled_in_model:null,native_driver_probe_verified:null,native_startup_verified:null};}}
"""


def workbench(r):
    return page("ROCm extensible KVconnector capability planner",
        "<h1>FDexport is not UCX/RDMAregistration;single physical chunk does not identify the access mechanism</h1>"
        "<p>Round157 · Original config/runtime-witness ledger.No POSIXFD,driver,VMM/ROCm/vLLMexecution.</p>"
        f'<form id="form"><label>Capabilities,explicit overrides and declared driver witness<textarea id="input" rows="9" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label>'
        '<button>Compare coarse and precise gates</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>FDimporter requires extensible allocation and cannot silently fall back if driver probe fails.Mixed FD+registration remains blocked onROCm."
        "Fabric/IMEX,custompool,legacyIPC and sizing overrides remain separate.Config enabled is not native startup verified.</p>",
        JS+r"""
function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),error=document.getElementById("error");out.textContent="";error.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const p=JSON.parse(document.getElementById("input").value),r=capture(p);
 out.textContent=JSON.stringify({before:capture(p,true),after:r},null,2);if(r.status==="rejected")error.textContent=r.reason_code+": "+r.detail;
 else if(r.warnings.length)error.textContent=r.warnings.join(";");}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    s=r["summary"];c=r["case"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        "+".join(x["inputs"]["capabilities"]) or "IN_PROCESS","auto" if x["inputs"]["requested"] is None else "on",
        x["after"]["status"],"✕ Coarse rejected" if x["before"]["status"]=="rejected" and x["after"]["status"]!="rejected" else "Confirm remaining gates"))+"</tr>" for x in r["matrix"])
    mx=max(max(v["bitmask_median_ms"],v["names_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['capability_sets']}capability sets</b><p>Bits{v['bitmask_median_ms']:.6f}ms</p>{bar(v['bitmask_median_ms'],mx)}"
                   f"<p>Names{v['names_median_ms']:.6f}ms</p>{bar(v['names_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · KV CAPABILITIES",f"{s['changed_configuration_results']}/{s['cases']}original configuration results change when ROCmgates registration precisely",
         "<div class='grid'><div><h3>Scenarios</h3><p>FD versus registration<br>Fabric/IMEX<br>Mandatory FDdriver</p></div>"
         "<div><h3>Input</h3><p>Access capability set<br>auto/on/off<br>Declared runtime witness</p></div>"
         "<div><h3>Decision</h3><p>Keep FDeligible<br>Reject mixed conflicts<br>No silent FDfallback</p><span class='pill'>Native startup/peak unknown</span></div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14ROCmconfig correction.Original Python{r['latency']['python']}/{r['latency']['machine']}capability classification only.</div>"),
        ("SCENARIO: ROCM FD",f"Original permitted configurations rise{s['before_enabled']}→{s['after_enabled']};FDstill requires the extensible cache",
         "<blockquote>FDexport needs one physical allocation per segment,but does not register thatallocation withUCX/RDMA.Coarse chunk requirements must not stand in for access capabilities.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · confirm driver and full capability set</span><h3>FDcase:{c['before']['status']}→{c['after']['status']}</h3></div>"
         f"<div class='metric'><span class='pill'>score · original configuration permits,not native success</span><h3>Before{s['before_enabled']}/{s['cases']}</h3>{bar(s['before_enabled'],s['cases'])}"
         f"<h3>After{s['after_enabled']}/{s['cases']}</h3>{bar(s['after_enabled'],s['cases'])}</div>"),
        ("WHAT I TESTED","Registration,FDandFABRIChave different allocation/export gates despite shared chunk requirements",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Mixed FD+REGISTERSstill rejects onROCm;FDis not a blanket exemption.</div><div class='callout'>FDexplicitoff and driver failure stop startup,not successful fallback.</div></div>"),
        ("UNDER THE HOOD","Precise registration predicate follows earlier sizing,elastic,legacyIPCandcustom-pool gates",
         "<pre>Source gate priority:\nCUDA-alike / sizing overrides / elasticEP\nLegacy IPC / custom memory pool\nROCm + REGISTERS only\nFDrequires extensible;FABRICneedsIMEX\nExplicit invalid config raises</pre><p>↓ Original runtime-witness model</p>"
         "<pre>Config enabled ≠ driver probe passed\nUnknown probe → no native startup claim\nFailed ordinary probe → model fallback\nFailed FDprobe → stop,not torch fallback\nNo actual POSIXFDor GPUexecution</pre>"),
        ("ROBUSTNESS CHECK","FDmandatory allocation blocks explicitoff,mixed-registration and failed-driver fallbacks",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>FDrequest</td><td>auto enabled</td><td>explicitoff</td><td>Rejected</td></tr>"
         "<tr><td>FDdriver</td><td>Unknown</td><td>Declared failure</td><td>Stop startup</td></tr>"
         "<tr><td>Ordinary driver</td><td>Enabled config</td><td>Declared failure</td><td>Model allocator fallback</td></tr>"
         "<tr><td>FDcapabilities</td><td>FDonly</td><td>AlsoREGISTERS</td><td>ROCmconflict retained</td></tr></table>"),
        ("LATENCY","CPUcapability classification is not native config-resolution time,VMMregistration or FDthroughput",
         timing+"<p>Prevalidated masks/names,2warmups,7alternating repeats,3operations/sample;median repeat means.Driver,allocation,exports/network and vLLM/GPUexcluded.</p>"),
        ("MY TAKE","Use explicit allocation capabilities without treating configuration eligibility as driver or transport support",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>FD/registration distinctions,combined capabilities,explicit errors and config-versus-runtime evidence.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>FDallowsmulti-physical chunks,FDoverridesregistration,ordinary fallback under an FDimporter,or equating config enabled with native readiness.</p></div></div>"
         "<div class='dark'>Apache-2.0rights retained,no implementation mirror.One physical allocation/segment remains an access requirement.Native POSIXFD/ROCm/VMM/peak/throughput unknown.No browser visual inspection.</div>"),
    ]
    return page("ROCm FD/registration capabilities — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    s=r["summary"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>FDexport versus registration capability cabinet</title><desc>Original sixty-solid cabinet metaphor;precise ROCmregistration gate does not reject FDonlyaccess.Config/runtime evidence is distinct.No native VMMor POSIXFDexport performed.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG157 · A SINGLE PHYSICAL CHUNK DOES NOT DEFINE REGISTRATION</text>
<text x="88" y="158" font-size="34" font-weight="700">Keep FDeligible,check combined capabilities and preserve startup errors</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Original changed configurations{s['changed_configuration_results']}/{s['cases']};permits{s['before_enabled']}to{s['after_enabled']}</text>
<text x="88" y="760" font-size="24">ROCmregistration remains blocked;FD+registration does not receive an exemption</text>
<text x="88" y="810" font-size="24">FDexplicitoff or failed VMMdriver cannot silently fall back toordinary allocation</text>
<text x="88" y="880" font-size="18">ActualNov14config correction · original capability model,no native VMM/ROCm/FDor GPU/network measurement</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
