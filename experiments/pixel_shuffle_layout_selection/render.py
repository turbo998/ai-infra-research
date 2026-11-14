import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="pixel-shuffle-depth-space-dcr-crd-layout"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==7||!["n","c","h","w","blocksize"].every(k=>Number.isInteger(p[k])&&p[k]>=1&&p[k]<=65536)||![1,2,4,8].includes(p.scalar_bytes)||!["DCR","CRD"].includes(p.mode))throw Error("Bounded positive exact NCHW/blocksize,mode and scalar width required");
 if(p.c%(p.blocksize**2)!==0)throw Error("Input channels must divide by blocksize squared");
 const elements=p.n*p.c*p.h*p.w,size=elements*p.scalar_bytes;
 if(!Number.isSafeInteger(elements*Math.max(8,2*p.scalar_bytes)))throw Error("Selected logical payload exceeds common exact integer budget");
 return {input_shape:[p.n,p.c,p.h,p.w],output_shape:[p.n,p.c/p.blocksize**2,p.h*p.blocksize,p.w*p.blocksize],elements_preserved:elements,input_payload_bytes:size,output_payload_bytes:size,selected_one_read_one_write_bytes:2*size,selected_separate_int64_index_map_bytes:8*elements,note:"logical materialized-copy/index-map accounting only;not native allocation,fusion,view,stride or GPU throughput"};
}
function simulate(p){
 const result=budget(p);if(result.elements_preserved>4096)throw Error("Interactive coordinate display limit4096elements;budget may be calculated separately");
 const [N,C,H,W]=result.input_shape,[,CO,OH,OW]=result.output_shape,b=p.blocksize,out=[],indices=[];
 for(let flat=0;flat<result.elements_preserved;flat++){
 let q=Math.floor(flat/OW),ox=flat%OW,oy=q%OH;q=Math.floor(q/OH);const oc=q%CO,n=Math.floor(q/CO),iy=Math.floor(oy/b),by=oy%b,ix=Math.floor(ox/b),bx=ox%b;
 const ic=p.mode==="DCR"?(by*b+bx)*CO+oc:oc*b*b+by*b+bx,index=((n*C+ic)*H+iy)*W+ix;indices.push(index);out.push(index);}
 const inverse=Array(indices.length);indices.forEach((v,i)=>inverse[v]=out[i]);
 return {contract:result,input_values:indices.map((_,i)=>i),output_values:out,gather_input_indices:indices,inverse_values:inverse,note:"Original integer coordinate permutation;not native ONNX/PyTorch or image quality"};
}
"""


def workbench(r):
    p=r["case"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" min="1" required></label>' for k,v in p.items() if k!="mode")
    return page("PixelShuffle DCR/CRD coordinate workbench",
        f"""<h1>Matching shape does not establish PixelShuffle layout equivalence</h1>
        <p>Round138 · Original integer NCHW coordinate permutation,not native image/tensor execution.</p>
        <form id="form"><div class="grid">{fields}<label>Mode<select id="mode"><option>CRD</option><option>DCR</option></select></label></div>
        <button>Map labeled input coordinates</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Toggle mode on8input channels/blocksize2 to expose channel mixing. Oneoutputchannel makes both modes coincide and can hide bugs.
        SpaceToDepth CRD direct-op requires opset28 or later in the selected contract;explicit older-opset lowering is a different path.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all positive exact fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,k==="mode"?document.getElementById(k).value:Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify(simulate(p),null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["rederive_median_ms"] for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['elements']} original scalar elements</b><p>Retained mapping {v['preplanned_median_ms']:.6f}ms</p>{bar(v['preplanned_median_ms'],mx)}"
                   f"<p>Re-derived mapping+gather {v['rederive_median_ms']:.6f}ms</p>{bar(v['rederive_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · TENSOR LAYOUT","DCR and CRD have identical1×2×2×4shape,but only4of16fixture outputs match",
         "<div class='grid'><div><h3>Scenarios</h3><p>Subpixel shuffle<br>Inverse and adjoint<br>Opset downgrade</p></div>"
         "<div><h3>Input</h3><p>NCHW1×8×1×2<br>blocksize2<br>Labeled values0..15</p></div>"
         "<div><h3>Decisions</h3><p>Explicit CRD export<br>Channel divisibility<br>Direct-op version gate</p><span class='pill'>Native GPU latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned ONNX28 SpaceToDepth mode/function-body addition is Aug31,not November.Labeled model/oracle synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: MULTICHANNEL SHUFFLE","PyTorch's pinned TorchScript exporter selects CRD;omitting it selects a different permutation",
         f"<blockquote>CRD output:{c['CRD_output']}<br>DCR output:{c['DCR_output']}</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact labeled values,not confidence</span><h3>Matching values{c['same_shape_wrong_mode_matching_values']} / {c['denominator']}</h3>{bar(c['same_shape_wrong_mode_matching_values'],c['denominator'])}<p>Shape-only validation misses the permutation error</p></div>"
         f"<div class='metric'><h3>Exact inverse with matching mode</h3>{bar(len(c['inverse_CRD']),c['denominator'])}<p>All16input labels recovered;inverse is permutation adjoint</p></div>"),
        ("WHAT I TESTED","Matching-mode inverse recovers all16labels;wrong mode and unsupported direct downgrade are rejected",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Same shape</td><td>Same layout?</td><td>✕ Only4/16labels match</td><td>Explicit mode</td></tr>"
         "<tr><td>Roundtrip</td><td>Recover input?</td><td>16/16withsame mode</td><td>Check actual coordinates</td></tr>"
         "<tr><td>One output channel</td><td>Expose mismatch?</td><td>✕ Modes coincide</td><td>Multichannel fixture</td></tr>"
         "<tr><td>SpaceToDepthCRD</td><td>Singleop at27?</td><td>✕ Not representable</td><td>Reject or explicit lowering</td></tr>"
         "<tr><td>Bad channels</td><td>Floor C/b²?</td><td>✕ Reject</td><td>Exact divisibility</td></tr></table>"
         "<div class='grid two'><div class='callout'>Positive bounded NCHW original model is narrower than all native tensor cases.</div><div class='callout'>No interpolation or image-quality claim.</div></div>"),
        ("UNDER THE HOOD","Block-position and output-channel order—not shape—select the source channel",
         "<pre>Output(y,x),blocksize=b:\niy,by=divmod(y,b)\nix,bx=divmod(x,b)\nDCR input_channel=(by*b+bx)*Cout+out_channel\nCRD input_channel=out_channel*b*b+by*b+bx\nGather original NCHW scalar</pre>"
         "<p>↓ Independent input-scatter oracle</p><pre>For each input channel:\nDCR:block,out_channel=divmod(channel,Cout)\nCRD:out_channel,block=divmod(channel,b*b)\nby,bx=divmod(block,b)\nScatter to(y*b+by,x*b+bx)\nExact inverse supplies permutation adjoint</pre>"),
        ("ROBUSTNESS CHECK","The single-output-channel control hides the bug;multichannel labeled inputs reveal it",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Cout1mode switch</td><td>Original permutation</td><td>Same labels</td><td>0 labels</td></tr>"
         "<tr><td>Cout2mode switch</td><td>16labels</td><td>12positions differ</td><td>12 labels</td></tr>"
         f"<tr><td>Adjoint inner product</td><td>{c['adjoint_fixture']['dot_forward_gradient']}</td><td>{c['adjoint_fixture']['dot_input_inverse_gradient']}</td><td>Exact equality</td></tr></table>"),
        ("LATENCY","Retaining an index map avoids repeated Python coordinate work but retains extra state",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means.Map construction excluded only for retained path;re-derived path allocates mapping plus gather.No ONNX/PyTorch/native/GPU/fusion measurement.</p>"),
        ("MY TAKE","Use explicit mode,divisibility and opset contracts;PixelShuffle rearranges values rather than creating information",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Export validation,exact subpixel permutations,inverse/adjoint checks and selected copy-budget accounting.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Shape-only parity,default-mode assumptions,unsupported direct downgrade or claims of native speed,image quality and zero-copy.</p></div></div>"
         "<div class='dark'>ONNX Apache-2.0 and PyTorch BSD-style source facts;original code/oracle/visuals.Native kernels and browser inspection unperformed;unknown fields null.</div>"),
    ]
    return page("PixelShuffle layouts — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>PixelShuffle sorting cabinet:DCR versus CRD coordinate permutations</title>
<desc>Original sixty-solid sorting cabinet metaphor:the same number of tickets can be routed in different channel/block orders while keeping output dimensions identical.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG138 · EXACT TENSOR LAYOUT</text>
<text x="88" y="158" font-size="36" font-weight="700">Same output shape does not prove the same routing</text>{sorter_scene()}
<text x="88" y="710" font-size="25">CRD:channel then block · DCR:block then channel</text>
<text x="88" y="758" font-size="25">Validate divisibility,actual coordinates,matching inverse and exporter mode</text>
<text x="88" y="806" font-size="25">SpaceToDepth CRD direct operator needs opset28 in the selected contract</text>
<text x="88" y="880" font-size="18">Original gather/scatter oracle and integer adjoint · not native kernels,zero-copy,image quality or GPU speed</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
