import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="dense-transpose-output-padding-finite-adjoint"
JS=r"""
function output_length(length,kernel_size,p){
 if(Object.keys(p).length!==4||!Number.isInteger(length)||!Number.isInteger(kernel_size)||length<1||length>1024||kernel_size<1||kernel_size>31||!Object.values(p).every(Number.isInteger)||p.stride<1||p.stride>16||p.dilation<1||p.dilation>16||p.padding<0||p.padding>128||p.output_padding<0||p.output_padding>=p.stride)throw Error("Bounded exact scalar contract;original output_padding must be below stride");
 const n=(length-1)*p.stride-2*p.padding+p.dilation*(kernel_size-1)+p.output_padding+1;
 if(n<1||n>32768)throw Error("Selected output length must be1..32768");return n;
}
function simulate(x,kernel,p){
 if(!Array.isArray(x)||!Array.isArray(kernel)||[...x,...kernel].some(v=>!Number.isInteger(v)||Math.abs(v)>1024))throw Error("Bounded exact integer arrays required");
 const length=output_length(x.length,kernel.length,p);if(length>4096)throw Error("Interactive output display limit4096");
 const output=Array(length).fill(0);
 for(let i=0;i<x.length;i++)for(let k=0;k<kernel.length;k++){const target=i*p.stride-p.padding+k*p.dilation;if(target>=0&&target<length)output[target]+=x[i]*kernel[k];}
 return {output_length:length,output_values:output,note:"Original onechannel integer tap scatter,no bias/native PyTorch/GPU"};
}
function budget(p){
 if(Object.keys(p).length!==11||!Object.values(p).every(Number.isInteger)||!["batch","in_channels","out_channels","groups"].every(k=>p[k]>=1&&p[k]<=65536)||![1,2,4,8].includes(p.scalar_bytes)||p.in_channels%p.groups||p.out_channels%p.groups)throw Error("Bounded grouped payload contract required");
 const n=output_length(p.input_length,p.kernel_size,{stride:p.stride,padding:p.padding,dilation:p.dilation,output_padding:p.output_padding}),a=p.batch*p.in_channels*p.input_length*p.scalar_bytes,b=p.batch*p.out_channels*n*p.scalar_bytes,w=p.in_channels*(p.out_channels/p.groups)*p.kernel_size*p.scalar_bytes;
 if(!Number.isSafeInteger(a+b+w))throw Error("Shared exact integer payload exceeded");
 return {output_length:n,input_payload_bytes:a,output_payload_bytes:b,weight_payload_bytes:w,selected_read_input_and_weights_once_write_output_bytes:a+w+b,note:"grouped logical tensor payload only;scalar evaluator is onechannel;no native traffic,workspace,allocator or convolution algorithm"};
}
"""


def workbench(r):
    p=r["case"]["parameters"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("ConvTranspose output-padding and finite-adjoint workbench",
        f"""<h1>output_padding can reveal a nonzero sample—not append a zero</h1>
        <p>Round139 · Original onechannel integer transpose scatter,finite zero boundary,no bias.</p>
        <form id="form"><label>Transpose input JSON<textarea id="x" required rows="2">[1,2]</textarea></label>
        <label>Kernel JSON<textarea id="kernel" required rows="2">[1,2,3]</textarea></label>
        <div class="grid">{fields}</div><button>Evaluate selected finite output domain</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Toggle output_padding0/1:output[2,5,4] becomes[2,5,4,6],not[2,5,4,0].
        Original policy output_padding&lt;stride is intentionally narrower than all native dilation-related cases.
        No bias,grouped evaluation,native execution or audio/image-quality claim.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)])),x=JSON.parse(document.getElementById("x").value),kernel=JSON.parse(document.getElementById("kernel").value);
 const result=simulate(x,kernel,p);out.textContent=JSON.stringify({result,budget:budget({batch:1,in_channels:1,out_channels:1,groups:1,input_length:x.length,kernel_size:kernel.length,...p,scalar_bytes:4})},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["dense_median_ms"] for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>Inputlength{v['input_length']}</b><p>Tap scatter {v['scatter_median_ms']:.6f}ms</p>{bar(v['scatter_median_ms'],mx)}"
                   f"<p>Prebuilt dense-matrix transpose {v['dense_median_ms']:.6f}ms</p>{bar(v['dense_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · FINITE CONVOLUTION","output_padding1changes[2,5,4]to[2,5,4,6]—the extra sample is not zero",
         "<div class='grid'><div><h3>Scenarios</h3><p>Stride ambiguity<br>Finite-domain adjoint<br>Not an inverse</p></div>"
         "<div><h3>Input</h3><p>x=[1,2],kernel=[1,2,3]<br>stride2,padding1<br>dilation1,no bias</p></div>"
         "<div><h3>Decisions</h3><p>Select output domain<br>Do not zero-append<br>Keep bias separate</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned established PyTorch contract,not a November invention.Synthetic original model/oracle,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: LENGTH AMBIGUITY","Both forward input lengths3and4map to length2;their finite transpose domains differ",
         f"<blockquote>Correctextendedoutput:{c['output_padding1']}<br>✕ Append-zero surrogate:{c['wrong_zero_appended_output']}</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact fixture outputs,not confidence</span><h3>Wrong surrogate matches{c['output_padding1_matching_wrong_append_positions']} / {c['output_position_denominator']}positions</h3>{bar(c['output_padding1_matching_wrong_append_positions'],c['output_position_denominator'])}<p>Finalsample6reveals real kernel contribution</p></div>"
         "<div class='metric'><h3>Action:select valid domain,then compute contributions</h3><p>Do not change only the array's reported length.</p></div>"),
        ("WHAT I TESTED","Adjoint equality48holds,while AᵀA[1,2,3,4]=[16,44,40,60]is not the input",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>outputpadding1</td><td>Appendzero?</td><td>✕ Lastvalue6</td><td>Reevaluate domain</td></tr>"
         "<tr><td>Adjoint</td><td>Innerproducts?</td><td>48=48</td><td>Tied linear kernel</td></tr>"
         "<tr><td>Numericalinverse</td><td>Recoverz?</td><td>✕ AᵀAz≠z</td><td>No inversion claim</td></tr>"
         "<tr><td>Outputsize</td><td>Length4?</td><td>Selectedoutputpadding1</td><td>Validate interval</td></tr></table>"
         "<div class='grid two'><div class='callout'>Original output_padding&lt;stride is a selected subset,not all native cases.</div><div class='callout'>Adjoint excludes bias and uses a matching finite forward domain.</div></div>"),
        ("UNDER THE HOOD","Transpose scatter uses the same cross-correlation weights with reversed matrix connectivity",
         "<pre>Output length:\n(Lin-1)*stride-2*padding\n+dilation*(K-1)+output_padding+1\n\nFor each input i and kernel tap k:\nposition=i*stride-padding+k*dilation\nif inside selected domain:\n  output[position]+=input[i]*weight[k]</pre>"
         "<p>↓ Independent dense forward matrix</p><pre>A[row,col]=weight[k]\nwhere col=row*stride-padding+k*dilation\nOriginal oracle computes A from each row/column pair\nTranspose result=A^T*x\nAdjoint identity:&lt;Az,x&gt;=&lt;z,A^T x&gt;\nNo claim that A^T A=I</pre>"),
        ("ROBUSTNESS CHECK","Changing finite length includes a boundary contribution;zero-input and zero-kernel controls remain linear",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>outputpadding0→1</td><td>[2,5,4]</td><td>[2,5,4,6]</td><td>Oneincluded sample</td></tr>"
         "<tr><td>Kernel→zeros</td><td>Nonzerooutput</td><td>Allcomputedzeros</td><td>Exact linear control</td></tr>"
         "<tr><td>Adjointinnerproducts</td><td>48</td><td>48</td><td>0</td></tr></table>"),
        ("LATENCY","Tap scatter avoids an educational dense zero-filled matrix,not a native cuDNN baseline",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means.Dense matrix construction excluded.Python scalar paths only;no BLAS/cuDNN/PyTorch/GPU,grouped evaluator,bias or streaming.</p>"),
        ("MY TAKE","Select output shape and tied adjoint semantics explicitly;transposed convolution is not numerical inversion",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Output-size debugging,finite boundary checks,linear adjoints and selected activation/weight payload budgets.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Append-zero shortcuts,unqualified inverse claims,native performance prediction or audio/image-quality guarantees.</p></div></div>"
         "<div class='dark'>Original scalar/matrix oracle and visuals;pinned BSD-style source facts.Native results/workspace unknown;browser inspection not performed.</div>"),
    ]
    return page("Transposed convolution boundaries — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Finite transpose contribution cabinet:output padding selects the boundary</title>
<desc>Original sixty-solid sorting cabinet metaphor for routing kernel contributions into a chosen finite output domain.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG139 · FINITE TRANSPOSE BOUNDARIES</text>
<text x="88" y="158" font-size="36" font-weight="700">A larger output domain can include a nonzero sample</text>{sorter_scene()}
<text x="88" y="710" font-size="25">output_padding selects shape;it does not just append zeros</text>
<text x="88" y="758" font-size="25">A transpose reverses linear connectivity—not the numerical operation</text>
<text x="88" y="806" font-size="25">Validate finite domain,stride ambiguity,kernel tying and bias boundaries</text>
<text x="88" y="880" font-size="18">Original tap scatter/dense matrix oracle · no native GPU,workspace,streaming or quality reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
