import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="numa-first-write-weighted-interleave-page-placement"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error(`Exact integer${lo}..${hi}required`);return v;}
function policy(nodes,weights){
 if(!Array.isArray(nodes)||nodes.length<1||nodes.length>16)throw Error("One to sixteen explicit node IDs required");
 nodes.forEach(n=>integer(n,0,4095));if(nodes.some((n,i)=>i&&n<=nodes[i-1]))throw Error("Node IDs must be unique and ascending");
 if(!Array.isArray(weights)||weights.length!==nodes.length)throw Error("One weight per node required");
 weights.forEach(w=>{if(!Number.isInteger(w)||w<1||w>255)throw Error("Each weight must be1..255");});
}
function weightedNode(index,nodes,weights){
 integer(index,0,2**32-1);policy(nodes,weights);let sum=0;const boundaries=weights.map(w=>(sum+=w)),target=index%sum;
 return nodes[boundaries.findIndex(v=>target<v)];
}
function summary(owners,readers,size){
 const counts={};let local=0;owners.forEach((n,i)=>{counts[n]=(counts[n]||0)+1;if(n===readers[i])local++;});
 const payload={};Object.keys(counts).forEach(n=>payload[n]=counts[n]*size);
 return {owners,node_pages:counts,node_payload_bytes:payload,local_reads:local,remote_reads:owners.length-local,read_events:owners.length,remote_fraction:(owners.length-local)/owners.length,selected_payload_bytes:owners.length*size};
}
function simulate(p){
 const keys=["pages","nodes","weights","initial_writer","reader_nodes","page_bytes"].sort();
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(keys))throw Error("Exact original input fields required");
 integer(p.pages,1,4096);integer(p.initial_writer,0,4095);integer(p.page_bytes,512,2**21);
 if(p.page_bytes&(p.page_bytes-1))throw Error("Selected page bytes must be a power of two");policy(p.nodes,p.weights);
 if(!Array.isArray(p.reader_nodes)||p.reader_nodes.length!==p.pages)throw Error("One reader node per page required");
 p.reader_nodes.forEach(n=>integer(n,0,4095));
 const indexes=Array.from({length:p.pages},(_,i)=>i),out={};
 out.serial_first_write=summary(indexes.map(()=>p.initial_writer),p.reader_nodes,p.page_bytes);
 out.parallel_owner_first_write=summary([...p.reader_nodes],p.reader_nodes,p.page_bytes);
 out.interleave=summary(indexes.map(i=>p.nodes[i%p.nodes.length]),p.reader_nodes,p.page_bytes);
 out.weighted=summary(indexes.map(i=>weightedNode(i,p.nodes,p.weights)),p.reader_nodes,p.page_bytes);
 return out;
}
"""


def workbench(r):
    p=r["case"]["inputs"]
    return page("NUMA first-write and weighted page placement",
        f"""<h1>Policy choice changes new page targets—not already allocated pages</h1>
        <p>Round141 · Original ideal page-placement model,not Linux or physical NUMA measurement.</p>
        <form id="form"><label>Selected page model JSON<textarea id="input" rows="12" required>{json.dumps(p)}</textarea></label>
        <button>Compare serial/owner first-write and interleave targets</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Nodes are actual synthetic node IDs,not list positions.Weights1..255apply only to the weighted path.
        Local first-write uses the writer node;selected interleave nodes are not a bind restriction.
        One explicit reader access per page;equal-size selected payloads.No allocator fallback,cpuset rebind,
        THP,read-zero-page fault,shared page cache,migration or auto-bandwidth weight derivation.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input JSON required");
 out.textContent=JSON.stringify(simulate(JSON.parse(document.getElementById("input").value)),null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];out=c["outputs"];total=c["inputs"]["pages"]
    metrics="".join(f"<div class='metric'><span class='pill'>score · exact model fraction,not confidence</span><h3>{name}: {row['remote_reads']}/{row['read_events']}remote reads</h3>"
                    f"{bar(row['remote_reads'],row['read_events'])}<p>Confirm · synthetic locality only;no latency estimate</p></div>"
                    for name,row in (("Serial first-write",out["serial_first_write"]),("Weighted1:3",out["weighted"])))
    mx=max(max(v["prefix_median_ms"],v["walk_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['lookups']}node-ID lookups</b><p>Prebuilt prefix/bisect {v['prefix_median_ms']:.6f}ms</p>{bar(v['prefix_median_ms'],mx)}"
                   f"<p>Weight walk {v['walk_median_ms']:.6f}ms</p>{bar(v['walk_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    matrix="".join(f"<tr><td>{name}</td><td>Remote/total reads</td><td>{row['remote_reads']}/{row['read_events']}</td><td>Confirm workload topology</td></tr>" for name,row in out.items())
    sections=[
        ("HANDS-ON · NUMA PAGE PLACEMENT","Serial initialization puts all12pages on one node despite later parallel readers",
         "<div class='grid'><div><h3>Scenarios</h3><p>First-write<br>Page-index striping<br>Policy change</p></div><div><h3>Input</h3><p>12pages×4096B<br>Readers:3onnode0,9onnode1<br>Weights1:3</p></div>"
         "<div><h3>Decisions</h3><p>Initialize on intended owners<br>Inspect actual placement<br>Do not assume migration</p><span class='pill'>Native bandwidth:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC original ideal allocation model;independently_reproduced=false.CPU records separately measured Python{r['latency']['python']} / {r['latency']['machine']}.Nov10allocator fixes are not a new weighted-interleave algorithm.</div>"),
        ("SCENARIO: ASYMMETRIC READERS","Weights1:3reduce synthetic remote reads from9/12to4/12—not to zero",
         "<blockquote>Original pages0..11;reader nodes[0,0,0,1,1,1,1,1,1,1,1,1].Every page is read once.</blockquote>"+metrics+
         "<div class='callout'>Actual reader ownership matters:with all readers onnode0,serial local initialization wins instead.</div>"),
        ("WHAT I TESTED","Four ideal policies produce9,0,5and4remote reads on the same12-page workload",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+
         "<tr><td>Set policy after allocation</td><td>Old pages moved?</td><td>✕ No automatic migration</td><td>Separate migration protocol</td></tr></table>"
         "<div class='grid two'><div class='callout'>No probability thresholds:deterministic page identities and read-count denominators.</div><div class='callout'>Weights express target ratios—not latency,capacity reservation or measured bandwidth.</div></div>"),
        ("UNDER THE HOOD","Weighted page offsets choose node-ID intervals;first-write ownership remains sticky",
         "<pre>Original selected page index:\nnodeIDs=[0,1],weights=[1,3]\nprefix=[1,4]\ntarget=page_index mod4\n[0,1)→node0;[1,4)→node1\nNot a per-task allocation-order counter</pre>"
         "<p>↓ Original independent expanded-cycle oracle</p><pre>cycle=[0,1,1,1]\nassert selector(i)==cycle[i mod4]\nif page already allocated:\n    keep existing owner\nelse:\n    allocate ideal selected target\nNo native fallback or page migration</pre>"),
        ("ROBUSTNESS CHECK","Reversing fault order preserves12/12page-index targets;policy changes move0/6old pages",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Weightedpage identities</td><td>Forward faults</td><td>Reverse faults</td><td>{c['reversed_fault_order']['matches']}/{total}same</td></tr>"
         f"<tr><td>Already allocated pages</td><td>6onnode0</td><td>Change policy</td><td>{c['policy_change']['old_pages_migrated']}moved</td></tr>"
         "<tr><td>SparseIDs[2,5]</td><td>[2,5,5,5]</td><td>Effective mask[5]</td><td>[5,5,5,5]</td></tr></table>"
         "<p>Fixed nonempty mask intersection is original policy input—not complete Linux static/relative cpuset remapping.</p>"),
        ("LATENCY","Prebuilt prefix lookup versus weight walking measures CPU selection—not NUMA access",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample,median repeat means.Eight synthetic nodes;both return complete ID lists.Preprocessing,validation,Linux faults,migration and native bandwidth excluded.</p>"),
        ("MY TAKE","Align first-write ownership with readers before treating weighted striping as a locality cure",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Reasoning about initialization,page-index selection,stable old ownership and explicit per-node selected payloads.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Predicting remote-access nanoseconds,claiming automatic migration,confusing cpuset authority with policy,or applying anonymous-page rules to all page-cache paths.</p></div></div>"
         "<div class='dark'>SYNTHETIC,not Linux execution.Kernel source GPL-2.0-only;no source mirrored.Auto/manual ABI and weights require live kernel checks.Visual browser inspection not performed.</div>"),
    ]
    return page("NUMA page placement — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    out=r["case"]["outputs"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Page allocation cabinet:first-write targets and sticky drawer ownership</title>
<desc>Original sixty-solid sorting cabinet metaphor;page tickets stay in assigned drawers after selector policy changes.Static Figure tier,not native NUMA topology.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG141 · FIRST-WRITE PLACEMENT IS NOT READER AFFINITY</text>
<text x="88" y="158" font-size="36" font-weight="700">A new drawer selector does not move old page tickets</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Serial initialization: {out['serial_first_write']['remote_reads']}/12 remote reads in the original workload</text>
<text x="88" y="758" font-size="25">Weighted page-index targets1:3: {out['weighted']['remote_reads']}/12 remote reads;not a bandwidth prediction</text>
<text x="88" y="806" font-size="25">Policy changes affect new allocations;old pages require a separate migration mechanism</text>
<text x="88" y="880" font-size="18">Original ideal allocation model · no Linux faults,capacity fallback,THP,shared page cache or native NUMA timing</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
