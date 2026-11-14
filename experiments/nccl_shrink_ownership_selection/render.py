import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="nccl-shrink-survivor-rank-shard-ownership"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error("Bounded exact integer required");return v;}
function validate(p){
 const fields=["parent_global_ranks","excluded","callers","mode","outstanding","shrink_share","rows","checkpoint","features","element_bytes"];
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(fields.sort()))throw Error("Exact original membership/row-ledger fields required");
 const globals=p.parent_global_ranks;
 if(!Array.isArray(globals)||globals.length<2||globals.length>16)throw Error("Two to sixteen parent entries required");
 globals.forEach(v=>integer(v,0,1e6));if(new Set(globals).size!==globals.length)throw Error("Unique global labels required");
 ["excluded","callers"].forEach(name=>{if(!Array.isArray(p[name]))throw Error("Parent-relative rank list required");
  p[name].forEach(v=>integer(v,0,globals.length-1));if(new Set(p[name]).size!==p[name].length)throw Error("Duplicate parent rank");});
 if(p.excluded.length<1||p.excluded.length>=globals.length)throw Error("Nonempty excludes must leave survivors");
 if(!["default","abort"].includes(p.mode)||typeof p.shrink_share!=="boolean")throw Error("Default/abort mode and exact share boolean required");
 integer(p.outstanding,0,64);integer(p.features,1,4096);integer(p.element_bytes,1,8);
 if(!Array.isArray(p.rows)||p.rows.length>128)throw Error("At most128scalar reference rows required");p.rows.forEach(v=>integer(v,-1e6,1e6));
 if(!Array.isArray(p.checkpoint)||p.checkpoint.length>p.rows.length)throw Error("Bounded checkpoint list required");
 const seen=new Set();p.checkpoint.forEach(r=>{if(!Array.isArray(r)||r.length!==2)throw Error("Checkpoint identity/value pair required");
  integer(r[0],0,p.rows.length-1);integer(r[1],-1e6,1e6);if(seen.has(r[0]))throw Error("Duplicate checkpoint row identity");
  if(r[1]!==p.rows[r[0]])throw Error("Checkpoint value disagrees with original synthetic reference");seen.add(r[0]);});
}
function membership(p){
 validate(p);const world=p.parent_global_ranks.length,excluded=new Set(p.excluded),survivors=Array.from({length:world},(_,i)=>i).filter(i=>!excluded.has(i));
 const callers=new Set(p.callers);if(callers.size!==survivors.length||survivors.some(i=>!callers.has(i)))throw Error("Callers must be exactly surviving parent-relative ranks;excluded ranks must not call");
 if(p.mode==="default"&&p.outstanding)throw Error("Default shrink requires no outstanding work;drain or select explicit abort semantics");
 const members=survivors.map(rank=>({parent_rank:rank,new_rank:rank-p.excluded.filter(x=>x<rank).length,global_id:p.parent_global_ranks[rank]}));
 return {parent_world:world,new_world:survivors.length,members,excluded:p.excluded.slice().sort((a,b)=>a-b),
 changed_rank_ids:members.filter(v=>v.parent_rank!==v.new_rank).length,abort_requested:p.mode==="abort",
 selected_sharing_request_permitted:p.mode==="default"&&p.shrink_share,
 scope:"original preflight and association ledger;not observed communicators,abort completion or actual resource sharing"};
}
function simulate(p){
 const group=membership(p),world=group.parent_world,n=group.new_world,remap=new Map(group.members.map(v=>[v.parent_rank,v.new_rank])),excluded=new Set(group.excluded);
 const retained=p.rows.map((value,i)=>({row:i,value,parent_owner:i%world,new_resident:remap.get(i%world),target_owner:i%n})).filter(v=>!excluded.has(v.parent_owner));
 const lost=p.rows.map((_,i)=>i).filter(i=>excluded.has(i%world)),checkpoint=new Map(p.checkpoint),missing=lost.filter(i=>!checkpoint.has(i));
 const moved=retained.filter(v=>v.new_resident!==v.target_owner).map(v=>v.row),rowBytes=p.features*p.element_bytes;
 const selected={retained_rows:retained.length,lost_rows:lost.length,already_at_target_rows:retained.length-moved.length,survivor_rows_needing_move:moved.length,
 selected_total_feature_bytes:p.rows.length*rowBytes,selected_survivor_move_feature_bytes:moved.length*rowBytes,selected_checkpoint_restore_feature_bytes:lost.length*rowBytes,
 scope:"selected feature payload only;scalar ledger does not execute feature tensors or model NCCL wire,checkpoint staging or allocator peak"};
 const error=missing.length?"missing checkpoint row IDs "+JSON.stringify(missing):null;
 let resharded=null;if(!missing.length){resharded=Array.from({length:n},()=>[]);p.rows.forEach((v,i)=>resharded[i%n].push([i,lost.includes(i)?checkpoint.get(i):v]));}
 return {membership:group,retained,lost_row_ids:lost,survivor_row_ids_needing_move:moved,selected_budget:selected,
 data_ready:!missing.length,data_error:error,resharded,
 scope:"membership rank compression is separate from explicit original cyclic-row recovery and redistribution;not training recovery"};
}
"""


def workbench(r):
    return page("Shrink membership versus cyclic-row recovery workbench",
        f"""<h1>Survivor rank renumbering is not shard recovery</h1>
        <p>Round146 · Original parent-relative membership and scalar row ledger.No NCCL,distributed processes or GPU execution.</p>
        <form id="form"><label>Original membership/checkpoint JSON<textarea id="input" rows="14" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <button>Check survivor participation and explicit data recovery</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Rows use the original cyclic rule row-ID modulo group size.Global labels do not determine group order.
        Missing checkpoint rows block data readiness;the model never fills lost data with zeros.
        Abort and sharing are metadata requests,not measured completion/resources.No optimizer,gradient,checkpoint I/O or training correctness certification.</p>""",
        JS+r"""
function update(e){if(e)e.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");const result=simulate(JSON.parse(document.getElementById("input").value));
 out.textContent=JSON.stringify(result,null,2);if(result.data_error)error.textContent=result.data_error;}
 catch(e){error.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    out=r["case"]["outputs"];g=out["membership"];b=out["selected_budget"]
    rank_table="".join(f"<tr><td>{v['parent_rank']}</td><td>{v['new_rank']}</td><td>{v['global_id']}</td></tr>" for v in g["members"])
    matrix=""
    for case in r["scenarios"]:
        result=case["outputs"]
        answer="✕ preflight rejected" if case["preflight_error"] else ("✕ lost rows unavailable" if not result["data_ready"] else "Original ledger ready")
        matrix+=f"<tr><td>{case['name'].replace('_',' ')}</td><td>Membership + rows</td><td>{answer}</td><td>Confirm native/app behavior</td></tr>"
    mx=max(v["counted_median_ms"] for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['parent_rank_entries']}parent-rank entries</b><p>Prebuilt prefix {v['prefix_median_ms']:.6f}ms</p>{bar(v['prefix_median_ms'],mx)}"
                   f"<p>Repeated lower-excluded counts {v['counted_median_ms']:.6f}ms</p>{bar(v['counted_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · SHRINK OWNERSHIP","Shrinking6ranks to4leaves four original rows needing checkpoint recovery",
         "<div class='grid'><div><h3>Scenarios</h3><p>Survivor call set<br>Group-relative remap<br>Missing cyclic shards</p></div><div><h3>Input</h3><p>Parent ranks0..5<br>Exclude1,4<br>12original scalar rows</p></div>"
         "<div><h3>Decisions</h3><p>Renumber survivors<br>Recover lost identities<br>Repartition explicitly</p><span class='pill'>Native shrink latency:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC,independently_reproduced=false.Original CPU Python{r['latency']['python']} / {r['latency']['machine']}.Nov14child-option/cleanup change is not a new shrink/recovery algorithm.</div>"),
        ("SCENARIO: RETAGGED RESIDENTS","Only2of8retained rows already satisfy the new cyclic ownership rule",
         f"<blockquote>Retained rows:{b['retained_rows']};lost IDs:{out['lost_row_ids']};survivor rows needing movement:{out['survivor_row_ids_needing_move']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact original ownership</span><h3>{b['already_at_target_rows']}/{b['retained_rows']}already at target</h3>{bar(b['already_at_target_rows'],b['retained_rows'])}<p>Confirm · move other retained rows explicitly</p></div>"
         "<div class='metric'><span class='pill'>choice · deterministic original gate</span><h3>Missing checkpoints block data readiness</h3><p>No zero-filled recovery or native-training claim</p></div>"),
        ("WHAT I TESTED","Seven original cases separate participation,abort metadata and data recovery",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Excluded ranks must not call shrink;not a NULL-return path for excluded callers.</div><div class='callout'>Default requires no pending work;abort request is not proof of completion.</div></div>"),
        ("UNDER THE HOOD","New rank equals old rank minus the number of lower excluded ranks",
         "<pre>Original parent-group ranks:[0,1,2,3,4,5]\nExclude:[1,4]\nSurvivors:[0,2,3,5]\nNew ranks:[0,1,2,3]\nGlobal labels stay associated—not globally sorted</pre>"
         "<p>↓ Original independent cyclic-row verifier</p><pre>Old owner=row_ID modulo6\nNew target=row_ID modulo4\nRetained rows need explicit movement\nLost row identities need supplied checkpoint\nVerify against original synthetic reference\nCommunicator metadata cannot regenerate row values</pre>"),
        ("ROBUSTNESS CHECK","Reversing exclusion/caller order preserves four survivor associations",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Exclusions</td><td>[1,4]</td><td>[4,1]</td><td>Same4members</td></tr>"
         "<tr><td>Caller traversal</td><td>[0,2,3,5]</td><td>[5,3,2,0]</td><td>Same associations</td></tr>"
         "<tr><td>Pending work</td><td>Default,2pending</td><td>Abort,2pending</td><td>Reject versus metadata request</td></tr>"
         "<tr><td>Checkpoints</td><td>Four lost rows supplied</td><td>None supplied</td><td>Data readiness blocked</td></tr></table>"
         "<p>Order perturbation is original metadata preflight,not native concurrent-call timing.</p>"),
        ("LATENCY","Prefix association planning is not native NCCL shrink latency or fault recovery",
         timing+"<p>One in four entries excluded;prebuilt prefix/exclusion/survivor lists.Two warmups,seven alternating repeats,three operations/sample,median repeat means.Standalone planner sizes exceed16-rank UI bound;no GPUs,ranks,network or state movement executed.</p>"),
        ("MY TAKE","Retagged communicator ranks do not restore lost shards or add replacement workers",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Parent-relative exclude lists,survivor participation,stable association labels and explicit cyclic-row checkpoint/reshard gates.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Excluded callers waiting for NULL,sorting global IDs as group order,automatic tensor/optimizer recovery or claiming abort/resource-sharing completion.</p></div></div>"
         "<div class='dark'>Original model;PyTorch BSD-style rights preserved,NVIDIA facts paraphrased.No native NCCL/GPU,training recovery,checkpoint I/O,wire/peak-memory certification.Visual browser inspection not performed.</div>"),
    ]
    return page("Shrink membership and row ownership — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure(r):
    b=r["case"]["outputs"]["selected_budget"]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Retagged sorting cabinet does not recover missing parcels</title>
<desc>Original sixty-solid sorting-cabinet metaphor;retagging survivor slots is distinct from recovering missing parcels and placing retained parcels in new ownership bins.Not NCCL hardware.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG146 · RETAGGING IS NOT PARCEL RECOVERY</text>
<text x="88" y="158" font-size="35" font-weight="700">Communicator membership and data ownership are separate ledgers</text>{sorter_scene()}
<text x="88" y="710" font-size="24">Parent ranks[0,2,3,5]become[0,1,2,3];global identities retain their associations</text>
<text x="88" y="760" font-size="24">{b['already_at_target_rows']}/{b['retained_rows']}retained rows already at target;{b['survivor_rows_needing_move']}need movement;{b['lost_rows']}need checkpoint recovery</text>
<text x="88" y="810" font-size="24">Excluded ranks must not call shrink;default and abort have different pending-work rules</text>
<text x="88" y="880" font-size="18">Original cyclic-row teaching ledger · no native communicator,checkpoint I/O,GPU,state/optimizer recovery or wire measurements</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure(r)}
