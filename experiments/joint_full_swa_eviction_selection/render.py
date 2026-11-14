import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="rust-joint-full-swa-eviction-recency-strip-only"
JS=r"""
function integer(v,max=1073741824){return Number.isInteger(v)&&v>=0&&v<=max;}
function validate(p){
 const keys="chain_allocator,chain_mamba,chain_window,core_supports_joint,nodes,policy,session_cache,state_slot,target_units,tree_enabled,windows";
 if(!p||Object.keys(p).sort().join(",")!==keys||!["lru","joint_lru"].includes(p.policy)||!integer(p.windows,4)||!integer(p.target_units))throw Error("Exact bounded joint profile required");
 for(const k of ["tree_enabled","state_slot","chain_allocator","chain_window","chain_mamba","core_supports_joint","session_cache"])if(typeof p[k]!=="boolean")throw Error("Explicit feature witnesses required");
 if(!Array.isArray(p.nodes)||p.nodes.length>64)throw Error("Bounded original forest required");
 const nk="full,full_locked,full_tick,full_units,nid,parent,swa,swa_locked,swa_rank,swa_tick,swa_units";
 for(const r of p.nodes){
 if(!r||Object.keys(r).sort().join(",")!==nk||!integer(r.nid,4095)||r.parent!==null&&!integer(r.parent,4095))throw Error("Exact node identity fields required");
 for(const k of ["full","swa","full_locked","swa_locked"])if(typeof r[k]!=="boolean")throw Error("Explicit component/lock witnesses required");
 for(const k of ["full_tick","swa_rank","full_units","swa_units"])if(!integer(r[k]))throw Error("Bounded ticks/resource units required");
 if(r.swa&&!integer(r.swa_tick)||!r.swa&&r.swa_tick!==null||r.full&&!r.full_units||r.swa&&!r.swa_units)throw Error("Live units/ticks and absent null SWA tick required");
 }
 const by=new Map(p.nodes.map(r=>[r.nid,r]));if(by.size!==p.nodes.length)throw Error("Unique nodes required");
 const ranks=p.nodes.filter(r=>r.swa).map(r=>r.swa_rank);if(new Set(ranks).size!==ranks.length)throw Error("Unique SWA LRU ranks required");
 for(const r of p.nodes){if(r.parent!==null&&!by.has(r.parent))throw Error("Missing parent");let id=r.nid,seen=new Set();while(id!==null){if(seen.has(id))throw Error("Forest cycle");seen.add(id);id=by.get(id).parent;}}
}
function mode(p){const active=p.policy==="joint_lru"&&p.tree_enabled&&p.windows===1&&!p.state_slot&&p.chain_allocator&&p.chain_window&&!p.chain_mamba;
 return active&&(p.session_cache||!p.core_supports_joint)?"reject_unsupported_joint":active?"joint":"component_lru";}
function cmp(a,b){for(let i=0;i<4;i++)if(a[i]!==b[i])return a[i]-b[i];return 0;}
function plan(p,collapse=false){
 validate(p);const m=mode(p),out={mode:m,trace:[],freed_units:0,retained_full_units:p.nodes.filter(r=>r.full).reduce((s,r)=>s+r.full_units,0),target_met:null,stale_entries_skipped:0,states:[],native_rust_verified:null,native_allocator_verified:null};
 if(m!=="joint")return out;
 const rows=p.nodes.map(r=>({...r})),by=new Map(rows.map(r=>[r.nid,r]));
 const full=r=>r.full&&!r.full_locked&&!r.swa_locked&&!rows.some(c=>c.parent===r.nid&&c.full);
 const eligible=(r,ct)=>ct===0?full(r):r.swa&&!r.swa_locked;
 const entry=(r,ct)=>[ct===0?r.full_tick:r.swa_tick,ct,ct===0?r.nid:r.swa_rank,r.nid];
 let queue=rows.flatMap(r=>[0,1].filter(ct=>eligible(r,ct)).map(ct=>entry(r,ct)));
 while(out.freed_units<p.target_units){
 queue.sort(cmp);let selected=null;
 while(queue.length){const v=queue.shift();if(eligible(by.get(v[3]),v[1])){selected=v;break;}out.stale_entries_skipped++;}
 if(!selected)break;const ct=selected[1],r=by.get(selected[3]);let freed=0;
 if(ct===1){freed=r.swa_units;r.swa=false;r.swa_tick=null;if(collapse&&r.full){freed+=r.full_units;r.full=false;}}
 else{freed=r.full_units;r.full=false;if(r.swa){freed+=r.swa_units;r.swa=false;r.swa_tick=null;}}
 out.freed_units+=freed;out.trace.push({component:ct===0?"full":"swa",nid:r.nid,freed_units:freed,full_retained:r.full,swa_tick_after:r.swa_tick});
 const parent=by.get(r.parent);if(parent&&full(parent))queue.push(entry(parent,0));
 }
 out.retained_full_units=rows.filter(r=>r.full).reduce((s,r)=>s+r.full_units,0);out.target_met=out.freed_units>=p.target_units;
 out.states=rows.map(r=>({nid:r.nid,full:r.full,swa:r.swa,swa_tick:r.swa_tick}));return out;
}
"""


def workbench(r):
    return page("Joint Full/SWA victim workbench","<h1>An SWA-selected leaf releases only SWA;Full can remain for a later victim</h1>"
        "<p>Original resource-unit forest,not native radix keys,allocator compaction or async backup.Native active joint mode requires one window/no state over the two-ended chain.</p>"
        f'<form id="form"><label>Feature witnesses,component ticks,locks and declared units<textarea id="input" rows="14" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Compare strip-only and wrong collapse proposal</button></form>'
        '<p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>JS uses an ordered queue;Python heap and independent eligible-min reference agree on original cases.No native Rust execution or cache-hit/throughput claim.</p>",
        JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),counterfactual_collapse:plan(p,true)},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["collapse_proposal"];p=r["case"]["inputs"];lost=a["retained_full_units"]-b["retained_full_units"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (x["name"],"victim/mode",x["outputs"]["trace"] or x["outputs"]["mode"],"Confirm "+str(x["outputs"]["target_met"])))+"</tr>" for x in r["matrix"])
    matrix+=f"<tr><td>Collapse proposal</td><td>Retain unselected Full</td><td>✕ loses{lost}units</td><td>Reject</td></tr>"
    mx=max(max(x["heap_median_ms"],x["linear_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['candidates']}keys</b><p>Heap{x['heap_median_ms']:.6f}ms</p>{bar(x['heap_median_ms'],mx)}"
                  f"<p>Linear min{x['linear_median_ms']:.6f}ms</p>{bar(x['linear_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sp=r["split_case"]
    sections=[
        ("HANDS-ON · SHARED KV ARENA","Rust now implements the joint walk;one window/no state and two-ended allocation still gate activation",
         "<div class='grid'><div>Scenarios<br>Strip-only SWA<br>Full parent reentry<br>Split recency</div><div>Input<br>Shared access ticks<br>Eligible locks/leaves<br>Declared units</div>"
         f"<div>Decision<br>Release{a['freed_units']}units<br>Retain Full{a['retained_full_units']}<br>Native unverified</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14Rust addition.CPUqueue keys onPython{r['latency']['python']}/{r['latency']['machine']},not native cache throughput.</div>"),
        ("SCENARIO: SWA FIRST",f"Original target{p['target_units']}releases{a['freed_units']};wrong collapse loses{lost}unselected Full units",
         "<blockquote>Heap compares Full and SWA access ticks,then component,then rank.SWA selection does not imply collapsing Full.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm resource model · confidence unmeasured</span>{html.escape(str(a['trace']))}</div>"
         f"<div class='metric'><span class='pill'>score · exact original retained Full units</span><p>Strip-only{a['retained_full_units']}</p>{bar(a['retained_full_units'],a['retained_full_units'])}<p>Wrong collapse{b['retained_full_units']}</p>{bar(b['retained_full_units'],a['retained_full_units'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original cases separate joint activation,strip-only safety and dynamic victim eligibility",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Cold/inactive joint profile remains component LRU,not a simulated native fallback.</div><div class='callout'>Wrong collapse is counterfactual;old active Rust joint mode was unsupported.</div></div>"),
        ("UNDER THE HOOD","A Full parent enters the heap only after child eviction makes it eligible",
         "<pre>Heap: (tick,component,rank,node)\nEligible Full leaves + live unlocked SWA\nFull rank=node ID;SWA rank=LRU enumeration\nValidate entry again when popped</pre><p>↓ Victim application</p>"
         "<pre>SWA → strip only;clear SWA tick\nFull → leaf reclamation/cascade\nRe-admit newly eligible Full parent\nSkip candidates removed by earlier cascade\nNative pending SWA finish retains strip-only too</pre>"),
        ("ROBUSTNESS CHECK",f"Original prefix-only access keeps suffix Full tick{sp['joint']['suffix_full_tick']},not{sp['other_policy']['suffix_full_tick']}",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Suffix Full tick</td><td>{sp['inputs']['full_tick']}</td><td>Prefix access{sp['inputs']['prefix_access']}</td><td>Joint suffix unchanged</td></tr>"
         f"<tr><td>SWA split tick</td><td>{sp['inputs']['swa_tick']}</td><td>Prefix fragment</td><td>Inherits same tick</td></tr>"
         "<tr><td>SWA removal</td><td>Live tick</td><td>Value removed</td><td>Tick null</td></tr>"
         "<tr><td>Core capability</td><td>Supports joint</td><td>Unsupported</td><td>Reject active profile</td></tr></table>"),
        ("LATENCY","Static CPUvictim-key queue drain excludes native tree,compaction and backup barriers",timing+
         "<p>2warmups,7alternating repeats,3drains/sample;median repeat means.No dynamic-parent work,native Rust/Torch/GPU or cache-hit timing.</p>"),
        ("MY TAKE","Use comparable component recency and component-only reclamation,not unconditional leaf collapse",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Shared Full/SWA arena,one window,no state,capable core,eligible victims and preserved split recency.</div>"
         "<div class='bad'><h3>⛔ Not for</h3>Several windows,state pools,unsupported/session cores,seconds-based ticks or native throughput claims.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Original victim forest;native Rust/allocator/async backup and browser acceptance unknown.</div>")]
    return page("Joint Full/SWA eviction — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["collapse_proposal"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Joint recency eviction cabinet</title><desc>Original sixty-solid shared cabinet;oldest eligible component is removed without collapsing unselected Full.Not native trie/allocator geometry.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG163 · JOINT COMPONENT RECENCY</text><text x="88" y="158" font-size="34">An SWA victim is strip-only;Full remains for its own later turn</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original release{a['freed_units']}units;retain Full{a['retained_full_units']}versus wrong collapse{b['retained_full_units']}</text><text x="88" y="760" font-size="24">Newly eligible Full parent re-enters;stale cascade victims are skipped</text><text x="88" y="810" font-size="24">Prefix split preserves suffix recency;removed SWA tick becomes null</text><text x="88" y="880" font-size="18">ActualNov14Rust support · original resource ledger,not native trie/compaction/backup or visual acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
