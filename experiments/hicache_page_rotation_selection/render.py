import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page, bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT = Path(__file__).resolve().parents[2]
SLUG = "hicache-logical-page-backend-staged-rotation"
JS = r"""
function integer(v,max=1048576){return Number.isInteger(v)&&v>=0&&v<=max;}
function validate(p){
 const keys="backend,base_at_admit,base_at_alloc,fetch_pages,found_at_admit,found_at_alloc,free_pages,hicache,joint_tokens,page_size,prefix_tokens,rank,registered,shards";
 if(!p||Object.keys(p).sort().join(",")!==keys)throw Error("Exact page-owner fields required");
 if(!integer(p.page_size)||!p.page_size||!integer(p.shards,16)||!p.shards||!integer(p.rank,p.shards-1)||typeof p.hicache!=="boolean")throw Error("Bounded page/shard/rank and HiCache witness required");
 if(typeof p.backend!=="string"||!p.backend||!Array.isArray(p.registered)||p.registered.length>16||p.registered.some(v=>typeof v!=="string"||!v)||new Set(p.registered).size!==p.registered.length)throw Error("Named backend/registry witnesses required");
 for(const k of ["prefix_tokens","found_at_alloc","found_at_admit","joint_tokens"])if(!integer(p[k])||p[k]%p.page_size)throw Error("Aligned observed token prefixes required");
 for(const k of ["base_at_alloc","base_at_admit"])if(p[k]!==null&&!integer(p[k],p.shards-1))throw Error("Null or valid observed owner class required");
 if(!integer(p.fetch_pages,128)||p.found_at_alloc>p.prefix_tokens||p.found_at_admit>p.prefix_tokens+p.fetch_pages*p.page_size||p.joint_tokens>p.prefix_tokens+p.fetch_pages*p.page_size)throw Error("Bounded span required");
 if(!Array.isArray(p.free_pages)||p.free_pages.length!==p.shards||p.free_pages.some(v=>!integer(v)))throw Error("One observed free-page count per class required");
}
function plan(p,detached=false){
 validate(p);const n=p.shards,supported=p.registered.includes(p.backend),startup=!(n>1&&p.hicache&&!supported),attach=p.hicache&&(n===1||supported);
 const out={startup_admitted:startup,runtime_attachment_admitted:attach,actual_legacy_startup_admitted:!(n>1&&p.hicache),action:p.hicache?"unregistered_backend":"not_initialized",raw_start_owner:null,owners:[],host_logical_pages:[],owned_local_host_pages:[],page_draw_counts:[],recovered_rotation_base:null,published_rotation_base:null,owner_misses:null,owned_token_rows:null,native_plugin_verified:null,native_copy_verified:null};
 if(!startup||!attach)return out;
 let least=0;for(let c=1;c<n;c++)if(p.free_pages[c]>p.free_pages[least])least=c;
 const base=p.found_at_alloc&&p.base_at_alloc!==null?p.base_at_alloc:least,offset=p.prefix_tokens/p.page_size,desired=base+offset,start=detached?least:desired;
 const mod=v=>((v%n)+n)%n,draws=Array.from({length:n},(_,c)=>Math.floor(p.fetch_pages/n)+Number(mod(c-start)<p.fetch_pages%n));
 Object.assign(out,{raw_start_owner:start,page_draw_counts:draws});
 if(draws.some((v,c)=>v>p.free_pages[c])){out.action="attempt_capacity_miss";return out;}
 const owners=Array.from({length:p.fetch_pages},(_,j)=>mod(start+j)),used=Array(n).fill(0),host=owners.map(c=>(++used[c])*n+c),own=host.filter(h=>h%n===p.rank).map(h=>Math.floor(h/n)),recovered=owners.length?mod(owners[0]-offset):null,misses=owners.filter((c,j)=>c!==mod(desired+j)).length;
 Object.assign(out,{owners,host_logical_pages:host,owned_local_host_pages:own,recovered_rotation_base:recovered,owner_misses:misses,owned_token_rows:own.length*p.page_size});
 if(p.joint_tokens>=p.prefix_tokens+p.fetch_pages*p.page_size)out.action="device_covered";
 else if(!owners.length)out.action="no_full_span";
 else if(!detached&&p.found_at_admit>0&&p.base_at_admit!==recovered)out.action="refetch_rotation";
 else{out.action="publish_rotation";out.published_rotation_base=detached?0:recovered;}
 return out;
}
"""


def workbench(r):
    return page("HiCache logical-page rotation workbench",
        "<h1>Registration admits a contract;staged page owners still must continue the device chain</h1>"
        "<p>Original aligned metadata ledger.No upstream backend is registered by this change.No plugin or Torch storage/copy execution.</p>"
        f'<form id="form"><label>Registry,observed prefix rotation and page-class capacity<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Check owners and staged admission</button></form>'
        '<p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
        "<p>Detached baseline is an intentionally wrong proposal,not native legacy execution;old native startup rejected sharded storage."
        "Host page labels are original synthetic slots,not real allocator IDs.Capacity miss describes this observed attempt,not eviction/retry outcome.</p>",
        JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),counterfactual_detached:plan(p,true)},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    p=r["case"]["inputs"];a=r["case"]["current"];b=r["case"]["detached_proposal"]
    matrix="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (x["name"],"registry/rotation",x["outputs"]["recovered_rotation_base"],x["outputs"]["action"]))+"</tr>" for x in r["matrix"])
    matrix+=f"<tr><td>Detached proposal</td><td>Page owners</td><td>✕ {b['owner_misses']}/{p['fetch_pages']} misses</td><td>Reject proposal</td></tr>"
    mx=max(max(x["single_median_ms"],x["scan_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['pages']}page labels</b><p>Single pass{x['single_median_ms']:.6f}ms</p>{bar(x['single_median_ms'],mx)}"
                  f"<p>Four scans{x['scan_median_ms']:.6f}ms</p>{bar(x['scan_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · KV PAGE OWNERS","No upstream storage backend is registered;out-of-tree admission is an explicit contract",
         "<div class='grid'><div>Scenarios<br>Registry guard<br>Prefix rotation<br>Staged refetch</div><div>Input<br>Observed owner classes<br>Aligned prefix<br>Whole-page fetch</div>"
         f"<div>Decision<br>{a['action']}<br>Rank{p['rank']} owns{len(a['owned_local_host_pages'])}pages<br>Plugin unverified</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14addition.CPUmetadata onPython{r['latency']['python']}/{r['latency']['machine']};not HBM/IO.</div>"),
        ("SCENARIO: CONTINUE THE PREFIX",f"Original detached ownership misses{b['owner_misses']}/{p['fetch_pages']}pages;chain-aware misses{a['owner_misses']}/{p['fetch_pages']}",
         f"<blockquote>start_owner = observed base{p['base_at_alloc']} + prefix_pages{p['prefix_tokens']//p['page_size']};native cyclic allocation normalizes mod{p['shards']}.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm plugin contract · confidence unmeasured</span><h3>Current owners{a['owners']}</h3><p>Recovered base{a['recovered_rotation_base']}</p></div>"
         f"<div class='metric'><span class='pill'>score · exact original owner mismatches</span><p>Detached{b['owner_misses']}</p>{bar(b['owner_misses'],p['fetch_pages'])}<p>Current{a['owner_misses']}</p>{bar(a['owner_misses'],p['fetch_pages'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original cases keep backend admission,rotation revalidation and capacity distinct",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+matrix+"</table>"
         "<div class='grid two'><div class='callout'>Actual old startup rejected sharded storage;detached owners are only a counterexample.</div><div class='callout'>Capacity miss is one observed attempt;native eviction/retry is not executed.</div></div>"),
        ("UNDER THE HOOD","Recover the staged chain base from first-page owner minus snapshotted prefix pages",
         "<pre>Registered names only;empty upstream registry\nstart = base + prefix_tokens // physical_page\nNew chain → most free class;lowest tie wins\nHost logical h → owner h % N,local h // N</pre><p>↓ Staged admission</p>"
         "<pre>Joint device coverage resolves first\nbase = (first_owner − staged_prefix_pages) % N\nExisting device chain has different base?\n  warning + refetch;do not publish\nElse insert the loaded prefix with recovered base</pre>"),
        ("ROBUSTNESS CHECK","A changed device-chain base requires refetch,not silently relabeling staged pages",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Backend</td><td>Registered</td><td>Unregistered</td><td>Reject</td></tr>"
         "<tr><td>Prefix base</td><td>Same</td><td>Different</td><td>Refetch</td></tr>"
         "<tr><td>Device prefix</td><td>Exists</td><td>Evicted</td><td>Insert staged base</td></tr>"
         "<tr><td>Joint cover</td><td>Incomplete</td><td>Full span</td><td>Device-covered first</td></tr></table>"),
        ("LATENCY","CPUowner-partition metadata is not a native storage bandwidth or copy benchmark",timing+
         "<p>2warmups,7alternating repeats,3partitions/sample;median repeat means.Setup,Torch allocator,plugin/network/storage/HBM excluded.</p>"),
        ("MY TAKE","Require rank-local IO plus consistent owner rotation before admitting a storage plugin",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Explicit plugin contract,whole-page owner ledger,prefix lifecycle checks and preserved insert base.</div>"
         "<div class='bad'><h3>⛔ Not for</h3>Assuming all HiCache backends work,ignoring chain rotation,claiming native throughput or real slot IDs.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.Original aligned metadata only.Native plugin/copy execution and browser acceptance unknown.</div>")]
    return page("HiCache page rotation — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"];b=r["case"]["detached_proposal"];p=r["case"]["inputs"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Owner-class staging cabinet</title><desc>Original sixty-solid cabinet metaphor;classed drawers preserve page-owner rotation.Not real slot geometry or native copy/IO.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG162 · OWNER-CLASS STAGING</text><text x="88" y="158" font-size="34">Registered storage must preserve the device chain's page owners</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original detached misses{b['owner_misses']}/{p['fetch_pages']};chain-aware misses{a['owner_misses']}/{p['fetch_pages']}</text><text x="88" y="760" font-size="24">Rank{p['rank']} gets{len(a['owned_local_host_pages'])}owned pages = {a['owned_token_rows']}logical token rows</text><text x="88" y="810" font-size="24">Changed prefix base:warning/refetch;loaded insert preserves recovered base</text><text x="88" y="880" font-size="18">Empty upstream registry · original metadata planner,not native plugin/allocator/copy or browser acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
