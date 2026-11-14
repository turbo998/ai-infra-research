import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="included-router-auth-level-leaf-resolution-policy"
JS=r"""
function pathOK(x,prefix=false){return typeof x==="string"&&x.length<=96&&((prefix&&x==="")||/^\/[A-Za-z0-9_/-]*$/.test(x));}
function plan(p,strategy="included"){
 const levels=["normal","admin_optional","admin_force"],methods=["GET","POST","DELETE","HEAD","OPTIONS"];
 if(!p||Object.keys(p).sort().join(",")!=="admin_configured,api_configured,credential,scope,tree")throw Error("Exact original fields required");
 if(typeof p.api_configured!=="boolean"||typeof p.admin_configured!=="boolean"||!["none","api","admin","both","other"].includes(p.credential))throw Error("Key flags and symbolic witness required;no secrets");
 const s=p.scope;
 if(!s||Object.keys(s).sort().join(",")!=="kind,method,path"||!pathOK(s.path)||!methods.includes(s.method)||!["http","websocket"].includes(s.kind))throw Error("Literal scope required");
 let nodes=0;function validate(xs,depth){if(!Array.isArray(xs)||!xs.length||depth>8)throw Error("Bounded nonempty tree required");
 for(const x of xs){if(++nodes>128||!x||typeof x!=="object")throw Error("At most128nodes required");const ks=Object.keys(x).sort().join(",");
 if(ks==="prefix,routes"){if(!pathOK(x.prefix,true))throw Error("Literal prefix required");validate(x.routes,depth+1);}
 else if(ks==="kind,level,matcher_error,methods,path,typed"){
 if(!pathOK(x.path)||![...levels,"invalid"].includes(x.level)||typeof x.typed!=="boolean"||typeof x.matcher_error!=="boolean"||!["http","websocket"].includes(x.kind)||!Array.isArray(x.methods)||!x.methods.length||x.methods.some(m=>!methods.includes(m)))throw Error("Original leaf metadata required");
 }else throw Error("Leaf or include node required");}}
 validate(p.tree,0);if(!["included","top_only"].includes(strategy))throw Error("Known strategy required");
 const rows=[];function flatten(xs,prefix){for(const x of xs){if("routes" in x){if(strategy==="included")flatten(x.routes,prefix+x.prefix);}
 else rows.push({...x,prefix,resolved_level:x.typed&&levels.includes(x.level)?x.level:"normal"});}}
 flatten(p.tree,"");let level="normal",path=null;const trace=[];
 for(const x of rows){let result="NONE";if(!s.path.startsWith(x.prefix))result="PREFIX_SKIP";else if(x.matcher_error)result="MATCHER_ERROR";
 else if(s.kind===x.kind&&s.path.slice(x.prefix.length)===x.path)result=x.kind==="websocket"||x.methods.includes(s.method)?"FULL":"PARTIAL";
 trace.push({path:x.prefix+x.path,result});if(result==="FULL"){level=x.resolved_level;path=x.prefix+x.path;break;}}
 let decision;if(s.method==="OPTIONS"||["/health","/ready","/metrics"].some(x=>s.path.startsWith(x)))decision={allowed:true,status:null,reason:"preflight_or_probe_prefix"};
 else if(level==="admin_force"&&!p.admin_configured)decision={allowed:false,status:403,reason:"admin_key_not_configured"};
 else{const expected=level==="admin_force"?"admin":level==="admin_optional"?(p.admin_configured?"admin":p.api_configured?"api":null):(p.api_configured?"api":null);
 const allowed=expected===null||p.credential===expected||p.credential==="both";decision={allowed,status:allowed?null:401,reason:expected===null?"no_key_required":allowed?"symbolic_credential_match":"symbolic_credential_mismatch"};}
 return {resolved_level:level,matched_path:path,registered_leaves:rows.length,examined_leaves:trace.length,resolution_trace:trace,decision,native_routing_verified:null,credential_comparison_verified:null,security_certification:null};
}
"""


def workbench(r):
    body="<h1>Resolve leaf metadata before applying the key profile</h1><p>Original literal route tree,not a server or exploit test.Use only symbolic credential roles,never keys.Enum metadata is represented by the typed witness;JSON strings are not native Enum instances.</p>"
    body+=f'<form id="form"><label>Original registry/deployment/scope<textarea id="input" rows="17" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Resolve symbolic policy</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
    body+="<p>API/admin mean exclusive matching witnesses;both covers matching both keys.Logical decision status is not WebSocket wire status.DFS registration order and first FULL match matter.No native templates,root_path,mounts,redirects,headers,constant-time token comparison or live HTTP/WebSocket execution.</p>"
    return page("Included route metadata workbench",body,JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),wrong_top_only_registry:plan(p,"top_only")},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["top_only"]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],x["outputs"]["resolved_level"],
        "allow" if x["outputs"]["decision"]["allowed"] else "✕ "+str(x["outputs"]["decision"]["status"]),
        "Confirm deployed profile"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["compiled_median_ms"],x["dynamic_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['leaves']}literal leaves</b><p>Compiled{x['compiled_median_ms']:.6f}ms</p>{bar(x['compiled_median_ms'],mx)}<p>Rebuild each lookup{x['dynamic_median_ms']:.6f}ms</p>{bar(x['dynamic_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · CONTROL PLANE","Included-router prefixes must reach leaf metadata before existing key policy applies",
         "<div class='grid'><div>Scenarios<br>Nested includes<br>Partial/full matches<br>Normal/admin profiles</div><div>Input<br>Literal route tree<br>Protocol/method<br>Symbolic role only</div>"
         f"<div>Decision<br>{a['registered_leaves']}leaves<br>API witness denied{a['decision']['status']}<br>Native auth unverified</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14source;original CPUlookup separately measured on Python{r['latency']['python']}/{r['latency']['machine']}.No secrets,network or security certification.</div>"),
        ("SCENARIO: NESTED ADMIN",f"The original five-leaf model resolves admin_optional,not top-only normal fallback",
         "<blockquote>Nested /api + /control + /pause.Both key profiles configured;symbolic API credential does not satisfy admin_optional when admin key exists.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm deployed control-plane profile · confidence unknown</span><p>Resolved{a['resolved_level']};deny{a['decision']['status']}.Top-only counterfactual allows the API witness.</p></div>"
         f"<div class='metric'><span class='pill'>score · original registered leaf count</span><p>Included registry{a['registered_leaves']}</p>{bar(a['registered_leaves'],a['registered_leaves'])}<p>Top-only{b['registered_leaves']}</p>{bar(b['registered_leaves'],a['registered_leaves'])}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original cases separate route resolution from normal/optional/forced policy",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>admin_optional selects configured admin over API;admin_force without admin configuration returns403.</div><div class='callout'>Normal uses API configuration only;admin-only does not make normal routes admin-protected.Native policy observation,not a recommendation.</div></div>"),
        ("UNDER THE HOOD","First FULL leaf wins in registration order;invalid/missing Enum metadata falls back to normal",
         "<pre>Walk included routers recursively\nAccumulate include prefixes\nStrip prefix for route matching\nSkip partial/nonmatching candidates\nRead first FULL leaf endpoint Enum</pre><p>↓ Independent policy</p>"
         "<pre>OPTIONS/probe raw-prefix bypass first\nNORMAL:API key when configured\nOPTIONAL:admin > API > no key\nFORCE:admin required;unset gives403\nNative headers/token comparison not modeled</pre>"),
        ("ROBUSTNESS CHECK","Changing protocol,method or metadata type can change the resolved leaf without changing policy code",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Metadata type</td><td>Typed admin_optional</td><td>Raw string witness</td><td>Normal fallback</td></tr>"
         "<tr><td>Duplicate literal</td><td>First FULL leaf</td><td>Reverse registration</td><td>Order-dependent</td></tr>"
         "<tr><td>Wrong method</td><td>PARTIAL candidate</td><td>Later FULL leaf</td><td>Later metadata wins</td></tr>"
         "<tr><td>Probe path</td><td>/health raw prefix</td><td>/api/health</td><td>Not the same bypass</td></tr></table>"),
        ("LATENCY","Original compiled literal lookup excludes registry build;native FastAPI still resolves dynamically",timing+
         "<p>2warmups,7alternating repeats,3batches/sample;median repeat means.Includes original trace allocation.No nativeframework/token/header/network.Routes are immutable snapshots;changes require rebuild.</p>"),
        ("MY TAKE","Audit included leaf declarations and deployment profiles,not just top-level route names",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Explicit leaf Enum metadata,full route inventory,protocol/method tests,and separately reviewed control-plane key configuration.</div><div class='bad'><h3>⛔ Not for</h3>Assuming optional means unauthenticated,admin-only protects normal routes,or compiled toy timings prove live server behavior.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.No implementation mirror,secret handling,exploitable-vulnerability review or security certification.Native framework/Rust/visual acceptance unknown.</div>")]
    return page("Included route metadata — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Leaf policy register cabinet</title><desc>Original sixty-solid register cabinet;leaf compartments carry metadata after nested include prefixes,not live authorization.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG167 · LEAF POLICY REGISTER</text><text x="88" y="158" font-size="34">Resolve included leaves before evaluating key configuration</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original{a['registered_leaves']}leaves / two includes / admin_optional denies API witness{a['decision']['status']}</text><text x="88" y="760" font-size="24">First FULL match + typed leaf metadata / no specificity sort</text><text x="88" y="810" font-size="24">Normal,admin_optional and admin_force remain different policies</text><text x="88" y="880" font-size="18">ActualNov14source · original symbolic literal model · native routing/auth/security/visual acceptance unknown</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
