import html
import json
from pathlib import Path
from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="managed-engine-launcher-sigterm-frontend-drain-budget"
JS=r"""
function integer(x,max=600000){return Number.isInteger(x)&&x>=0&&x<=max;}
function plan(p,strategy="drain_first"){
 if(!p||Object.keys(p).sort().join(",")!=="already_started,api_drain_ms,budget_ms,has_api,launcher_exit_after_ms,pid_present,reason,requests,serve_error,stragglers")throw Error("Exact original shutdown fields required");
 if(["has_api","pid_present","already_started","serve_error"].some(k=>typeof p[k]!=="boolean")||!["signal","server","engine_exit"].includes(p.reason))throw Error("Explicit lifecycle flags/reason required");
 if(!integer(p.budget_ms)||!integer(p.api_drain_ms)||!integer(p.stragglers,16)||(p.launcher_exit_after_ms!==null&&!integer(p.launcher_exit_after_ms))||!Array.isArray(p.requests)||p.requests.length<1||p.requests.length>64||p.requests.some(x=>!integer(x)))throw Error("Bounded nonnegative original times/counts required");
 if(!p.has_api&&p.api_drain_ms!==0)throw Error("Headless profile has no frontend drain");
 if(p.reason==="engine_exit"&&p.pid_present)throw Error("Observed-exit profile cannot claim live launcher PID");
 if(!["drain_first","legacy_group_first"].includes(strategy))throw Error("Known strategy required");
 if(p.already_started||!p.pid_present)return {outcome:p.already_started?"no_op_already_started":"no_op_no_pid",events:[],engine_start_ms:null,remaining_budget_ms:null,launcher_wait_ms:null,modeled_end_ms:null,modeled_deadline_overrun_ms:null,completed_requests:null,failed_requests:null,stragglers_remaining:null,native_shutdown_verified:null};
 const drain=p.has_api&&p.reason!=="server"?p.api_drain_ms:0,start=strategy==="drain_first"?drain:0;
 const remaining=Math.max(0,p.budget_ms-start),wait=Math.max(strategy==="drain_first"?remaining:p.budget_ms,5000);
 const timeout=p.launcher_exit_after_ms===null||p.launcher_exit_after_ms>wait,engineEnd=start+(timeout?wait:p.launcher_exit_after_ms);
 const events=[{at_ms:0,event:"cancel_serving"}];
 if(strategy==="drain_first"&&drain)events.push({at_ms:drain,event:"frontend_await_complete"});
 events.push({at_ms:start,event:strategy==="drain_first"?"TERM_launcher":"TERM_group"});
 events.push({at_ms:engineEnd,event:timeout?"KILL_group_timeout":"launcher_exit_observed"});
 if(strategy==="drain_first")events.push({at_ms:engineEnd,event:"KILL_group_post_exit"});
 const end=Math.max(engineEnd,drain);
 if(strategy==="legacy_group_first"&&drain)events.push({at_ms:drain,event:"frontend_await_complete"});
 if(p.serve_error||p.reason==="server")events.push({at_ms:end,event:"preserve_serve_error"});
 const stop=strategy==="drain_first"?engineEnd:0,complete=p.requests.filter(t=>t<=stop).length;
 return {outcome:p.serve_error||p.reason==="server"?"serve_error_after_cleanup":timeout?"forced_group_cleanup":"launcher_exit_cleanup",
 events:events.sort((a,b)=>a.at_ms-b.at_ms),engine_start_ms:start,remaining_budget_ms:remaining,launcher_wait_ms:wait,modeled_end_ms:end,
 modeled_deadline_overrun_ms:Math.max(0,end-p.budget_ms),completed_requests:complete,failed_requests:p.requests.length-complete,
 stragglers_remaining:strategy==="drain_first"||timeout?0:p.stragglers,native_shutdown_verified:null};
}
"""


def workbench(r):
    body="<h1>Drain the frontend,TERM only the launcher,then clean its group</h1><p>Signal-free original timeline.No PID lookup,process launch,signal delivery or wait.No real OS timing;finish-before-stop ties and instant modeled reap are declared assumptions.</p>"
    body+=f'<form id="form"><label>Original times and lifecycle flags<textarea id="input" rows="12" required>{html.escape(json.dumps(r["case"]["inputs"]))}</textarea></label><button>Inspect signal-free ledger</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>'
    body+="<p>Caller remaining budget is clamped by the native5000mslauncher minimum;it is not a hard end-to-end deadline.No-op completion/cleanup stays unknown,not zero.</p>"
    return page("Managed engine drain workbench",body,JS+r"""function update(e){if(e)e.preventDefault();const out=document.getElementById("result"),err=document.getElementById("error");out.textContent="";err.textContent="";
try{const p=JSON.parse(document.getElementById("input").value);out.textContent=JSON.stringify({current:plan(p),legacy_group_first_model:plan(p,"legacy_group_first")},null,2);}catch(e){err.textContent=e.message;}}
document.getElementById("form").addEventListener("submit",update);update();""")


def cards(r):
    a=r["case"]["current"];b=r["case"]["legacy"];hung=r["matrix"][1]["outputs"]
    table="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        x["name"],"Original lifecycle",x["outputs"]["outcome"],
        "✕ preserve error" if x["inputs"]["serve_error"] else "Confirm real lifecycle"))+"</tr>" for x in r["matrix"])
    mx=max(max(x["prefix_median_ms"],x["scan_median_ms"]) for x in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{x['requests']}original finishes</b><p>Sorted prefix{x['prefix_median_ms']:.6f}ms</p>{bar(x['prefix_median_ms'],mx)}<p>Linear count{x['scan_median_ms']:.6f}ms</p>{bar(x['scan_median_ms'],mx,True)}</div>" for x in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · PROCESS LIFECYCLE","Frontend drain precedes launcher-only TERM;group KILL remains final cleanup",
         "<div class='grid'><div>Scenarios<br>Frontend drain<br>Launcher timeout<br>No-op/serve error</div><div>Input<br>Original finish times<br>Caller budget<br>No real PID/signals</div>"
         f"<div>Decision<br>{a['completed_requests']}modeled completions<br>{a['launcher_wait_ms']}mswait floor<br>Native drain unknown</div></div>"
         f"<div class='dark'>Synthetic,independently_reproduced=false.ActualNov14source;original CPUfinish counting separately measured on Python{r['latency']['python']}/{r['latency']['machine']}.No process/signals/GPU.</div>"),
        ("SCENARIO: TWO REQUESTS",f"Original drain-first completes{a['completed_requests']}/{len(r['case']['inputs']['requests'])};legacy early group TERM completes{b['completed_requests']}",
         "<blockquote>Finish times2s/6s,frontend drain6s,caller budget10s,launcher exit1safterTERM.One original TERM-ignoring straggler.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · Confirm actual DP drain · confidence unknown</span><p>Launcher TERM at{a['engine_start_ms']}ms;post-exit group cleanup at{a['modeled_end_ms']}ms.</p></div>"
         f"<div class='metric'><span class='pill'>score · original completion count</span><p>Drain-first{a['completed_requests']}</p>{bar(a['completed_requests'],len(r['case']['inputs']['requests']))}<p>Legacy{b['completed_requests']}</p>{bar(b['completed_requests'],len(r['case']['inputs']['requests']))}</div>"),
        ("WHAT I TESTED",f"{len(r['matrix'])}original cases separate drain,timeout,minimum wait and no-op uncertainty",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Positive launcher PID for TERM;negative process-group target for KILL.Native Rust tests read,not executed.</div><div class='callout'>Serve errors remain errors after cleanup.No-PID/already-started returns do not prove workers exited.</div></div>"),
        ("UNDER THE HOOD","Remaining budget saturates at zero,but the engine still applies a five-second minimum",
         "<pre>Cancel serving\nAwait frontend drain unless already done\nremaining=max(0,deadline-now)\nwait=max(remaining,5000ms)\nTERM launcher only</pre><p>↓ Cleanup</p>"
         "<pre>Wait for launcher observation\nTimeout:KILL group and await exit\nAfter exit:KILL group stragglers\nThen preserve serve result/error\nNative polling/reaping duration unknown</pre>"),
        ("ROBUSTNESS CHECK",f"A never-exiting original launcher ends at{hung['modeled_end_ms']}ms,over its10sbudget by{hung['modeled_deadline_overrun_ms']}ms",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Caller remaining</td><td>{a['remaining_budget_ms']}ms</td><td>Wait{a['launcher_wait_ms']}ms</td><td>Minimum applies</td></tr>"
         f"<tr><td>Launcher observed</td><td>{a['modeled_end_ms']}msend</td><td>{hung['modeled_end_ms']}msend</td><td>No hard deadline claim</td></tr>"
         f"<tr><td>Stragglers</td><td>{a['stragglers_remaining']}after cleanup</td><td>{b['stragglers_remaining']}legacy graceful return</td><td>Post-exit group cleanup</td></tr>"
         "<tr><td>No-op</td><td>Cleanup branch</td><td>Completion null</td><td>Unknown is not zero</td></tr></table>"),
        ("LATENCY","Original sorted finish counting excludes sorting,signals,OS waits and native drain",timing+
         "<p>2warmups,7alternating repeats,100counts/sample;median repeat means.Pre-sorted input,finish<=cutoff tie rule.This metadata kernel is not graceful shutdown latency.</p>"),
        ("MY TAKE","Keep engine workers available while requests drain;measure the real shutdown budget separately",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3>Explicit frontend/launcher/worker ownership,remaining-budget accounting,post-exit cleanup and preserved failure outcomes.</div><div class='bad'><h3>⛔ Not for</h3>Group TERM at cancellation,assuming no-op means fully reaped,hard deadline claims despite minimum waits,or running toy signals on production.</div></div>"
         "<div class='dark'>Apache-2.0rights retained.No native processes/signals executed.Original instant reap/tie assumptions;native DP/token streaming/OS/visual acceptance unknown.</div>")]
    return page("Managed engine drain — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {x}</p><h1>{y}</h1>{z}</section>' for i,(x,y,z) in enumerate(sections,1)))


def outputs(r):
    a=r["case"]["current"]
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000"><title>Drain interlock cabinet</title><desc>Original sixty-solid cabinet;worker compartments stay available until frontend drain,then the launcher interlock closes before group cleanup.</desc><rect width="1600" height="1000" fill="#f1f5f9"/><rect x="44" y="40" width="1512" height="920" rx="28" fill="white" stroke="#cbd5e1"/><g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG168 · DRAIN INTERLOCK CABINET</text><text x="88" y="158" font-size="34">Keep workers serving until frontend drain has completed</text>{sorter_scene()}<text x="88" y="710" font-size="24">Original two completions / TERM launcher at{a['engine_start_ms']}ms / cleanup at{a['modeled_end_ms']}ms</text><text x="88" y="760" font-size="24">Remaining{a['remaining_budget_ms']}ms / native minimum wait{a['launcher_wait_ms']}ms</text><text x="88" y="810" font-size="24">Group KILL after timeout or launcher exit;no-op state is not cleanup proof</text><text x="88" y="880" font-size="18">ActualNov14source · original signal-free timeline · no native process/signals/DP/visual acceptance</text></g></svg>\n"""
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":svg}
