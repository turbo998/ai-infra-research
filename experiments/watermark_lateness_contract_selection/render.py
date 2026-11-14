import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="event-time-watermark-idleness-lateness-cleanup"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==4||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||![4,8].includes(p.count_bytes))throw Error("Positive exact dimensions and4/8byte count required");
 const out={selected_partition_watermark_int64_bytes:Number(BigInt(p.partitions)*8n),selected_window_count_payload_bytes:Number(BigInt(p.active_keys)*BigInt(p.retained_windows_per_key)*BigInt(p.count_bytes))};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(maxA,maxB,bound,size,lateness,currentWM,eventTimestamp){
 if(![maxA,maxB,bound,size,lateness,currentWM,eventTimestamp].every(v=>Number.isSafeInteger(v)&&Math.abs(v)<=1000000)||bound<0||size<1||lateness<0)throw Error("Bounded exact integer timestamps,bound>=0,size>=1,lateness>=0 required");
 const a=maxA-bound-1,b=maxB-bound-1,start=Math.floor(eventTimestamp/size)*size,fire=start+size-1,cleanup=fire+lateness;
 return {partition_watermarks:[a,b],active_min:Math.min(a,b),after_B_idle:a,
 event_window:[start,start+size],fire_timestamp:fire,cleanup_timestamp:cleanup,
 event_status:currentWM>=cleanup?"rejected_window_cleaned":currentWM>=fire?"accepted_late_update":"accepted_before_fire",
 note:"Selected two-partition known-watermark controls;explicit idleness,not native timeout detection;resume realignment and all-idle flush tested in Python"};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    controls={"maxA":29,"maxB":9,"bound":2,"size":10,"lateness":5,"currentWM":14,"eventTimestamp":8}
    toy="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in controls.items())
    return page("Event-time watermark and cleanup workbench",
        f"""<h1>Idleness can advance watermarks without making every future event on time</h1>
        <p>Round129 · Original discrete event model,not native Flink/timers/checkpoints/exactly-once.</p>
        <form id="form"><h2>Selected two-partition and event-window controls</h2><div class="grid">{toy}</div>
        <h2>Independent logical count payload budget</h2><div class="grid">{fields}</div><button>Inspect watermark and cleanup boundary</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Bounded disorder is an assumption,not proof that late events cannot arrive. Unknown watermarks
        remain null in sealed data,never0. Pinned valve flushes maximum known watermark when all inputs become idle;
        resumed lagging inputs remain unaligned until they catch up.</p>""",
        "const DEFAULT="+json.dumps(p)+";\nconst CONTROLS="+json.dumps(list(controls))+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const values=CONTROLS.map(k=>Number(document.getElementById(k).value));
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(...values),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["scan_median_ms"] for v in r["latency"]["rows"])
    mx=max(mx,max(v["heap_median_ms"] for v in r["latency"]["rows"]))
    chart="".join(f"<div class='metric'><b>{v['partitions']} partitions ·128updates</b><p>Lazy heap {v['heap_median_ms']:.6f}ms</p>{bar(v['heap_median_ms'],mx)}"
                  f"<p>List minimum {v['scan_median_ms']:.6f}ms</p>{bar(v['scan_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · EVENT-TIME STATE","Partition watermarks26 and6 combine to6;marking the slow input idle permits26",
         "<div class='grid'><div><h3>Scenarios</h3><p>Bounded disorder<br>Idle/resume alignment<br>Window cleanup</p></div>"
         "<div><h3>Input</h3><p>Max timestamps29,9<br>Bound2<br>Window[0,10),lateness5</p></div>"
         "<div><h3>Top decisions</h3><p>Source WM=max−bound−1<br>Fire9,cleanup14</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned FlinkNov13tree;latest savepoint-format change is not new watermark theory. Original CPU Python{r['latency']['python']} / {r['latency']['machine']};synthetic,independently_reproduced=false.</div>"),
        ("SCENARIO: WINDOW BOUNDARY","Window[0,10) still accepts an update at WM13;it rejects the same timestamp at WM14",
         f"<blockquote>Allowed lateness5;max timestamp9;cleanup timestamp14. Before:{c['window']['wm13_event']['status']};at boundary:{c['window']['wm14_event']['status']}.</blockquote>"
         "<div class='metric'><span class='pill'>choice · deterministic policy,not confidence</span><h3>WM13:accepted late update</h3><p>Act automatically while window state is retained</p></div>"
         "<div class='metric'><span class='pill'>choice · deterministic policy,not confidence</span><h3>WM14:window cleaned</h3><p>Confirm late-data handling;do not silently recreate final state</p></div>"
         "<p>Trigger deadline and cleanup deadline are separate. No end-to-end exactly-once or sink retraction reproduction.</p>"),
        ("WHAT I TESTED","Idleness removes a minimum constraint;it is not a guarantee that a resumed source has caught up",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Unknown active source</td><td>Emit finite minimum?</td><td>✕ Still unknown</td><td>Keep null</td></tr>"
         "<tr><td>Slow source idle</td><td>Minimum can advance?</td><td>6→26</td><td>Declare policy</td></tr>"
         "<tr><td>Resume old6</td><td>Roll back global26?</td><td>No,unaligned</td><td>Wait for catch-up</td></tr>"
         "<tr><td>All idle</td><td>Always freeze old minimum?</td><td>✕ Pinned max flush</td><td>Read actual version</td></tr>"
         "<tr><td>WM14,window0</td><td>Accept timestamp8?</td><td>✕ Cleanup reached</td><td>Explicit rejection</td></tr></table>"
         "<div class='grid two'><div class='callout'>Idleness transitions are explicit controls,not wall-clock timeout detection.</div><div class='callout'>Window rejection is modeled;native side-output element predicate and triggers are distinct.</div></div>"),
        ("UNDER THE HOOD","Output watermarks are monotonic minima of aligned active inputs,with a pinned all-idle flush",
         "<pre>Source periodic candidate:max_timestamp−B−1\nActive aligned minimum → advance only\nIdle input → remove from aligned set\nResume below output → active but unaligned\nCatch up to output → realign\nAll idle transition → flush known maximum</pre>"
         "<p>↓ Original window boundary oracle</p><pre>Discrete interval[start,end)\nfire=end−1\ncleanup=end−1+allowed_lateness\nreject when currentWM &gt;= cleanup\n\nNo checkpoint/sink transaction model</pre>"),
        ("ROBUSTNESS CHECK","Both all-idle orders finish at20 with pinned flush;without it one order stops at10",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Idle order0,1</td><td>{c['all_idle_pinned_flush'][0]}</td><td>{c['wrong_no_flush'][0]}</td><td>Event-time units</td></tr>"
         f"<tr><td>Idle order1,0</td><td>{c['all_idle_pinned_flush'][1]}</td><td>{c['wrong_no_flush'][1]}</td><td>Final20versus10</td></tr></table>"
         "<p>Pinned source behavior,not timeless statement for every Flink release. Long integer overflow/triggers/network/idleness clocks not reproduced.</p>"),
        ("LATENCY","Original heap/list minimum timings cover active aggregation only,not watermark propagation latency",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means.128monotonic updates,initialization included.Heap retains stale entries until needed;native Flink/idleness/network/window backend excluded.</p>"),
        ("MY TAKE","Declare disorder,idleness,alignment and retained-state policy before trusting streaming feature freshness",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Event-time count/window audits,late-update boundaries,idle/resume and known-versus-unknown watermark checks.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Watermarks as certainty,all-idle assumptions across versions,exactly-once claims or native latency predictions.</p></div></div>"
         "<div class='dark'>Apache-2.0Flinksource;original discrete model/oracles/art.Native timing/checkpointing/exactly-once/peakstate unknown.Browser inspection not performed.</div>"),
    ]
    return page("Event-time watermark contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Event-time window ticket cabinet: minimum,idleness and retained state</title>
<desc>Original sixty-solid sorting cabinet metaphor for event-time window drawers and watermark ownership.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG129 · EVENT-TIME WINDOW CONTRACTS</text>
<text x="88" y="158" font-size="36" font-weight="700">A watermark advances policy,not certainty</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Aligned active inputs constrain minimum;idle/resume changes ownership</text>
<text x="88" y="758" font-size="25">Fire and cleanup are separate deadlines;allowed lateness retains state</text>
<text x="88" y="806" font-size="25">Unknown remains null;all-idle flush behavior is version-specific</text>
<text x="88" y="880" font-size="18">Original discrete event model / independent invariants / CPU aggregation · no native exactly-once or latency reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
