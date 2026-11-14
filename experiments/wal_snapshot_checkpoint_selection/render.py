import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="sqlite-wal-reader-endmark-checkpoint-visibility"
JS=r"""
function play(commands){
 if(!Array.isArray(commands)||commands.length>128)throw Error("At most128commands required");
 let db={"1":"A0","2":"B0"},frames=[],head=0,backfill=0;const readers=new Map(),used=new Set(),trace=[];
 function read(page,end){for(let i=end-1;i>=0;i--)if(frames[i].page===page)return frames[i].value;return db[page];}
 function values(end){return {"1":read("1",end),"2":read("2",end)};}
 function snapshot(){
  const entries=[...readers].sort(([a],[b])=>a<b?-1:a>b?1:0);
  return {committed_frames:head,model_frames:frames.length,backfilled:backfill,database:{...db},readers:Object.fromEntries(entries),current:values(head),
   reader_values:Object.fromEntries(entries.map(([name,end])=>[name,values(end)]))};
 }
 for(const cmd of commands){
  if(!Array.isArray(cmd)||!cmd.length)throw Error("Nonempty command array required");const op=cmd[0];let result=null;
  if(op==="commit"&&cmd.length===2){
   const data=cmd[1];if(!data||Array.isArray(data)||typeof data!=="object"||!Object.keys(data).length)throw Error("Original page update object required");
   const pairs=Object.entries(data).sort(([a],[b])=>a<b?-1:a>b?1:0);
   if(pairs.some(([k,v])=>!["1","2"].includes(k)||typeof v!=="string"||!/^[\x20-\x7e]{0,64}$/.test(v)))throw Error("Page keys1or2and printable-ASCII values required");
   if(frames.length+pairs.length>4096)throw Error("Original frame cap exceeded");
   pairs.forEach(([page,value],i)=>frames.push({page,value,commit_end:i===pairs.length-1}));head=frames.length;
  }else if(op==="reader"&&cmd.length===2){
   const name=cmd[1];if(typeof name!=="string"||!/^[A-Za-z0-9_-]{1,32}$/.test(name)||used.has(name)||readers.size>=16)throw Error("Bounded nonreused reader name required");
   readers.set(name,head);used.add(name);
  }else if(op==="release"&&cmd.length===2){
   if(!readers.has(cmd[1]))throw Error("Unknown original reader");readers.delete(cmd[1]);
  }else if(op==="checkpoint"&&cmd.length===1){
   const safe=Math.min(head,...readers.values());for(let i=backfill;i<safe;i++)db[frames[i].page]=frames[i].value;backfill=safe;
   result={eligible_head:safe,backfilled:backfill,committed_frames:head,fully_backfilled:backfill===head};
  }else if(op==="reset"&&cmd.length===1){
   if(readers.size||backfill!==head||frames.length!==head)throw Error("Original reset needs no readers and complete backfill");
   frames=[];head=0;backfill=0;
  }else throw Error("Unsupported original command or argument count");
  trace.push({command:cmd,result,snapshot:snapshot()});
 }
 return trace;
}
"""


def workbench(r):
    return page("SQLite WAL snapshots and checkpoint workbench",
        f"""<h1>Commit visibility is not completed checkpoint or WAL truncation</h1>
        <p>Round142 · Original sequential fixed-page model.No SQL engine,real locks,fsync or binary recovery in this workbench.</p>
        <form id="form"><label>Original command arrays<textarea id="input" rows="12" required>{json.dumps(r['case']['commands'])}</textarea></label>
        <button>Trace reader end marks and checkpoint eligibility</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Commands:reader(name),commit(page-value object),checkpoint,release(name),reset.Page keys1/2,printable ASCII values.
        Original reset conservatively requires no active reader—not every native SQLite read-mark rule.
        Sealed separate SQLite{r['native_sqlite_contract_fixture']['sqlite_version']}fixture is benign single-thread execution;patch/backport status not certified.
        WAL-reset fix versions must be verified for production.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Commands required");out.textContent=JSON.stringify(play(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];native=r["native_sqlite_contract_fixture"]
    held=c["trace"][4]["result"];middle=c["trace"][6]["result"];end=c["trace"][8]["result"]
    mx=max(max(v["indexed_median_ms"],v["scan_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['frames']}frames / {v['queries']}queries</b><p>Prebuilt per-page index {v['indexed_median_ms']:.6f}ms</p>{bar(v['indexed_median_ms'],mx)}"
                   f"<p>Reverse scan {v['scan_median_ms']:.6f}ms</p>{bar(v['scan_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · WAL SNAPSHOTS","Old readers retainA0/B0while the latest committed view isA2/B2",
         "<div class='grid'><div><h3>Scenarios</h3><p>Reader end marks<br>Commit versus backfill<br>Checkpoint versus reset</p></div><div><h3>Input</h3><p>2original pages<br>2committed transactions<br>2held readers</p></div>"
         "<div><h3>Decisions</h3><p>Bound read transactions<br>Inspect checkpoint progress<br>Verify patch versions</p><span class='pill'>Crash durability:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC ledger,independently_reproduced=false.Separate benign measured SQLite{native['sqlite_version']}contract fixture;CPU Python{r['latency']['python']} / {r['latency']['machine']}.No race or corruption test.</div>"),
        ("SCENARIO: HELD READERS","Releasing old readers advances original eligible backfill from0to1to3frames",
         "<blockquote>Original reader old pins end0;middle pins end1;current committed head3.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact model progress,not confidence</span><h3>Both held:{held['backfilled']}/{held['committed_frames']}backfilled</h3>{bar(held['backfilled'],held['committed_frames'])}<p>Confirm · inspect active read transactions</p></div>"
         f"<div class='metric'><h3>Old released:{middle['backfilled']}/{middle['committed_frames']};all released:{end['backfilled']}/{end['committed_frames']}</h3>{bar(middle['backfilled'],middle['committed_frames'])}<p>Complete backfill still does not itself truncate the log</p></div>"),
        ("WHAT I TESTED","PASSIVE can return status0while checkpoint progress remains incomplete",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Held old reader</td><td>Latest committed view?</td><td>✕ Still A0/B0</td><td>Snapshot stays stable</td></tr>"
         f"<tr><td>SQLite PASSIVE</td><td>Fully checkpointed?</td><td>✕ Tuple{native['passive_held_reader']}</td><td>Compare log/backfill</td></tr>"
         f"<tr><td>SQLite TRUNCATE held</td><td>Reset completed?</td><td>✕ Tuple{native['truncate_held_reader']}</td><td>Readers still matter</td></tr>"
         f"<tr><td>TRUNCATE after release</td><td>File truncated?</td><td>{native['wal_bytes_after_successful_truncate']}bytes</td><td>Observed fixture only</td></tr></table>"
         "<div class='grid two'><div class='callout'>No probability thresholds:exact reader identities and progress counts.</div><div class='callout'>Commit visibility,checkpoint completion,reset and crash durability are different contracts.</div></div>"),
        ("UNDER THE HOOD","Read the newest page version at or before a reader's fixed committed end mark",
         "<pre>Original commit frames:\n1:page1=A1,commit_end=true\n2:page1=A2,commit_end=false\n3:page2=B2,commit_end=true\nReader end0:base A0/B0\nReader end1:A1/B0\nCurrent end3:A2/B2</pre>"
         "<p>↓ Original independent transaction-version oracle</p><pre>Keep immutable expected transaction states\nRead snapshot selected when reader begins\nCheckpoint through conservative safe end\nDo not expose an uncommitted tail\nNo salts/checksums/native WAL parser\nNo fsync or crash-persistence witness</pre>"),
        ("ROBUSTNESS CHECK","An uncommitted fourth model frame does not change the3-frame committed view",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Current page2</td><td>B2</td><td>Append dirty tail</td><td>Still B2</td></tr>"
         "<tr><td>Logical recovery</td><td>4model frames</td><td>Drop uncommitted tail</td><td>3remain</td></tr>"
         "<tr><td>Held snapshots</td><td>A0/B0,A1/B0</td><td>Later commit</td><td>Unchanged</td></tr></table>"
         "<p>Original recovery only follows commit booleans.It does not validate binary headers,salts,checksums,torn writes or native corruption handling.</p>"),
        ("LATENCY","Prebuilt page-version lookup versus reverse scan measures Python—not SQLite checkpoints",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means.64queries at each tier,32page IDs.C-backed bisect versus Python scan.Index build,SQLite SQL/WAL-index,locks,files,fsync and durability excluded.</p>"),
        ("MY TAKE","Bound reader lifetimes and verify progress before treating WAL checkpoint calls as cleanup success",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Snapshot visibility,commit/backfill separation,explicit checkpoint progress and selected WAL frame budgeting for AI metadata stores.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Calling status0complete,copying only a live main DB as backup,predicting crash durability,network-filesystem WAL or certifying unpatched runtimes.</p></div></div>"
         f"<div class='dark'>SYNTHETIC ledger plus separate benign SQLite fixture.Public-domain source,no implementation mirror.Documented WAL-reset fixes3.51.3or named backports;local{native['sqlite_version']}patch status not verified.Visual browser inspection absent.</div>"),
    ]
    return page("SQLite WAL snapshots — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Snapshot archive cabinet:readers keep old tickets while new commits enter the log</title>
<desc>Original sixty-solid sorting cabinet metaphor;held version tickets constrain conservative backfill.Static Figure tier,not a SQLite lock implementation.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG142 · COMMIT IS NOT CHECKPOINT OR TRUNCATE</text>
<text x="88" y="158" font-size="36" font-weight="700">A pinned snapshot can retain old pages while writers commit</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Reader end marks select committed versions—not the newest physical tail</text>
<text x="88" y="758" font-size="25">PASSIVE status zero need not mean every frame was checkpointed</text>
<text x="88" y="806" font-size="25">Backfill,reset,truncation and crash durability require separate evidence</text>
<text x="88" y="880" font-size="18">Original ledger and benign sequential SQLite fixture · no crash/race/corruption test or native throughput benchmark</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
