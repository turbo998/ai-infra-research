import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="io-uring-ring-credits-wraparound-buffer-lifetime"
COMMANDS=[["submit",100,0],["submit",101,1],["consume"],["submit",102,2],["consume"],["consume"],["complete",101,-5],["complete",100,4],["complete",102,4],["reap"],["reap"],["reap"]]
JS=r"""
const U32=2**32;
function depth(tail,head){if(![tail,head].every(v=>Number.isInteger(v)&&v>=0&&v<U32))throw Error("Unsigned32bit counters required");return (tail-head+U32)%U32;}
function capacity(n){if(!Number.isInteger(n)||n<1||n>1024||(n&(n-1)))throw Error("Poweroftwo ring entries1..1024required");return n;}
function play(p,commands){
 if(Object.keys(p).length!==4||!Number.isInteger(p.buffers)||p.buffers<1||p.buffers>65536)throw Error("Bounded ring/buffer parameters required");
 capacity(p.entries);capacity(p.cq_entries);depth(p.start,p.start);
 if(!Array.isArray(commands)||commands.length>128)throw Error("At most128commands required");
 let head=p.start,tail=p.start,ch=p.start,ct=p.start;const sq=Array(p.entries).fill(null),cq=Array(p.cq_entries).fill(null),pending=new Map(),owned=new Map(),used=new Set(),results=[];
 const snapshot=()=>({sq_head:head,sq_tail:tail,sq_depth:depth(tail,head),sq_free:p.entries-depth(tail,head),cq_head:ch,cq_tail:ct,cq_depth:depth(ct,ch),sq_slots:sq.map(v=>v?[...v]:null),pending:[...pending].sort((a,b)=>a[0]-b[0]).map(([request_id,v])=>({request_id,...v})),buffer_owners:[...owned].sort((a,b)=>a[0]-b[0]).map(([buffer,request_id])=>({buffer,request_id}))});
 for(const cmd of commands){
 if(!Array.isArray(cmd))throw Error("Each command must be an array");const [op,id,arg]=cmd;let result=null;
 if(op==="submit"&&cmd.length===3){
 if(!Number.isSafeInteger(id)||id<0||used.has(id))throw Error("Original request IDs cannot be reused");
 if(!Number.isInteger(arg)||arg<0||arg>=p.buffers||owned.has(arg))throw Error("Buffer outside pool or still owned");
 if(depth(tail,head)>=p.entries)throw Error("Original SQ full");const slot=tail&(p.entries-1);sq[slot]=[id,arg];tail=(tail+1)%U32;pending.set(id,{buffer:arg,state:"queued",requested_bytes:4});owned.set(arg,id);used.add(id);result=slot;
 }else if(op==="consume"&&cmd.length===1){
 if(!depth(tail,head))throw Error("Original SQ empty");const slot=head&(p.entries-1),[request,buffer]=sq[slot];sq[slot]=null;head=(head+1)%U32;pending.get(request).state="inflight";result=request;
 }else if(op==="complete"&&cmd.length===3){
 if(!Number.isSafeInteger(id)||!pending.has(id)||pending.get(id).state!=="inflight")throw Error("Unknown,unconsumed or completed request");
 if(!Number.isInteger(arg)||arg< -4095||arg>pending.get(id).requested_bytes)throw Error("Original read result outside bounded contract");
 if(depth(ct,ch)>=p.cq_entries)throw Error("Original CQ full;native overflow not modeled");cq[ct&(p.cq_entries-1)]=[id,arg];ct=(ct+1)%U32;pending.get(id).state="completion-queued";
 }else if(op==="reap"&&cmd.length===1){
 if(!depth(ct,ch))throw Error("Original CQ empty");const slot=ch&(p.cq_entries-1),[request,value]=cq[slot],item=pending.get(request);cq[slot]=null;ch=(ch+1)%U32;pending.delete(request);owned.delete(item.buffer);
 result={request_id:request,buffer:item.buffer,result:value,requested_bytes:item.requested_bytes,status:value<0?"error":value===item.requested_bytes?"complete":"short_or_eof"};
 }else throw Error("Unsupported command or argument count");
 results.push({command:cmd,result,snapshot:snapshot()});
 }
 return results;
}
function budget(p){
 if(Object.keys(p).length!==6||!Object.values(p).every(v=>Number.isInteger(v)&&v>=1))throw Error("Positive exact selected payload parameters required");
 capacity(p.sq_entries);capacity(p.cq_entries);if(p.buffer_count>65536||p.buffer_bytes>2**30||p.sqe_bytes>256||p.cqe_bytes>256)throw Error("Selected payload parameters exceed bounded contract");
 const sq=p.sq_entries*p.sqe_bytes,cq=p.cq_entries*p.cqe_bytes;
 return {selected_sqe_payload_bytes:sq,selected_cqe_payload_bytes:cq,selected_ring_entry_payload_bytes:sq+cq,selected_buffer_pool_payload_bytes:p.buffer_count*p.buffer_bytes,note:"selected entry/pool payload only;no ring indexes,SQ indirection,mmap pages,kernel state,pinning,allocator or native peak memory"};
}
"""


def workbench(r):
    p={"entries":2,"cq_entries":4,"buffers":3,"start":0}
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("io_uring ring credits and buffer lifetime workbench",
        f"""<h1>SQ credit is not an ordinary-read buffer release</h1><p>Round140 · Original sequential state model on this host,not Linux/io_uring execution.</p>
        <form id="form"><div class="grid">{fields}</div><label>Original commands JSON<textarea id="commands" rows="9" required>{json.dumps(COMMANDS)}</textarea></label>
        <button>Trace ring credit and completion identities</button></form><p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Commands:submit(requestID,buffer),consume,complete(requestID,result),reap. Ordinary requests use4bytes;negative result remains an explicit error,short result is not full completion.
        No SQPOLL,multishot,skip-success,zero-copy-send notifications,real atomics or native CQ overflow.
        Submitted IDs are never reused in this original bounded session policy.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify(play(p,JSON.parse(document.getElementById("commands").value)),null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(max(v["cursor_median_ms"],v["shifting_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['entries']} copied-list entries</b><p>Head cursor {v['cursor_median_ms']:.6f}ms</p>{bar(v['cursor_median_ms'],mx)}"
                   f"<p>pop(0)shifting {v['shifting_median_ms']:.6f}ms</p>{bar(v['shifting_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    event=c["events"][3]
    sections=[
        ("HANDS-ON · ASYNC READ OWNERSHIP","The SQ can be empty while all3ordinary-read buffers remain owned",
         "<div class='grid'><div><h3>Scenarios</h3><p>SQ/CQ credits<br>Unsigned wraparound<br>Completion identity</p></div>"
         "<div><h3>Input</h3><p>2SQslots,3buffers<br>3one-shot reads<br>IDs100/101/102</p></div>"
         "<div><h3>Decisions</h3><p>Separate lifetimes<br>Assign user_data<br>Check result values</p><span class='pill'>Native IOPS:unknown</span></div></div>"
         f"<div class='dark'>Pinned liburing docs;Nov12SQPOLLtestfix is not new ring semantics.Synthetic original state/oracle,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: REUSED SQ SLOT","Request100ownsbuffer0even after its SQslot0holdsrequest102",
         f"<blockquote>Correct CQE ID100→buffer0;✕ current SQslot0→ID{c['wrong_slot_current_request']}→buffer{c['wrong_slot102_buffer']}.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact model state,not confidence</span><h3>{event['sq_depth']}queued SQentries,{len(event['buffer_owners'])}owned buffers</h3>{bar(len(event['buffer_owners']),len(event['buffer_owners']))}<p>Empty SQ does not mean I/O has completed</p></div>"
         "<div class='metric'><h3>Completion order101,100,102</h3><p>Negative result-5remains an error;do not assume FIFO request completion.</p></div>"),
        ("WHAT I TESTED","Unsigned counters wrap at2³²without losing4-entry occupancy",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         f"<tr><td>Wrappedcounters</td><td>Depth?</td><td>{c['wraparound_correct_unsigned_depth']}</td><td>Unsigned difference</td></tr>"
         f"<tr><td>Plaindifference</td><td>Depth?</td><td>✕ {c['wrong_plain_tail_minus_head']}</td><td>Not signed/unbounded subtraction</td></tr>"
         "<tr><td>SQconsumed</td><td>Releasebuffer?</td><td>✕ Still owned</td><td>Wait operation completion</td></tr>"
         "<tr><td>Shortread</td><td>Fullrequestedbytes?</td><td>✕ Explicit short/EOF</td><td>Application handles result</td></tr></table>"
         "<div class='grid two'><div class='callout'>Ring capacity is not total pending I/O or buffer capacity.</div><div class='callout'>Original hard CQ limit is not native overflow/backlog semantics.</div></div>"),
        ("UNDER THE HOOD","Publish filled payload before tail visibility;track ordinary buffers until completion observation",
         "<pre>Original unsigned32bit ring:\ndepth=(tail-head)mod2^32\nslot=counter &amp; (entries-1)\nFill entry → release publish tail\nAcquire observe tail → read entry\nConsume SQ entry → SQ credit only</pre>"
         "<p>↓ Separate operation identity and ownership</p><pre>Original one-shot read:\nrequestID → buffer,requestedbytes,state\nCQE user_data returns request identity\nReap matched completion → release owner\nresult&lt;0:error\n0&lt;=result&lt;requested:short/EOF\nNo multishot/F_MORE/zero-copy-send model</pre>"),
        ("ROBUSTNESS CHECK","Out-of-order completion preserves buffer identity;reusing a bare slot index does not",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Completionorder</td><td>100,101,102</td><td>101,100,102</td><td>Same exact owners</td></tr>"
         "<tr><td>Slot0reuse</td><td>Request100</td><td>Request102</td><td>Must not reuse identity</td></tr>"
         "<tr><td>Publishbeforefill</td><td>Filledpayload</td><td>✕ Unfilled witness</td><td>Not race probability</td></tr></table>"),
        ("LATENCY","Python cursor versus shifting measures list movement—not io_uring,syscalls or storage",
         timing+"<p>Both include list copy;two warmups,seven alternating repeats,three operations/sample,median repeat means.Cursor may lose on small lists.C-backed pop movement and Python control differ.No atomic,ring-class,kernel or IOPS measurement.</p>"),
        ("MY TAKE","Separate ring credits,opaque operation IDs and buffer lifetime before tuning asynchronous storage",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>One-shot read-state reasoning,wraparound boundaries,explicit request identities and selected buffer-pool budgeting.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Assuming SQ consumption releases data,implicit user_data initialization,FIFO completion or extending one-shot rules to multishot/notification modes.</p></div></div>"
         "<div class='dark'>Read manpages LGPL-2.0-or-later;repository mixed licenses preserved.Original model/art,not Linux/atomic execution.Native results and visual browser inspection absent.</div>"),
    ]
    return page("io_uring ownership contracts — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Async read ticket cabinet:ring credits and buffer ownership have different lifetimes</title>
<desc>Original sixty-solid sorting cabinet metaphor:ticket slots can be reused while corresponding operation-owned buffers remain checked out.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG140 · RING CREDIT IS NOT BUFFER RELEASE</text>
<text x="88" y="158" font-size="36" font-weight="700">A reused ticket slot is not the old operation identity</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Unsigned depth and masked slot preserve ring occupancy across wraparound</text>
<text x="88" y="758" font-size="25">SQ consumption releases a slot—not an ordinary read/write data buffer</text>
<text x="88" y="806" font-size="25">Match completion IDs,check result,and preserve mode-specific lifetimes</text>
<text x="88" y="880" font-size="18">Original one-shot state/list oracle · not Linux,real atomics,SQPOLL,multishot,IOPS or storage benchmark</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
