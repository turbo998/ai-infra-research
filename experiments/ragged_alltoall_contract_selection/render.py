import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="ragged-all-to-all-split-matrix-token-inverse"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error(`Exact integer${lo}..${hi}required`);return v;}
function restore(responses,lengths){
 if(!Array.isArray(lengths)||lengths.length<1||lengths.length>8||!Array.isArray(responses)||responses.length!==lengths.length)throw Error("Response grid and lengths required");
 lengths.forEach(v=>integer(v,0,128));const out=lengths.map(n=>Array(n).fill(null)),seen=new Set();
 responses.forEach((records,d)=>{if(!Array.isArray(records))throw Error("Response list required");records.forEach(r=>{
  if(!r||Array.isArray(r)||JSON.stringify(Object.keys(r).sort())!==JSON.stringify(["owner","position","value","destination","result"].sort()))throw Error("Exact response fields required");
  integer(r.owner,0,lengths.length-1);integer(r.position,0,127);integer(r.value,-1e6,1e6);integer(r.result,-2e6,2e6+700);
  const key=r.owner+":"+r.position;if(!Number.isInteger(r.destination)||r.destination!==d||r.position>=lengths[r.owner]||seen.has(key))throw Error("Bad response identity");
  seen.add(key);out[r.owner][r.position]=r.result;
 });});
 if(seen.size!==lengths.reduce((a,b)=>a+b,0))throw Error("Missing response identity");return out;
}
function dispatch(p,declared=null){
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(["destinations","tokens"]))throw Error("Exact tokens/destinations fields required");
 const tokens=p.tokens,targets=p.destinations;
 if(!Array.isArray(tokens)||tokens.length<1||tokens.length>8||!Array.isArray(targets)||targets.length!==tokens.length)throw Error("One to eight group-relative rank rows required");
 const n=tokens.length,send=Array.from({length:n},()=>Array(n).fill(0)),packed=[];
 tokens.forEach((values,rank)=>{
  if(!Array.isArray(values)||values.length>128||!Array.isArray(targets[rank])||targets[rank].length!==values.length)throw Error("Matched bounded rank lists required");
  const records=values.map((value,position)=>{const destination=targets[rank][position];integer(value,-1e6,1e6);integer(destination,0,n-1);send[rank][destination]++;return {owner:rank,position,value,destination};});
  packed.push(records.sort((a,b)=>a.destination-b.destination));
 });
 const recv=Array.from({length:n},(_,d)=>Array.from({length:n},(_,s)=>send[s][d]));
 if(declared!==null){
  if(!Array.isArray(declared)||declared.length!==n)throw Error("Receive-split grid required");
  declared.forEach((row,d)=>{if(!Array.isArray(row)||row.length!==n)throw Error("World-size receive row required");
   row.forEach((count,s)=>{integer(count,0,128);if(count!==send[s][d])throw Error(`Pairwise split mismatch source${s}→destination${d};row totals insufficient`);});});
 }
 const received=Array.from({length:n},(_,d)=>packed.flatMap(records=>records.filter(r=>r.destination===d)));
 const responses=received.map((records,d)=>records.map(r=>({...r,result:2*r.value+100*d}))),restored=restore(responses,tokens.map(v=>v.length));
 const naive=tokens.map((_,owner)=>responses.flatMap(records=>records.filter(r=>r.owner===owner).map(r=>r.result)));
 return {send_splits:send,recv_splits:recv,packed,received,responses,restored,naive_return_order:naive,
 naive_positions_matching:restored.reduce((sum,row,owner)=>sum+row.filter((v,i)=>v===naive[owner][i]).length,0),
 tokens:tokens.reduce((sum,row)=>sum+row.length,0),send_rows:send.map(row=>row.reduce((a,b)=>a+b,0)),receive_rows:recv.map(row=>row.reduce((a,b)=>a+b,0))};
}
"""


def workbench(r):
    return page("Ragged all-to-all row-split and inverse workbench",
        f"""<h1>Equal receive totals do not establish pairwise split agreement</h1>
        <p>Round144 · Original single-destination scalar transport model—not distributed PyTorch/NCCL execution.</p>
        <form id="form"><label>Token/target JSON<textarea id="input" rows="8" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <label>Declared receive matrix JSON;blank uses derived transpose<textarea id="receive" rows="4">{json.dumps(r['case']['outputs']['recv_splits'])}</textarea></label>
        <button>Validate pairwise row splits and inverse identities</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Rank IDs are group-relative.Split counts represent token rows,not all feature scalars.
        Original policy allows zero-receive ranks;native backend zero-length support remains unverified.
        No top-k replication,routing policy,expert capacity,async stream/buffer lifetime or native wire model.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");
 const text=document.getElementById("receive").value.trim();out.textContent=JSON.stringify(dispatch(JSON.parse(document.getElementById("input").value),text?JSON.parse(text):null),null,2);}
 catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];out=c["outputs"];n=len(out["send_splits"])
    mx=max(max(v["indexed_median_ms"],v["searching_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['tokens']}inverse token lookups</b><p>Prebuilt ID dictionary {v['indexed_median_ms']:.6f}ms</p>{bar(v['indexed_median_ms'],mx)}"
                   f"<p>Repeated list.index {v['searching_median_ms']:.6f}ms</p>{bar(v['searching_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · RAGGED TOKEN TRANSPORT","Rank1receives zero rows but still sends two rows to other ranks",
         "<div class='grid'><div><h3>Scenarios</h3><p>Pairwise split matrix<br>Zero receive bucket<br>Inverse token order</p></div><div><h3>Input</h3><p>4group-relative ranks<br>8scalar tokens<br>One destination per token</p></div>"
         "<div><h3>Decisions</h3><p>Transpose counts<br>Preserve token identity<br>Separate row/byte units</p><span class='pill'>Native collective latency:unknown</span></div></div>"
         f"<div class='dark'>SYNTHETIC original transport,independently_reproduced=false.CPU Python{r['latency']['python']} / {r['latency']['machine']}.Nov14pinned nccl2capability change is not new split semantics.No backend/encryption executed.</div>"),
        ("SCENARIO: SAME TOTAL WRONG PEERS","Correct and wrong receive rows both total3—but pairwise peer counts disagree",
         f"<blockquote>Correct rank0receive splits:{out['recv_splits'][0]};wrong:{c['bad_receive_row']}.Each totals{c['correct_receive_row_total']}rows.</blockquote>"
         f"<div class='metric'><span class='pill'>choice · deterministic preflight,not confidence</span><h3>Require recv[d][s]=send[s][d]</h3><p>Confirm · global metadata exchange itself is outside this simulator</p></div>"
         f"<div class='metric'><span class='pill'>score · exact model positions</span><h3>Naive return append matches{out['naive_positions_matching']}/{out['tokens']}positions</h3>{bar(out['naive_positions_matching'],out['tokens'])}<p>Carry owner/position or inverse-permutation metadata</p></div>"),
        ("WHAT I TESTED","Four ranks require16pairwise agreements—not just four matching totals",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Same output row total</td><td>Enough agreement?</td><td>✕ Peer counts differ</td><td>Pairwise validation</td></tr>"
         "<tr><td>Rank1receives0</td><td>Can skip all work?</td><td>✕ Still sends2</td><td>Preserve participation protocol</td></tr>"
         "<tr><td>Append return buckets</td><td>Original token order?</td><td>✕4/8positions</td><td>Inverse identity</td></tr>"
         "<tr><td>Dim0split counts</td><td>Feature scalar count?</td><td>✕ Rows,not every scalar</td><td>Multiply only for byte budget</td></tr></table>"
         "<div class='grid two'><div class='callout'>Exact identities/counts,no model-probability thresholds.</div><div class='callout'>Native mismatch/zero-length outcomes are not executed or certified.</div></div>"),
        ("UNDER THE HOOD","Pack by destination;receive source blocks;restore original owner positions",
         "<pre>Original matrix:\nS[source][destination]=row count\nR[destination][source]=S[source][destination]\nPack source rows destination-major\nReceive each destination source-major\nSplit units:dimension0rows</pre>"
         "<p>↓ Original independent direct-owner oracle</p><pre>Carry(owner,original_position)\nToy expert result=2*value+100*destination\nReturn via original owner\nScatter to original_position\nReject duplicate/missing identities\nNo top-k replication or native tensor execution</pre>"),
        ("ROBUSTNESS CHECK","Reversing local response traversal preserves all8owner-position results",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Identity restoration</td><td>Forward local traversal</td><td>Reverse local traversal</td><td>Same8/8values</td></tr>"
         "<tr><td>Payloadvalues</td><td>Distinct</td><td>Duplicate values</td><td>Identity still needed</td></tr>"
         "<tr><td>Receive row total</td><td>3</td><td>Wrong peers,total3</td><td>Explicit rejection</td></tr></table>"
         "<p>This perturbs original local processing traversal,not native collective output ordering.PyTorch's output retains source-block semantics.</p>"),
        ("LATENCY","Prebuilt inverse-ID lookup avoids repeated searches—not network communication",
         timing+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means.Both return original-order values.Dictionary build,packing,PyTorch,NCCL/MPI,GPUs,experts and network excluded.</p>"),
        ("MY TAKE","Validate pairwise row counts and inverse metadata before tuning expert-token transport",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Group-relative ranks,ragged split preflight,zero-receive reasoning and exact nonreplicating inverse permutation.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Skipping a rank because it receives nothing,using scalar counts for dim0splits,assuming row totals suffice,or claiming native backend performance/support.</p></div></div>"
         "<div class='dark'>SYNTHETIC original model;PyTorch BSD-style source rights preserved.No implementation mirror or GPU/distributed/TLS execution.API and backend-specific config require live checks.Visual browser inspection absent.</div>"),
    ]
    return page("Ragged all-to-all — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Token mail-sorting cabinet:peer counts and owner return tickets</title>
<desc>Original sixty-solid sorting cabinet metaphor;destination slots group outgoing tickets while owner-position tags preserve their return order.Static Figure tier,not NCCL hardware.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG144 · PEER SPLITS AND INVERSE TOKEN IDENTITY</text>
<text x="88" y="158" font-size="36" font-weight="700">A receive-total match does not prove sender agreement</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Receive matrix is the send transpose;counts measure token rows,not all feature scalars</text>
<text x="88" y="758" font-size="25">A zero-receive rank can still send;it is not permission to skip the group protocol</text>
<text x="88" y="806" font-size="25">Owner-position metadata restores order;append-by-destination does not</text>
<text x="88" y="880" font-size="18">Original single-destination transport and CPU inverse lookups · no PyTorch/NCCL/MPI/GPU,wire,async stream or TLS execution</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
