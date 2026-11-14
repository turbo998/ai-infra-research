import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="checkpoint-multipart-completion-etag-integrity"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==3||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||p.part_bytes>5*1024**3)throw Error("Positive exact bytes/concurrency,part<=5GiB required");
 const b=BigInt(p.object_bytes),s=BigInt(p.part_bytes),c=BigInt(p.concurrency),n=(b+s-1n)/s;
 if(n>10000n||(n>1n&&s<5242880n))throw Error("Plan exceeds10000parts or nonfinal minimum5MiB");
 const out={selected_part_count:Number(n),selected_final_part_bytes:Number(b-s*(n-1n)),selected_create_upload_complete_calls:Number(n+2n),selected_equal_size_buffer_upper_bytes:Number((c<n?c:n)*s)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function simulate(events,manifest,consecutive,expectedLength){
 if(!Array.isArray(events)||events.length>64||!Array.isArray(manifest)||manifest.length<1||manifest.length>64||!Number.isSafeInteger(expectedLength)||expectedLength<0)throw Error("Bounded events/manifest and nonnegative expected length required");
 const parts=new Map(),history=[];events.forEach(([n,value],i)=>{if(!Number.isInteger(n)||n<1||n>10000||typeof value!=="string"||value.length>64||!/^[\x00-\x7f]*$/.test(value))throw Error("Part1..10000 and bounded ASCII toy payload required");const token="opaque-revision-"+(i+1);parts.set(n,{token,value});history.push([n,token]);});
 const numbers=manifest.map(row=>row[0]);if(!numbers.every((n,i)=>Number.isInteger(n)&&(i===0||n>numbers[i-1])))throw Error("Unique ascending completion list required");
 if(consecutive&&!numbers.every((n,i)=>n===i+1))throw Error("Selected checksum mode requires consecutive part IDs");
 const values=manifest.map(([n,token])=>{if(!parts.has(n)||parts.get(n).token!==token)throw Error("Missing part or stale opaque ETag token");return parts.get(n).value;}),assembled=values.join("");
 if(expectedLength!==assembled.length)throw Error("Application expected-length mismatch");
 return {token_history:history,assembled_ascii:assembled,assembled_length:assembled.length,note:"Synthetic opaque tokens;no SHA/MD5/nativeETag or multipart network execution"};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Checkpoint multipart completion and integrity workbench",
        f"""<h1>Uploaded parts are not a completed checkpoint object</h1>
        <p>Round135 · Original ASCII/opaque-revision toy,not S3/SDK/ETag implementation. Tiny parts are not valid native nonfinal sizes.</p>
        <form id="form"><label>Upload events JSON<textarea id="events" rows="4" required>[[2,"BBBB"],[1,"AAAA"],[1,"CCCC"],[3,"DD"]]</textarea></label>
        <label>Completion number/token manifest JSON<textarea id="manifest" rows="4" required>{json.dumps(r['case']['completion_manifest'])}</textarea></label>
        <label>Expected object byte length<input id="length" value="10" type="number" min="0" step="1" required></label>
        <label>Selected consecutive-checksum mode<input id="consecutive" type="checkbox" checked></label>
        <h2>Independent native-size planning budget</h2><div class="grid">{fields}</div><button>Validate supplied completion list</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>Same part number replaces old token/payload. Supplied completion list controls assembly;omissions need expected-plan/hash checks.
        HTTP200requires parsing actual completion body;this browser toy does not parse network XML.
        ETag is opaque,not a general full-object MD5 guarantee.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("events").value),JSON.parse(document.getElementById("manifest").value),document.getElementById("consecutive").checked,Number(document.getElementById("length").value)),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["copy_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['parts']} local1KiBparts</b><p>Immutable bytes join {v['join_median_ms']:.6f}ms</p>{bar(v['join_median_ms'],mx)}"
                  f"<p>Bytearray copy+conversion {v['copy_median_ms']:.6f}ms</p>{bar(v['copy_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · MULTIPART CHECKPOINTS","Four toy uploads replace part1once;the final manifest assembles10bytes in part-number order",
         "<div class='grid'><div><h3>Scenarios</h3><p>Replacement tokens<br>Manifest omission<br>HTTP200body errors</p></div>"
         "<div><h3>Input</h3><p>Upload2/1/1/3<br>Final1/2/3<br>Lengths4/4/2</p></div>"
         "<div><h3>Decisions</h3><p>Current opaque tokens<br>Expected length/hash<br>Parse completion body</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>AWS live docs read Nov14;original toy,not native S3. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: PART REPLACEMENT","Old part1token no longer completes the updated payload;the selected final object is10bytes",
         f"<blockquote>Token{c['old_part1_token']}→{c['current_part1_token']};beforecompleteobjectnull;expectedSHA256{c['expected_sha256']}.</blockquote>"
         "".join(f"<div class='metric'><span class='pill'>score · toy bytes,not confidence</span><h3>Part{n}:{length}bytes</h3>{bar(length,max(c['toy_part_lengths']))}<p>Current token and ordered manifest required</p></div>" for n,length in zip(c["part_numbers"],c["toy_part_lengths"]))),
        ("WHAT I TESTED","HTTP200can carry an Error body;supplied-list completion can omit an intended part",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Staletoken</td><td>Old part1accepted?</td><td>✕ Rejected</td><td>Record replacement</td></tr>"
         "<tr><td>Suppliedsubset</td><td>Every uploaded part included?</td><td>✕ Onlylisted</td><td>Expectedplan/hash</td></tr>"
         "<tr><td>HTTP200</td><td>Guaranteedcomplete?</td><td>✕ BodymayError</td><td>Parse or use SDK</td></tr>"
         "<tr><td>ETag</td><td>AlwaysfullMD5?</td><td>✕ Opaquecontract</td><td>Supportedfullchecksum</td></tr>"
         "<tr><td>Checkpoint</td><td>Singleobjectcompletionglobalatomic?</td><td>✕ Notestablished</td><td>Manifest/generationprotocol</td></tr></table>"
         "<div class='grid two'><div class='callout'>Toy bytes are below native nonfinal minimum;size plan separate.</div><div class='callout'>XML success-root check is bounded fixture,not a full SDK/retry implementation.</div></div>"),
        ("UNDER THE HOOD","Completion binds uploadID,ordered part identities and current opaque tokens—not transport status alone",
         "<pre>Original state:\nactive upload→replace numbered parts\nvalidate current tokens and ascending supplied list\napplication:expected length/full SHA256\ncomplete→single object visible;parts cleared</pre>"
         "<p>↓ Original independent assembly/body oracle</p><pre>Independent:flatten selected payload byte sequences\nCompare expected original bytes/hash\n\nstatus200 + &lt;Error&gt; = explicit failure\nstatus200 + CompleteMultipartUploadResult\n  = bounded result-root acceptance\nNo native SDK,response schema or retry reproduction</pre>"),
        ("ROBUSTNESS CHECK","Replacing part1changes identity;omitting part2shrinks the selected object from10to4bytes",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Part1token</td><td>{c['old_part1_token']}</td><td>{c['current_part1_token']}</td><td>Staletokenfails</td></tr>"
         f"<tr><td>Suppliedlistlength</td><td>{c['assembled_length']}bytes</td><td>{len(c['subset_supplied_manifest_bytes'])}bytes</td><td>Expectedlengthgate</td></tr>"
         "<tr><td>Responsebody</td><td>Successroot</td><td>ErrorunderHTTP200</td><td>Not probability points</td></tr></table>"),
        ("LATENCY","Local bytes assembly timing measures neither multipart upload throughput nor S3 server completion",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Original local1KiBparts;S3/network/SDK/checksum/server/disk/retries excluded.</p>"),
        ("MY TAKE","Persist a part ledger and verify object integrity before publishing a checkpoint-generation manifest",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Part replacement/order audits,expected-plan validation,HTTPbody error controls and bounded transfer planning.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>ETag-as-MD5 assumptions,HTTPstatus-only success,globalcheckpointatomicity or native upload latency claims.</p></div></div>"
         "<div class='dark'>AWS documentation links;original state/oracle/art.No cloud requests or credentials.Nativeperformance/globalatomicity unknown;browser inspection not performed.</div>"),
    ]
    return page("Multipart checkpoint completion — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Multipart checkpoint ledger cabinet:current part tokens,integrity and completion visibility</title>
<desc>Original sixty-solid sorting cabinet metaphor for ordered multipart records and a separate published checkpoint manifest.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG135 · MULTIPART COMPLETION</text>
<text x="88" y="158" font-size="36" font-weight="700">Uploaded parts are not a committed checkpoint</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Part replacement invalidates the old opaque token;assemble supplied order</text>
<text x="88" y="758" font-size="25">Expected length/full checksum and actual response body are separate gates</text>
<text x="88" y="806" font-size="25">Publish multi-object checkpoint metadata only after verified asset completion</text>
<text x="88" y="880" font-size="18">Original multipart/ledger toy and local CPU assembly · no S3/SDK/network/globalatomicity reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
