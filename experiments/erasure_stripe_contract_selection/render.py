import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="checkpoint-erasure-stripe-gf256-failure-domain"
JS=r"""
function budget(p){
 if(Object.keys(p).length!==4||!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0)||p.data_shards+p.parity_shards>16)throw Error("Positive bounded profile/payload/stripe parameters required");
 const b=BigInt(p.object_bytes),k=BigInt(p.data_shards),m=BigInt(p.parity_shards),u=BigInt(p.stripe_unit_bytes),stripes=(b+k*u-1n)/(k*u),chunk=stripes*u;
 const out={selected_classic_stripes:Number(stripes),selected_chunk_payload_bytes:Number(chunk),selected_classic_encoded_bytes:Number(chunk*(k+m)),selected_data_padding_bytes:Number(chunk*k-b),selected_minimum_k_chunk_recovery_read_bytes:Number(chunk*k),selected_three_replica_payload_bytes:Number(b*3n)};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("Exact budget range exceeded");return out;
}
function gfMul(a,b){let r=0;for(let i=0;i<8;i++){if(b&1)r^=a;b>>=1;a<<=1;if(a&256)a^=0x11b;}return r;}
function gfPow(a,n){let r=1;while(n){if(n&1)r=gfMul(r,a);a=gfMul(a,a);n>>=1;}return r;}
function gfInv(a){if(!a)throw Error("Zero field inverse");return gfPow(a,254);}
function invert(matrix){
 const n=matrix.length,rows=matrix.map((row,i)=>[...row,...Array.from({length:n},(_,j)=>Number(i===j))]);
 for(let c=0;c<n;c++){let p=c;while(p<n&&!rows[p][c])p++;if(p===n)throw Error("Singular survivor matrix");[rows[c],rows[p]]=[rows[p],rows[c]];
 const f=gfInv(rows[c][c]);rows[c]=rows[c].map(v=>gfMul(v,f));for(let r=0;r<n;r++){if(r===c)continue;const t=rows[r][c];rows[r]=rows[r].map((v,j)=>v^gfMul(t,rows[c][j]));}}
 return rows.map(row=>row.slice(n));
}
function generator(k,m){const v=Array.from({length:k+m},(_,x)=>Array.from({length:k},(_,j)=>gfPow(x,j))),base=invert(v.slice(0,k));
 return v.map(row=>Array.from({length:k},(_,c)=>row.reduce((s,v,t)=>s^gfMul(v,base[t][c]),0)));}
function simulate(data,k,m,missing){
 if(!Array.isArray(data)||data.length>64||!data.every(v=>Number.isInteger(v)&&v>=0&&v<=255)||!Number.isInteger(k)||!Number.isInteger(m)||k<1||m<1||k+m>16||!Array.isArray(missing)||new Set(missing).size!==missing.length||!missing.every(v=>Number.isInteger(v)&&v>=0&&v<k+m))throw Error("Bytes0..255,length<=64,k/mpositive,sum<=16,unique lost IDs required");
 const width=Math.ceil(data.length/k),padded=data.concat(Array(width*k-data.length).fill(0)),g=generator(k,m);
 const shards=g.map(row=>Array.from({length:width},(_,j)=>row.reduce((s,c,i)=>s^gfMul(c,padded[i*width+j]),0)));
 const chosen=Array.from({length:k+m},(_,i)=>i).filter(i=>!missing.includes(i)).slice(0,k);
 if(chosen.length<k)throw Error("Fewer than k known survivors;original erasure decoder cannot recover");
 const decode=invert(chosen.map(i=>g[i])),recovered=decode.flatMap(row=>Array.from({length:width},(_,j)=>row.reduce((s,c,i)=>s^gfMul(c,shards[chosen[i]][j]),0))).slice(0,data.length);
 return {generator:g,encoded_chunks:shards,chosen_survivors:chosen,recovered_bytes:recovered,matches:JSON.stringify(data)===JSON.stringify(recovered),padding_bytes:width*k-data.length,note:"Original GF256 polynomial0x11b,not ISA-L/Ceph byte parity;known erasures only,not silent-corruption correction"};
}
"""


def workbench(r):
    p=r["default_budget"]["inputs"]
    fields="".join(f'<label>{k}<input id="{k}" value="{v}" type="number" step="1" required></label>' for k,v in p.items())
    return page("Checkpoint erasure stripes and failure-domain workbench",
        f"""<h1>Two parity shards recover two known erasures—not arbitrary rack loss or silent corruption</h1>
        <p>Round133 · Original GF256 systematic Vandermonde code,polynomial0x11b. Not native Ceph/ISA-L/FastEC.</p>
        <form id="form"><label>Original bytes JSON<input id="data" value="[1,2,3,4,5,6,7,8,9,10]" required></label>
        <label>Toy data shards<input id="k" type="number" value="3" min="1" max="15" step="1" required></label>
        <label>Toy parity shards<input id="m" type="number" value="2" min="1" max="15" step="1" required></label>
        <label>Known missing shard IDs JSON<input id="missing" value="[0,1]" required></label>
        <h2>Independent classic full-stripe logical payload budget</h2><div class="grid">{fields}</div><button>Encode and recover known erasures</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>The toy packs one byte-aligned object;the independent budget assumes full stripe-unit padding.
        This is not a prediction of Ceph FastEC layout. Fault-domain mapping,checksums,write acknowledgement,
        metadata and rebuild I/O remain separate requirements.</p>""",
        "const DEFAULT="+json.dumps(p)+";\n"+JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Complete all fields");
 const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
 out.textContent=JSON.stringify({result:simulate(JSON.parse(document.getElementById("data").value),Number(document.getElementById("k").value),Number(document.getElementById("m").value),JSON.parse(document.getElementById("missing").value)),budget:budget(p)},null,2);
 }catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];mx=max(v["bitwise_median_ms"] for v in r["latency"]["rows"])
    chart="".join(f"<div class='metric'><b>{v['chunk_bytes']} bytes/chunk ·3+2</b><p>Coefficient lookup {v['lookup_median_ms']:.6f}ms</p>{bar(v['lookup_median_ms'],mx)}"
                  f"<p>Bitwise multiplication {v['bitwise_median_ms']:.6f}ms</p>{bar(v['bitwise_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · CHECKPOINT ERASURES","Original3+2code recovers10/10two-erasure combinations with three known survivors",
         "<div class='grid'><div><h3>Scenarios</h3><p>Known erasures<br>Failure domains<br>Classic stripe padding</p></div>"
         "<div><h3>Input</h3><p>10originalbytes<br>3data+2parity<br>4bytes/chunk,2padding</p></div>"
         "<div><h3>Decisions</h3><p>Distinct failure domains<br>Expected checksum<br>Version-specific layout</p><span class='pill'>Native latency:unknown</span></div></div>"
         f"<div class='dark'>Pinned Ceph EC/FastEC constraints;originalGF256model,not native byte parity. Synthetic,independently_reproduced=false;CPU Python{r['latency']['python']} / {r['latency']['machine']}.</div>"),
        ("SCENARIO: KNOWN ERASURES","All10selected two-shard losses recover;one badly placed rack can erase3shards instead",
         f"<blockquote>Original bytes{c['original_bytes']};k3,m2;chunk4bytes with2padding.</blockquote>"
         f"<div class='metric'><span class='pill'>score · recovery cases,not confidence</span><h3>Known two-erasure recovery{c['recovered_cases']}/{c['total_cases']}</h3>{bar(c['recovered_cases'],c['total_cases'])}<p>Act automatically only with known-good survivors</p></div>"
         f"<div class='metric'><span class='pill'>score · missing shards,not probability</span><h3>One bad domain erases{len(c['bad_single_domain_missing'])}/5</h3>{bar(len(c['bad_single_domain_missing']),5,True)}<p>✕ Confirm fewer than k survivors</p></div>"),
        ("WHAT I TESTED","Two parity chunks do not detect or repair an unknown corrupted survivor",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Twoerasures</td><td>Threegoodremain?</td><td>Recover10/10</td><td>Decode</td></tr>"
         "<tr><td>Bad rack mapping</td><td>One rack loses3?</td><td>✕ Only2survive</td><td>Reject recovery</td></tr>"
         "<tr><td>Corrupted survivor</td><td>Decoder guarantees original?</td><td>✕ Expectedhashmismatch</td><td>External integrity gate</td></tr>"
         "<tr><td>Padding</td><td>Original length needed?</td><td>Yes,10not12</td><td>Preserve manifest length</td></tr>"
         "<tr><td>FastEC</td><td>Classic padded budget universal?</td><td>✕ Version/layout-specific</td><td>Inspect live pool</td></tr></table>"
         "<div class='grid two'><div class='callout'>Erasure tolerance counts missing chunks,not a durability probability.</div><div class='callout'>Checksum fixture uses known expected digest;not a full checksum storage protocol.</div></div>"),
        ("UNDER THE HOOD","An invertible survivor submatrix—not ordinary integer division—recovers original byte columns",
         "<pre>Original field:GF(2^8),polynomial0x11b\nV[i,j] = point_i^j in the field\nG = V * inverse(V[first k rows])\nfirst k rows of G = identity\n\nencoded_column = G * data_column</pre>"
         "<p>↓ Original independent interpolation oracle</p><pre>Select k known-good surviving rows\ninvert G[survivor rows]\nrecover each original column\ntrim using original byte length\n\nIndependent encode:Lagrange interpolation\nat distinct field points,not matrix inversion</pre>"),
        ("ROBUSTNESS CHECK","Changing only failure-domain mapping turns one rack outage from1missing shard into3",
         f"<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Chunk domains</td><td>{c['good_domains']}</td><td>{c['bad_domains']}</td><td>Topology,notcode</td></tr>"
         f"<tr><td>Onebad-domainloss</td><td>One shard per domain</td><td>{c['bad_single_domain_missing']}</td><td>{c['bad_domain_survivors']}survivors&lt;k3</td></tr>"
         f"<tr><td>Silentxor1corruption</td><td>Known good</td><td>Matchesexpected:{c['corruption_control']['matches_expected']}</td><td>Not probability points</td></tr></table>"),
        ("LATENCY","Original lookup tables replace bitwise GF multiply only in a parity-only CPU toy",
         chart+"<p>Two warmups,seven alternating repeats,three operations/sample;median repeat means. Coefficient/generator setup,Ceph/ISA-L/SIMD,network,disk,recovery,FastEC/checksums excluded.</p>"),
        ("MY TAKE","Choose k/m with failure domains and integrity metadata,then measure real write/rebuild paths",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Known-erasure algebra,systematic shards,original-length tracking,classic padding and domain-loss audits.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Unknown-error correction,durability SLAs,native codec byte parity,FastEC layout or production throughput prediction.</p></div></div>"
         "<div class='dark'>Original field/matrix/oracle/art;linked Ceph docs,not copied codec.NativeCeph performance/storage/rebuild/durability unknown;browser inspection not performed.</div>"),
    ]
    return page("Checkpoint erasure stripe contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Checkpoint shard ticket cabinet:finite-field redundancy and real failure domains</title>
<desc>Original sixty-solid sorting cabinet metaphor for systematic data tickets,parity tickets and known-erasure recovery.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG133 · CHECKPOINT ERASURE STRIPES</text>
<text x="88" y="158" font-size="36" font-weight="700">Two parity shards do not mean any two rack failures</text>{sorter_scene()}
<text x="88" y="710" font-size="25">Three known-good survivors recover original3+2byte stripes</text>
<text x="88" y="758" font-size="25">Colocated chunks turn one domain outage into several erasures</text>
<text x="88" y="806" font-size="25">Expected integrity metadata and original length remain separate gates</text>
<text x="88" y="880" font-size="18">Original GF256 polynomial0x11b,matrix/Lagrange oracle,CPU timing · no Ceph/ISA-L/FastEC reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
