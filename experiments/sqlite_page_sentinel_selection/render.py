import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="sqlite-64k-page-size-content-offset-sentinels"
JS=r"""
function integer(v,lo,hi){if(!Number.isInteger(v)||v<lo||v>hi)throw Error(`Exact integer${lo}..${hi}required`);return v;}
function layout(p){
 const keys=["page_bytes","first_page","reserved_bytes","cells","content_raw","fragmented_bytes"].sort();
 if(!p||Array.isArray(p)||JSON.stringify(Object.keys(p).sort())!==JSON.stringify(keys))throw Error("Exact original leaf-header fields required");
 const size=integer(p.page_bytes,512,65536);if(size&(size-1))throw Error("Power-of-two page size required");
 if(typeof p.first_page!=="boolean")throw Error("Explicit first_page boolean required");
 const reserved=integer(p.reserved_bytes,0,255),cells=integer(p.cells,0,32767),raw=integer(p.content_raw,0,65535),fragments=integer(p.fragmented_bytes,0,60);
 const usable=size-reserved,base=p.first_page?100:0,content=raw===0?65536:raw,end=base+8+2*cells;
 if(usable<480)throw Error("Usable SQLite page bytes must be at least480");
 if(!(end<=content&&content<=usable)||fragments>usable-content)throw Error("Invalid original leaf geometry");
 if(!cells&&(content!==usable||fragments))throw Error("Original empty leaf must have unused usable body");
 if(cells&&content===usable)throw Error("Nonempty leaf needs content");
 return {file_header_bytes:base,leaf_header_bytes:8,pointer_array_bytes:2*cells,usable_page_bytes:usable,decoded_content_offset:content,
 unallocated_bytes:content-end,fragmented_bytes:fragments,unused_without_freeblocks_bytes:content-end+fragments,
 naive_raw_offset_unused_bytes:raw-end+fragments,opaque_content_region_bytes:usable-content,
 scope:"original header arithmetic only;leaf0x0d,no freeblocks;not cell/record/overflow/tree/file validation"};
}
"""


def workbench(r):
    return page("SQLite64KiB field-context accounting",
        f"""<h1>Page-size1and content-offset0are different field-specific sentinels</h1>
        <p>Round143 · Original no-freeblock table-leaf header geometry,not a SQLite file validator.</p>
        <form id="form"><label>Selected header JSON<textarea id="input" rows="9" required>{json.dumps(r['case']['inputs'])}</textarea></label>
        <button>Decode content offset and selected free space</button></form>
        <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
        <p>first_page adds the100-byte file header;leaf header8B;cell pointers2B each.
        No freeblock chains,cell record decoding,interior pages or overflow pages.
        Native raw-page headers were observed separately;local dbstat unavailable remains unknown,null—not zero.</p>""",
        JS+r"""
function update(event){if(event)event.preventDefault();const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{if(!document.getElementById("form").checkValidity())throw Error("Input required");out.textContent=JSON.stringify(layout(JSON.parse(document.getElementById("input").value)),null,2);}
 catch(e){error.textContent=e.message}}
document.getElementById("form").addEventListener("submit",update);update();
""")


def cards(r):
    c=r["case"];out=c["outputs"];native=r["native_raw_page_fixture"]
    table="".join(f"<tr><td>{v['inputs']['page_bytes']}B / first={v['inputs']['first_page']}</td><td>raw offset{v['inputs']['content_raw']}</td>"
                  f"<td>{v['decoded']['outputs']['unused_without_freeblocks_bytes']}B unused</td><td>{'✕ Raw formula negative' if v['decoded']['outputs']['naive_raw_offset_unused_bytes']<0 else 'Decode field context'}</td></tr>" for v in c["scenarios"])
    mx=max(max(v["unpacked_median_ms"],v["manual_median_ms"]) for v in r["latency"]["rows"])
    timing="".join(f"<div class='metric'><b>{v['headers']}repeated headers</b><p>struct.unpack {v['unpacked_median_ms']:.6f}ms</p>{bar(v['unpacked_median_ms'],mx)}"
                   f"<p>Manual shifts {v['manual_median_ms']:.6f}ms</p>{bar(v['manual_median_ms'],mx,True)}</div>" for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · BINARY FIELD CONTEXT","An empty64KiB leaf has65528unused bytes—not negative8",
         "<div class='grid'><div><h3>Scenarios</h3><p>Two sentinel fields<br>First-page prefix<br>Reserved tail</p></div><div><h3>Input</h3><p>65536B page<br>8B leaf header<br>raw content offset0</p></div>"
         "<div><h3>Decisions</h3><p>Decode before arithmetic<br>Do not global-rewrite zero<br>Keep missing dbstat unknown</p><span class='pill'>Native dbstat:unavailable</span></div></div>"
         f"<div class='dark'>SYNTHETIC original geometry,independently_reproduced=false.Separate measured SQLite{native['sqlite_version']}raw headers;no patched dbstat binary.Build CPU Python{r['latency']['python']} / {r['latency']['machine']}.Nov14actual upstream accounting fix.</div>"),
        ("SCENARIO: EMPTY64KLEAF","Content-offset0decodes to65536before subtracting the8-byte header",
         "<blockquote>Original header bytes[13,0,0,0,0,0,0,0]:leaf flag13,zero cells,no freeblock,content raw0.</blockquote>"
         f"<div class='metric'><span class='pill'>score · exact byte arithmetic,not confidence</span><h3>{out['unused_without_freeblocks_bytes']}/{c['inputs']['page_bytes']}B unused</h3>{bar(out['unused_without_freeblocks_bytes'],c['inputs']['page_bytes'])}<p>Act automatically only within validated original header domain</p></div>"
         f"<div class='metric'><h3>✕ Raw-offset formula:{out['naive_raw_offset_unused_bytes']}B</h3><p>Negative result is displayed—not clamped to a fabricated zero</p></div>"),
        ("WHAT I TESTED","Five layouts separate64KiB sentinels,first-page overhead and reserved bytes",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"+table+"</table>"
         "<div class='grid two'><div class='callout'>Deterministic geometry,not a probability threshold or full-record validity score.</div><div class='callout'>First-freeblock0still means none;cell-count0still means empty.</div></div>"),
        ("UNDER THE HOOD","File page-size1means65536;cell-content-offset0means65536;other zeros stay zero",
         "<pre>Original selected field decoding:\nfile header offset16,u16be:1→65536\nleaf header offset5,u16be:0→65536\nleaf first-freeblock0→none\nleaf cell-count0→zero\nNever apply one universal zero rule</pre>"
         "<p>↓ Original independent interval accounting</p><pre>pointer_end=file_header+8+2*cells\nunallocated=decoded_content-pointer_end\nunused=unallocated+fragmented\nNo freeblock chain in model\nCheck usable page and pointer bounds\nOpaque cells are not SQLite records</pre>"),
        ("ROBUSTNESS CHECK","First-page overhead removes100free bytes;32reserved bytes disable the zero content sentinel",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         "<tr><td>Empty leaf unused</td><td>65528B</td><td>First page65428B</td><td>−100B</td></tr>"
         "<tr><td>Contentoffset</td><td>raw0→65536</td><td>Reserve32:raw65504</td><td>Field no longer zero</td></tr>"
         "<tr><td>Native dbstat</td><td>Module absent</td><td>Raw headers available</td><td>Metric stays null</td></tr></table>"),
        ("LATENCY","Repeated8-byte decoding measures Python/C conversion—not SQLite page scans",
         timing+"<p>Same hot prevalidated header reused—not distinct physical pages.Two warmups,seven alternating repeats,three operations/sample,median repeat means.Both return tuples;full-page I/O,validation,SQLite/dbstat and cache-miss behavior excluded.</p>"),
        ("MY TAKE","Decode representation sentinels before trusting storage-accounting dashboards",
         "<div class='grid two'><div class='good'><h3>✅ Good fit</h3><p>Field-context decoding,selected header budgets,first-page offsets and explicit missing observability capabilities.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Universal zero replacement,negative free-space clamps,record/tree corruption certification or claiming the new dbstat fix was natively executed.</p></div></div>"
         "<div class='dark'>SYNTHETIC model plus measured original raw pages.SQLite public-domain source,no implementation mirror.Local dbstat absent;new patched binary not built.Visual browser inspection not performed.</div>"),
    ]
    return page("SQLite64KiB sentinels — seven cards","".join(f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>' for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
<title>Storage accounting cabinet:field labels determine what sentinel tickets mean</title>
<desc>Original sixty-solid sorting cabinet metaphor;identical zero tickets have different meanings in content-offset,freeblock and cell-count drawers.Static Figure tier,not physical SQLite storage.</desc>
<rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
<g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG143 · FIELD CONTEXT BEFORE BYTE ACCOUNTING</text>
<text x="88" y="158" font-size="36" font-weight="700">A zero content offset can describe an entirely empty64KiB page</text>{sorter_scene()}
<text x="88" y="710" font-size="25">File page-size1 and content-offset0 both represent65536—in different fields</text>
<text x="88" y="758" font-size="25">First-freeblock0 means none;cell-count0 means zero—not65536</text>
<text x="88" y="806" font-size="25">Decode the sentinel before subtracting headers;do not clamp negative evidence</text>
<text x="88" y="880" font-size="18">Original header geometry and benign raw-page fixture · dbstat unavailable stays null;no native patch or full-file validation</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
