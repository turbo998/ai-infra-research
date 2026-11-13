"""Reproducible, dependency-free views of the sealed synthetic report."""

from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SLUG = "ddp-gradient-compression-powersgd-error-feedback"
THEME = """
:root {
  color-scheme: light;
  --cp-bg: #f7f4ef;
  --cp-bg-elevated: #fcfbf8;
  --cp-surface: #ffffff;
  --cp-surface-soft: #f5f5f5;
  --cp-border: #dedede;
  --cp-border-strong: #919191;
  --cp-text: #242424;
  --cp-text-muted: #5c5c5c;
  --cp-text-soft: #6f6f6f;
  --cp-accent: #b11f4b;
  --cp-accent-hover: #9a1a41;
  --cp-accent-soft: rgba(177, 31, 75, 0.08);
  --cp-accent-fg: #ffffff;
  --cp-success: #16a34a;
  --cp-danger: #dc2626;
  --cp-warning: #f59e0b;
  --cp-link: #0078d4;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.12);
  --cp-overlay: rgba(255, 255, 255, 0.8);
  --cp-panel: rgba(255, 255, 255, 0.86);
  --cp-panel-strong: rgba(255, 255, 255, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.55);
  --cp-highlight: rgba(177, 31, 75, 0.12);
  --cp-code: #0F172A;
  --cp-primary: #3B82F6;
  --cp-baseline: #CBD5E1;
}
html[data-theme="dark"] {
  color-scheme: dark;
  --cp-bg: #3d3b3a;
  --cp-bg-elevated: #343231;
  --cp-surface: #292929;
  --cp-surface-soft: #2e2e2e;
  --cp-border: #474747;
  --cp-border-strong: #5f5f5f;
  --cp-text: #dedede;
  --cp-text-muted: #919191;
  --cp-text-soft: #b0b0b0;
  --cp-accent: #fd8ea1;
  --cp-accent-hover: #fb7b91;
  --cp-accent-soft: rgba(253, 142, 161, 0.14);
  --cp-accent-fg: #1a1a1a;
  --cp-success: #4ade80;
  --cp-danger: #f87171;
  --cp-warning: #fbbf24;
  --cp-link: #4da6ff;
  --cp-shadow: 0 18px 48px rgba(0, 0, 0, 0.32);
  --cp-overlay: rgba(41, 41, 41, 0.88);
  --cp-panel: rgba(41, 41, 41, 0.72);
  --cp-panel-strong: rgba(41, 41, 41, 0.96);
  --cp-sheen: rgba(255, 255, 255, 0.04);
  --cp-highlight: rgba(253, 142, 161, 0.12);
}
"""
CSS = """
*{box-sizing:border-box}body{margin:0;padding:24px;background:var(--cp-bg);color:var(--cp-text);
font-family:"Segoe UI",Aptos,Calibri,-apple-system,BlinkMacSystemFont,sans-serif}
main{max-width:1180px;margin:auto}h1{font-size:44px;line-height:1.13;margin:18px 0}
h2{font-size:25px}p,li{line-height:1.55}.muted{color:var(--cp-text-muted)}
.eyebrow{font-size:14px;letter-spacing:.12em;font-weight:700;color:var(--cp-accent)}
.accent{background:linear-gradient(90deg,var(--cp-accent),var(--cp-primary));
background-clip:text;color:transparent}
.panel,blockquote{border:1px solid var(--cp-border);border-radius:16px;padding:20px;
background:var(--cp-surface-soft);margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.pill{display:inline-block;border:1px solid var(--cp-border);border-radius:.625rem;
padding:5px 9px;margin:4px 4px 4px 0;font-size:13px}
.good{border-top:4px solid var(--cp-success)}.bad{border-top:4px solid var(--cp-danger)}
pre,code{font-family:Consolas,"Courier New",Courier,monospace}
pre{background:var(--cp-code);color:var(--cp-accent-fg);padding:24px;border-radius:16px;
font-size:16px;line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;font-size:16px}th,td{text-align:left;padding:14px 8px;
border-bottom:1px solid var(--cp-border)}th{font-size:12px;letter-spacing:.04em}
.bar{height:14px;background:var(--cp-surface-soft);border-radius:5px;overflow:hidden;margin:6px 0 12px}
.bar i{display:block;height:100%;background:var(--cp-primary)}.bar.baseline i{background:var(--cp-baseline)}
.card{position:relative;width:900px;height:1125px;max-width:100%;margin:0 auto 32px;
padding:46px;background:var(--cp-surface);border:1px solid var(--cp-border);border-radius:16px;
display:flex;flex-direction:column;overflow:hidden}
.card:before{content:"";position:absolute;top:0;left:0;right:0;height:10px;
background:linear-gradient(90deg,var(--cp-accent),var(--cp-primary))}
.evidence{flex:1}.callout{background:var(--cp-surface-soft);padding:18px;border-radius:12px;margin:14px 0}
footer{border-top:1px solid var(--cp-border);padding-top:16px;display:flex;justify-content:space-between;font-size:13px}
label{display:flex;flex-direction:column;gap:8px;font-size:15px}
input,button,select{font:inherit;border:1px solid var(--cp-border-strong);border-radius:.625rem;
padding:10px;background:var(--cp-surface);color:var(--cp-text);max-width:100%}
button{cursor:pointer;background:var(--cp-accent);color:var(--cp-accent-fg)}
input:focus-visible,button:focus-visible,select:focus-visible{outline:3px solid var(--cp-link);outline-offset:3px}
.error{color:var(--cp-danger);font-weight:700}output{display:block}
@media(max-width:700px){body{padding:12px}.grid{grid-template-columns:1fr}
.card{height:auto;min-height:1125px;padding:24px}h1{font-size:34px}table{font-size:13px}th,td{padding:10px 4px}}
@media print{body{padding:0;background:var(--cp-surface)}.card{margin:0;border:0;border-radius:0;break-after:page}
@page{size:900px 1125px;margin:0}}
"""
DETECTION = """
(() => {
  const param = new URLSearchParams(window.location.search).get("scoutTheme");
  const theme = param === "light" || param === "dark" ? param :
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
})();
"""


def page(title: str, body: str, script: str | None = None) -> str:
    csp = "default-src 'none'; style-src 'unsafe-inline'"
    if script is not None:
        csp += "; script-src 'unsafe-inline'"
    # The static deck deliberately has no script, preserving its stricter CSP.
    js = f"<script>{DETECTION}\n{script}</script>" if script is not None else ""
    return (
        '<!doctype html>\n<html lang="en" data-theme="light"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<title>{escape(title)}</title><style>{THEME}{CSS}</style></head>"
        f"<body>{body}{js}</body></html>\n"
    )


def bar(value: float, maximum: float, baseline: bool = False) -> str:
    width = 100 * value / maximum if maximum else 0
    return f'<div class="bar{" baseline" if baseline else ""}"><i style="width:{width:.6f}%"></i></div>'


def deck(report: dict) -> str:
    rows = report["scenarios"]
    first = rows[0]
    profile = first["input"]
    f = lambda value: f"{value:.2f}"
    cards = []

    def card(eyebrow: str, title: str, evidence: str, takeaway: str) -> None:
        number = len(cards) + 1
        cards.append(
            f'<section class="card" aria-label="Card {number} of 7">'
            f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
            '<p class="muted">SYNTHETIC DEMO · independently_reproduced=false · no GPU benchmark</p>'
            f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
            f'<footer><span>Round {report["round"]} · DDP communication model</span><span>{number} / 7</span></footer></section>'
        )

    card(
        "HANDS-ON · TRAINING COMMUNICATION",
        f'<span class="accent">{f(first["payload_compression_ratio"])}× fewer payload bytes.</span><br>Not the same step speedup.',
        '<div class="grid"><div class="panel"><h2>Scenarios</h2>'
        + "".join(f'<p>{escape(row["id"])}</p>' for row in rows)
        + '</div><div class="panel"><h2>Contract</h2>'
        + f'<p>{profile["matrices"]} matrices<br>{profile["rows"]} × {profile["columns"]}<br>'
        + f'FP32 → rank {profile["rank"]}<br>{profile["workers"]} workers</p>'
        + '<p>Timing gate: lower modeled step.<br>Quality gate: pending.</p></div>'
        + f'<div class="panel"><h2>Outputs</h2><p>{f(first["dense"]["step_ms"])} → '
        + f'{f(first["powersgd"]["step_ms"])} ms</p><p>{f(first["powersgd_step_speedup"])}× modeled step speedup</p>'
        + '<span class="pill">Tokens: N/A</span><span class="pill">Confidence: null</span>'
        + '</div></div><div class="panel"><b>Workbench:</b> the companion H5 exposes bandwidth, rank, '
        + 'overlap and codec costs. No API key, downloads or telemetry.</div>',
        f'<b>Provenance:</b> {escape(report["hardware"])}. All coordinates come from report.json; '
        'the model does not evaluate training convergence.',
    )
    evidence = (
        f'<blockquote>Assume {profile["bandwidth_gbps"]} GB/s effective bandwidth, '
        f'{profile["overlap_ms"]} ms overlap and {profile["codec_ms"]} ms PowerSGD codec cost.</blockquote>'
    )
    for label, key in (("FP32 dense", "dense"), ("BF16 cast", "bf16"), ("PowerSGD", "powersgd")):
        row = first[key]
        evidence += (
            f'<div class="panel"><span class="pill">score · analytical</span><span class="pill">Confirm</span>'
            f'<h2>{label}: {f(row["step_ms"])} ms step</h2>'
            f'<p>Service {f(row["service_ms"])} ms · exposed {f(row["exposed_ms"])} ms · confidence: null</p>'
            + bar(row["step_ms"], first["dense"]["step_ms"])
            + '</div>'
        )
    card("SCENARIO: BANDWIDTH-BOUND", f'{f(first["powersgd_saved_ms"])} ms saved in the assumed critical path.',
         evidence, '<b>Confirm, never auto-deploy.</b> Timing scores are not probabilities. '
         'Compression removes exposed communication only; production quality remains unmeasured.')
    table = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in rows:
        match = row["timing_only_winner"] == row["expected_timing_winner"]
        table += (
            f'<tr><td>{escape(row["id"])}</td><td>Lowest modeled step?</td>'
            f'<td><b>{row["timing_only_winner"]}</b>{" ✕" if not match else ""}</td><td>Confirm</td></tr>'
        )
    table += '</table>'
    card("WHAT I TESTED", f'{report["summary"]["scenario_count"]} timing cases. No quality approvals.',
         table + '<div class="grid two"><div class="callout"><b>Gating:</b> strict shape saving, '
         'then exposed-step comparison. Ties retain dense.</div><div class="callout"><b>Failure analysis:</b> '
         'fast links hide the baseline; vectors and high rank erase compression savings. No fabricated misses.</div></div>',
         f'<b>{report["summary"]["timing_policy_matches"]} expected timing policies matched.</b> '
         f'{report["summary"]["production_approved"]} production approvals; hand-authored analytical cases are not a model benchmark score.')
    card(
        "UNDER THE HOOD", "Shared factors preserve averaging. Error feedback preserves a ledger.",
        '<pre>' + escape(json.dumps({
            "gradient": "M_i = G_i + E_i", "P": "orth(mean_i(M_i Q_shared))",
            "Q_next": "mean_i(M_i^T P)", "output": "P Q_next^T"
        }, indent=2)) + '</pre><p>↓ two sequential factor reductions; vectors stay dense</p>'
        '<pre>E_i(next) = M_i - output\n'
        'mean_i(sum_t G_i,t) - sum_t output_t\n'
        '  = mean_i E_i(final)   # initial E = 0\n\n'
        'step = backward + max(0, comm + codec - overlap)</pre>',
        '<b>No independent local SVDs.</b> Factors must share their basis. '
        'The telescoping identity is accounting, not convergence or loss parity.',
    )
    robustness = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for seed in (7, 11):
        off, on = [r for r in report["feedback"] if r["seed"] == seed]
        robustness += (
            f'<tr><td>Seed {seed}: cumulative gradient error</td><td>EF off: '
            f'{f(off["cumulative_relative_error_pct"])}%</td><td>EF on: '
            f'{f(on["cumulative_relative_error_pct"])}%</td><td>'
            f'{f(on["cumulative_relative_error_pct"] - off["cumulative_relative_error_pct"])} pts</td></tr>'
        )
    robustness += '</table><div class="panel"><h2>What changed?</h2><p>Only error feedback and the '
    robustness += 'shared-Q seed. This is a fixed synthetic gradient sequence, not SGD on a trained model.</p>'
    robustness += f'<p>{report["feedback"][0]["workers"]} workers · {report["feedback"][0]["steps"]} steps · '
    robustness += f'{report["feedback"][0]["matrix_shape"]} matrices · rank {report["feedback"][0]["rank"]}</p></div>'
    card("ROBUSTNESS CHECK", "Residual feedback reduces cumulative error in the toy sequence.",
         robustness, '<b>Limited perturbation coverage.</b> Only the sealed seeds and gradient sequence '
         'are tested. No general accuracy, optimizer or dataset claim is supported.')
    maximum = max(r["dense_ms"] for r in report["robustness"])
    latency = '<p>Blue: PowerSGD · gray: FP32 dense · units: modeled step milliseconds.</p>'
    for row in report["robustness"]:
        latency += (
            f'<div class="panel"><b>{row["bandwidth_gbps"]} GB/s</b><p>'
            f'PowerSGD <b>{f(row["powersgd_ms"])} ms</b></p>'
            + bar(row["powersgd_ms"], maximum)
            + f'<p>Dense <b>{f(row["dense_ms"])} ms</b></p>'
            + bar(row["dense_ms"], maximum, True) + '</div>'
        )
    card("LATENCY", "Faster fabric can reduce the step saving to zero.", latency,
         '<b>Not measured latency.</b> One deterministic evaluation per tier; median/repeats: N/A. '
         'Ranks, codec cost and overlap are held fixed. No CPU runtime is presented as GPU time.')
    card(
        "MY TAKE", "Compress exposed bandwidth, not already-hidden collectives.",
        '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2><p>Replicated DDP gradients.</p>'
        '<p>Large compressible matrices.</p><p>Bandwidth-bound cross-node path.</p>'
        '<p>Room for residual buffers.</p><p>Matched-quality A/B evaluation.</p></div>'
        '<div class="panel bad"><h2>⛔ Not for</h2><p>Automatic FSDP reduce-scatter replacement.</p>'
        '<p>Tiny tensors or mostly vectors.</p><p>Already-hidden communication.</p>'
        '<p>Assuming BF16 buckets halve twice.</p><p>Unverified FP8 bytewise reductions.</p></div></div>'
        '<pre>READINESS: analytical prototype only\nQUALITY / CONVERGENCE: not measured\n'
        'DATA: experiments/ddp_gradient_compression_selection/report.json\n'
        'SOURCE: pinned PyTorch v2.10.0 (BSD-3-Clause)\n'
        'ARTIFACTS: original repository material; no new license grant</pre>',
        '<b>SYNTHETIC DEMO · independently_reproduced=false.</b> Restore per-rank residual state and '
        'bucket mapping when resuming; revalidate after world-size or layout changes.',
    )
    return page("Round 92 · DDP compression · synthetic seven-card report", '<main>' + "".join(cards) + '</main>')


CALCULATOR = r"""
function estimateProfile(p) {
  for (const k of ["rows","columns","matrices","workers","rank"])
    if (!Number.isSafeInteger(p[k]) || p[k] < 1) throw Error(k + " must be a positive safe integer");
  if (!Number.isSafeInteger(p.vector_elements) || p.vector_elements < 0)
    throw Error("vector_elements must be a nonnegative safe integer");
  if (p.rank > Math.min(p.rows,p.columns)) throw Error("rank exceeds the smaller matrix dimension");
  for (const k of ["bandwidth_gbps","min_compression_rate","backward_ms"])
    if (!Number.isFinite(p[k]) || p[k] <= 0) throw Error(k + " must be finite and positive");
  for (const k of ["alpha_us","overlap_ms","codec_ms","cast_ms"])
    if (!Number.isFinite(p[k]) || p[k] < 0) throw Error(k + " must be finite and nonnegative");
  if (p.overlap_ms > p.backward_ms) throw Error("Overlap cannot exceed backward time");
  const elements = p.rows*p.columns*p.matrices+p.vector_elements;
  const factorElements = (p.rows+p.columns)*p.rank*p.matrices;
  if (!Number.isSafeInteger(4*elements) || !Number.isSafeInteger(4*factorElements))
    throw Error("Payload exceeds exact JavaScript integer range");
  const eligible = p.rows>1 && p.columns>1 &&
    (p.rows+p.columns)*p.rank*p.min_compression_rate < p.rows*p.columns;
  function path(payload,collectives,codec) {
    const wire = payload*2*(p.workers-1)/p.workers;
    const communication = wire/(p.bandwidth_gbps*1e6)+collectives*2*(p.workers-1)*p.alpha_us/1000;
    const service = communication+codec, exposed = Math.max(0,service-p.overlap_ms);
    return {payload_bytes:payload,ring_send_bytes_per_rank:wire,collectives,
      communication_ms:communication,codec_ms:codec,service_ms:service,
      exposed_ms:exposed,step_ms:p.backward_ms+exposed};
  }
  const dense=path(4*elements,1,0), bf16=path(2*elements,1,p.cast_ms);
  const powersgd=path(eligible ? 4*(factorElements+p.vector_elements) : 4*elements,
    eligible ? 2+Number(p.vector_elements>0) : 1, eligible ? p.codec_ms : 0);
  return {compression_eligible:eligible,dense,bf16,powersgd,
    payload_compression_ratio:dense.payload_bytes/powersgd.payload_bytes,
    powersgd_step_speedup:dense.step_ms/powersgd.step_ms,
    powersgd_saved_ms:dense.step_ms-powersgd.step_ms};
}
const form=document.getElementById("calculator"), output=document.getElementById("result"),
  error=document.getElementById("error");
function update() {
  error.textContent="";
  if (!form.reportValidity()) { output.textContent="Invalid input: correct the highlighted field."; return; }
  try {
    const p={...BASE};
    for (const control of form.elements) if (control.name) {
      if (control.value.trim()==="") throw Error(control.name+" is required");
      p[control.name]=Number(control.value);
    }
    const r=estimateProfile(p);
    output.textContent=JSON.stringify({
      provenance:"synthetic; not GPU-measured",...r,
      persistent_residual_gib:r.dense.payload_bytes/2**30,
      residual_note:"one FP32 gradient buffer; temporary copies and factors are additional",
      quality:"pending; no production approval"},null,2);
  } catch (e) { error.textContent=e.message; output.textContent="No estimate: invalid input."; }
}
form.addEventListener("submit",event=>{event.preventDefault();update();});
update();
"""


def workbench(report: dict) -> str:
    p = report["scenarios"][0]["input"]
    fields = [
        ("rows", "Matrix rows / 矩阵行数", 1, 1),
        ("columns", "Matrix columns / 矩阵列数", 1, 1),
        ("matrices", "Matrix count / 矩阵数量", 1, 1),
        ("vector_elements", "Dense vector elements / 向量元素", 0, 1),
        ("workers", "DDP workers / 进程数", 1, 1),
        ("rank", "Factor rank / 压缩秩", 1, 1),
        ("min_compression_rate", "Minimum payload ratio / 最低压缩比", 0.01, 0.01),
        ("bandwidth_gbps", "Effective bandwidth GB/s / 有效带宽", 0.01, 0.01),
        ("alpha_us", "Per-hop startup µs / 单跳启动", 0, 0.01),
        ("overlap_ms", "Hideable overlap ms / 可隐藏窗口", 0, 0.01),
        ("backward_ms", "Backward floor ms / 反向基础时间", 0.01, 0.01),
        ("codec_ms", "PowerSGD codec ms / 压缩开销", 0, 0.01),
        ("cast_ms", "BF16 cast ms / 类型转换开销", 0, 0.01),
    ]
    controls = "".join(
        f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
        f'min="{minimum}" step="{step}" value="{p[key]}"></label>'
        for key, label, minimum, step in fields
    )
    return page(
        "DDP compression break-even workbench / 梯度通信压缩工作台",
        '<main><div class="eyebrow">ROUND 92 · SYNTHETIC DEMO</div>'
        '<h1>Fewer bytes ≠ faster training.<br>梯度压缩：先算暴露在关键路径上的时间。</h1>'
        '<p>FP32 dense vs BF16 cast vs PowerSGD. No GPU measurements, network requests or model-quality scores.</p>'
        '<div class="panel"><b>单位注意：</b>GB/s 是十进制字节带宽，不是 Gb/s。'
        'Codec、有效带宽与 overlap 都是输入假设。FP32 梯度作为基线；原生 BF16 bucket 不会再减半。</div>'
        f'<form id="calculator"><div class="grid">{controls}</div>'
        '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
        '<p id="error" class="error" role="alert"></p>'
        '<output id="result" aria-live="polite" aria-label="Analytical estimate"></output>'
        '<noscript>This calculator requires inline JavaScript. No estimate is available without it.</noscript>'
        '<div class="panel"><b>模型边界：</b>一个聚合 bucket，理想 Ring，P/Q 顺序执行；'
        '向量另行稠密归约。固定 codec 与通信共享一个隐藏窗口，不模拟真实 NCCL 调度。'
        '必须另测收敛、loss、HBM 峰值与 matched-quality time-to-train。'
        '<p><b>Error feedback:</b> one persistent FP32 gradient buffer, plus factor/cache/temporary memory. '
        'Per-rank residuals are not interchangeable checkpoints.</p></div></main>',
        "const BASE=" + json.dumps(p) + ";\n" + CALCULATOR,
    ).replace('<output id="result"', '<output style="white-space:pre-wrap;font-family:Consolas,monospace" id="result"')


def summary_svg(report: dict) -> str:
    first = report["scenarios"][0]
    # Physical metaphor: a patch-panel chassis, full-width input cards, two slim
    # factor cards and a residual drawer. Every solid sits on the chassis.
    parts = []

    def point(x: float, y: float, z: float) -> tuple[float, float]:
        return 800 + (x - y) * 1.65, 410 + (x + y) * 0.55 - z * 1.6

    def polygon(points: list[tuple], fill: str) -> str:
        return '<polygon points="' + " ".join(f"{a:.2f},{b:.2f}" for a, b in points) + f'" fill="{fill}" stroke="#475569" stroke-width="1"/>'

    def block(x: float, y: float, z: float, w: float, d: float, h: float, lit: bool = False) -> None:
        a, b, c, e = [point(*v) for v in ((x, y, z+h), (x+w, y, z+h), (x+w, y+d, z+h), (x, y+d, z+h))]
        af, bf, cf, ef = [point(*v) for v in ((x, y, z), (x+w, y, z), (x+w, y+d, z), (x, y+d, z))]
        markup = polygon([b, c, cf, bf], "#475569") + polygon([e, c, cf, ef], "#64748b")
        markup += polygon([a, b, c, e], "#b11f4b" if lit else "#e2e8f0")
        parts.append(((z, x + y + w + d), markup))

    for x in (-190, 170):
        for y in (-100, 90):
            block(x, y, -16, 22, 22, 10)
    block(-215, -120, -6, 430, 250, 12)
    for y in (-110, 110):
        block(-200, y, 6, 400, 7, 10)
    for x in (-200, 190):
        block(x, -103, 6, 7, 213, 10)
    for row in range(4):
        for column in range(4):
            x, y = -165 + column * 36, -77 + row * 38
            block(x, y, 6, 29, 30, 12)
            block(x + 10, y + 9, 18, 9, 12, 3)
    for x in (20, 54):
        block(x, -77, 6, 20, 144, 20, True)
        for pin in range(4):
            block(x + 4, -70 + pin * 35, 26, 12, 6, 3)
    block(108, -77, 6, 62, 144, 15)
    for row in range(4):
        block(118, -67 + row * 33, 21, 42, 24, 5)
    for x in (-200, 187):
        for y in (-105, 105):
            block(x, y, 16, 5, 5, 2)
    scene = "".join(markup for _, markup in sorted(parts))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description">'
        '<title id="title">DDP compression: fewer bytes do not guarantee a shorter step</title>'
        '<desc id="description">An isometric patch-panel metaphor: full matrix cards, two slim factor '
        'cards and an equally important residual drawer. Synthetic analytical estimates, not GPU measurements.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 92 · SYNTHETIC DEMO · DDP</text>'
        '<text x="64" y="128" font-size="44" font-weight="700">A smaller wire payload is only one part of the training step.</text>'
        '<text x="64" y="175" font-size="24">Shared P/Q factors need two reductions; residual feedback needs a gradient-sized ledger.</text>'
        + scene
        + '<text x="92" y="650" font-size="24" font-weight="700">Full matrix cards → factor cards + residual drawer</text>'
        + f'<text x="92" y="700" font-size="26">Payload ratio: {first["payload_compression_ratio"]:.2f}× · '
        + f'Modeled step: {first["dense"]["step_ms"]:.2f} → {first["powersgd"]["step_ms"]:.2f} ms</text>'
        + '<rect x="64" y="750" width="1472" height="156" rx="18" fill="#0F172A"/>'
        + '<g fill="#ffffff"><text x="96" y="797" font-size="23">Gate 1 · shape saving     Gate 2 · exposed time     Gate 3 · matched training quality</text>'
        + '<text x="96" y="843" font-size="22">E(next) = G + E − reconstruction. Feedback is an accounting identity, not a convergence guarantee.</text>'
        + '<text x="96" y="881" font-size="20">FP32 baseline · ideal Ring · assumed codec/overlap · no FSDP claim · no physical GPU measurements</text></g>'
        + '<text x="64" y="952" font-size="20">Data: experiments/ddp_gradient_compression_selection/report.json · independently_reproduced=false</text>'
        + '</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path, str]:
    return {
        ROOT / "docs" / "reports" / f"{SLUG}-selection-cards.html": deck(report),
        ROOT / "docs" / "reports" / f"{SLUG}-sizer.html": workbench(report),
        ROOT / "assets" / "svg" / f"{SLUG}-selection-faq-summary.svg": summary_svg(report),
    }
