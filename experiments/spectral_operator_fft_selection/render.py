"""Self-contained Fourier-operator cards, calculator and frequency-sorter figure."""

from html import escape
import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "fourier-neural-operator-real-fft-hermitian"


def deck(report: dict) -> str:
    b = report["budget"]
    cases = {row["id"]:row for row in report["cases"]}
    cards = []

    def card(eyebrow, title, evidence, takeaway):
        n = len(cards) + 1
        cards.append(f'<section class="card" aria-label="Card {n} of 7">'
                     f'<div class="eyebrow">{eyebrow}</div><h1>{title}</h1>'
                     '<p class="muted">SYNTHETIC · independently_reproduced=false · no GPU or PDE-quality measurements</p>'
                     f'<div class="evidence">{evidence}</div><div class="callout">{takeaway}</div>'
                     f'<footer><span>Round 98 · real-grid spectral operators</span><span>{n} / 7</span></footer></section>')

    card("HANDS-ON · SCIENTIFIC AI",
         'Only the last <span class="accent">real-FFT axis</span> is one-sided.',
         '<div class="grid"><div class="panel"><h2>Workbench</h2>'
         '<p>3 × 5 odd real grid<br>4 × 6 even check<br>Gate: error &lt; 1e-12</p>'
         '<span class="pill">PDE quality: null</span></div><div class="panel"><h2>Contract</h2>'
         '<p>Signed first-axis rows</p><p>Explicit inverse size</p><p>Paired FFT normalization</p>'
         '</div><div class="panel"><h2>Separate budget</h2>'
         f'<p>{b["last_axis_bins"]} last-axis FFT bins<br>{b["retained_last_axis_bins"]} retained last-axis bins<br>'
         f'{b["dense_weight_complex_values"]:,} dense complex weights</p>'
         '<span class="pill">GPU latency: null</span></div></div>',
         '<b>Original standard-library complex DFTs, not an FFT benchmark.</b> '
         'A direct spatial circular convolution is the independent numerical reference. '
         'Multi-channel bytes are a separate analytical configuration.')
    exact = cases["staged-real-inverse"]
    odd = cases["odd-last-bin-treated-as-nyquist"]
    detail = '<blockquote>Five real cells produce three stored frequency bins. '
    detail += 'The last bin is paired, not an even-length Nyquist endpoint.</blockquote>'
    for row in (exact, odd):
        detail += '<div class="panel"><span class="pill">score · max output error</span>'
        detail += '<span class="pill">'+('Confirm' if row["matches_reference"] else 'Send to a human')+'</span>'
        detail += f'<h2>{escape(row["id"])}</h2><p>{row["maximum_absolute_error"]:.3e} · confidence: null</p>'
        detail += bar(row["maximum_absolute_error"],odd["maximum_absolute_error"])+'</div>'
    card("SCENARIO: ODD REAL GRID",
         f'Treating an odd-grid bin as Nyquist adds {odd["maximum_absolute_error"]:.4f} error.',
         detail,'<b>Shape is part of the contract.</b> Omitting the odd inverse length separately '
         'returns 3 × 4 instead of 3 × 5; that case has error=null, not zero.')
    names = {
        "staged-real-inverse":"Staged inverse",
        "premature-frequency-column-clear":"Clear 2D boundary columns early",
        "odd-last-bin-treated-as-nyquist":"Invent odd Nyquist",
        "implicit-even-inverse-length":"Infer even output length",
        "missing-conjugate-pair-factor":"Omit pair factor",
        "mismatched-inverse-normalization":"Mix FFT norms",
        "incorrectly-half-every-axis":"Halve both axes",
    }
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in report["cases"]:
        value = "null; wrong shape" if row["maximum_absolute_error"] is None else f'{row["maximum_absolute_error"]:.3e}'
        matrix += f'<tr><td>{names[row["id"]]}</td><td>Spatial circular convolution</td>'
        matrix += f'<td>{"Match" if row["matches_reference"] else "Mismatch ✕"}<br>{value}</td>'
        matrix += f'<td>{"Confirm" if row["matches_reference"] else "Review contract"}</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout"><b>Gate:</b> same shape and error &lt; 1e-12.</div>'
    matrix += '<div class="callout"><b>Failure scope:</b> original controls, not upstream bug reproductions.</div></div>'
    card("WHAT I TESTED","Six wrong contracts fail; the staged real inverse matches.",
         matrix,'<b>Separate intentional truncation:</b> retaining fewer frequency bins changes the full '
         'operator, but still matches an independent reference using the same retained spectrum.')
    card("UNDER THE HOOD","Clear real endpoints after the other axes have been inversed.",
         '<pre>input: real[Nx, Ny]\n'
         'spectrum = rfftn(input, norm="forward")\n'
         'shape = [Nx, floor(Ny/2)+1]\n'
         'out[kx,ky] = SUM_c input[c,kx,ky] * W[c,o,kx,ky]\n\n'
         'Only non-last axes use centered signed-frequency slicing.</pre>'
         '<p>↓ inverse the complex axes before enforcing real endpoints</p>'
         '<pre>rows = ifft(spectrum, axis=0, norm="forward")\n'
         'rows[:,0].imag = 0\n'
         'if Ny is even: rows[:,-1].imag = 0\n'
         'output = irfft(rows, n=Ny, axis=1, norm="forward")\n\n'
         'Raw 2D boundary columns can be complex across kx.\n'
         'Clearing them early erases valid first-axis phase.</pre>',
         '<b>Pinned source:</b> neuraloperator SpectralConv defaults to forward normalization and '
         'staged Hermitian enforcement. Its reported hardware artifacts were not reproduced here.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in report["normalization_checks"]:
        robust += f'<tr><td>Same convolution</td><td>Forward/forward pair</td><td>{row["norm"]} pair</td>'
        robust += f'<td>Max reference error {row["maximum_absolute_error"]:.2e}</td></tr>'
    trunc = report["mode_truncation"]
    robust += '</table><div class="panel"><h2>Truncation is an intentional operator change</h2>'
    robust += f'<p>Full operator difference: {trunc["full_operator_difference"]:.4f}</p>'
    robust += f'<p>Same retained-spectrum reference error: {trunc["same_retained_spectrum_reference_error"]:.2e}</p>'
    robust += '<p>Kernel multiplier is fixed and unnormalized while the input/inverse convention changes.</p></div>'
    card("ROBUSTNESS CHECK","Three paired FFT conventions preserve the same spatial operator.",
         robust,'<b>Units are raw output error, not accuracy points.</b> A mismatched inverse rescales '
         'the 3 × 5 result by 1/15. Mode truncation does not establish a PDE-error guarantee.')
    maximum = max(row["full_complex_one_pass_read_modeled_ms"] for row in report["scaling"])
    timing = '<p>Modeled one-pass spectrum read only: blue one-sided layout; gray full complex layout.</p>'
    for row in report["scaling"]:
        timing += f'<div class="latency-row"><b>Batch {row["input"]["batch"]} · 128 × 128 · 32 → 64 channels</b>'
        for key,label,baseline in (("one_sided_one_pass_read_modeled_ms","One-sided",False),
                                   ("full_complex_one_pass_read_modeled_ms","Full complex",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","One-sided buffers reduce byte reads, not proven GPU execution time.",
         timing,'<b>Assumed 1000 GB/s + 5 µs setup.</b> One uncached read of input/output spectra, '
         'not an FFT execution model; workspace, transforms and contraction excluded. Median/repeats N/A.')
    card("MY TAKE","Preserve FFT semantics before tuning retained modes or precision.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Real-grid spectral-layer correctness reviews.</p><p>Odd/even and phase-sensitive fixtures.</p>'
         '<p>Explicit complex-buffer budgeting.</p></div><div class="panel bad"><h2>⛔ Not for</h2>'
         '<p>Halving every frequency axis.</p><p>Calling every last bin Nyquist.</p>'
         '<p>GPU speedup or PDE-quality claims.</p></div></div>'
         '<pre>READINESS: original numerical prototype\n'
         'SOURCE: pinned neuraloperator SpectralConv\n'
         'LICENSE: upstream MIT; no implementation copied\n'
         'MODEL: one-channel circular convolution fixture\n'
         'BUDGET: separate dense multi-channel formulas\n'
         'GPU / PDE QUALITY / PEAK HBM: not measured</pre>',
         '<b>Synthetic, independently_reproduced=false.</b> The February 2026 Hermitian-enforcement '
         'change is verified source context, not a local reproduction of its hardware symptom.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}'
              'td{overflow-wrap:anywhere}</style>')
    return page("Round 98 · Fourier neural operator real FFT · seven cards",
                styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["batch","in_channels","out_channels","grid_x","grid_y","modes_x","modes_y","real_scalar_bytes"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(![4,8].includes(p.real_scalar_bytes)||p.modes_x>p.grid_x||p.modes_y>p.grid_y)
    throw Error("Modes must fit the grid; scalar bytes must be 4 or 8");
  if(!Number.isFinite(p.assumed_read_gbps)||p.assumed_read_gbps<=0)throw Error("Bandwidth must be finite and positive");
  if(!Number.isFinite(p.setup_us)||p.setup_us<0)throw Error("Setup must be finite and nonnegative");
  const half=Math.floor(p.grid_y/2)+1,retained=Math.floor(p.modes_y/2)+1;
  const complex=2*p.real_scalar_bytes,channels=p.in_channels+p.out_channels;
  const spectra=p.batch*channels*p.grid_x*half*complex;
  const full=p.batch*channels*p.grid_x*p.grid_y*complex;
  const weights=p.in_channels*p.out_channels*p.modes_x*retained;
  const real=p.batch*channels*p.grid_x*p.grid_y*p.real_scalar_bytes;
  if(![spectra,full,weights,weights*complex,real].every(Number.isSafeInteger))
    throw Error("Byte budget exceeds exact numerical range");
  return {last_axis_bins:half,retained_last_axis_bins:retained,complex_scalar_bytes:complex,
    dense_weight_complex_values:weights,dense_weight_bytes:weights*complex,
    input_output_spectrum_bytes:spectra,full_complex_spectrum_bytes:full,real_input_output_bytes:real,
    one_sided_one_pass_read_modeled_ms:spectra/(p.assumed_read_gbps*1e6)+p.setup_us/1000,
    full_complex_one_pass_read_modeled_ms:full/(p.assumed_read_gbps*1e6)+p.setup_us/1000,
    peak_hbm_bytes:null,gpu_latency_ms:null,pde_error:null};
}
const form=document.getElementById("calculator"),result=document.getElementById("result"),
  error=document.getElementById("error");
form.addEventListener("submit",event=>{event.preventDefault();update();});
function update(){
  error.textContent="";
  if(!form.reportValidity()){result.textContent="No estimate: correct invalid input.";return;}
  try{
    const p={...BASE};
    for(const control of form.elements)if(control.name){
      if(control.value.trim()==="")throw Error(control.name+" is required");
      p[control.name]=Number(control.value);
    }
    result.textContent=JSON.stringify({provenance:"synthetic selected buffers and one-pass byte-read assumptions",
      ...estimateBudget(p),production_approved:false},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(report: dict) -> str:
    p = report["budget"]["input"]
    labels = {
        "batch":"Batch / 批量","in_channels":"Input channels / 输入通道",
        "out_channels":"Output channels / 输出通道","grid_x":"Grid first axis / 首轴网格",
        "grid_y":"Grid last axis / 末轴网格","modes_x":"Requested first-axis modes / 首轴模式数",
        "modes_y":"Requested last-axis modes / 末轴模式数",
        "real_scalar_bytes":"Real scalar bytes (4/8) / 实数字节",
        "assumed_read_gbps":"Assumed one-pass GB/s / 假定读取带宽",
        "setup_us":"Assumed setup µs / 假定启动时间",
    }
    controls = ""
    for key,label in labels.items():
        minimum = 0 if key == "setup_us" else 0.001 if key == "assumed_read_gbps" else 1
        step = 0.001 if key in ("setup_us","assumed_read_gbps") else 1
        controls += f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" min="{minimum}" step="{step}" value="{p[key]}"></label>'
    return page("Real-FFT spectral layer / 实数频谱层工作台",
                '<main><div class="eyebrow">ROUND 98 · SYNTHETIC BUFFER ACCOUNTING</div>'
                '<h1>One-sided means one axis, not every axis.<br>只压缩末轴，不是每轴减半。</h1>'
                '<div class="panel"><b>契约：</b>先逆变换其他轴，再清零 DC 与偶数 Nyquist 的虚部；'
                '奇数末轴必须显式给出 inverse length。相同 FFT norm 必须成对使用。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>末轴存储 floor(Ny/2)+1 个复数 bin；参数末轴为 '
                'floor(requested_modes_y/2)+1。只计算 dense 复数权重与 input/output spectra，'
                '不含 FFT workspace、激活、梯度、优化器或框架分配。'
                '<p>时间只是一次读取所选字节的假设公式，不代表 FFT/收缩耗时、PDE 误差或峰值 HBM。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(report: dict) -> str:
    b = report["budget"]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">Real spectral frequency-card sorter</title>'
        '<desc id="description">A physical card sorter represents signed first-axis frequencies, '
        'paired last-axis frequencies and real endpoints. Endpoint handling follows the first inverse.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 98 · SYNTHETIC DEMO · FOURIER NEURAL OPERATORS</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A real FFT halves only its final frequency axis.</text>'
        '<text x="64" y="180" font-size="24">Retain signed rows; inverse other axes before enforcing real DC and even Nyquist endpoints.</text>'
        +sorter_scene()
        +'<text x="80" y="650" font-size="26" font-weight="700">A five-cell last axis stores three bins, but its final bin is not Nyquist.</text>'
        +f'<text x="80" y="700" font-size="25">Separate dense budget: {b["last_axis_bins"]} last-axis bins; '
        +f'{b["dense_weight_bytes"]:,} weight bytes; peak HBM unknown.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">Pair forward/inverse normalization and explicitly preserve odd output lengths.</text>'
        +'<text x="96" y="842" font-size="23">Truncating modes changes the operator; it does not prove a PDE-quality bound.</text>'
        +'<text x="96" y="883" font-size="21">Direct DFT checks are not FFT/GPU benchmarks; modeled byte reads are not execution latency.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/spectral_operator_fft_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(report: dict) -> dict[Path,str]:
    return {
        ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(report),
        ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(report),
        ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(report),
    }
