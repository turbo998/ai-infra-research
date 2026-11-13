from fractions import Fraction
import json
from pathlib import Path

from experiments.ddp_gradient_compression_selection.render import page,bar
from experiments.contrastive_gradcache_selection.render import sorter_scene

ROOT=Path(__file__).resolve().parents[2]
SLUG="kernel-cholesky-jitter-factor-reuse-variance"
JS=r"""
function estimate(p){
 if(!Object.values(p).every(v=>Number.isSafeInteger(v)&&v>0))throw Error("positive bounded exact counts required");
 const n=p.training_points,q=p.query_points,out={selected_float64_covariance_bytes:n*n*8,selected_float64_factor_bytes:n*n*8,
 selected_float64_cross_covariance_bytes:n*q*8,selected_float64_rhs_bytes:n*8};
 if(!Object.values(out).every(Number.isSafeInteger))throw Error("byte result exceeds exact arithmetic range");return out;
}
function calculate(shift){
 if(!Number.isFinite(shift)||shift<0)throw Error("finite nonnegative diagonal shift required");
 const a=1+shift,l00=Math.sqrt(a),l10=1/l00,pivot=a-l10*l10;
 if(!Number.isFinite(a)||pivot<=0||!Number.isFinite(pivot))throw Error("duplicate kernel is not numerically positive definite at this shift; no silent fallback");
 const l11=Math.sqrt(pivot),y0=1/l00,y1=(1-l10*y0)/l11,x1=y1/l11,x0=(y0-l10*x1)/l00;
 const mean=x0+x1,latent=1-mean;
 return {solution:[x0,x1],mean,latent_variance:latent,noisy_variance:latent+shift,logdet:2*(Math.log(l00)+Math.log(l11)),residual:Math.max(Math.abs(a*x0+x1-1),Math.abs(x0+a*x1-1))};
}
"""


def workbench(r):
    fields="".join(f'<label>{k}<input id="{k}" type="number" step="1" value="{v}" required></label>'
                   for k,v in r["default_budget"]["inputs"].items())
    body=f"""<h1>A diagonal shift changes the duplicated-observation kernel</h1>
    <p>Round 123 · Original fixed K=[[1,1],[1,1]], y=[1,1], test covariance k=[1,1], self covariance=1.
    No native linear_operator/GPU rescue reproduction.</p>
    <form id="form"><label>Selected observation-noise variance (diagonal shift)<input id="shift" type="number" value="0.1" step="any" required></label>
    <div class="grid">{fields}</div><button>Inspect factor, solve and variance</button></form>
    <p id="error" role="alert"></p><pre id="result" aria-live="polite"></pre>
    <p>Shift here is declared observation-noise variance, not an automatically selected numerical rescue.
    Native rescue jitter is not automatically observation noise.
    Matrix/factor budgets are separate logical payloads, not summed peak HBM.</p>"""
    script=f"const DEFAULT={json.dumps(r['default_budget']['inputs'])};\n"+JS+r"""
function update(event){
 if(event)event.preventDefault();
 const error=document.getElementById("error"),out=document.getElementById("result");error.textContent="";out.textContent="";
 try{
  if(!document.getElementById("form").checkValidity())throw Error("Complete all required controls");
  const p=Object.fromEntries(Object.keys(DEFAULT).map(k=>[k,Number(document.getElementById(k).value)]));
  out.textContent=JSON.stringify({result:calculate(Number(document.getElementById("shift").value)),budget:estimate(p),
 note:"Native rescue policy, dtype, warnings and tracing paths not reproduced. Latent variance differs from noisy observation variance; changing matrix invalidates cached factor."},null,2);
 }catch(e){error.textContent=e.message}
}
document.getElementById("form").addEventListener("submit",update);update();
"""
    return page("Kernel Cholesky shift and factor workbench",body,script)


def cards(r):
    c=r["case"]
    latent=float(Fraction(c["rational_latent_variance"]));double=float(Fraction(c["double_shift_latent_variance"]))
    maximum=max(v["refactor_median_ms"] for v in r["latency"]["rows"])
    charts="".join(f"<div class='metric'><b>Dimension {v['dimension']} · {v['right_hand_sides']} right-hand sides</b>"
                   f"<p>Cached factor solves {v['cached_median_ms']:.6f} ms</p>{bar(v['cached_median_ms'],maximum)}"
                   f"<p>Refactor each RHS {v['refactor_median_ms']:.6f} ms</p>{bar(v['refactor_median_ms'],maximum,True)}</div>"
                   for v in r["latency"]["rows"])
    sections=[
        ("HANDS-ON · KERNEL FACTORIZATION","Duplicated observations are singular without a shift; the shift changes predictions",
         "<div class='grid'><div><h3>Scenarios</h3><p>Positive definite gate<br>Factor cache<br>Latent/noisy variance</p></div>"
         "<div><h3>Structured input</h3><p>K=[[1,1],[1,1]]<br>y=[1,1]<br>shift=1/10</p></div>"
         "<div><h3>Top decisions</h3><p>Declare diagonal shift<br>Invalidate stale factor</p><span class='pill'>Original rational/CPU model</span></div></div>"
         f"<div class='dark'>Original CPU records: Python {r['latency']['python']}, {r['latency']['machine']}. Existing linear_operator rescue source inspected; March API migration is not a new factorization algorithm. Synthetic, independently_reproduced=false; no native rescue/GPU test.</div>"),
        ("SCENARIO: DIAGONAL SHIFT",f"Shift 1/10 gives latent variance {c['rational_latent_variance']}; doubling it gives {c['double_shift_latent_variance']}",
         f"<blockquote>Mean {c['rational_mean']}; noisy observation variance {c['rational_noisy_variance']}. These are different synthetic quantities, not calibrated model quality.</blockquote>"
         f"<div class='metric'><span class='pill'>score · original variance</span><h3>Single shift: {latent:.6f}</h3>{bar(latent,double)}</div>"
         f"<div class='metric'><span class='pill'>score · changed model</span><h3>Double shift: {double:.6f}</h3>{bar(double,double,True)}</div>"
         "<p>All values from independent rational elimination and the sealed model. No confidence or accuracy interpretation.</p>"),
        ("WHAT I TESTED","Factor validity, matrix identity and variance meaning are separate contracts",
         "<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>"
         "<tr><td>Unshifted duplicate K</td><td>Strict Cholesky possible?</td><td>✕ Singular</td><td>Explicit error</td></tr>"
         f"<tr><td>Shift 1/10</td><td>Rational solve</td><td>{c['rational_solution']}</td><td>Act automatically</td></tr>"
         "<tr><td>Matrix changes</td><td>Keep cached factor?</td><td>✕ Stale system</td><td>Invalidate</td></tr>"
         "<tr><td>Latent/noisy variance</td><td>Same output?</td><td>No; selected noise added</td><td>Confirm meaning</td></tr>"
         "<tr><td>Native rescue</td><td>Automatic free fix?</td><td>Policy/warnings matter</td><td>Inspect actual source</td></tr></table>"
         "<div class='grid'><div class='metric'>Gate: original model checks full symmetry; native trusted-triangle behavior is not reproduced.</div><div class='metric'>Failure analysis: numerical rescue is not permission to ignore its effect on the matrix.</div></div>"),
        ("UNDER THE HOOD","Reuse triangular solves only while the effective covariance matrix is unchanged",
         "<pre>original scalar path:\n  A = K + explicit_shift*I\n  A = L*Lᵀ\n  solve L*z = y\n  solve Lᵀ*x = z\n\nmean = kᵀ*x\nlatent variance = k_self - kᵀ*A⁻¹*k\nnoisy variance = latent + selected noise</pre>"
         "<p>↓ Independent original oracle</p><pre>exact Fraction Gaussian elimination\ncompare all solve coordinates and residual\nlogdet = 2*sum(log(diag(L)))\n\nnative rescue increments shift by new-old\nnot blindly adding each full retry value</pre>"),
        ("ROBUSTNESS CHECK","After changing the shift, a stale factor leaves a nonzero system residual",
         "<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE (PTS)</th></tr>"
         f"<tr><td>Fresh solve residual</td><td>{c['residual']:.3g}</td><td>{c['fresh_factor_shifted_matrix_residual']:.3g}</td><td>Numerical tolerance</td></tr>"
         f"<tr><td>✕ Reuse old factor</td><td>{c['residual']:.3g}</td><td>{c['stale_factor_shifted_matrix_residual']:.9f}</td><td>Matrix contract changed</td></tr></table>"
         "<p>These are residual values, not probability points. Cache identity includes effective matrix/data/kernel/noise configuration; source warning/tracing/rescue behavior not executed.</p>"),
        ("LATENCY","Original cached-factor solves avoid repeated scalar factorizations across eight right-hand sides",
         charts+"<p>Cached setup excluded; baseline rebuilds same factor per RHS. Two warmups, seven alternating repeats, three operations/sample; median repeat means. Kernel construction/fitting, initial cache setup, native library and GPU excluded.</p>"),
        ("MY TAKE","Keep numerical rescue visible and distinguish latent uncertainty from observation noise",
         "<div class='grid'><div class='good'><h3>✅ Good fit</h3><p>Small dense kernel audits, explicit shifts, factor reuse and independent solve/variance verification.</p></div>"
         "<div class='bad'><h3>⛔ Not for</h3><p>Native rescue guarantees, indefinite covariance acceptance, stale-cache reuse, calibrated quality or GPU throughput.</p></div></div>"
         "<div class='dark'>MIT pinned linear_operator source; original model/oracle/art. Native rescue/latency, posterior model quality and peak HBM not reproduced. Browser inspection not performed.</div>"),
    ]
    return page("Kernel Cholesky contracts — seven cards","".join(
        f'<section class="card"><p class="eyebrow">{i} / 7 · {label}</p><h1>{title}</h1>{body}</section>'
        for i,(label,title,body) in enumerate(sections,1)))


def figure():
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000">
    <title>Kernel-factor ticket cabinet: factor identity follows the effective covariance matrix</title>
    <desc>Figure-tier original 60-solid magazine sorter maps matrix/factor tickets and distinguishes latent/noisy variance drawers.</desc>
    <rect width="1600" height="1000" fill="#f1f5f9"/><rect x="48" y="48" width="1504" height="904" rx="28" fill="white" stroke="#cbd5e1"/>
    <g fill="#0f172a" font-family="system-ui,sans-serif"><text x="88" y="104" font-size="18">FIG 123 · KERNEL CHOLESKY CONTRACTS</text>
    <text x="88" y="158" font-size="36" font-weight="700">A diagonal shift changes the matrix and invalidates its cache</text>{sorter_scene()}
    <text x="88" y="710" font-size="25">Positive semidefinite duplicate kernels need a declared shift for strict factorization</text>
    <text x="88" y="758" font-size="25">Reuse the factor only for the same effective covariance matrix</text>
    <text x="88" y="806" font-size="25">Latent variance is not the variance of a noisy observation</text>
    <text x="88" y="880" font-size="18">Original rational / scalar model · no native jitter rescue / GPU / calibrated model-quality reproduction</text></g></svg>\n"""


def outputs(r):
    return {ROOT/f"docs/reports/{SLUG}-selection-cards.html":cards(r),
            ROOT/f"docs/reports/{SLUG}-sizer.html":workbench(r),
            ROOT/f"assets/svg/{SLUG}-selection-faq-summary.svg":figure()}
