"""Self-contained gradient-contract evidence and selected-buffer calculator."""

import json
from pathlib import Path

from experiments.contrastive_gradcache_selection.render import sorter_scene
from experiments.ddp_gradient_compression_selection.render import bar, page


ROOT = Path(__file__).resolve().parents[2]
SLUG = "neural-ode-discrete-continuous-adjoint-checkpoint"


def deck(r: dict) -> str:
    cards = []

    def card(eyebrow,title,body,note):
        n = len(cards)+1
        cards.append(f'<section class="card" aria-label="Card {n} of 7"><div class="eyebrow">{eyebrow}</div>'
                     f'<h1>{title}</h1><p class="muted">Synthetic scalar ODE · independently_reproduced=false · '
                     'Original CPU timings separately measured; no framework or GPU execution.</p>'
                     f'<div class="evidence">{body}</div><div class="callout">{note}</div>'
                     f'<footer><span>Round 103 · Gradient target before memory</span><span>{n} / 7</span></footer></section>')

    full,continuous = r["full_tape"],r["continuous"]
    card("HANDS-ON · NEURAL ODE",
         'Continuous and four-step <span class="accent">Euler objectives</span> give opposite gradient signs.',
         '<div class="grid"><div class="panel"><h2>Fixture</h2>'
         '<p>dy/dt = .7y<br>y(0)=1, T=1<br>Target=2, four steps</p>'
         '<span class="pill">Discrete gate: &lt;1e-8</span></div><div class="panel"><h2>Contract</h2>'
         '<pre>y_next = y + h*theta*y\nloss = .5*(y_T-2)^2</pre>'
         '<p>Differentiate numerical steps or the exact continuous flow?</p></div>'
         '<div class="panel"><h2>Signs differ</h2>'
         f'<p>Discrete: {full["theta_gradient"]:+.6f}</p>'
         f'<p>Continuous: {continuous["theta_gradient"]:+.6f}</p>'
         '<span class="pill">Model quality: null</span></div></div>',
         '<b>Established source distinction.</b> Diffrax prefers discrete recursive checkpointing for '
         'most problems; its continuous backsolve warning concerns approximate gradients, not a local library result.')
    scores = '<blockquote>Euler ends below target 2; the exact continuous solution ends above it. '
    scores += 'Opposite signs here reflect different objectives, not a fabricated failure.</blockquote>'
    maximum = max(abs(full["theta_gradient"]),abs(continuous["theta_gradient"]))
    for label,result in (("Numerical Euler objective",full),("Exact continuous objective",continuous)):
        scores += '<div class="panel"><span class="pill">score · gradient magnitude, not probability</span>'
        scores += f'<span class="pill">Confirm objective</span><h2>{label}</h2>'
        scores += f'<p>Terminal {result["terminal"]:.6f}; signed gradient {result["theta_gradient"]:+.6f}; confidence: null</p>'
        scores += bar(abs(result["theta_gradient"]),maximum)+'</div>'
    card("SCENARIO: DIFFERENT GRADIENT TARGETS",
         'The numerical endpoint is 1.906125; the continuous endpoint is 2.013753.',
         scores,'<b>Reference choice matters.</b> Finite differences of the fixed Euler computation '
         'validate its discrete reverse, not a different continuous loss.')
    names = {"full-discrete-reverse":"Discrete reverse","block-checkpoint-replay":"Block replay",
             "analytic-continuous-objective":"Exact continuous loss","continuous-backsolve-coarse-euler":"Coarse continuous backsolve",
             "continuous-backsolve-reset-every-step":"Backsolve + all state resets"}
    matrix = '<table><tr><th>SCENARIO</th><th>QUESTION/TASK</th><th>MODEL ANSWER</th><th>ACTION</th></tr>'
    for row in r["cases"]:
        interpretation = ("Different objective" if row["different_continuous_objective"] else
                          "Discrete match" if row["matches_discrete_reference"] else "Not discrete gradient ✕")
        matrix += f'<tr><td>{names[row["id"]]}</td><td>θ gradient</td><td>{row["theta_gradient"]:+.6f}<br>{interpretation}</td>'
        matrix += '<td>'+('Choose objective' if row["different_continuous_objective"] else
                         'Confirm' if row["matches_discrete_reference"] else 'Review numerics')+'</td></tr>'
    matrix += '</table><div class="grid two"><div class="callout">'
    matrix += f'<b>Independent gate:</b> θ and initial-state finite differences, max error {r["finite_difference_error"]:.2e}.</div>'
    matrix += '<div class="callout"><b>Boundary:</b> restoring observations is not discrete autodiff.</div></div>'
    card("WHAT I TESTED","Replay preserves the Euler gradient; continuous backsolve need not.",
         matrix,'<b>Controls use original explicit Euler backward in time.</b> That is not the '
         'implicit backward-Euler method, nor a reproduction of adaptive Diffrax/torchdiffeq solvers.')
    card("UNDER THE HOOD","Discrete reverse uses the pre-step state in the parameter VJP.",
         '<pre># Discrete Euler, y_next = y + h*theta*y\n'
         'theta_gradient += adjoint_next * h * y_pre_step\n'
         'adjoint_previous = adjoint_next * (1 + h*theta)\n\n'
         '# Block replay regenerates identical pre-step states.\n'
         '# Same step sequence and backward order preserve the gradient.</pre>'
         '<p>↓ a continuous augmented ODE is a different numerical object</p>'
         '<pre>forward physical time:\n'
         '    dy/dt = theta*y\n'
         '    da/dt = -theta*a\n'
         '    dg/dt = -a*y\n\n'
         'Integrate (y,a,g) backward in time.\n'
         'Use selected solver/tolerances; restore saved observations.\n'
         'Do not equate this discretization with discrete reverse.</pre>',
         '<b>Source scope:</b> torchdiffeq saves requested output states and parameters, solves augmented '
         'VJPs backward, resets y at observations, and has separate adjoint options/norm handling.')
    robust = '<table><tr><th>QUESTION/ANSWER</th><th>ORIGINAL</th><th>PERTURBED</th><th>CHANGE</th></tr>'
    for row in r["replay_blocks"]:
        robust += f'<tr><td>θ gradient</td><td>{full["theta_gradient"]:+.6f}</td><td>Block {row["block"]}</td>'
        robust += f'<td>{row["theta_gradient"]-full["theta_gradient"]:+.2e}</td></tr>'
    robust += '</table><div class="panel"><h2>Refinement, not a quality score</h2>'
    first,last = r["convergence"][0],r["convergence"][-1]
    robust += f'<p>4 → 256 Euler steps reduce the continuous-gradient gap '
    robust += f'from {first["absolute_gradient_difference"]:.6f} to {last["absolute_gradient_difference"]:.6f}.</p>'
    robust += '<p>State resets reduce reconstruction error but do not guarantee a closer discrete gradient.</p></div>'
    card("ROBUSTNESS CHECK","All four block partitions reproduce the same discrete gradient.",
         robust,'<b>Deterministic replay only.</b> Adaptive decisions, events, stochastic vector fields '
         'and nondeterministic kernels are excluded. Fixed-block replay is not Diffrax binomial checkpointing.')
    latency = r["latency"]
    maximum = max(max(row["full_tape_median_ms"],row["block_replay_median_ms"]) for row in latency["rows"])
    timing = '<p>Measured original CPU medians: blue block replay, gray full state tape.</p>'
    for row in latency["rows"]:
        timing += f'<div class="latency-row"><b>{row["steps"]} steps · block {row["block"]}</b>'
        for key,label,baseline in (("block_replay_median_ms","Block replay",False),
                                   ("full_tape_median_ms","Full tape",True)):
            timing += f'<div class="latency-pair"><span>{label} <b>{row[key]:.4f} ms</b></span>'
            timing += bar(row[key],maximum,baseline)+'</div>'
        timing += '</div>'
    card("LATENCY","Replay is slower in all four measured scalar cases; it trades work for storage.",
         timing,'<b>Windows ARM64, Python '+latency["environment"]["python"]+'.</b> Two warmups, seven '
         'alternating repeats, forward plus reverse and validation. Not adaptive-library/GPU latency '
         'or a timing extrapolation to the calculator state dimension.')
    card("MY TAKE","Low internal-step storage does not mean constant total memory.",
         '<div class="grid two"><div class="panel good"><h2>✅ Good fit</h2>'
         '<p>Choose the gradient objective.</p><p>Finite-difference discrete computations.</p>'
         '<p>Trade deterministic replay for state storage.</p></div>'
         '<div class="panel bad"><h2>⛔ Not for</h2><p>Backsolve called exact discrete autodiff.</p>'
         '<p>Erasing observation/parameter buffers.</p><p>Toy scalar timing called GPU speed.</p></div></div>'
         '<pre>READINESS: fixed-Euler arithmetic and source review\n'
         'DIFFRAX: Apache-2.0; discrete checkpoint default\n'
         'TORCHDIFFEQ: MIT; saved observations + augmented VJPs\n'
         'CPU TIMING: original deterministic scalar functions\n'
         'ADAPTIVE SOLVERS / EVENTS / SCIENTIFIC QUALITY: not tested\n'
         'PROVENANCE: synthetic inputs, independently_reproduced=false</pre>',
         '<b>Selected buffers only.</b> The calculator includes observations and augmented state but '
         'excludes solver stages, model activations, optimizer state and actual peak allocation.')
    styles = ('<style>.latency-row{padding:14px 0;border-bottom:1px solid var(--cp-border)}'
              '.latency-pair{display:grid;grid-template-columns:230px 1fr;gap:14px;align-items:center}'
              '.latency-pair .bar{margin:8px 0}table{font-size:14px}th,td{padding:10px 6px}</style>')
    return page("Round 103 · Neural ODE gradient contracts · seven cards",styles+'<main>'+"".join(cards)+'</main>')


CALCULATOR = r"""
function estimateBudget(p){
  for(const key of ["steps","block","state_elements","element_bytes","observation_times","parameter_elements"])
    if(!Number.isSafeInteger(p[key])||p[key]<1)throw Error(key+" must be a positive safe integer");
  if(p.block>p.steps||![2,4,8].includes(p.element_bytes)||p.observation_times<2)
    throw Error("Block must fit steps, bytes 2/4/8 and at least two observations");
  const checkpoints=Number((BigInt(p.steps)+BigInt(p.block)-1n)/BigInt(p.block)+1n);
  const replay=p.block+1,state=p.state_elements*p.element_bytes;
  const values={full_tape_state_bytes:(p.steps+1)*state,block_checkpoint_slots:checkpoints,
    maximum_replay_slots:replay,block_selected_state_bytes:(checkpoints+replay)*state,
    saved_observation_state_bytes:p.observation_times*state,
    augmented_y_adjoint_parameter_bytes:(2*p.state_elements+p.parameter_elements)*p.element_bytes,
    block_extra_forward_evaluations:p.steps};
  if(!Object.values(values).every(Number.isSafeInteger))throw Error("Logical budget exceeds exact numerical range");
  return {...values,peak_memory_bytes:null,gpu_latency_ms:null};
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
    result.textContent=JSON.stringify({provenance:"synthetic selected-buffer formulas",
      ...estimateBudget(p),production_approved:false,
      note:"Fixed-block replay, not binomial checkpointing. Saved observations and parameter adjoints are explicit. Selected buffer bounds do not certify peak allocation."},null,2);
  }catch(e){error.textContent=e.message;result.textContent="No estimate: invalid input.";}
}
update();
"""


def workbench(r: dict) -> str:
    p = r["budget"]["input"]
    labels = {"steps":"Fixed forward steps / 固定前向步数","block":"Replay block / 重放块大小",
              "state_elements":"State elements / 状态元素数","element_bytes":"Bytes (2/4/8) / 元素字节",
              "observation_times":"Saved observation times / 保存观测时刻",
              "parameter_elements":"Parameter adjoint elements / 参数伴随元素数"}
    controls = "".join(f'<label for="{key}">{label}<input required type="number" id="{key}" name="{key}" '
                       f'min="1" step="1" value="{p[key]}"></label>' for key,label in labels.items())
    return page("Neural ODE / 梯度目标与 Checkpoint 工作台",
                '<main><div class="eyebrow">ROUND 103 · FIXED-EULER GRADIENTS · SYNTHETIC BUFFERS</div>'
                '<h1>Choose the gradient target before the memory strategy.<br>先确认梯度目标，再选省内存策略。</h1>'
                '<div class="panel">四步 Euler 的 θ 梯度为 -0.152287，精确连续目标为 +0.027695。'
                '它们回答不同问题；deterministic block replay 保持离散目标，而 coarse continuous backsolve 不必相同。</div>'
                f'<form id="calculator"><div class="grid">{controls}</div>'
                '<p><button type="submit">Recalculate / 重新计算</button></p></form>'
                '<p id="error" class="error" role="alert"></p>'
                '<output id="result" aria-live="polite" style="white-space:pre-wrap;font-family:Consolas,monospace"></output>'
                '<noscript>Inline JavaScript is required; no estimate without it.</noscript>'
                '<div class="panel"><b>边界：</b>full tape 仅 Euler 状态，不是框架完整 Autograd tape。'
                '固定 block 的 checkpoint 与 local replay payload 含边界重复计数上界；不模拟 Diffrax online/binomial schedule。'
                '<p>Observation 与 augmented buffers 分项显示，不能随意相加成 peak HBM；'
                '排除 solver stages、网络激活、参数存储和优化器。GPU latency 未测。</p></div></main>',
                "const BASE="+json.dumps(p)+";\n"+CALCULATOR)


def summary_svg(r: dict) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1000" viewBox="0 0 1600 1000" '
        'role="img" aria-labelledby="title description"><title id="title">ODE checkpoint trajectory sorter</title>'
        '<desc id="description">A solid card sorter stores checkpoints and replays local trajectory cards. '
        'The gradient objective is selected separately from the storage strategy.</desc>'
        '<rect width="1600" height="1000" rx="28" fill="#f7f4ef"/>'
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#242424">'
        '<text x="64" y="65" font-size="21" letter-spacing="3">ROUND 103 · SYNTHETIC ODE · DISCRETE VERSUS CONTINUOUS</text>'
        '<text x="64" y="130" font-size="44" font-weight="700">A smaller tape does not define the gradient target.</text>'
        '<text x="64" y="180" font-size="24">Replay the same numerical steps for discrete reverse; continuous backsolve is a different route.</text>'
        +sorter_scene()
        +f'<text x="80" y="650" font-size="26" font-weight="700">Four-step Euler gradient {r["full_tape"]["theta_gradient"]:+.6f}; '
        +f'exact continuous gradient {r["continuous"]["theta_gradient"]:+.6f}.</text>'
        +'<text x="80" y="700" font-size="25">Saved-state resets improve reconstruction, not equivalence to discrete differentiation.</text>'
        +'<rect x="64" y="752" width="1472" height="154" rx="18" fill="#0F172A"/>'
        +'<g fill="#ffffff"><text x="96" y="799" font-size="23">All four block partitions preserve the original fixed-step gradient.</text>'
        +'<text x="96" y="842" font-size="23">Observation states and parameter adjoints remain real logical buffers.</text>'
        +'<text x="96" y="883" font-size="21">Original scalar CPU timing only; no adaptive solver, scientific quality or peak GPU memory claim.</text></g>'
        +'<text x="64" y="955" font-size="20">Data: experiments/neural_ode_adjoint_selection/report.json · independently_reproduced=false</text>'
        +'</g></svg>\n'
    )


def render_outputs(r: dict) -> dict[Path,str]:
    return {ROOT/"docs"/"reports"/f"{SLUG}-selection-cards.html":deck(r),
            ROOT/"docs"/"reports"/f"{SLUG}-sizer.html":workbench(r),
            ROOT/"assets"/"svg"/f"{SLUG}-selection-faq-summary.svg":summary_svg(r)}
