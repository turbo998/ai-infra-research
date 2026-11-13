"""Original fixed-Euler gradients, block replay and continuous-adjoint controls."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import time


HERE = Path(__file__).resolve().parent
SOURCE = "https://github.com/patrick-kidger/diffrax/blob/6321f02fbd6d65a95f745ed9a05245394023fb08/diffrax/_adjoint.py"
MAX_SAFE = 9007199254740991


@dataclass(frozen=True)
class Problem:
    theta: float = 0.7
    initial: float = 1.0
    target: float = 2.0
    duration: float = 1.0
    steps: int = 4


def validate(p: Problem):
    if type(p.steps) is not int or not 1 <= p.steps <= 100000:
        raise ValueError("fixed-step reference needs 1..100000 steps")
    if any(type(value) not in (int,float) or not math.isfinite(value)
           for value in (p.theta,p.initial,p.target,p.duration)) or p.duration <= 0:
        raise ValueError("finite scalar parameters and positive duration required")
    if not math.isfinite(p.theta*p.duration) or abs(p.theta*p.duration) > 100:
        raise ValueError("toy drift-duration product must stay within +/-100")


def step(y: float,p: Problem) -> float:
    value = y+(p.duration/p.steps)*p.theta*y
    if not math.isfinite(value):
        raise ValueError("Euler state exceeds finite numerical range")
    return value


def forward(p: Problem) -> list:
    validate(p)
    states = [p.initial]
    for _ in range(p.steps):
        states.append(step(states[-1],p))
    return states


def loss(y: float,p: Problem) -> float:
    value = 0.5*(y-p.target)**2
    if not math.isfinite(value):
        raise ValueError("loss exceeds finite numerical range")
    return value


def discrete_reverse(p: Problem) -> dict:
    states = forward(p)
    h = p.duration/p.steps
    adjoint = states[-1]-p.target
    gradient = 0.0
    for i in range(p.steps-1,-1,-1):
        gradient += adjoint*h*states[i]
        adjoint *= 1+h*p.theta
    return {"terminal":states[-1],"loss":loss(states[-1],p),"theta_gradient":gradient,
            "initial_gradient":adjoint,"stored_state_slots":len(states),
            "forward_evaluations":p.steps,"replay_evaluations":0}


def checkpoint_reverse(p: Problem,block: int) -> dict:
    validate(p)
    if type(block) is not int or not 1 <= block <= p.steps:
        raise ValueError("block must be an integer within the step count")
    checkpoints = {0:p.initial}
    value = p.initial
    for i in range(1,p.steps+1):
        value = step(value,p)
        if i%block == 0 or i == p.steps:
            checkpoints[i] = value
    terminal = value
    adjoint,gradient = terminal-p.target,0.0
    h,replays,maximum = p.duration/p.steps,0,0
    boundaries = list(checkpoints)
    for start,end in reversed(list(zip(boundaries,boundaries[1:]))):
        local = [checkpoints[start]]
        for _ in range(end-start):
            local.append(step(local[-1],p))
            replays += 1
        if local[-1] != checkpoints[end]:
            raise ValueError("deterministic replay endpoint differs from its checkpoint")
        maximum = max(maximum,len(local))
        for i in range(len(local)-2,-1,-1):
            gradient += adjoint*h*local[i]
            adjoint *= 1+h*p.theta
    return {"terminal":terminal,"loss":loss(terminal,p),"theta_gradient":gradient,"initial_gradient":adjoint,
            "checkpoint_slots":len(checkpoints),"maximum_local_replay_slots":maximum,
            "selected_state_slot_bound":len(checkpoints)+maximum,
            "forward_evaluations":p.steps,"replay_evaluations":replays}


def continuous_exact(p: Problem) -> dict:
    validate(p)
    terminal = p.initial*math.exp(p.theta*p.duration)
    residual = terminal-p.target
    return {"terminal":terminal,"loss":loss(terminal,p),
            "theta_gradient":residual*p.duration*terminal,
            "initial_gradient":residual*math.exp(p.theta*p.duration)}


def continuous_backsolve_euler(p: Problem,reset_stride: int | None = None) -> dict:
    """Explicit Euler on the continuous augmented ODE, stepping backwards."""
    states = forward(p)
    if reset_stride is not None and (type(reset_stride) is not int or not 1 <= reset_stride <= p.steps):
        raise ValueError("observation reset stride must fit the step count")
    h = p.duration/p.steps
    y,adjoint,gradient = states[-1],states[-1]-p.target,0.0
    maximum,initial_before_reset,resets = 0.0,y,0
    for i in range(p.steps,0,-1):
        gradient += h*adjoint*y
        y,adjoint = y-h*p.theta*y,adjoint+h*p.theta*adjoint
        maximum = max(maximum,abs(y-states[i-1]))
        initial_before_reset = y
        if reset_stride is not None and (i-1)%reset_stride == 0:
            y = states[i-1]
            resets += 1
    return {"theta_gradient":gradient,"initial_gradient":adjoint,
            "reconstructed_initial_before_final_reset":initial_before_reset,
            "maximum_state_difference_before_reset":maximum,"observation_resets":resets,
            "scope":"original explicit Euler stepping backward in time on continuous adjoint; not discrete reverse or library execution"}


def finite_difference(p: Problem,field: str) -> float:
    from dataclasses import replace
    if field not in ("theta","initial"):
        raise ValueError("finite difference supports theta and initial only")
    epsilon = 1e-5
    plus = replace(p,**{field:getattr(p,field)+epsilon})
    minus = replace(p,**{field:getattr(p,field)-epsilon})
    return (loss(forward(plus)[-1],plus)-loss(forward(minus)[-1],minus))/(2*epsilon)


@dataclass(frozen=True)
class Budget:
    steps: int = 1000
    block: int = 32
    state_elements: int = 1048576
    element_bytes: int = 4
    observation_times: int = 2
    parameter_elements: int = 1048576


def estimate(b: Budget) -> dict:
    for name,value in asdict(b).items():
        if type(value) is not int or not 1 <= value <= MAX_SAFE:
            raise ValueError(f"{name} must be a positive safe integer")
    if b.block > b.steps or b.element_bytes not in (2,4,8) or b.observation_times < 2:
        raise ValueError("block must fit steps, bytes 2/4/8, and at least two observation times")
    checkpoints = (b.steps+b.block-1)//b.block+1
    replay_slots = b.block+1
    one_state = b.state_elements*b.element_bytes
    values = {
        "full_tape_state_bytes":(b.steps+1)*one_state,
        "block_checkpoint_slots":checkpoints,"maximum_replay_slots":replay_slots,
        "block_selected_state_bytes":(checkpoints+replay_slots)*one_state,
        "saved_observation_state_bytes":b.observation_times*one_state,
        "augmented_y_adjoint_parameter_bytes":(2*b.state_elements+b.parameter_elements)*b.element_bytes,
        "block_extra_forward_evaluations":b.steps,
    }
    if any(value > MAX_SAFE for value in values.values()):
        raise ValueError("logical budget exceeds exact numerical range")
    return {"input":asdict(b),**values,"peak_memory_bytes":None,"gpu_latency_ms":None,
            "scope":"fixed block checkpointing, not Diffrax binomial schedule; logical selected buffers only"}


def measure_latency():
    from dataclasses import replace
    rows = []
    for count in (32,128,512,2048):
        p = replace(Problem(),steps=count)
        block = math.ceil(math.sqrt(count))
        functions = {"full_tape":lambda:discrete_reverse(p),
                     "block_replay":lambda:checkpoint_reverse(p,block)}
        if abs(functions["full_tape"]()["theta_gradient"]-functions["block_replay"]()["theta_gradient"]) > 1e-12:
            raise ValueError("timed gradients disagree")
        for function in functions.values():
            for _ in range(2):
                function()
        samples = {key:[] for key in functions}
        for repeat in range(7):
            for key in list(functions) if repeat%2 == 0 else list(reversed(functions)):
                start = time.perf_counter_ns()
                functions[key]()
                samples[key].append(time.perf_counter_ns()-start)
        rows.append({"steps":count,"block":block,"samples_ns":samples,
                     **{key+"_median_ms":statistics.median(values)/1e6 for key,values in samples.items()}})
    data = {"provenance":"measured","measured_at_utc":datetime.now(timezone.utc).isoformat(),
            "environment":{"os":platform.system(),"machine":platform.machine(),
                           "processor":platform.processor(),"python":platform.python_version()},
            "scope":"original scalar fixed-Euler forward plus reverse, including validation; not ODE library/GPU runtime",
            "timer":"perf_counter_ns","warmups_per_path":2,"repeats_per_path":7,
            "order":"alternating function order by repeat","rows":rows}
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8",newline="\n")


def load_latency() -> dict:
    path = HERE/"latency.json"
    if not path.exists():
        raise ValueError("missing sealed timing file; explicitly run --measure")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["provenance"] != "measured" or data["repeats_per_path"] != 7:
        raise ValueError("invalid measured timing contract")
    if [row["steps"] for row in data["rows"]] != [32,128,512,2048]:
        raise ValueError("sealed timing tiers do not match")
    for row in data["rows"]:
        for key in ("full_tape","block_replay"):
            samples = row["samples_ns"][key]
            if len(samples) != 7 or any(type(n) is not int or n <= 0 for n in samples):
                raise ValueError("samples must be seven positive nanosecond integers")
            if row[key+"_median_ms"] != statistics.median(samples)/1e6:
                raise ValueError("sealed median differs from samples")
    return data


def simulate() -> dict:
    from dataclasses import replace
    p = Problem()
    full,checkpoint,continuous = discrete_reverse(p),checkpoint_reverse(p,2),continuous_exact(p)
    backward,reset = continuous_backsolve_euler(p),continuous_backsolve_euler(p,1)
    fd = {field:finite_difference(p,field) for field in ("theta","initial")}
    cases = []
    for name,result,objective_change in (
        ("full-discrete-reverse",full,False),("block-checkpoint-replay",checkpoint,False),
        ("analytic-continuous-objective",continuous,True),
        ("continuous-backsolve-coarse-euler",backward,False),
        ("continuous-backsolve-reset-every-step",reset,False),
    ):
        error = abs(result["theta_gradient"]-full["theta_gradient"])
        cases.append({"id":name,"theta_gradient":result["theta_gradient"],
                      "difference_from_discrete_gradient":error,"matches_discrete_reference":error < 1e-12,
                      "different_continuous_objective":objective_change,"confidence":None})
    convergence = []
    for count in (4,16,64,256):
        discrete = discrete_reverse(replace(p,steps=count))
        convergence.append({"steps":count,"discrete_gradient":discrete["theta_gradient"],
                            "continuous_gradient":continuous["theta_gradient"],
                            "absolute_gradient_difference":abs(discrete["theta_gradient"]-continuous["theta_gradient"]),
                            "terminal_difference":abs(discrete["terminal"]-continuous["terminal"])})
    return {
        "schema_version":1,"round":103,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,"problem":asdict(p),
        "full_tape":full,"checkpoint":checkpoint,"continuous":continuous,
        "finite_differences":fd,
        "finite_difference_error":max(abs(fd[field]-full[field+"_gradient"]) for field in fd),
        "backsolve":backward,"backsolve_reset_each_step":reset,
        "continuous_backsolve_exact_from_numerical_terminal":{
            "theta_gradient":(full["terminal"]-p.target)*full["terminal"]*p.duration,
            "reconstructed_initial":full["terminal"]*math.exp(-p.theta*p.duration),
            "scope":"exact continuous augmented flow with numerical terminal, not the true-forward continuous objective"},
        "cases":cases,"convergence":convergence,
        "replay_blocks":[{"block":block,**checkpoint_reverse(p,block)} for block in (1,2,3,4)],
        "budget":estimate(Budget()),"latency":load_latency(),"scientific_quality":None,
        "notes":[
            "Original fixed explicit Euler only; no adaptive solver, event, stochastic vector field or framework execution.",
            "Continuous objective and discrete numerical objective differ; an opposite gradient sign need not be a coding bug.",
            "Observation-state resets improve state reconstruction but do not turn a continuous adjoint discretization into discrete reverse.",
            "Block replay preserves identical forward operations; no stochastic/nondeterministic replay is modeled.",
            "Logical saved observations and augmented adjoint/parameter states prevent an unqualified constant-total-memory claim.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {HERE/"report.json":json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--measure",action="store_true")
    args = parser.parse_args()
    if args.check and args.measure:
        parser.error("--check cannot replace sealed measurements")
    if args.measure:
        measure_latency()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 103 artifacts match." if args.check else "Wrote Round 103 artifacts.")


if __name__ == "__main__":
    main()
