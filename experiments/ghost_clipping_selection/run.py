"""Original per-example clipping and sequence ghost-norm correctness model."""

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


PARAMETERS = [0.2,-0.1,0.3,0.4,0.1,-0.2]
ACTIVATIONS = [
    [[1.0,0.0],[1.0,0.0]],
    [[1.0,0.0],[1.0,0.0]],
    [[1.0,2.0],[-1.0,0.5]],
]
DESIRED_BACKPROPS = [[[2.0,0.0],[2.0,0.0]],[[3.0,0.0],[-3.0,0.0]],[[0.5,-1.0],[1.0,0.25]]]
SOURCE = "https://github.com/pytorch/opacus/blob/a196a93a8326089242d738e7110fbb625608f114/opacus/grad_sample/linear.py"
EPSILON = 1e-6


def dot(a: list, b: list) -> float:
    if len(a) != len(b):
        raise ValueError("dot product dimensions must match")
    return math.fsum(x*y for x,y in zip(a,b))


def norm(values: list) -> float:
    return math.sqrt(dot(values,values))


def predict(parameters: list, activation: list) -> list:
    if len(parameters) != 6 or len(activation) != 2:
        raise ValueError("toy linear model needs six parameters and two input features")
    if any(not math.isfinite(v) for v in parameters+activation):
        raise ValueError("parameters and activations must be finite")
    return [dot(parameters[2*o:2*o+2],activation)+parameters[4+o] for o in range(2)]


TARGETS = [[
    [value-delta for value,delta in zip(predict(PARAMETERS,a),d)]
    for a,d in zip(activations,backprops)]
    for activations,backprops in zip(ACTIVATIONS,DESIRED_BACKPROPS)]


def example(parameters: list, activations: list, targets: list) -> dict:
    if not activations or len(activations) != len(targets):
        raise ValueError("an example needs matching nonempty token activations and targets")
    if any(len(row) != 2 or any(not math.isfinite(v) for v in row) for row in targets):
        raise ValueError("targets must be finite two-dimensional vectors")
    deltas = [[p-y for p,y in zip(predict(parameters,a),target)]
              for a,target in zip(activations,targets)]
    gradient = [math.fsum(d[o]*a[i] for a,d in zip(activations,deltas))
                for o in range(2) for i in range(2)]
    gradient += [math.fsum(d[o] for d in deltas) for o in range(2)]
    return {"loss":0.5*math.fsum(v*v for d in deltas for v in d),
            "gradient":gradient,"backprops":deltas}


def ghost_norm(activations: list, backprops: list, method: str = "gram") -> float:
    if not activations or len(activations) != len(backprops):
        raise ValueError("ghost norm needs matching nonempty token sequences")
    if any(len(row) != 2 or any(not math.isfinite(v) for v in row)
           for row in activations+backprops):
        raise ValueError("ghost norm needs finite two-dimensional token vectors")
    bias_squared = math.fsum(math.fsum(row[o] for row in backprops)**2 for o in range(2))
    if method == "gram":
        weight_squared = math.fsum(dot(a,b)*dot(g,h)
                                  for a,g in zip(activations,backprops)
                                  for b,h in zip(activations,backprops))
    elif method == "diagonal-only":
        weight_squared = math.fsum(dot(a,a)*dot(g,g) for a,g in zip(activations,backprops))
    elif method == "frobenius-product":
        weight_squared = (math.fsum(dot(a,a) for a in activations)
                          * math.fsum(dot(g,g) for g in backprops))
    else:
        raise ValueError("unknown ghost norm method")
    total = weight_squared+bias_squared
    if total < -1e-10:
        raise ValueError("computed squared norm is materially negative")
    return math.sqrt(max(total,0.0))


def coefficient(length: float, clip: float = 2.0) -> float:
    if not math.isfinite(length) or length < 0 or not math.isfinite(clip) or clip <= 0:
        raise ValueError("norm must be finite nonnegative and clipping bound finite positive")
    return min(1.0,clip/(length+EPSILON))


def clipped_sum(gradients: list, coefficients: list) -> list:
    if not gradients or len(gradients) != len(coefficients):
        raise ValueError("gradients and coefficients must match")
    width = len(gradients[0])
    if not width or any(len(row) != width for row in gradients):
        raise ValueError("gradient widths must match")
    return [math.fsum(c*g[k] for c,g in zip(coefficients,gradients)) for k in range(width)]


def batch(parameters: list = PARAMETERS) -> list:
    return [example(parameters,a,y) for a,y in zip(ACTIVATIONS,TARGETS)]


def weighted_loss(parameters: list, coefficients: list) -> float:
    rows = batch(parameters)
    if len(coefficients) != len(rows):
        raise ValueError("weighted replay needs one stopped coefficient per example")
    return math.fsum(c*row["loss"] for c,row in zip(coefficients,rows))


def finite_difference(function, parameters: list = PARAMETERS) -> list:
    step = 1e-5
    gradient = []
    for k in range(len(parameters)):
        plus,minus = parameters[:],parameters[:]
        plus[k] += step
        minus[k] -= step
        gradient.append((function(plus)-function(minus))/(2*step))
    return gradient


def partition_sum(gradients: list, sizes: list[int], clip: float = 2.0) -> list:
    if any(type(size) is not int or size < 1 for size in sizes) or sum(sizes) != len(gradients):
        raise ValueError("physical partitions must cover all examples exactly")
    partials = []
    index = 0
    for size in sizes:
        chunk = gradients[index:index+size]
        partials.append(clipped_sum(chunk,[coefficient(norm(g),clip) for g in chunk]))
        index += size
    return [math.fsum(row[k] for row in partials) for k in range(len(gradients[0]))]


@dataclass(frozen=True)
class Budget:
    logical_examples: int = 512
    physical_examples: int = 64
    tokens: int = 128
    input_dimension: int = 4096
    output_dimension: int = 4096
    scalar_bytes: int = 4
    clip: float = 2.0
    noise_multiplier: float = 1.0
    expected_batch_size: int = 512
    assumed_read_gbps: float = 1000.0
    setup_us: float = 5.0


def estimate(b: Budget) -> dict:
    names = ("logical_examples","physical_examples","tokens","input_dimension",
             "output_dimension","scalar_bytes","expected_batch_size")
    if any(type(getattr(b,n)) is not int or getattr(b,n) < 1 for n in names):
        raise ValueError("example counts and dimensions must be positive integers")
    if b.physical_examples > b.logical_examples or b.scalar_bytes not in (4,8):
        raise ValueError("physical examples must fit the logical batch; scalar bytes must be 4 or 8")
    if not math.isfinite(b.clip) or b.clip <= 0:
        raise ValueError("clip must be finite positive")
    if not math.isfinite(b.noise_multiplier) or b.noise_multiplier < 0:
        raise ValueError("noise multiplier must be finite nonnegative")
    if not math.isfinite(b.assumed_read_gbps) or b.assumed_read_gbps <= 0:
        raise ValueError("assumed bandwidth must be finite positive")
    if not math.isfinite(b.setup_us) or b.setup_us < 0:
        raise ValueError("setup must be finite nonnegative")
    dense = b.physical_examples*b.input_dimension*b.output_dimension*b.scalar_bytes
    gram = 2*b.physical_examples*b.tokens*b.tokens*b.scalar_bytes
    activation = b.physical_examples*b.tokens*(b.input_dimension+b.output_dimension)*b.scalar_bytes
    chunks = (b.logical_examples+b.physical_examples-1)//b.physical_examples
    noise_scale = b.noise_multiplier*b.clip
    std = noise_scale/b.expected_batch_size
    repeated_std = std*math.sqrt(chunks)
    if not all(math.isfinite(value) for value in (noise_scale,std,repeated_std)):
        raise ValueError("noise formula exceeds finite numerical range")
    return {
        "input":asdict(b),"physical_chunks":chunks,
        "per_example_weight_gradient_bytes":dense,"two_sequence_gram_bytes":gram,
        "activation_backprop_bytes":activation,"dense_to_gram_ratio":dense/gram,
        "mean_release_noise_std":std,"repeated_chunk_noise_std":repeated_std,
        "dense_one_pass_read_modeled_ms":dense/(b.assumed_read_gbps*1e6)+b.setup_us/1000,
        "gram_one_pass_read_modeled_ms":gram/(b.assumed_read_gbps*1e6)+b.setup_us/1000,
        "privacy_epsilon":None,"privacy_delta":None,"peak_hbm_bytes":None,"gpu_latency_ms":None,
        "scope":"one linear layer, physical chunk buffers; no optimizer, other layers, bias scratch or DP accountant",
    }


def simulate() -> dict:
    rows = batch()
    gradients = [row["gradient"] for row in rows]
    exact_norms = [norm(g) for g in gradients]
    ghost = [ghost_norm(a,row["backprops"]) for a,row in zip(ACTIVATIONS,rows)]
    coefficients = [coefficient(n) for n in exact_norms]
    reference = clipped_sum(gradients,coefficients)
    cases = []

    def case(name, output, output_norms=None, scope="per-example-clipped-sum"):
        error = max(abs(a-b) for a,b in zip(output,reference))
        cases.append({
            "id":name,"output":output,"maximum_absolute_error":error,"matches_reference":error < 1e-8,
            "individual_clipped_norms":output_norms,"reference_scope":scope,"confidence":None,
        })

    ghost_coefficients = [coefficient(n) for n in ghost]
    case("sequence-gram-ghost",clipped_sum(gradients,ghost_coefficients),
         [norm([c*v for v in g]) for c,g in zip(ghost_coefficients,gradients)])
    for method in ("diagonal-only","frobenius-product"):
        factors = [coefficient(ghost_norm(a,row["backprops"],method)) for a,row in zip(ACTIVATIONS,rows)]
        case(method,clipped_sum(gradients,factors),
             [norm([c*v for v in g]) for c,g in zip(factors,gradients)])
    combined = clipped_sum(gradients,[1.0]*len(gradients))
    case("clip-after-example-aggregation",[coefficient(norm(combined))*v for v in combined])
    layerwise = []
    for g in gradients:
        cw,cb = coefficient(norm(g[:4])),coefficient(norm(g[4:]))
        layerwise.append([cw*v for v in g[:4]]+[cb*v for v in g[4:]])
    case("independent-weight-bias-bounds",clipped_sum(layerwise,[1.0]*len(layerwise)),
         [norm(g) for g in layerwise])
    fixed_replay = finite_difference(lambda p:weighted_loss(p,ghost_coefficients))
    case("stopped-coefficient-replay",fixed_replay)

    def changing_loss(p):
        current = batch(p)
        return weighted_loss(p,[coefficient(norm(row["gradient"])) for row in current])

    case("differentiate-through-coefficients",finite_difference(changing_loss))
    std_vector = [0.3,-0.4,0.1,0.2,-0.25,0.5]
    clip,sigma,expected = 2.0,1.0,4
    release = [(g+sigma*clip*z)/expected for g,z in zip(reference,std_vector)]
    b = Budget()
    return {
        "schema_version":1,"round":99,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,
        "fixture":{"parameters":PARAMETERS,"activations":ACTIVATIONS,"targets":TARGETS,
                   "clip":clip,"coefficient_epsilon":EPSILON,"loss":"half squared error summed across tokens and outputs",
                   "per_example_gradients":gradients,"materialized_norms":exact_norms,"gram_norms":ghost,
                   "coefficients":coefficients,"reference_clipped_sum":reference},
        "cases":cases,
        "partition_checks":[{"physical_sizes":sizes,"maximum_absolute_error":max(
            abs(a-b) for a,b in zip(partition_sum(gradients,sizes),reference))}
                            for sizes in ([3],[1,2],[2,1],[1,1,1])],
        "synthetic_release":{
            "unit_noise_vector":std_vector,"noise_is_gaussian_draw":False,
            "noise_multiplier":sigma,"clip":clip,"expected_batch_size":expected,"actual_examples":len(rows),
            "release":release,"mean_noise_std_if_gaussian":sigma*clip/expected,
            "wrong_actual_count_release":[(g+sigma*clip*z)/len(rows) for g,z in zip(reference,std_vector)],
            "privacy_epsilon":None,"privacy_delta":None,"secure_rng":False,
        },
        "budget":estimate(b),
        "scaling":[estimate(replace(b,tokens=t)) for t in (64,128,1024,4096)],
        "notes":[
            "Original float64 scalar linear model, not Opacus/PyTorch execution or a private training implementation.",
            "Ghost sequence norms retain all token-pair cross terms; product of whole-sequence Frobenius norms is only a bound.",
            "Global flat norm combines weight and bias before clipping; independent per-parameter bounds define another policy.",
            "Fixed coefficient replay matches clipped gradients; differentiating coefficients defines a different objective.",
            "Fixed noise vector is synthetic, not a random Gaussian sample, secure randomness or a DP guarantee.",
            "Expected denominator is explicit; source may additionally scale by accumulated_iterations; this toy models one logical release.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {Path(__file__).with_name("report.json"):json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args = parser.parse_args()
    for path,content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 99 artifacts match." if args.check else "Wrote Round 99 artifacts.")


if __name__ == "__main__":
    main()
