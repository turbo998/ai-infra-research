"""Original affine shape contracts; not JAX execution or a general solver."""

import argparse
import itertools
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = "99904d3b3b0e0b138f9a86f6115594243cd983f8"
MAX_SAFE = 2**53-1
DEFAULT = dict(request_batches=[1,2,3,5,8,13,21],width=64,value_bytes=2,bucket_multiple=8)


class ShapeError(ValueError):
    pass


class Unsolvable(ShapeError):
    pass


class Inconclusive(ShapeError):
    pass


def integer(value,minimum=0):
    return isinstance(value,int) and not isinstance(value,bool) and value>=minimum


@dataclass(frozen=True,eq=False)
class Scope:
    lower_bounds: tuple = ()

    def __post_init__(self):
        if not isinstance(self.lower_bounds,tuple):
            raise ShapeError("scope bounds must be immutable")
        names = []
        for name,value in self.lower_bounds:
            if not isinstance(name,str) or not name.isidentifier() or not integer(value,1):
                raise ShapeError("scope bounds require named positive integer variables")
            names.append(name)
        if len(set(names)) != len(names):
            raise ShapeError("duplicate bound")

    def minimum(self,name):
        return dict(self.lower_bounds).get(name,1)


@dataclass(frozen=True)
class Affine:
    scope: Scope
    coefficients: tuple
    constant: int = 0

    def __post_init__(self):
        if not isinstance(self.scope,Scope) or not isinstance(self.constant,int) or isinstance(self.constant,bool):
            raise ShapeError("affine expression requires a scope and integer constant")
        for name,coefficient in self.coefficients:
            if (not isinstance(name,str) or not name.isidentifier() or
                    not isinstance(coefficient,int) or isinstance(coefficient,bool)):
                raise ShapeError("integer named coefficients required")
        merged = {}
        for name,coefficient in self.coefficients:
            merged[name] = merged.get(name,0)+coefficient
        object.__setattr__(self,"coefficients",tuple(sorted((name,c) for name,c in merged.items() if c)))

    def evaluate(self,env):
        return self.constant+sum(c*env[name] for name,c in self.coefficients)

    def add(self,other):
        if not isinstance(other,Affine) or self.scope is not other.scope:
            raise ShapeError("symbolic expressions must share scope identity")
        return Affine(self.scope,self.coefficients+other.coefficients,self.constant+other.constant)


def affine(scope,constant=0,**coefficients):
    return Affine(scope,tuple(coefficients.items()),constant)


@dataclass(frozen=True)
class Equation:
    expression: Affine
    actual: int

    def __post_init__(self):
        if not isinstance(self.expression,Affine) or not integer(self.actual):
            raise ShapeError("shape equations need an affine expression and nonnegative actual size")


def solve(equations):
    if not equations:
        raise ShapeError("at least one shape equation is required")
    scope = equations[0].expression.scope
    if any(eq.expression.scope is not scope for eq in equations):
        raise ShapeError("shape equations mix scope identities")
    env,pending = {},list(equations)
    while pending:
        deferred = []
        for eq in pending:
            unknown = [(name,c) for name,c in eq.expression.coefficients if name not in env]
            remaining = eq.actual-eq.expression.constant-sum(
                c*env[name] for name,c in eq.expression.coefficients if name in env)
            if len(unknown)>1:
                deferred.append(eq)
            elif len(unknown)==1:
                name,coefficient = unknown[0]
                value,remainder = divmod(remaining,coefficient)
                if remainder:
                    raise ShapeError(f"division remainder {remainder} for {name}")
                if value<scope.minimum(name):
                    raise ShapeError(f"{name} violates positive/lower-bound domain")
                env[name] = value
            elif remaining:
                raise ShapeError("inconsistent concrete input dimensions")
        if len(deferred)==len(pending):
            raise Unsolvable("no remaining univariate affine equation; not a proof of infeasibility")
        pending = deferred
    if any(name not in env for name,_ in scope.lower_bounds):
        raise Unsolvable("an explicit bound refers to a variable absent from solved inputs")
    return env


def exhaustive(equations,limit):
    if not integer(limit,1) or not equations:
        raise ShapeError("bounded oracle requires equations and a positive limit")
    scope = equations[0].expression.scope
    if any(eq.expression.scope is not scope for eq in equations):
        raise ShapeError("oracle equations mix scopes")
    names = sorted({name for eq in equations for name,_ in eq.expression.coefficients} |
                   {name for name,_ in scope.lower_bounds})
    result = []
    for values in itertools.product(*(range(scope.minimum(name),limit+1) for name in names)):
        env = dict(zip(names,values))
        if all(eq.expression.evaluate(env)==eq.actual for eq in equations):
            result.append(env)
    return result


def prove_nonnegative(expression):
    lower = upper = expression.constant
    lower_finite = upper_finite = True
    for name,c in expression.coefficients:
        if c<0:
            lower_finite = False
        else:
            lower += c*expression.scope.minimum(name)
        if c>0:
            upper_finite = False
        else:
            upper += c*expression.scope.minimum(name)
    if lower_finite and lower>=0:
        return True
    if upper_finite and upper<0:
        return False
    raise Inconclusive("bounded affine proof cannot decide for every allowed dimension")


def estimate_budget(p):
    batches = p["request_batches"]
    if not isinstance(batches,list) or not 1<=len(batches)<=512 or any(not integer(b,1) for b in batches):
        raise ShapeError("one to 512 positive integer request batches required")
    if not integer(p["width"],1) or not integer(p["bucket_multiple"],1):
        raise ShapeError("positive integer width and bucket multiple required")
    if not integer(p["value_bytes"],1) or p["value_bytes"] not in (1,2,4,8):
        raise ShapeError("value_bytes must be 1/2/4/8")
    if any(v>MAX_SAFE for v in batches+[p["width"],p["bucket_multiple"]]):
        raise ShapeError("inputs exceed exact integer range")
    padded = [((b+p["bucket_multiple"]-1)//p["bucket_multiple"])*p["bucket_multiple"] for b in batches]
    actual = sum(batches)*p["width"]*p["value_bytes"]
    allocated = sum(padded)*p["width"]*p["value_bytes"]
    if any(v>MAX_SAFE for v in padded+[actual,allocated]):
        raise ShapeError("budget exceeds exact integer range")
    return dict(concrete_request_signatures=len(set(batches)),
                assumed_export_artifact_signatures=1,
                bucketed_request_signatures=len(set(padded)),padded_batches=padded,
                actual_batch_rows=sum(batches),padded_batch_rows=sum(padded),
                logical_input_bytes=actual,padded_logical_input_bytes=allocated,
                padding_bytes=allocated-actual,
                cache_policy="one dtype, width, target and shard layout; count batch signatures only",
                observed_jax_compile_events=None,native_jax_latency_ms=None,
                executable_bytes=None,peak_hbm_bytes=None)


def fixture(b=8,c=5):
    scope = Scope()
    return (Equation(affine(scope,1,b=3),3*b+1),
            Equation(affine(scope,b=1,c=1),b+c),
            Equation(affine(scope,c=2),2*c))


def measure_latency():
    rows = []
    operations = 3
    for limit in (8,16,32,64):
        equations = fixture(limit//2,max(1,limit//3))
        methods = (lambda: solve(equations),lambda: exhaustive(equations,limit))
        samples = [[],[]]
        for method in methods:
            for _ in range(2):
                method()
        for repeat in range(7):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                begin = time.perf_counter_ns()
                for _ in range(operations):
                    methods[index]()
                samples[index].append((time.perf_counter_ns()-begin)/1e6/operations)
        rows.append(dict(oracle_variable_bound=limit,oracle_candidate_environments=limit**2,
                         primary_samples_ms=samples[0],baseline_samples_ms=samples[1],
                         primary_median_ms=statistics.median(samples[0]),
                         baseline_median_ms=statistics.median(samples[1]),
                         oracle_matches_solution=methods[1]()==[methods[0]()]))
    data = dict(provenance="measured",scope="Original bounded affine solver versus independent exhaustive oracle",
                hardware=platform.machine(),processor=platform.processor(),python=platform.python_version(),
                warmups=2,repeats=7,operations_per_repeat=operations,alternating_order=True,
                statistic="median of seven repeat means; each mean spans three operations",
                rows=rows,native_jax_measured=False)
    (HERE/"latency.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")


def simulate():
    equations = fixture()
    solution = solve(equations)
    invalid = []
    for name,actuals in (("nondivisible",[24,13,10]),("nonpositive",[1,5,10]),
                         ("inconsistent",[25,13,11])):
        altered = tuple(Equation(eq.expression,size) for eq,size in zip(equations,actuals))
        try:
            solve(altered)
        except ShapeError as error:
            invalid.append(dict(case=name,actual_shapes=actuals,rejected=True,reason=str(error)))
        else:
            raise AssertionError(f"negative control accepted: {name}")
    scope = Scope()
    coupled = (Equation(affine(scope,a=1,b=1),10),Equation(affine(scope,a=1,b=-1),2))
    try:
        solve(coupled)
    except Unsolvable:
        rejected = True
    else:
        raise AssertionError("bounded univariate solver unexpectedly solved coupled system")
    plain = affine(Scope(),-5,b=2)
    try:
        prove_nonnegative(plain)
    except Inconclusive:
        proof = "inconclusive"
    else:
        raise AssertionError("non-universal inequality was decided")
    return dict(schema_version=1,round=109,provenance="synthetic",independently_reproduced=False,
                native_jax_measured=False,
                source=dict(pin=PIN,date="2026-11-04",license="Apache-2.0",
                            capability="Symbolic polynomial arithmetic and comparison optimization"),
                solve=dict(actual_shapes=[25,13,10],equations=["3*b+1","b+c","2*c"],
                           solution=solution,oracle=exhaustive(equations,16),
                           residuals=[eq.expression.evaluate(solution)-eq.actual for eq in equations]),
                negative_controls=invalid,
                coupled=dict(equations=["a+b=10","a-b=2"],univariate_solver_rejected=rejected,
                             bounded_oracle_solutions=exhaustive(coupled,16)),
                proof=dict(expression="2*b-5 >= 0",unconstrained=proof,
                           at_b1=plain.evaluate({"b":1}),at_b8=plain.evaluate({"b":8}),
                           with_b_at_least_3=prove_nonnegative(affine(Scope((("b",3),)),-5,b=2))),
                budget=dict(input=DEFAULT,**estimate_budget(DEFAULT)),
                latency=json.loads((HERE/"latency.json").read_text(encoding="utf-8")))


def outputs():
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
            if not path.exists() or path.read_text(encoding="utf-8")!=content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 109 artifacts match." if args.check else "Wrote Round 109 artifacts.")


if __name__ == "__main__":
    main()
