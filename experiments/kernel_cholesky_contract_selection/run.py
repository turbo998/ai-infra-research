"""Original dense kernel Cholesky/cache contracts, not native linear_operator."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import statistics
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PIN="1c5e24bc774352461be818b9d347e93ddee7d7f7"
DEFAULT={"training_points":1024,"query_points":64}


def finite(value):
    if type(value) not in (int,float,Fraction):
        raise ValueError("finite scalar required")
    value=float(value)
    if not math.isfinite(value):
        raise ValueError("finite scalar required")
    return value


def factor(matrix):
    if not matrix or any(len(row)!=len(matrix) for row in matrix):
        raise ValueError("nonempty square matrix required")
    a=[[finite(value) for value in row] for row in matrix]
    if any(a[i][j]!=a[j][i] for i in range(len(a)) for j in range(i)):
        raise ValueError("original model requires explicit full symmetry, not just a trusted triangle")
    l=[[0.0]*len(a) for _ in a]
    for i in range(len(a)):
        for j in range(i+1):
            value=a[i][j]-math.fsum(l[i][k]*l[j][k] for k in range(j))
            if i==j:
                if value<=0 or not math.isfinite(value):
                    raise ValueError("original matrix is not numerically positive definite; no silent jitter fallback")
                l[i][j]=math.sqrt(value)
            else:
                l[i][j]=value/l[j][j]
    return l


def solve_factor(l,rhs):
    if len(rhs)!=len(l):
        raise ValueError("right-hand side dimension mismatch")
    rhs=[finite(value) for value in rhs]
    y=[]
    for i in range(len(l)):
        y.append((rhs[i]-math.fsum(l[i][j]*y[j] for j in range(i)))/l[i][i])
    x=[0.0]*len(l)
    for i in reversed(range(len(l))):
        x[i]=(y[i]-math.fsum(l[j][i]*x[j] for j in range(i+1,len(l))))/l[i][i]
    return x


def shifted(matrix,shift):
    if type(shift) not in (int,Fraction) or shift<0:
        raise ValueError("original explicit shift requires nonnegative exact rational input")
    return [[Fraction(value)+(shift if i==j else 0) for j,value in enumerate(row)]
            for i,row in enumerate(matrix)]


def rational_solve(matrix,rhs):
    n=len(matrix)
    augmented=[[Fraction(value) for value in row]+[Fraction(rhs[i])] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=next((row for row in range(column,n) if augmented[row][column]),None)
        if pivot is None:
            raise ValueError("independent rational reference is singular")
        augmented[column],augmented[pivot]=augmented[pivot],augmented[column]
        divisor=augmented[column][column]
        augmented[column]=[value/divisor for value in augmented[column]]
        for row in range(n):
            if row!=column:
                multiple=augmented[row][column]
                augmented[row]=[a-multiple*b for a,b in zip(augmented[row],augmented[column])]
    return [row[-1] for row in augmented]


def residual(matrix,x,rhs):
    return max(abs(math.fsum(finite(value)*x[j] for j,value in enumerate(row))-finite(rhs[i]))
               for i,row in enumerate(matrix))


def estimate_budget(p):
    if any(type(p[key]) is not int or not 1<=p[key]<=2**53-1 for key in DEFAULT):
        raise ValueError("positive bounded exact point counts required")
    n,q=p["training_points"],p["query_points"]
    out={"selected_float64_covariance_bytes":n*n*8,"selected_float64_factor_bytes":n*n*8,
         "selected_float64_cross_covariance_bytes":n*q*8,"selected_float64_rhs_bytes":n*8}
    if any(value>2**53-1 for value in out.values()):
        raise ValueError("byte result exceeds exact arithmetic range")
    return out


def measure():
    rows=[]
    for n in (4,8,16,32):
        matrix=shifted([[Fraction(1,2)**abs(i-j) for j in range(n)] for i in range(n)],Fraction(1,4))
        l=factor(matrix)
        queries=[[Fraction((i+j)%7,7) for i in range(n)] for j in range(8)]
        functions=(lambda:[solve_factor(l,rhs) for rhs in queries],
                   lambda:[solve_factor(factor(matrix),rhs) for rhs in queries])
        assert functions[0]()==functions[1]()
        assert max(residual(matrix,x,rhs) for x,rhs in zip(functions[0](),queries))<1e-10
        for _ in range(2):
            for function in functions:
                function()
        samples=[[],[]]
        for repeat in range(7):
            for i in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter_ns()
                for _ in range(3):
                    functions[i]()
                samples[i].append((time.perf_counter_ns()-started)/3/1e6)
        rows.append({"dimension":n,"right_hand_sides":8,"cached_samples_ms":samples[0],
                     "refactor_samples_ms":samples[1],"cached_median_ms":statistics.median(samples[0]),
                     "refactor_median_ms":statistics.median(samples[1])})
    return {"provenance":"measured","warmups":2,"repeats":7,"operations_per_sample":3,
            "method":"original cached Cholesky triangular solves versus rebuilding same scalar factor for each of eight right-hand sides; cached setup excluded",
            "excludes":"kernel construction, hyperparameter fitting, initial cache setup, native linear_operator and GPU",
            "python":platform.python_version(),"machine":platform.machine(),"rows":rows}


def build_report(latency):
    kernel=[[1,1],[1,1]];rhs=[1,1];shift=Fraction(1,10)
    matrix=shifted(kernel,shift);double=shifted(kernel,2*shift)
    l=factor(matrix);x=solve_factor(l,rhs);exact=rational_solve(matrix,rhs)
    doubled=rational_solve(double,rhs)
    mean=sum(exact);latent=1-sum(exact);observed=latent+shift
    return {"schema_version":1,"round":123,"provenance":"synthetic","independently_reproduced":False,
            "source":{"repository":"cornellius-gp/linear_operator","pin":PIN,"tree_date_utc":"2026-09-11T04:13:37Z",
                      "license":"MIT","cholesky_source_change":"2026-03-04 modern torch.linalg API migration, not new factorization theory",
                      "inspected":"normal-mode Cholesky rescue with warnings, failed-batch mask, incremental jitter deltas and final error; tracing path not reproduced"},
            "case":{"kernel":kernel,"rhs":rhs,"diagonal_shift":str(shift),
                    "diagonal_shift_role":"selected observation-noise variance, not automatically native rescue jitter",
                    "rational_solution":[str(value) for value in exact],
                    "float_solution":x,"rational_mean":str(mean),"rational_latent_variance":str(latent),
                    "rational_noisy_variance":str(observed),
                    "logdet":2*sum(math.log(l[i][i]) for i in range(2)),
                    "rational_determinant":str(matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]),
                    "residual":residual(matrix,x,rhs),"unshifted_singular":True,
                    "double_shift_mean":str(sum(doubled)),"double_shift_latent_variance":str(1-sum(doubled)),
                    "stale_factor_shifted_matrix_residual":residual(double,x,rhs),
                    "fresh_factor_shifted_matrix_residual":residual(double,solve_factor(factor(double),rhs),rhs),
                    "note":"one synthetic duplicated-observation kernel; a diagonal shift changes the model, not only numerical status"},
            "default_budget":{"inputs":DEFAULT,"outputs":estimate_budget(DEFAULT)},
            "latency":latency,"native_factorization_latency_ms":None,"native_rescue_reproduced":False,
            "posterior_model_quality":None,"peak_hbm_bytes":None}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--measure",action="store_true")
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    if args.measure:
        (HERE/"latency.json").write_text(json.dumps(measure(),indent=2)+"\n",encoding="utf-8")
    report=build_report(json.loads((HERE/"latency.json").read_text(encoding="utf-8")))
    from .render import outputs
    for path,text in {HERE/"report.json":json.dumps(report,indent=2)+"\n",**outputs(report)}.items():
        if args.check:
            if path.read_text(encoding="utf-8")!=text:
                raise SystemExit(f"stale artifact: {path}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding="utf-8")
    print("Round 123 artifacts match." if args.check else "Wrote Round 123 artifacts.")


if __name__=="__main__":
    main()
