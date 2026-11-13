"""Original real-grid spectral operator: direct DFT, symmetry and byte accounting."""

import argparse
import cmath
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path


SOURCE = "https://github.com/neuraloperator/neuraloperator/blob/a6648eeeb60fe0837f2d7d021517a945a159a0c8/neuralop/layers/spectral_convolution.py"


def shape_of(grid: list) -> tuple[int, int]:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be nonempty and rectangular")
    if any(not (math.isfinite(complex(v).real) and math.isfinite(complex(v).imag))
           for row in grid for v in row):
        raise ValueError("grid values must be finite")
    return len(grid), len(grid[0])


def scale(size: int, inverse: bool, norm: str) -> float:
    if norm not in ("forward", "backward", "ortho"):
        raise ValueError("norm must be forward, backward or ortho")
    if norm == "ortho":
        return 1 / math.sqrt(size)
    return 1 / size if (norm == "backward") == inverse else 1.0


def dft(values: list, inverse: bool = False, norm: str = "forward") -> list[complex]:
    if not values:
        raise ValueError("DFT needs at least one value")
    n = len(values)
    sign = 1 if inverse else -1
    factor = scale(n, inverse, norm)
    return [factor * sum(v * cmath.exp(sign * 2j * math.pi * k * j / n)
                         for j, v in enumerate(values)) for k in range(n)]


def fft2(grid: list, inverse: bool = False, norm: str = "forward") -> list:
    nx, ny = shape_of(grid)
    rows = [dft(row, inverse, norm) for row in grid]
    columns = [dft([rows[x][y] for x in range(nx)], inverse, norm) for y in range(ny)]
    return [[columns[y][x] for y in range(ny)] for x in range(nx)]


def direct_dft2(grid: list, inverse: bool = False, norm: str = "forward") -> list:
    """Independent four-loop transform, not the separable implementation."""
    nx, ny = shape_of(grid)
    factor = scale(nx * ny, inverse, norm)
    sign = 1 if inverse else -1
    return [[factor * sum(grid[x][y] * cmath.exp(sign * 2j * math.pi * (kx*x/nx + ky*y/ny))
                          for x in range(nx) for y in range(ny))
             for ky in range(ny)] for kx in range(nx)]


def rfft2(grid: list, norm: str = "forward") -> list:
    _, ny = shape_of(grid)
    if any(complex(v).imag != 0 for row in grid for v in row):
        raise ValueError("real FFT input must be real")
    return [row[:ny//2 + 1] for row in fft2(grid, norm=norm)]


def irfft_row(values: list, size: int, norm: str, pair_factor: float = 2.0) -> list:
    if type(size) is not int or size < 1 or len(values) != size//2 + 1:
        raise ValueError("one-sided spectrum must match the explicit inverse size")
    paired_end = len(values) - 1 if size % 2 == 0 else len(values)
    factor = scale(size, True, norm)
    return [factor * (values[0].real
                      + sum(pair_factor * (values[k] * cmath.exp(2j*math.pi*k*j/size)).real
                            for k in range(1, paired_end))
                      + (values[-1].real * (-1)**j if size % 2 == 0 else 0))
            for j in range(size)]


def staged_inverse(spectrum: list, shape: tuple[int, int], norm: str = "forward",
                   premature_clear: bool = False, odd_as_nyquist: bool = False,
                   infer_size: bool = False, missing_pairs: bool = False) -> list:
    nx, ny = shape
    sx, sy = shape_of(spectrum)
    if sx != nx or sy != ny//2 + 1:
        raise ValueError("spectrum shape does not match real-grid inverse shape")
    work = [[complex(v) for v in row] for row in spectrum]
    if premature_clear:
        for row in work:
            row[0] = complex(row[0].real)
            if ny % 2 == 0:
                row[-1] = complex(row[-1].real)
    columns = [dft([work[x][y] for x in range(nx)], True, norm) for y in range(sy)]
    spatial_rows = [[columns[y][x] for y in range(sy)] for x in range(nx)]
    output_size = 2*(sy - 1) if infer_size else ny
    for row in spatial_rows:
        row[0] = complex(row[0].real)
        if output_size % 2 == 0 or odd_as_nyquist:
            row[-1] = complex(row[-1].real)
    return [irfft_row(row, output_size, norm, 1.0 if missing_pairs else 2.0)
            for row in spatial_rows]


def full_spectrum_inverse(spectrum: list, shape: tuple[int, int], norm: str = "forward") -> list:
    nx, ny = shape
    sx, sy = shape_of(spectrum)
    if sx != nx or sy != ny//2 + 1:
        raise ValueError("invalid one-sided spectrum shape")
    full = [[spectrum[kx][ky] if ky < sy else spectrum[(-kx) % nx][ny-ky].conjugate()
             for ky in range(ny)] for kx in range(nx)]
    return [[v.real for v in row] for row in direct_dft2(full, True, norm)]


def fixture(nx: int = 3, ny: int = 5) -> tuple[list, list]:
    if type(nx) is not int or type(ny) is not int or nx < 1 or ny < 1:
        raise ValueError("fixture dimensions must be positive integers")
    grid = [[0.3 + 0.8*math.sin(2*math.pi*x/nx) + 0.4*math.cos(2*math.pi*y/ny)
             + 0.6*math.sin(4*math.pi*y/ny) + 0.2*math.cos(2*math.pi*x/nx + 2*math.pi*y/ny)
             + (0.25*(-1)**y*math.sin(2*math.pi*x/nx) if ny % 2 == 0 else 0)
             for y in range(ny)] for x in range(nx)]
    kernel = [[0.0]*ny for _ in range(nx)]
    for x, y, value in ((0,0,0.5),(1,0,0.2),(0,1,-0.15),(nx-1,ny-1,0.1),(1,ny-1,0.05)):
        kernel[x % nx][y % ny] += value
    return grid, kernel


def circular_convolution(grid: list, kernel: list) -> list:
    nx, ny = shape_of(grid)
    if shape_of(kernel) != (nx, ny):
        raise ValueError("circular convolution grid and kernel shapes must match")
    return [[sum(kernel[a][b]*grid[(x-a) % nx][(y-b) % ny]
                 for a in range(nx) for b in range(ny))
             for y in range(ny)] for x in range(nx)]


def spectrum_product(grid: list, kernel: list, norm: str = "forward") -> list:
    if shape_of(grid) != shape_of(kernel):
        raise ValueError("grid and kernel shapes must match")
    x = rfft2(grid, norm)
    # Unnormalized kernel transform defines one fixed multiplier across FFT conventions.
    weight = rfft2(kernel, "backward")
    return [[a*b for a,b in zip(row, w)] for row,w in zip(x, weight)]


def maximum_error(actual: list, expected: list) -> float | None:
    if shape_of(actual) != shape_of(expected):
        return None
    return max(abs(a-b) for row, ref in zip(actual, expected) for a,b in zip(row, ref))


@dataclass(frozen=True)
class Budget:
    batch: int = 4
    in_channels: int = 32
    out_channels: int = 64
    grid_x: int = 128
    grid_y: int = 128
    modes_x: int = 32
    modes_y: int = 32
    real_scalar_bytes: int = 4
    assumed_read_gbps: float = 1000.0
    setup_us: float = 5.0


def estimate(b: Budget) -> dict:
    names = ("batch","in_channels","out_channels","grid_x","grid_y","modes_x","modes_y","real_scalar_bytes")
    if any(type(getattr(b,n)) is not int or getattr(b,n) < 1 for n in names):
        raise ValueError("dimensions, modes and scalar bytes must be positive integers")
    if b.real_scalar_bytes not in (4,8) or b.modes_x > b.grid_x or b.modes_y > b.grid_y:
        raise ValueError("modes must fit the grid; scalar bytes must be 4 or 8")
    if not math.isfinite(b.assumed_read_gbps) or b.assumed_read_gbps <= 0:
        raise ValueError("assumed bandwidth must be finite and positive")
    if not math.isfinite(b.setup_us) or b.setup_us < 0:
        raise ValueError("setup must be finite and nonnegative")
    half_y = b.grid_y//2 + 1
    retained_y = b.modes_y//2 + 1
    complex_bytes = 2*b.real_scalar_bytes
    combined_channels = b.in_channels + b.out_channels
    spectra = b.batch*combined_channels*b.grid_x*half_y*complex_bytes
    full = b.batch*combined_channels*b.grid_x*b.grid_y*complex_bytes
    return {
        "input":asdict(b),"last_axis_bins":half_y,"retained_last_axis_bins":retained_y,
        "complex_scalar_bytes":complex_bytes,
        "dense_weight_complex_values":b.in_channels*b.out_channels*b.modes_x*retained_y,
        "dense_weight_bytes":b.in_channels*b.out_channels*b.modes_x*retained_y*complex_bytes,
        "input_output_spectrum_bytes":spectra,"full_complex_spectrum_bytes":full,
        "real_input_output_bytes":b.batch*combined_channels*b.grid_x*b.grid_y*b.real_scalar_bytes,
        "one_sided_one_pass_read_modeled_ms":spectra/(b.assumed_read_gbps*1e6)+b.setup_us/1000,
        "full_complex_one_pass_read_modeled_ms":full/(b.assumed_read_gbps*1e6)+b.setup_us/1000,
        "peak_hbm_bytes":None,"gpu_latency_ms":None,"pde_error":None,
        "scope":"selected dense weights and input/output spectra only; no FFT workspace, activations, gradients or optimizer",
    }


def simulate() -> dict:
    grid, kernel = fixture()
    shape = shape_of(grid)
    reference = circular_convolution(grid, kernel)
    spectrum = spectrum_product(grid, kernel)
    cases = []
    configurations = [
        ("staged-real-inverse", {}, "forward"),
        ("premature-frequency-column-clear", {"premature_clear":True}, "forward"),
        ("odd-last-bin-treated-as-nyquist", {"odd_as_nyquist":True}, "forward"),
        ("implicit-even-inverse-length", {"infer_size":True}, "forward"),
        ("missing-conjugate-pair-factor", {"missing_pairs":True}, "forward"),
        ("mismatched-inverse-normalization", {}, "backward"),
    ]
    for name, options, norm in configurations:
        output = staged_inverse(spectrum, shape, norm, **options)
        error = maximum_error(output, reference)
        cases.append({
            "id":name,"output_shape":shape_of(output),"output":output,
            "maximum_absolute_error":error,"matches_reference":error is not None and error < 1e-12,
            "status":"shape-mismatch" if error is None else "numeric-comparison","confidence":None,
        })
    cropped = [[v if kx <= shape[0]//2 else 0j for v in row] for kx,row in enumerate(spectrum)]
    output = staged_inverse(cropped, shape)
    error = maximum_error(output, reference)
    cases.append({
        "id":"incorrectly-half-every-axis","output_shape":shape_of(output),"output":output,
        "maximum_absolute_error":error,"matches_reference":error < 1e-12,
        "status":"numeric-comparison","confidence":None,
    })
    norms = []
    for norm in ("forward","backward","ortho"):
        result = staged_inverse(spectrum_product(grid,kernel,norm), shape, norm)
        norms.append({"norm":norm,"maximum_absolute_error":maximum_error(result,reference)})
    truncated = [[v if ky < 2 else 0j for ky,v in enumerate(row)] for row in spectrum]
    low = staged_inverse(truncated, shape)
    b = Budget()
    return {
        "schema_version":1,"round":98,"provenance":"synthetic","independently_reproduced":False,
        "physical_gpu_measurement":False,"source":SOURCE,
        "fixture":{"shape":shape,"grid":grid,"kernel":kernel,"reference":reference,"tolerance":1e-12},
        "cases":cases,"normalization_checks":norms,
        "mode_truncation":{
            "policy":"retain last-axis bins 0 and 1; intentional low-pass operator",
            "full_operator_difference":maximum_error(low,reference),
            "same_retained_spectrum_reference_error":maximum_error(low,full_spectrum_inverse(truncated,shape)),
            "pde_quality":None,
        },
        "even_grid":{
            "shape":[4,6],
            "maximum_absolute_error":maximum_error(
                staged_inverse(spectrum_product(*fixture(4,6)),(4,6)),
                circular_convolution(*fixture(4,6))),
        },
        "budget":estimate(b),
        "scaling":[estimate(replace(b,batch=batch)) for batch in (1,2,4,8)],
        "notes":[
            "Original direct complex DFTs, not an FFT performance implementation or neuraloperator reproduction.",
            "One-channel real-kernel convolution fixture; dense multi-channel byte formulas are a separate budget.",
            "Only the last real-FFT axis is one-sided; other axes keep signed frequency rows.",
            "Clear DC/even Nyquist imaginary parts after inversing the other axes, not on raw 2D boundary columns.",
            "Odd spatial lengths need an explicit inverse shape; last stored odd-grid bin is not Nyquist.",
            "The pinned source discusses GPU artifacts; no such hardware symptom was reproduced here.",
        ],
    }


def outputs() -> dict[Path,str]:
    from .render import render_outputs
    report = simulate()
    return {Path(__file__).with_name("report.json"):json.dumps(report,indent=2)+"\n",**render_outputs(report)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"stale artifact: {path.name}")
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding="utf-8",newline="\n")
    print("Round 98 artifacts match." if args.check else "Wrote Round 98 artifacts.")


if __name__ == "__main__":
    main()
