# Round 109: symbolic shape contracts and export/compilation layers

Original affine integer solver and finite independent exhaustive oracle.
Synthetic contracts, `independently_reproduced=false`; original Python CPU
timings separately measured, not native JAX or compiler performance.

- [中文 FAQ](../../docs/components/symbolic-shape-export-dimension-contract-selection-faq.md)
- [English FAQ](../../docs/components/symbolic-shape-export-dimension-contract-selection-faq.en.md)
- [H5 signature/padding budget](../../docs/reports/symbolic-shape-export-dimension-contract-sizer.html)
- [Seven cards](../../docs/reports/symbolic-shape-export-dimension-contract-selection-cards.html)
- [SVG](../../assets/svg/symbolic-shape-export-dimension-contract-selection-faq-summary.svg)

```powershell
python -m experiments.symbolic_shape_contract_selection.run
python -m unittest discover -s tests -p test_symbolic_shape_contract_selection.py -q
python -m experiments.symbolic_shape_contract_selection.run --check
```

`--measure` explicitly replaces sealed samples; generation/checking does not.
Two warmups, seven alternating repeats, three operations per sample. Each
sample is mean ms/operation; chart uses median across seven repeat means.
The exhaustive oracle visits all bounded candidate environments, not JAX's
native solver strategy. Agreement is checked outside timing.

## Contracts and boundaries

Exact divisibility, positive solved variables and repeated-dimension
consistency are separate guards. A feasible coupled system can be outside
the univariate strategy; a bounded oracle is not a global feasibility proof.
The original model implements affine expressions only, not JAX's broader
polynomial/constraint semantics. Unknown inequalities raise explicitly.
Scope identity is required; immutable lower bounds change accepted inputs.

H5 signature counts assume one dtype/width/target/shard-layout family and
count only concrete batch keys. One assumed symbolic export is not one
executable or observed compilation event. Padding adds logical input bytes
and requires application-correct valid-position semantics. Compiler cost,
executable cache bytes, intermediates, allocator and peak HBM are unknown.

Pinned JAX `99904d3b3b0e0b138f9a86f6115594243cd983f8`, November 4, 2026,
optimizes symbolic polynomial arithmetic/comparisons. No upstream
optimization speedup is independently measured. Apache-2.0 upstream rights
retained; no source implementation, datasets or model weights copied.
