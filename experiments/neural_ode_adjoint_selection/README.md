# Round 103: neural ODE gradient contracts and checkpoint replay

Original scalar fixed-Euler forward, discrete reverse, deterministic
block replay, polynomial/finite-difference references and continuous
backsolve controls. No adaptive solver/framework execution, scientific
model quality or GPU result.

```powershell
python -m experiments.neural_ode_adjoint_selection.run
python -m experiments.neural_ode_adjoint_selection.run --check
python -m unittest tests.test_neural_ode_adjoint_selection
```

`latency.json` seals two warmups and seven alternating CPU repeats.
Rendering/checking never remeasures; explicit `--measure` replaces
samples and requires updating timing prose. `--check --measure` is invalid.

See [English FAQ](../../docs/components/neural-ode-discrete-continuous-adjoint-checkpoint-selection-faq.en.md),
[中文 FAQ](../../docs/components/neural-ode-discrete-continuous-adjoint-checkpoint-selection-faq.md),
[H5](../../docs/reports/neural-ode-discrete-continuous-adjoint-checkpoint-sizer.html),
[seven cards](../../docs/reports/neural-ode-discrete-continuous-adjoint-checkpoint-selection-cards.html)
and [SVG](../../assets/svg/neural-ode-discrete-continuous-adjoint-checkpoint-selection-faq-summary.svg).

Fixed block replay is not Diffrax's online/binomial schedule. Backsolve
controls use explicit Euler backward in time, not implicit backward Euler.
Selected logical buffers are not full autograd tape or peak allocations.
Pinned Diffrax/torchdiffeq rights remain Apache-2.0/MIT respectively;
no implementation or model weights were copied.
