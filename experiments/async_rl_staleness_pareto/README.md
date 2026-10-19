# Asynchronous RL Rollout Staleness, Partial-Rollout Continuation & Weight-Sync Pareto Simulator

Deterministic simulation of `THUDM/slime` (`slime/rollout/fully_async_rollout.py` and `docs/en/advanced/rollout-scheduling.md`), comparing Synchronous Colocated Rollout ($\tau=0$), Synchronous + Partial Rollout (`--partial-rollout`), and Fully Asynchronous Rollout across staleness bounds $\tau_{\text{max}} \in \{1, 2, 4, 8\}$ with Token-Aligned Truncated Importance Sampling (`TIS`, $p_\theta(y_t\mid h_t)/q_{v(t)}(y_t\mid h_t)$) vs prefix loss masking (`--mask-offpolicy-in-partial-rollout`).

## Run

```bash
python experiments/async_rl_staleness_pareto/run.py
python experiments/async_rl_staleness_pareto/run.py --check
```
