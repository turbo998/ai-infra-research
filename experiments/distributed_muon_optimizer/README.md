# Distributed Muon Optimizer: 5-Step Quintic Newton-Schulz Orthogonalization & 2D FSDP/TP Sharding

Deterministic offline simulator backed by upstream source verification of `KellerJordan/Muon` (`commit f98f1cacc0263b04290753e32be8d498c1efc806`, `muon.py`) and `MoonshotAI/Moonlight` (`commit c2ad5b20c605086526a179d36901bfc41b52b44b`, `examples/toy_train.py`).

## Quick Run

```bash
python experiments/distributed_muon_optimizer/run.py
python experiments/distributed_muon_optimizer/run.py --check
```
