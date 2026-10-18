# In-Network Collectives (`NVLS` + `IB/UEC SHARP`) Offline Experiment

Deterministic stdlib evaluation (`experiments/sharp_in_network_collectives/run.py`) modeling:
1. AllReduce latency and algorithm bandwidth (`algbw`) across `64–16,384` GPUs (`1 GiB` payload) for `Ring`, `Tree`, `NVLS_TREE`, and `NVLS + IB/UEC SHARP` (`CollNet`).
2. Multi-tenant switch SHARP resource saturation (`64` hardware trees) and fallback tail latency (`Ring` vs `NVLS_TREE`).
3. Floating-point non-associativity bit-wise drift (`BF16` vs `FP32` internal switch accumulation vs fixed-order reduction).

## Usage

```powershell
python experiments\sharp_in_network_collectives\run.py
python experiments\sharp_in_network_collectives\run.py --check
```
