# Test-Time Compute (TTC) & Reasoning Scaling Selection Experiment (`Long CoT` vs. `Best-of-N` vs. `PRM + CoW KV Forking`)

Reproducible capacity and latency model comparing 4 Test-Time Compute (`TTC`) reasoning architectures across average generated tokens per task, peak KV Cache HBM footprint, end-to-end `P95` reasoning latency, `Pass@1` accuracy, and cost per 1,000 solved tasks.

## Run
```bash
python experiments/test_time_compute_scaling_reasoning_prm_mcts_best_of_n_kv_forking_selection/run.py
python experiments/test_time_compute_scaling_reasoning_prm_mcts_best_of_n_kv_forking_selection/run.py --check
```
