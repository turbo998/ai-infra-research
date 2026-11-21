# Fault-Tolerant LLM Training & Sub-10s Recovery Selection Experiment (`TorchFT / ReCycle` vs. `In-Memory P2P Ring Checkpoint` vs. `Async DCP/Orbax`)

Reproducible Young-Daly Effective Training Time Ratio (`ETTR / Goodput`) and cost model comparing 4 fault-tolerant training architectures on an `8,192`-GPU `H200` cluster (`MTBF = 3.2 hours`, `5.2 TB` state).

## Run
```bash
python experiments/llm_fault_tolerant_training_async_checkpoint_torchft_local_replica_selection/run.py
python experiments/llm_fault_tolerant_training_async_checkpoint_torchft_local_replica_selection/run.py --check
```
