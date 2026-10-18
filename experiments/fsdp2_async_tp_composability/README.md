# FSDP2 (`fully_shard` DTensor) & `SymmetricMemory` `Async-TP` 5D Composability Experiment

Deterministic offline simulator (`measurement_class: own-offline`) grounded in PyTorch (`commit 1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340`):
- `torch/distributed/fsdp/_fully_shard/_fully_shard.py` (`blob bbdaf77bed399b2b769253ff4cd8e336b262d995`)
- `torch/distributed/_symmetric_memory/__init__.py` (`blob 44294523a2da34f4fe546a0a4a0b80b18da0eadb`)

## Usage
```bash
python experiments/fsdp2_async_tp_composability/run.py
python experiments/fsdp2_async_tp_composability/run.py --check
```
