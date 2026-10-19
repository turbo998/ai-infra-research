# PyTorch Distributed Checkpoint (`DCP`) `AsyncStager`, `StateDictStager`, & `_ReplicationStager` Simulator

Deterministic simulator verifying `DefaultStager`, `StateDictStager` (`WeakIdKeyDictionary` cached pinned/shared CPU `UntypedStorage`), and `_ReplicationStager` (half-world partner exchange) in `pytorch/pytorch` (`torch/distributed/checkpoint/`).

## Run & Check

```bash
python experiments/async_dcp_checkpoint_stager/run.py --check
```
