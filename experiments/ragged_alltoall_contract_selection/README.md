# Ragged all-to-all split matrices and token inverses

Original single-destination scalar token transport: destination-major source packing, source-block receive ordering, transposed split matrices and owner/original-position restoration. Independent direct-owner oracle covers24deterministic random cases.

```powershell
python -m experiments.ragged_alltoall_contract_selection.run --measure
python -m experiments.ragged_alltoall_contract_selection.run --check
python -m unittest discover -s tests -p test_ragged_alltoall_contract_selection.py -q
```

Zero-receive ranks are permitted by original policy, not certified on a native backend. No PyTorch/NCCL/MPI/GPU, network, encryption, top-k replication, capacity/drop policy or async stream lifetime execution. CPU timings only compare inverse-ID lookup with construction excluded. Selected feature/identity/split payloads are not actual wire or heap/peak memory. [English FAQ](../../docs/components/ragged-all-to-all-split-matrix-token-inverse-selection-faq.en.md) / [中文 FAQ](../../docs/components/ragged-all-to-all-split-matrix-token-inverse-selection-faq.md).
