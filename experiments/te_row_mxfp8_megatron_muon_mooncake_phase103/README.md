# Phase 103 Track C: TransformerEngine Row-Only MXFP8, Megatron-LM MFSDP v2 Muon, TorchTitan Ulysses/RoPE & Mooncake TENT

Deterministic Python 3.12 standard-library model reproducing:
- `NVIDIA/TransformerEngine@3850986745` (#3488) & `0f8d92f5a7` (#3744): Row-only MXFP8 master-weight casts (`50.0%` FP8 write reduction, `25.38%` total HBM I/O reduction).
- `NVIDIA/Megatron-LM@19895dafba` (#8512) & `63f687dc83` (#8517): MFSDP v2 Muon (`RowAtomic` + `FsdpMuon` P2P gather/scatter `8.0x` communication reduction vs boundary all-gather, and `TensorAtomic` zero-optimizer-communication tradeoff).
- `pytorch/torchtitan@b44774915d` (#5590) & `e111f10d4d` (#5583): Ulysses CP for KDA/GDN & Tied RoPE cache (`80.0x` RoPE buffer reduction across 80 layers).
- `LMCache/LMCache@0395e677c4` (#5870), `kvcache-ai/Mooncake@1dd37f57ba` (#4383), `ddda75d351` (#4708), and `volcengine/verl@6b2b417a05` (#8315).

## Usage
```bash
python experiments/te_row_mxfp8_megatron_muon_mooncake_phase103/run.py
python experiments/te_row_mxfp8_megatron_muon_mooncake_phase103/run.py --check
```
