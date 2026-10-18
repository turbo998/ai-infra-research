# HBM3e ECC Row Remapping, `Xid 63/64/94/95` & Predictive Node Drain Experiment

Deterministic offline simulator (`measurement_class: own-offline`) grounded in `NVIDIA/DCGM` (`commit 64df9f894541e426e416131a9820cae97aa4dd81`):
- `dcgmlib/dcgm_fields.h` (`blob aa65ab23eb0d0ca3f60a062e61e6a62148ba1b21`)
- `dcgmlib/dcgm_structs.h` (`blob cc49146e2d21d9ccd0304966d89ecaf4fcdf750a`)
- `modules/health/DcgmHealthWatch.cpp` (`blob e3c7a5f811eebba2b0c8ae4190eea3d6f90d5b36`)

## Usage
```bash
python experiments/gpu_ecc_predictive_drain/run.py
python experiments/gpu_ecc_predictive_drain/run.py --check
```
