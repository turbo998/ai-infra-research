# Activation Checkpointing & Async PCIe Offloading Selection Experiment (`Full` vs. `Selective` vs. `Async PCIe Gen5 Offload`)

Reproducible analytical model comparing 4 activation memory management tiers for `70B` training at `16K` context on `NVIDIA H100 80GB`:
- **Tier 1 (`No Activation Checkpointing`)**: `98.0 GB/GPU` activation memory (`144 GB` total -> immediate `CUDA OOM` on `80GB H100`).
- **Tier 2 (`Full Layer Activation Checkpointing`)**: `11.5 GB/GPU` activation memory, but `+33.3%` extra recomputation FLOPs drags effective MFU down to `42.5%`.
- **Tier 3 (`Selective Activation Checkpointing` — save Linear GEMMs, recompute Attention/SwiGLU)**: `21.0 GB/GPU` activation memory, `+4.5%` extra FLOPs, `53.6%` effective MFU.
- **Tier 4 (`Selective Checkpointing + Async PCIe Gen5 Pinned-DRAM Offload`, Recommended)**: `7.8 GB/GPU` HBM footprint (`-92.0%` vs Tier 1), `+3.2%` extra FLOPs, and `57.8%` effective MFU (`+36.0%` faster training than Full Checkpointing).

## Run & Verify

```bash
python experiments/activation_checkpointing_selective_recomputation_pcie_offload_mfu_selection/run.py
python experiments/activation_checkpointing_selective_recomputation_pcie_offload_mfu_selection/run.py --check
```
