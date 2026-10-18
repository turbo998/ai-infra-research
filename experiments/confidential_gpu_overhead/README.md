# Confidential GPU (CC-On / PPCIE) Attestation, Topology & Bounce-Buffer Overhead Model

- **Measurement type**: `own-offline` (deterministic CPU analytical and protocol verification model derived from `NVIDIA/nvtrust` source code).
- **Clear Boundary**: This experiment does not run on physical Hopper/Blackwell TEE hardware or call NVML/NSCQ drivers.

## What It Models

1. **RIM vs Runtime SPDM Measurement Verification (`verifier/verifier.py`)**:
   - Models Driver + VBIOS Reference Integrity Manifest (RIM) index merging, `InvalidMeasurementIndexError` when both RIMs declare an active measurement at the same index, conditional skipping of index `35` when `OPAQUE_FIELD_ID_NVDEC0_STATUS == DISABLED`, and digest size/value matching.
2. **PPCIE 8-GPU + 4-NVSwitch Bidirectional PDI Topology Validation (`validate_topology.py`, `verification.py`)**:
   - Validates `x-nvidia-gpu-switch-pdis`, `x-nvidia-switch-pdi`, and `x-nvidia-switch-gpu-pdis` claims across `8` GPUs and `4` NVSwitches, ignoring `DISABLED_PDI = "0000000000000000"` and enforcing fail-closed `set_gpu_ready_state(False)` upon any rogue switch or GPU PDI mismatch.
3. **Confidential Computing Bounce-Buffer & AES-GCM Data-Plane Overhead**:
   - Compares `Standard Non-CC` (`104.174 ms` TTFT with `1 GiB` host KV load, `11.6678 ms` TPOT), `Single-GPU CC-On SW Bounce Buffer` (`212.3079 ms` TTFT, `27.6029 ms` TPOT due to host-routed P2P), `CC-On Pinned + Pipelined CE AES-GCM` (`143.8107 ms` TTFT, `18.8513 ms` TPOT), and `8-GPU + 4-NVSwitch PPCIE` (`143.5831 ms` TTFT, `11.6811 ms` TPOT — restoring `99.89%` of non-CC multi-GPU decode throughput).

## Usage

```bash
python experiments/confidential_gpu_overhead/run.py
python experiments/confidential_gpu_overhead/run.py --check
```
