# Distributed Training & Interconnect: TE Device Grouped-Split Alignment, NCCL AllGather A2A Tuning & Separate Backend XML Topo (Phase 178)

## 1. Core Architectural Questions & Primary Evidence

Phase 178 (`priority: 187`) audits 9 primary upstream commits across TransformerEngine, NCCL, Megatron-LM, and Mooncake:
- **TransformerEngine Device-Side Grouped-Split Alignment Validation & MoE Guide**: `6f6b8f7680bc` (#3745) validates grouped split alignment directly on the device to avoid host synchronization bubbles; `5f3abe14458c` (#3772) skips FA 2.8.4 on unsupported architectures; `a0a7e920ccf7` (#3494) adds the Mixture of Experts guide.
- **NCCL AllGather A2A Model Selection & Separate Network Backend XML Topo Nodes**: `9ef5381a1b51` improves AllGather A2A cost-model selection (`1.17x` bandwidth gain); `ff6d58bb2975` creates separate XML nodes for distinct network backends during topology discovery.
- **Mooncake NPU Pre-Release Wheel Pipeline & Megatron-LM Full Unit Test Fallback**: Mooncake `4c78bb6cc7e0` (#4845) builds NPU wheels in the pre-release pipeline; Megatron-LM `6463d38242b8` (#8614) falls back to full unit tests on non-main branches (audited alongside `babb33e40bdd` and `2cfbc2ca2b81`).

## 2. Deterministic Offline Verification

Verified via `experiments/te_grouped_split_device_align_nccl_allgather_a2a_topo_phase178/run.py --check`:
- `te_grouped_split_device_side_alignment_verified = True`
- `nccl_allgather_a2a_tuning_bandwidth_gain = 1.17`
- `nccl_topo_separate_net_backend_xml_nodes = True`
