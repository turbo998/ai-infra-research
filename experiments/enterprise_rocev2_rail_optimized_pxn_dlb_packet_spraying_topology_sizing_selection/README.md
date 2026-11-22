# Enterprise 8-Rail `RoCEv2` Topology (`Rail-Optimized` vs. `Fat-Tree`), `NCCL PXN` & `DLB / Packet Spraying` Sizing Model (`#458`)

Reproducible analytical model comparing oversubscribed server-ToR Fat-Tree, non-blocking static-ECMP Fat-Tree, Rail-Optimized Leaf-Spine with `NCCL PXN` + Flowlet `DLB`, and Rail-Optimized + Per-Packet Spraying with Out-of-Order Direct Data Placement (`Spectrum-X / UEC`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection/run.py --check`
