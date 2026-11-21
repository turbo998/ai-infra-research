# Experiment: Single-Node 8-GPU HGX vs. Multi-Node RDMA vs. Rack-Scale Supernode (GB200/GB300 NVL72) Selection & TCO (`#402`)

Reproducible analytical model comparing four hardware cluster form factors (`Dual-Node 16x H100`, `Single-Node 8x H200`, `Multi-Node 32x HGX B200 + RDMA`, and `GB200/GB300 NVL72 Rack Supernode`) across Scale-Up domain radius, `EP=72` copper backplane bandwidth, rack power/cooling (`10.5 kW` air vs. `132 kW` DLC), and `$/1M Goodput tokens`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/single_node_hgx_vs_multinode_rdma_vs_rack_nvl72_tco_goodput_selection/run.py`
- **Verify**: `python experiments/single_node_hgx_vs_multinode_rdma_vs_rack_nvl72_tco_goodput_selection/run.py --check`
