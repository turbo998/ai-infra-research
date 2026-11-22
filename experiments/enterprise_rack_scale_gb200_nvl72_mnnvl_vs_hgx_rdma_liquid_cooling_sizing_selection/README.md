# Enterprise Rack-Scale `GB200 / GB300 NVL72` (`MNNVL` 72-GPU Copper Domain) vs. 8-GPU `HGX H200 / B200` + RDMA & `45kW/132kW` Liquid Cooling Sizing Model (`#455`)

Reproducible analytical model comparing multi-node H100 over 200G RoCE, single-node 8x H200 (air-cooled `10.5 kW/rack`), multi-node 8x B200, and Rack-Scale `GB200 NVL72` (`72-GPU` NVLink 5 `1,800 GB/s/GPU` copper backplane + `132 kW` Direct-to-Chip Liquid Cooling).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_rack_scale_gb200_nvl72_mnnvl_vs_hgx_rdma_liquid_cooling_sizing_selection/run.py --check`
