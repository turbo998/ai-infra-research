# Enterprise Air-Gapped / Branch On-Premises AI Appliance Sizing (`4x L40S` vs. `2x RTX PRO 6000 Blackwell 96GB` vs. `2x/4x H200 NVL PCIe`) (`#476`)

Reproducible analytical model comparing `4x L40S (48GB GDDR6)`, `2x RTX PRO 6000 Blackwell Server Edition (96GB GDDR7)`, `2x H200 NVL (141GB HBM3e + NVLink Bridge)`, and `4x H200 NVL` under branch-office air-cooled `< 3.2 kW` power envelopes.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_onprem_edge_ai_appliance_rtx_pro_6000_l40s_h200_nvl_selection/run.py --check`
