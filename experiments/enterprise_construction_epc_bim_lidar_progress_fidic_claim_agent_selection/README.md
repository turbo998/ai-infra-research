# Enterprise Construction & Infrastructure EPC Project Controls, `BIM IFC 4.3`, LiDAR Progress `EVM` & `FIDIC` Claim Agent (`R440`)

Deterministic evaluation suite comparing four megaproject construction/EPC project controls, 4D BIM clash pruning, LiDAR reality-capture Earned Value Management (`EVM`), and `FIDIC / AIA` contract claim architectures across an `$850M` active EPC capital program (`480,000` `IFC 4.3` elements, `320` monthly subcontractor pay apps, `$95,000/day` delay liquidated damages):

1. **Tier 1**: Manual Site Walk + 2D PDF Takeoff Excel + Subjective Pay Apps (`14.5%` subcontractor overbilling leakage, `11.8%` field rework, `22.0%` FIDIC time-bar claim forfeiture, `28.0` days deviation lag, `$840,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Subcontractor Pay-App & 2D Photo Reviewer (`16.2%` overbilling leakage, `10.5%` rework, `18.5%` claim forfeiture, `$910,000/mo` loss)
3. **Tier 3**: Static 3D BIM Clash Checker Without LiDAR Alignment or CPM Float AST (`6.8%` overbilling leakage, `4.9%` rework, `9.2%` claim forfeiture, `7.0` days lag, `$380,000/mo` loss)
4. **Tier 4**: `BIM IFC 4.3` Semantic Graph + Terrestrial/Drone LiDAR `ICP` 3D Voxel Progress Verifier + `Primavera P6` Critical-Path `TIA` + `FIDIC Sub-Clause 20.2` 28-Day Time-Bar `AST` Agent (`0.0%` overbilling leakage, `0.4%` rework, `0.0%` claim forfeiture, `0.5` days deviation lag, `$72,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_construction_epc_bim_lidar_progress_fidic_claim_agent_selection/run.py
python experiments/enterprise_construction_epc_bim_lidar_progress_fidic_claim_agent_selection/run.py --check
python -m unittest tests/test_enterprise_construction_epc_bim_lidar_progress_fidic_claim_agent_selection.py
```
