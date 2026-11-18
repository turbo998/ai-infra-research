# Enterprise Steel, Cement & Heavy Process Metallurgy (`BOF / EAF / Rotary Kiln`) Alloy `MILP`, Refractory Wear & Kiln `MPC` Agent (`R448`)

Deterministic evaluation suite comparing four steelmaking ladle furnace (`LF`) ferroalloy addition optimization, refractory lining wear monitoring, and cement rotary-kiln burning-zone control architectures across a `4.2M tonnes/year` plant (`1,400` 250-ton steel heats/month, `$8.5M/month` ferroalloy & kiln fuel spend):

1. **Tier 1**: Manual Melter Aim-Midpoint Recipe + Fixed 75-Heat Ladle Reline (`14.8%` ferroalloy overconsumption waste, `3.8%` off-spec heat downgrade, `11.5%` kiln fuel overburn, `$1,180,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Spectrometer & Kiln Operator Chatbot (`16.5%` alloy waste, `8.4%` off-spec heat downgrade from trace carbon/phosphorus impurities, `$1,390,000/mo` loss)
3. **Tier 3**: Unconstrained Black-Box Regression (`5.4%` alloy waste, `4.9%` off-spec heat downgrade from targeting minimum spec line at 50% quantile, `$520,000/mo` loss)
4. **Tier 4**: `99.5%` Chance-Constrained Ferroalloy Blend `MILP` (Slag `FeO` Recovery) + 3D Laser/`16-Bit RJPEG` Refractory Twin + Rotary-Kiln `Free-Lime / NOx MPC` Agent (`1.1%` alloy waste, `0.0%` off-spec downgrade, `1.2%` kiln fuel overburn, `1.8%` refractory waste, `$112,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_steel_cement_metallurgy_alloy_milp_kiln_mpc_agent_selection/run.py
python experiments/enterprise_steel_cement_metallurgy_alloy_milp_kiln_mpc_agent_selection/run.py --check
python -m unittest tests/test_enterprise_steel_cement_metallurgy_alloy_milp_kiln_mpc_agent_selection.py
```
