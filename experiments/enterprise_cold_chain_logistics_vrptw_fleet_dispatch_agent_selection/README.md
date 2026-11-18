# Enterprise Cold-Chain Logistics Fleet Dispatch, Multi-Compartment OR-Tools VRPTW & Reefer Rescue Agent Evaluator

Deterministic evaluator comparing four cold-chain logistics and refrigerated fleet dispatch architectures across 650 multi-compartment reefer trucks, 18,000 daily hospital/retail delivery drops, and 12 metro hubs:
- **Tier 1**: Manual Fixed-Zone Dispatch + Post-Trip USB Temperature Logger Review (`78.5%` on-time in-window delivery, `3.4%` thermal spoilage rate, `29.0%` deadhead mileage, `$360,000/mo`)
- **Tier 2**: Prompt-Only LLM Directly Sequencing 500 Delivery Stops in Text Space (`56.0%` on-time rate, `4.8%` spoilage rate, `38.5%` deadhead mileage, `$520,000/mo`)
- **Tier 3**: Static Morning VRP Solver + Reactive `+8°C` Threshold SMS Alarm (`88.2%` on-time rate, `1.6%` spoilage rate, `18.4%` deadhead mileage, `$165,000/mo`)
- **Tier 4**: LLM Dock-Constraint Compiler + Dynamic Multi-Compartment `OR-Tools / VROOM VRPTW` Solver + Reefer Thermal-Inertia (`dT/dt`) Mid-Route Rescue Agent (`98.4%` on-time rate, `0.0%` cargo spoilage rate, `11.2%` deadhead mileage, `$52,000/mo`)

## Usage
```bash
python experiments/enterprise_cold_chain_logistics_vrptw_fleet_dispatch_agent_selection/run.py
python experiments/enterprise_cold_chain_logistics_vrptw_fleet_dispatch_agent_selection/run.py --check
```
