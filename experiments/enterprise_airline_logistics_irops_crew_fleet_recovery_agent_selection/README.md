# Enterprise Airline & Logistics Irregular Operations (IROPS) Fleet/Crew Recovery, FAR Part 117 & Column-Generation Evaluator

Deterministic evaluator comparing four airline and express-logistics Irregular Operations (`IROPS`) recovery architectures across 420 aircraft, 1,800 daily flights, and 14,000 pilots and cabin crew:
- **Tier 1**: Manual OCC Dispatcher Gantt Recovery (`62.0%` on-time recovery rate, `0.6%` crew legality violation rate, `135.0 min` solve time, `$680,000/mo`)
- **Tier 2**: Prompt-Only LLM Directly Reassigning Aircraft Tails & Crew Rosters (`54.0%` recovery rate, `21.5%` `FAR Part 117 / CCAR-121` crew duty violation rate, `3.5 min` solve time, `$920,000/mo`)
- **Tier 3**: Standalone Aircraft Swap MIP Solver + Text Manual RAG (`76.5%` recovery rate, `5.8%` downstream crew violation rate, `28.0 min` solve time, `$310,000/mo`)
- **Tier 4**: `NOTAM / Weather GDP` LLM Constraint Parser + Integrated Aircraft-Crew **Column Generation (`Branch-and-Price`)** Solver + Deterministic `FAR Part 117` Legality Gate Agent (`93.8%` recovery rate, `0.0%` crew legality violation rate, `4.2 min` solve time, `$115,000/mo`)

## Usage
```bash
python experiments/enterprise_airline_logistics_irops_crew_fleet_recovery_agent_selection/run.py
python experiments/enterprise_airline_logistics_irops_crew_fleet_recovery_agent_selection/run.py --check
```
