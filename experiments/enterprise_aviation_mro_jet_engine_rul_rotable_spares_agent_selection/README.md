# Enterprise Commercial Aviation `MRO`, Jet Engine `EGT` Margin `RUL`, Rotable Spares (`METRIC`) & `FAA Part 145` Agent (`R447`)

Deterministic evaluation suite comparing four commercial aviation Maintenance, Repair & Overhaul (`MRO`), turbofan engine Remaining Useful Life (`RUL`), rotable spare-parts pooling, and `FAA Part 145 / EASA` airworthiness sign-off architectures across a `280-aircraft` fleet (`620` turbofan engines, `$240M` rotable LRU pool, `$150,000/day` AOG cost):

1. **Tier 1**: Fixed Cycle Hard-Time Overhaul + Reactive AOG Charter Expediting (`18.5` AOG events/mo, `9.4%` IPC wrong-part dispatch, `21.0%` wasted engine green-time cycles, `2.8%` FAA 8130 audit escape, `$1,280,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Pilot Logbook (`PIREP`) & AMM/IPC Chatbot (`16.0` AOG/mo, `17.8%` IPC wrong-part dispatch on MSN effectivity blocks, `6.5%` audit escape, `$1,410,000/mo` loss)
3. **Tier 3**: Uncorrected Raw Sensor RUL Neural Net (`7.2` AOG/mo, `4.8%` wrong-part dispatch, `12.4%` wasted green time on hot-and-high takeoffs, `$590,000/mo` loss)
4. **Tier 4**: `ISA`-Corrected Gas-Path Thermodynamic Twin (`ΔEGT / ΔN2 / ΔFF`) + Multi-Echelon `Sherbrooke METRIC` Rotable Solver + `MSN/SB IPC Effectivity AST` + `FAA Form 8130-3` LLP Back-to-Birth Gate (`0.6` AOG/mo, `0.0%` wrong-part dispatch, `1.9%` wasted green time, `0.0%` airworthiness escape, `$142,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection/run.py
python experiments/enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection/run.py --check
python -m unittest tests/test_enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection.py
```
