# Enterprise Maritime Shipping Weather-Routing (`IMO CII / EU ETS`), Port Congestion & Demurrage Laytime Agent (`R438`)

Deterministic evaluation suite comparing four maritime fleet routing, Just-in-Time (`JIT`) port berth arrival, and charterparty Demurrage & Detention (`D&D`) laytime audit architectures across a `65-vessel` fleet (`850,000 TEU/yr`, `280` port calls/month, `$720/MT` VLSFO + EU ETS carbon price):

1. **Tier 1**: Fixed Great-Circle Route + Hurry-Up-and-Wait Speed + Manual `SOF` Laytime Excel (`18.2%` bunker/ETS overburn, `32.0h` berth ETA MAE, `24.5%` demurrage overpayment, `$780,000/mo` waste)
2. **Tier 2**: Prompt-Only General LLM Charterparty & Port-News Assistant (`16.5%` overburn, `28.5h` ETA MAE, `29.0%` demurrage overpayment, `$820,000/mo` waste)
3. **Tier 3**: Shortest-Distance Weather Router Without JIT Port Sync or Laytime AST (`8.4%` overburn, `11.5h` ETA MAE, `14.2%` demurrage overpayment, `$365,000/mo` waste)
4. **Tier 4**: 3D Metocean Wave-Resistance Isochrone + `AIS` Port-Queue `JIT` Cubic Speed Optimizer + Deterministic `BIMCO Laytime` (`WWD / SHINC / Once on Demurrage`) Interval State Machine (`1.2%` overburn, `1.8h` berth ETA MAE, `0.0%` demurrage overpayment, `0.0%` IMO CII downgrade risk, `$68,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/run.py
python experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/run.py --check
python -m unittest tests/test_enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection.py
```
