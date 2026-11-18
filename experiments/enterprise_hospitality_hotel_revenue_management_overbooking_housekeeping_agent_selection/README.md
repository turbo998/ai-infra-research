# Enterprise Hospitality, Hotel Chain & Theme Park Dynamic Revenue Management (`RMS`), Overbooking & Housekeeping Agent (`R439`)

Deterministic evaluation suite comparing four hospitality revenue management (`RMS`), overbooking walk-risk optimization, group block wash forecasting, and housekeeping dispatch architectures across a `120-property` hotel portfolio (`36,000` rooms/keys, `24,000` daily check-ins/turnovers, `$210` baseline ADR):

1. **Tier 1**: Static Seasonal BAR Tiers + Flat 10% Overbooking + FIFO Floor Housekeeping (`0.0%` RevPAR lift, `8.4%` walk/spoilage rate, `29.5%` early check-in lobby wait breach, `$540,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Competitor Rate Matcher (`-3.2%` RevPAR race-to-the-bottom, `11.2%` walk/spoilage, `27.0%` wait breach, `$635,000/mo` loss)
3. **Tier 3**: Standard Censored Occupancy Regression (`+4.8%` RevPAR lift, `4.1%` walk/spoilage, `16.5%` wait breach, `$265,000/mo` loss)
4. **Tier 4**: `EM`-Detruncated Network Length-of-Stay (`LOS`) Bid-Price `DP` + Binomial Asymmetric Walk-Cost `Newsvendor` Solver + Mobile-ETA Synced Housekeeping `MILP` (`+14.6%` RevPAR lift, `0.3%` walk/spoilage, `0.8%` lobby wait breach, `2.1%` group wash error, `$48,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection/run.py
python experiments/enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection/run.py --check
python -m unittest tests/test_enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection.py
```
