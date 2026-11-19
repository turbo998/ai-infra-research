# Enterprise Automotive Aftermarket Parts (`TSB/Croston`), 17-Digit `VIN` Collision `VLM` & Service-Bay `MILP` Agent (`R460`)

Deterministic evaluation suite comparing four automotive dealership network (`DMS`), aftermarket spare-parts inventory, collision photo estimation, and service-bay scheduling architectures across `180` dealerships + `4` regional PDCs (`145,000` OEM part SKUs, `2,800` daily repair orders):

1. **Tier 1**: DMS Min/Max Moving-Average Reorder + Manual Collision Tear-Down + FIFO Service Bays (`18.4%` VOR stockout, `21.6%` dead-stock write-off, `26.5%` supplement rework, `$840,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Photo Estimator & Free-Text Parts Ordering Bot (`15.8%` stockout, `31.2%` supplement rework, `9.6%` wrong-VIN part escape, `$950,000/mo` loss)
3. **Tier 3**: Standard Retail Gaussian Demand ML + 2D Photo Segmentation (`7.4%` stockout, `11.2%` dead-stock write-off, `12.8%` supplement rework, `$380,000/mo` loss)
4. **Tier 4**: `TSB` Intermittent-Demand Multi-Echelon Pooling + 17-Digit `VIN EPC VLM` + Skill/Lift Bay `MILP` Agent (`0.6%` VOR stockout, `1.8%` dead-stock write-off, `1.9%` supplement rework, `0.0%` wrong-VIN part escape, `$74,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/run.py
python experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/run.py --check
python -m unittest tests/test_enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection.py
```
