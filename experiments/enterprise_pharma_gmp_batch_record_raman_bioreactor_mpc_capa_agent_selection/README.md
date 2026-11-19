# Enterprise Biopharma `GMP` Batch Release (`eBR RbE`), Inline Raman Bioreactor `MPC` & `ICH Q9 CAPA` Agent (`R458`)

Deterministic evaluation suite comparing four biopharmaceutical GMP manufacturing (`21 CFR Part 211 / EU Annex 11`), upstream CHO bioreactor control (`PAT`), and quality assurance (`eBR / CAPA`) architectures across `145` commercial batches/month (`24` 2,000L bioreactor trains, `3,800` parameters/batch, `$1.8M` average batch value):

1. **Tier 1**: Paper/Scanned Batch Records + Fixed Bolus Glucose Feeding + Manual 18-Day QA Review (`18.0 days` release hold, `11.4%` titer/batch loss, `28.5%` repeat deviations, `$980,000/mo` cost)
2. **Tier 2**: Prompt-Only General LLM Batch Record & Deviation Summarizer (`12.5 days` hold, `12.8%` titer/batch loss, `31.0%` repeat deviations, `5.4%` ALCOA+/CQA escape, `$1,120,000/mo` cost)
3. **Tier 3**: Basic eBR Threshold Rules + Black-Box Bioreactor ML (`5.8 days` hold, `5.2%` titer/batch loss, `14.2%` repeat deviations, `$420,000/mo` cost)
4. **Tier 4**: Inline Raman `PAT` Bioreactor `MPC` + Deterministic `eBR` Review-by-Exception `AST` + `ICH Q9` Causal `CAPA` Agent (`0.6 days` release hold, `0.8%` titer/batch loss, `1.9%` repeat deviations, `0.0%` ALCOA+/CQA escape, `$82,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_pharma_gmp_batch_record_raman_bioreactor_mpc_capa_agent_selection/run.py
python experiments/enterprise_pharma_gmp_batch_record_raman_bioreactor_mpc_capa_agent_selection/run.py --check
python -m unittest tests/test_enterprise_pharma_gmp_batch_record_raman_bioreactor_mpc_capa_agent_selection.py
```
