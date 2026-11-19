# Enterprise Commercial Real Estate (`CRE`), `ASC 842` Lease, `CAM` Audit & Workplace `CMMS` Agent (`R455`)

Deterministic evaluation suite comparing four corporate real estate (`CRE` / `IWMS`), `ASC 842 / IFRS 16` lease accounting, landlord `NNN CAM` audit, and facility `CMMS` architectures across a `450-site` portfolio (`18M sq ft`, `2,400` lease/amendment documents, `65,000` annual facility work orders):

1. **Tier 1**: Manual Lease Spreadsheets + Unaudited Landlord CAM Invoices + Static Desk Assignments (`14.2%` CAM/option loss, `6.4%` ASC 842 error, `38.5%` ghost-space waste, `$890,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM 120-Page Lease & Invoice Summarizer (`16.5%` CAM/option loss, `21.8%` ASC 842 error, `20.4%` warranty double-billing, `$1,020,000/mo` loss)
3. **Tier 3**: Standard OCR + Flat Vector Lease RAG (`5.8%` CAM/option loss, `2.9%` ASC 842 error, `19.4%` space waste, `$360,000/mo` loss)
4. **Tier 4**: Amendment-Supersession Lease `AST` + `ASC 842` Engine + Deterministic `CAM` Cap Auditor + Space/`CMMS MILP` Agent (`0.1%` CAM/option loss, `0.0%` ASC 842 error, `4.2%` space waste, `0.0%` warranty double-billing, `$78,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_commercial_real_estate_lease_asc842_space_cmms_agent_selection/run.py
python experiments/enterprise_commercial_real_estate_lease_asc842_space_cmms_agent_selection/run.py --check
python -m unittest tests/test_enterprise_commercial_real_estate_lease_asc842_space_cmms_agent_selection.py
```
