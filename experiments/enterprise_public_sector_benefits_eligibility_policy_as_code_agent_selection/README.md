# Enterprise Public-Sector Benefits Eligibility, `Catala / OpenFisca` Policy-as-Code & Audit Agent (`R449`)

Deterministic evaluation suite comparing four public-sector/sovereign social-benefits eligibility (`SNAP / Medicaid / Unemployment / Tax Credits`), statutory Policy-as-Code (`Catala / OpenFisca`), and administrative due-process audit architectures across `220,000` monthly household applications/recertifications (`8,400` statutory subclauses, `$145M/month` benefits disbursed):

1. **Tier 1**: Manual Caseworker Paperwork Queue + Legacy Mainframe Tables (`11.4%` improper payment & wrongful denial rate, `9.8%` deduction math error, `28.5%` appeal reversal rate, `42.0` days backlog, `$920,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Policy Manual Reader & Calculator (`19.6%` improper payment/wrongful denial, `24.2%` statutory math error, `39.0%` appeal reversal, `$1,240,000/mo` loss)
3. **Tier 3**: Policy Manual Vector RAG Without Rules-as-Code Compiler (`6.9%` improper payment/denial, `7.5%` math error, `16.4%` appeal reversal, `2.0` days lag, `$440,000/mo` loss)
4. **Tier 4**: Multimodal Paystub/Lease Fact Extractor + `Catala / OpenFisca` Policy-as-Code `AST` + State Wage/IRS Cross-Match + `Goldberg v. Kelly` Proof-Tree Notice Agent (`0.2%` improper payment/denial, `0.0%` statutory math error, `0.5%` appeal reversal, `0.1` days / 2.4h turnaround, `$82,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection/run.py
python experiments/enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection/run.py --check
python -m unittest tests/test_enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection.py
```
