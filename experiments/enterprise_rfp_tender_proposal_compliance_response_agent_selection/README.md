# Enterprise RFP/Tender Proposal Agent: Mandatory Star-Clause AST, Version-Gated Truth Graph & Certificate Gate Evaluator

Deterministic evaluator comparing four enterprise RFP/RFI tender response and proposal generation architectures across 320 monthly enterprise tenders, 160,000 technical/security compliance items, and 7,300 product feature and qualification records:
- **Tier 1**: Manual Presales Copy-Paste from Old Proposals + Manual Compliance Check (`89.5%` matrix accuracy, `2.8%` star-clause disqualification / overpromise rate, `76.0 h` turnaround, `$290,000/mo`)
- **Tier 2**: Prompt-Only LLM Auto-Answering `"Fully Compliant"` over Unfiltered Old Proposals (`64.0%` accuracy, `14.5%` disqualification / false-GA promise rate, `1.2 h` turnaround, `$440,000/mo`)
- **Tier 3**: Flat Vector Search over Historical Q&A Bank + Copilot Drafting (`79.8%` accuracy, `5.4%` stale-spec / expired-case error rate, `14.0 h` turnaround, `$145,000/mo`)
- **Tier 4**: Mandatory `★` Star-Clause AST + Version-Gated `GA vs Roadmap` Product Truth Graph + Certificate/Contract SQL Gate + SME Sign-Off Agent (`98.2%` accuracy, `0.0%` disqualification / false-promise rate, `1.5 h` turnaround, `$42,000/mo`)

## Usage
```bash
python experiments/enterprise_rfp_tender_proposal_compliance_response_agent_selection/run.py
python experiments/enterprise_rfp_tender_proposal_compliance_response_agent_selection/run.py --check
```
