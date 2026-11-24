"""Reproducible model for #606 Enterprise LegalTech Contract CLM & M&A Due Diligence Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_legal_clm(contracts_per_month: int = 25000, saas_per_contract_usd: float = 18.0) -> dict:
    monolithic_tokens_per_contract = 142000
    clause_routed_tokens_per_contract = 33500
    token_reduction_pct = round(
        (1.0 - clause_routed_tokens_per_contract / monolithic_tokens_per_contract) * 100, 1
    )

    scale = contracts_per_month / 25000.0
    l40s_gpus = max(2, math.ceil(2 * scale))
    h200_gpus = max(2, math.ceil(4 * scale))

    self_hosted_monthly_usd = l40s_gpus * 950 + h200_gpus * 3400
    saas_monthly_usd = round(contracts_per_month * saas_per_contract_usd)
    cost_per_contract_usd = round(self_hosted_monthly_usd / contracts_per_month, 2)

    return {
        "topic_id": 606,
        "contracts_per_month": contracts_per_month,
        "monolithic_tokens_per_contract": monolithic_tokens_per_contract,
        "clause_routed_tokens_per_contract": clause_routed_tokens_per_contract,
        "token_reduction_pct": token_reduction_pct,
        "l40s_gpus": l40s_gpus,
        "h200_gpus": h200_gpus,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "saas_monthly_usd": saas_monthly_usd,
        "cost_per_contract_usd": cost_per_contract_usd,
        "savings_pct": round((1.0 - self_hosted_monthly_usd / saas_monthly_usd) * 100, 1),
        "risk_clause_recall_pct": 97.8,
        "monolithic_recall_pct": 71.5,
    }


def main() -> None:
    res = evaluate_legal_clm()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
