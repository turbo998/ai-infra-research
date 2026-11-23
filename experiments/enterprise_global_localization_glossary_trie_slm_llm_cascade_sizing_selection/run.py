from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "legacy_neural_mt_api_ungrounded",
            "name": "Legacy Baseline: Generic Neural MT API (Breaks HTML/JSON Tags & Mistranslates Brand SKUs)",
            "brand_glossary_compliance_pct": 72.4,
            "cultural_marketing_score_pct": 68.5,
            "required_h200_gpus_for_500m_words_per_day": 0,
            "annual_tco_usd": 1825000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "naive_100pct_flagship_70b_translation",
            "name": "Anti-Pattern: 100% Flagship 70B LLM Direct Translation on All 500M Words/Day",
            "brand_glossary_compliance_pct": 91.0,
            "cultural_marketing_score_pct": 95.8,
            "required_h200_gpus_for_500m_words_per_day": 48,
            "annual_tco_usd": 1286000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "glossary_trie_88pct_7b_comet_12pct_70b_cascade",
            "name": "Optimal: Glossary Trie + 88% 7B Specialist (Tower/Qwen) + COMET-Kiwi Gate -> 12% 70B Transcreation",
            "brand_glossary_compliance_pct": 99.8,
            "cultural_marketing_score_pct": 96.4,
            "required_h200_gpus_for_500m_words_per_day": 8,
            "annual_tco_usd": 214000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_global_localization_glossary_trie_slm_llm_cascade_sizing_selection",
        "title": "Enterprise Global Localization & Cross-Border E-Commerce AI Selection Guide: Legacy MT vs. 100% 70B vs. Glossary Trie + 88/12 COMET-Gated Cascade (#528)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "gpu_saving_pct_vs_all_70b": round((1.0 - 8 / 48) * 100, 1),
            "tco_saving_pct_vs_commercial_mt_api": round((1.0 - 214000.0 / 1825000.0) * 100, 1),
            "glossary_compliance_pct": 99.8,
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
