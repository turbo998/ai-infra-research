from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_global_localization_glossary_trie_cascade_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_global_localization_glossary_trie_slm_llm_cascade_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_global_localization_glossary_trie_slm_llm_cascade_sizing_selection"
    )
    assert data["summary_metrics"]["gpu_saving_pct_vs_all_70b"] == 83.3
    assert data["summary_metrics"]["tco_saving_pct_vs_commercial_mt_api"] == 88.3
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[2]["brand_glossary_compliance_pct"] == 99.8
    assert archs[2]["required_h200_gpus_for_500m_words_per_day"] == 8
