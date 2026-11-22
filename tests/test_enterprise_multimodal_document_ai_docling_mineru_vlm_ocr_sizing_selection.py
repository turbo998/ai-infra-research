from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multimodal_document_ai_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_multimodal_document_ai_docling_mineru_vlm_ocr_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_multimodal_document_ai_docling_mineru_vlm_ocr_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[1]["gpus_for_10m_pages_per_day"] == 64
    assert tiers[3]["table_teds_accuracy_pct"] == 96.1
    assert tiers[3]["effective_pages_per_sec_per_gpu"] == 74.0
    assert tiers[3]["gpus_for_10m_pages_per_day"] == 2
