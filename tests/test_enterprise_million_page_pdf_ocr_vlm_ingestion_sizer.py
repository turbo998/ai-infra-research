from experiments.enterprise_million_page_pdf_ocr_vlm_ingestion_sizer.run import (
    evaluate_pdf_ingestion_pipeline,
)


def test_enterprise_million_page_pdf_ocr_vlm_ingestion_sizer() -> None:
    res = evaluate_pdf_ingestion_pipeline(10_000_000)
    assert res["Deliverable_ID"] == "#658"
    opt = res["Comparison"]["Layout_Cropped_7B_VLM_FP8"]
    assert opt["pages_per_sec"] == 285
    assert opt["throughput_speedup"] == 6.79
    assert opt["table_teds_f1_pct"] == 96.2
    assert opt["cost_per_1k_pages_usd"] == 0.14
    assert opt["cost_saving_vs_full_vlm_pct"] == 92.4
