from pathlib import Path
from experiments.enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection.run import build_results


def test_enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection"
    engines = res["index_engines"]
    assert len(engines) == 4
    assert engines[2]["dram_footprint_1b_vectors_tb"] < engines[0]["dram_footprint_1b_vectors_tb"] * 0.05
    assert engines[3]["index_build_hours_1b"] < engines[0]["index_build_hours_1b"] * 0.10
    assert engines[2]["annual_infra_tco_1b_usd"] < engines[0]["annual_infra_tco_1b_usd"] * 0.10

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-rag-vector-search-hnsw-diskann-cuvs-memory-tco-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-rag-vector-search-hnsw-diskann-cuvs-memory-tco-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/rag-vector-hnsw-diskann-cuvs-memory-tco-lab.html").exists()
