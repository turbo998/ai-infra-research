from pathlib import Path
from experiments.enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection.run import build_results


def test_enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection"
    modes = res["cni_modes"]
    assert len(modes) == 4
    assert modes[3]["achieved_nccl_busbw_gbps_400g"] > modes[0]["achieved_nccl_busbw_gbps_400g"] * 10
    assert modes[3]["multi_tenant_network_namespace_isolation"] is True

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-k8s-ai-cni-multus-sriov-rdma-network-operator-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-k8s-ai-cni-multus-sriov-rdma-network-operator-selection-faq.en.md").exists()
    assert (root / "docs/h5/k8s-ai-cni-multus-sriov-rdma-operator-lab.html").exists()
