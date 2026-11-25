from experiments.enterprise_gpu_host_server_bom_numa_pcie_switch_ddr5_nvme.run import (
    compute_ddr5_bandwidth_gbs,
    evaluate_gpu_host_server_bom,
)


def test_enterprise_gpu_host_server_bom_numa_pcie_switch_ddr5_nvme() -> None:
    res = evaluate_gpu_host_server_bom()
    assert res["Deliverable_ID"] == "#657"
    assert compute_ddr5_bandwidth_gbs(24, 5600, 0.85) == 914
    assert res["Memory_Bandwidth_Comparison"]["l2_kv_32k_restore_ms_24ch"] == 97
    assert res["PCIe_Gen5_Topology"]["pix_rdma_busbw_gbs"] == 382
    assert res["Local_NVMe_Pool"]["aggregate_read_gbs"] == 56.0
