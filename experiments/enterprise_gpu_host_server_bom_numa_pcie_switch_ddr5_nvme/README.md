# `#657` AI GPU Server Host BoM Sizer (`4× PCIe Gen5 Switch + 24-Channel DDR5 >= 2× HBM + 8× Gen5 NVMe`)

Reproducible analytical verification for `#657`:
- **Chinese FAQ**: [`../../docs/components/enterprise-gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-faq.md`](../../docs/components/enterprise-gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-faq.en.md`](../../docs/components/enterprise-gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-lab.html`](../../docs/h5/gpu-host-server-bom-numa-pcie-switch-ddr5-nvme-lab.html)

```bash
python experiments/enterprise_gpu_host_server_bom_numa_pcie_switch_ddr5_nvme/run.py
pytest tests/test_enterprise_gpu_host_server_bom_numa_pcie_switch_ddr5_nvme.py -q
```
