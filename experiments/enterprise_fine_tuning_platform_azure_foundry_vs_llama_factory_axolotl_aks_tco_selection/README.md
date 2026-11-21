# Experiment: Enterprise LLM Fine-Tuning & Domain Adaptation Platform Selection (`#410`)

Reproducible analytical model comparing `Full Fine-Tuning + Siloed Pods`, `Dedicated Managed Endpoints`, `Azure AI Foundry Serverless/Global Fine-Tuning`, and `AKS Tidal LLaMA-Factory/Axolotl + vLLM Multi-LoRA Serving` across 40 domain adapters.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_fine_tuning_platform_azure_foundry_vs_llama_factory_axolotl_aks_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_fine_tuning_platform_azure_foundry_vs_llama_factory_axolotl_aks_tco_selection/run.py --check`
