# Track B Experiment (Phase 109): SGLang DSV4.1 KV Capacity & Disagg Roles, vLLM NVFP4 MLA / QSA / GLM-5.3 KDA / Hetero TP, and TRT-LLM Agg Helix & GDN Replay

Deterministic Python 3.12 standard-library model of Phase 109 (`priority: 118`) Track B runtime commits:

- `sgl-project/sglang` `dea05ca16e` (`#46718`), `7f8ad67373` (`#46144`), `20ff40cb84` (`#41275`)
- `vllm-project/vllm` `741584f8ca` (`#58744`), `3bc521d9e6` (`#61543`), `90309f6b74` (`#60490`), `de096755ad` (`#60991`), `115b59a2da` (`#63857`)
- `NVIDIA/TensorRT-LLM` `4b027a381d` (`#20759`), `ae91047ac3` (`#20008`)

## Run

```bash
python experiments/sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109/run.py
python experiments/sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109/run.py --check
```
