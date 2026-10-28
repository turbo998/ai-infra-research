# Phase 124 Track B Experiment: SGLang ComfyUI / StaticKVStorage / Routed Draft KV, vLLM P2P Retain / SWA Offload / DSA Indexer / NIXL Push & TensorRT-LLM 64-Bit AttnRes Stride

Deterministic verification of:
- `sgl-project/sglang` commits `4517a5ff1b`, `b1c6945659`, `146378629d`, `53592ea5ff`, `ca8184e13a`, `240862923e`, `eb79ecca11`, `14fea75124`
- `vllm-project/vllm` commits `a1cfd2c1b5`, `77f58e5304`, `317081fb6d`, `066d979934`, `1cb31be702`, `5118d990fb`
- `NVIDIA/TensorRT-LLM` commits `e2b1544160`, `99466ea9ad`, `2fe91cdfc9`

## Run
```bash
python experiments/sglang_comfyui_kv_storage_routed_views_vllm_p2p_retain_swa_dsa_nixl_rocm_trtllm_phase124/run.py
python experiments/sglang_comfyui_kv_storage_routed_views_vllm_p2p_retain_swa_dsa_nixl_rocm_trtllm_phase124/run.py --check
```
