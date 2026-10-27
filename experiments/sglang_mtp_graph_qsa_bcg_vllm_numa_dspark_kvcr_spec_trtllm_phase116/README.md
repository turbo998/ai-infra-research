# SGLang In-Graph MTP Decode / Zero-Copy QSA KV Gather / vLLM NUMA & GLM5.3 FP8 Draft KV / KVCR Prefix Recency & TRT-LLM GH200 Model (Phase 116)

Reproducible CPU/offline analytical model (`evidence_mode: own-offline`) for Phase 116 Track B (`priority: 125`):
- SGLang Qwen3.5 MTP in-graph `small_field_copy.cuh` (`b74fd392`), QSA prefill generated KV gather D2D removal (`148bea73`), Breakable CUDA Graph `on_capture` (`010eb729`), and AMD mono-kernel model-type spec registry (`5044779f`)
- vLLM best-effort NUMA binding (`4d95f222`), GLM5.3 `dspark` FP8 draft KV cache saving `5.38 GB/rank` (`c6f6e6a9`), KVCR prefix-aligned cache recency (`e3883b7a`), and Speculative Decode grammar-invalid draft preservation (`c4f2bb4a`)
- TensorRT-LLM GH200 default model storage restoration (`e805aac3`) and comm op-matrix installed package import isolation (`8c08d34d`)
