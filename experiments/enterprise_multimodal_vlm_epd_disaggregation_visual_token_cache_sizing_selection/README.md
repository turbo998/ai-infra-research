# Enterprise Multimodal VLM / Video-LLM Visual Token Sizing & 3-Stage `E-P-D` (`Encode-Prefill-Decode`) Disaggregation Model (`#451`)

Reproducible analytical model comparing Monolithic Colocated VLM serving, Colocated + Image Embedding Cache, 2-Stage PD Disaggregation, and 3-Stage Heterogeneous `E-P-D` Disaggregation (`L40S ViT Encode + H100 Prefill + H200 Decode + AnyRes Token Pruning`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_multimodal_vlm_epd_disaggregation_visual_token_cache_sizing_selection/run.py --check`
