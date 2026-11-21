# Enterprise Edge-Cloud Video VLM Cascade Architecture Selection (`Cloud VLM API` vs. `Edge Jetson/L40S + 2x2 Keyframe Sprite Grid + VLM Verifier`)

Reproducible analytical model comparing four enterprise video analytics and multimodal VLM architectures across 500 1080p RTSP camera streams (`WAN uplink bandwidth`, `complex SOP/safety event F1 score`, `false alarm rate`, and `monthly TCO`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_edge_cloud_video_vlm_jetson_l40s_cascade_bandwidth_tco_selection/run.py`
