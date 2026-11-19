# Enterprise Multimodal Video & Visual Asset Understanding, Keyframe Indexing & Search Selection (`R491`)

Deterministic evaluation suite comparing four enterprise multimodal video understanding, keyframe indexing, and Video-RAG architectures across `50,000` monthly ingested video hours (`180,000,000` raw 1-FPS frames/month across meetings, training, retail, and industrial video):

1. **Tier 1**: Audio Transcript (`Whisper ASR`) Only (`18.0%` visual/screen-share recall, `34.5%` temporal IoU, `$360/1K hrs`, `$142,000/mo` TCO incl. manual video review rework)
2. **Tier 2**: Uniform `1 FPS` Frame Extraction + Frontier VLM on Every Frame (`95.8%` visual recall, `$14,400/1K hrs`, `$720,000/mo` TCO)
3. **Tier 3**: Scene-Change Keyframes + `CLIP / SigLIP` Visual Embedding Only (`71.5%` visual recall, `76.0%` temporal IoU, `$680/1K hrs`, `$68,000/mo` TCO)
4. **Tier 4**: Scene-Change + `pHash` Deduplication (`-95%` Frames) + Tri-Modal `Whisper + OCR + SigLIP` + Selective 8-Frame `Qwen2.5-VL / Video AI` (`96.8%` visual recall, `94.5%` temporal IoU, `$920/1K hrs`, `460ms` P95 search latency, `$46,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_multimodal_video_image_understanding_indexing_rag_selection/run.py
python experiments/enterprise_multimodal_video_image_understanding_indexing_rag_selection/run.py --check
python -m unittest tests/test_enterprise_multimodal_video_image_understanding_indexing_rag_selection.py
```
