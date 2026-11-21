# LLM Trillion-Token Data Curation & Deduplication Selection Experiment (`NVIDIA NeMo Curator + Ray Data` vs. `Datatrove` vs. `Apache Spark`)

Reproducible performance and cost model comparing 4 data curation architectures on a `20 TB` (`5 Trillion` raw tokens) pretraining/continual-SFT corpus across wall-clock curation hours, `MinHash LSH + SemDeDup` token pruning ratio, curation compute cost, and downstream training GPU savings.

## Run
```bash
python experiments/llm_data_curation_dedup_minhash_sem_dedup_nemo_curator_ray_data_selection/run.py
python experiments/llm_data_curation_dedup_minhash_sem_dedup_nemo_curator_ray_data_selection/run.py --check
```
