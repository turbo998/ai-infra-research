# Enterprise Kubernetes GPU Orchestration (`Kueue TAS + DRA`), `KEDA` KV/Queue Autoscaling & Fast GPU Cold-Start (`Run:ai Streamer / GDS / Sleep Mode`) Model (`#452`)

Reproducible analytical model comparing Vanilla K8s Device Plugin + `GPU_UTIL` HPA, `KEDA` + local NVMe, `Kueue TAS + DRA + GDS Streamer`, and `Kueue TAS + DRA + KEDA + Level-1 Sleep Wakeup (2.8s)`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection/run.py --check`
