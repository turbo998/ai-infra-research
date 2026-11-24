# Experiment #594: Enterprise GPU Cold-Start Acceleration & Elastic Autoscaling Buffer Sizer

Reproduces the `29.1x` cold-start speedup (`17.5s` vs. `510.0s` for `70B-FP8`) via `Nydus P2P + Run:ai Model Streamer + Warm TorchInductor Cache`, reducing required surge buffer GPUs on a `32× H100` base fleet from `+20` (`52` total) to `+4` (`36` total, saving `$36,800/mo` / `-30.8%`) (`docs/components/enterprise-gpu-cold-start-model-streamer-p2p-autoscaling-sizing-faq.md`).
