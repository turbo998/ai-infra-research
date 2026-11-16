# Round 303: PyTorch Elastic (`torchrun / rdzv`) `TCPStore` Barrier Storm & Zombie CUDA Context Recovery Selection

Deterministic evaluator comparing 4 4,096-GPU (`512` nodes) elastic rendezvous and fault recovery architectures across recovery latency (`s`), restart `CUDA OOM` failure rate (`%`), and cluster goodput (`%`).
