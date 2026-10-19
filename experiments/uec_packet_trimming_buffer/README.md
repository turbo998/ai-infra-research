# UEC Packet Trimming & SONiC Dynamic-Alpha Switch Buffer Sizing Simulator

Deterministic model comparing SONiC/Broadcom SAI RoCEv2 (`PFC + DCQCN`, `dynamic_th` $\alpha=2^k$, `XOFF` lossless headroom reserve) against Ultra Ethernet Consortium (UEC 1.0) `Packet Trimming` (`64 B` header-only `CP` notification in `0.5 RTT`) + `EVB` path spraying under 800 Gbps MoE All-to-All and Checkpoint Incast bursts.

## Run

```bash
python experiments/uec_packet_trimming_buffer/run.py
python experiments/uec_packet_trimming_buffer/run.py --check
```
