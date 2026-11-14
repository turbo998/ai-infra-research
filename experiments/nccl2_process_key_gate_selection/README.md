# NCCL2 TCP TLS process-key capability gates

Original metadata-only sequential configuration automaton with fixed synthetic labels, byte-length/NUL fields and explicitly injected API outcomes. Never accepts real keys. Independent four-state declarative oracle covers2,801traces.

```powershell
python -m experiments.nccl2_process_key_gate_selection.run --measure
python -m experiments.nccl2_process_key_gate_selection.run --check
python -m unittest discover -s tests -p test_nccl2_process_key_gate_selection.py -q
```

Source-verified Nov14feature boundaries include compiled NCCL2.32.3+, no ROCm, process lifetime, first nccl2group, environment precedence and at-least32bytes. UI length0..256is an original bound, not native maximum. TLS build/runtime metadata does not prove successful communication; exact native failure timing is not predicted.

No credentials, crypto, native NCCL/PyTorch/GPU, key rotation, concurrent construction, actual communicator registry or complete data-plane security certification. CPU timings compare synthetic label agreement, excluding metadata collection and native TLS. Static60-solid latching cabinet is original teaching art; visual browser inspection was not performed. [English FAQ](../../docs/components/nccl2-tcp-tls-process-key-capability-gates-selection-faq.en.md) / [中文](../../docs/components/nccl2-tcp-tls-process-key-capability-gates-selection-faq.md).
