# Experiment #604: Enterprise Telecom & NOC AIOps 3-Stage Log Funnel & Topology RCA Sizer

Reproduces the `680,828:1` event compression funnel (`578,704` raw logs/s -> `420` `Drain3` anomaly bursts/s -> `0.85` CMDB topology-collapsed root incidents/s), enabling `4× H200` (`$15,200/mo`, `-97.2%` vs. `$544,000/mo` uncollapsed LLM routing) to achieve `95.6%` Top-1 RCA accuracy and `4.2-minute` MTTR (`docs/components/enterprise-telecom-aiops-log-funnel-topology-rca-sizing-faq.md`).
