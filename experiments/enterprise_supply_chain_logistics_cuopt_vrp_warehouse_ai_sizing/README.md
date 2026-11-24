# Experiment #611: Enterprise Supply Chain & Logistics AI (LLM Constraint Compiler + cuOpt VRP) Sizer

Reproduces the `2.2s` dynamic 15,000-stop re-routing pipeline (`0.4s` LLM constraint parsing + `1.8s` `NVIDIA cuOpt` GPU VRP solve, `1,227.3x` faster than `2,700s` CPU MIP) and `$13,000/mo` (`4× L40S + 4× H100`) fleet sizing delivering `$11.20M/yr` fuel savings across `5,000` trucks (`docs/components/enterprise-supply-chain-logistics-cuopt-vrp-warehouse-ai-sizing-faq.md`).
