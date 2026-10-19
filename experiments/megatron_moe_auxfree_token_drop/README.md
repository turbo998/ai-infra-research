# Megatron-Core MoE Aux-Loss-Free Routing (`expert_bias`), Quantile Balancing & Token Dropping Simulator

Deterministic CPU/offline simulator grounded in `NVIDIA/Megatron-LM` (`commit e8a302ac5bfeb98c6701020cc6cba977d05ab1d4`):
- `megatron/core/transformer/moe/router.py` (`TopKRouter`, `expert_bias`, `qb_beta`)
- `megatron/core/transformer/moe/moe_utils.py` (`get_updated_expert_bias`, `qb_dual_update`, `get_capacity`, `apply_router_token_dropping`)

## Usage

```bash
python experiments/megatron_moe_auxfree_token_drop/run.py
python experiments/megatron_moe_auxfree_token_drop/run.py --check
```
