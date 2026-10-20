# FlashInfer Cake Grouped MXFP8 In-Tree Loader, vLLM DeepSeek-V4 EPLB Redundant Slots & Kimi-K3 GB200 AgentX Profile

Deterministic Python 3.12 standard-library model verifying:
1. FlashInfer PR #6868 (`d8ddf1a367`): in-tree `CakeProgram` loader for `cake_grouped_mxfp8_quantize` (`d8aa920df5a6252450bd` BF16 and `f84d4eab2905894ac4c2` FP16 on `sm_100a` and `sm_103a`), internal 128-tile padding, and `[B, pm_tiles, pk_tiles, 32, 4, 4]` E8M0 scale layout.
2. vLLM PR #59611 (`9d4192aaa1`): DeepSeek-V4 `get_expert_mapping()` `num_redundant_experts` propagation and `EplbCheckExtension.check_eplb_replica_slots` pre-traffic verification.
3. vLLM PR #62761 (`56e67b46a3`): Kimi-K3 GB200 4-tray 16-GPU `TP8 x PP2 + DCP8` (`a2a`, `TRTLLM_RAGGED`, DSpark `K=4`) AgentX profile.
