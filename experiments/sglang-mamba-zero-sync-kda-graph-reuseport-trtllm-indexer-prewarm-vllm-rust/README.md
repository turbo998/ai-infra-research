# SGLang Zero-Sync Mamba Extend, KDA Prefill Graph & `_ReusePortMultiprocess`, TRT-LLM Indexer Prewarm, and vLLM Rust `PromptFillsContext` Experiment

Deterministic Python 3.12 standard-library verification for Phase 50 Topic 2 (`sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust`):

- **`sglang#45472` (`740d5f9f9e`), `#45473` (`f06d1bb1f4`), `#45487` (`a3d00eff58`), `#45482` (`7e5256d1ba`)**: Simulates `torch.isin` sync threshold `P >= 10 * N**0.145` vs zero-sync `mamba_ping_pong_empty: list[bool]` + `searchsorted`, `FlashInferKDAPrefillGraph` (`RecurrentKDAPrefillWrapper`), TBO `contains_last_prefill_chunk` preservation, and `_ReusePortMultiprocess` (`1` reserved parent + `N` worker sockets).
- **`TensorRT-LLM#20734` (`370fe4c509`)**: Verifies `_prewarm_cute_dsl_indexer_q` PP stage filtering (`wq_b.weight is not None`) and `(4, 8, 16)` token prewarming across all `3` `_fallback_tactic()` branches.
- **`vllm#57393` (`4a3ee9a0f6`) & `#62822`/`#62825`/`#62416`/`#62824` (`e0ca7d9cc1`)**: Verifies Cosmos3-Edge ModelOpt state-dict cleanup and Rust `PromptFillsContext` (`Ordering::Equal`, `param="input_tokens"`) vs `PromptTooLong` (`Ordering::Greater`).

```powershell
python experiments\sglang-mamba-zero-sync-kda-graph-reuseport-trtllm-indexer-prewarm-vllm-rust\run.py --check
```
