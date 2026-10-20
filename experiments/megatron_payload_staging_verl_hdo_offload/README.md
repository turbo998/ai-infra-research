# Experiment: Megatron-LM DynamicInferenceEngine Async Payload Staging & Idle Rerun Strict Load + veRL HDO Optimizer Offload

## Purpose (研究目的 / Bilingual Overview)
- **中文**：复现并量化验证 `NVIDIA/Megatron-LM` `DynamicInferenceEngine` 将已完成请求的 `RequestPayloadStager.stage()` 移出同步解码主循环（通过 `OffloadedRequestPayload.from_request` 立即释放 VLM 媒体与 Prompt GPU 显存、并在 `asyncio.to_thread` 中异步写出后再回主事件循环线程发送 ZMQ 响应，`#8169`），`checkpointing.py` 在 `RerunState.NOT_RUNNING_YET` 空闲态下的严格分布式 Checkpoint 加载（`#7222`），以及 `volcengine/verl` 在 `offload_megatron_optimizer` 中跳过 `HybridDeviceOptimizer`（`HDO`）子优化器状态迁移以防止下一步训练设备不匹配崩溃（`#8286` / `#8237`）。
- **English**: Deterministic Python 3.12 standard-library model verifying `NVIDIA/Megatron-LM` `DynamicInferenceEngine` off-loop asynchronous payload staging (`#8169`), strict checkpoint loading with idle `rerun_state_machine` (`#7222`), and `volcengine/verl` `HybridDeviceOptimizer` (`HDO`) state placement preservation during RL offload (`#8286`/`#8237`).

## Run & Check
```bash
python experiments/megatron_payload_staging_verl_hdo_offload/run.py --check
```
